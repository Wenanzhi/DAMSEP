import argparse
import json
import os
from math import gcd

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

from constants import SAMPLERATE
from utils import read_scaled_wav, quantize, fix_length, create_wham_mixes
from wham_room import WhamRoom
from tqdm.auto import tqdm


FILELIST_STUB = os.path.join('prepared', 'metadata', 'mix_2_spk_filenames_{}.csv')

SINGLE_DIR = 'mix_single'
BOTH_DIR = 'mix_both'
CLEAN_DIR = 'mix_clean'
S1_DIR = 's1'
S2_DIR = 's2'
RIR_DIR = 'rir'
SUFFIXES = ['_anechoic', '_reverb']

MONO = True  # Generate mono audio, change to false for stereo audio
SPLITS = ['tt']
SAMPLE_RATES = ['16k', '8k'] # Remove element from this list to generate less data
DATA_LEN = ['max', 'min'] # Remove element from this list to generate less data


def _as_mono(x):
    if x.ndim == 1:
        return x.astype(np.float64)
    return x.mean(axis=1).astype(np.float64)


def _resample_to(x, fs_in, fs_out):
    if fs_in == fs_out:
        return x.astype(np.float64, copy=False)
    g = gcd(int(fs_in), int(fs_out))
    up = fs_out // g
    down = fs_in // g
    return resample_poly(x, up=up, down=down).astype(np.float64)


def _align_minmax(a, b, mode):
    if mode == 'max':
        max_len = max(len(a), len(b))
        if len(a) < max_len:
            a = np.pad(a, (0, max_len - len(a)))
        if len(b) < max_len:
            b = np.pad(b, (0, max_len - len(b)))
        return a, b
    min_len = min(len(a), len(b))
    return a[:min_len], b[:min_len]


def _mix_scaling(s1, s2, peak=0.9):
    mix = s1 + s2
    if len(mix) == 0:
        return 1.0
    max_amp = float(np.max(np.abs(np.concatenate([mix, s1, s2]))))
    return (peak / max_amp) if max_amp > 0 else 1.0

def _pack_rir_to_multich(rir_list, mono=True, left_mic=0):
    """
    rir_list: room.rir_anechoic 或 room.rir_reverberant
              结构: rir_list[mic][source] -> 1D np.array
    返回: (n_samples, n_channels) 适合直接 sf.write
          MONO=True  -> 2通道(2个source, 只取left mic)
          MONO=False -> 4通道(2个source x 2个mic)
    """
    if mono:
        mic_ids = [left_mic]
    else:
        mic_ids = list(range(len(rir_list)))

    chans = []
    for m in mic_ids:
        for s in range(len(rir_list[m])):
            chans.append(np.asarray(rir_list[m][s], dtype=np.float64))

    # 保险：对齐长度（正常情况下 length 都一样）
    L = max(len(c) for c in chans) if chans else 0
    chans = [np.pad(c, (0, L - len(c))) for c in chans]

    return np.stack(chans, axis=1) if chans else np.zeros((0, 1), dtype=np.float64)

# def _parse_mix_gains(output_name):
#     stem = os.path.splitext(output_name)[0]
#     parts = stem.rsplit('_', 3)
#     if len(parts) != 4:
#         raise ValueError(f"Cannot parse gains from output filename: {output_name}")
#     import pdb;pdb.set_trace()
#     return float(parts[1]), float(parts[3])

def _parse_mix_gains(output_name):
    """
    Robustly parse snr1_db, snr2_db from filenames like:
      <anything>_<snr1>_<anything>_<snr2>.wav
    where <anything> may contain underscores.

    We assume snr values are floats (typically with decimal point), e.g. 1.42427, -1.42427.
    """
    stem = os.path.splitext(output_name)[0]

    # 1) 先尝试原来的 fast-path（兼容旧命名：id2 不含 '_'）
    parts = stem.rsplit('_', 3)
    if len(parts) == 4:
        try:
            return float(parts[1]), float(parts[3])
        except ValueError:
            pass  # fallback to robust parsing

    # 2) robust：gain2 一定是最后一个 '_' 后的 token
    head, sep, snr2_str = stem.rpartition('_')
    if not sep:
        raise ValueError(f"Cannot parse gains from output filename: {output_name}")
    try:
        snr2_db = float(snr2_str)
    except ValueError as e:
        raise ValueError(f"Cannot parse snr2 from output filename: {output_name}") from e

    # 3) robust：在 head 里从右往左找“像 snr1 的浮点数 token”
    #    为避免把 dialog_2 这种 '2' 误当成 snr1，这里优先找包含 '.' / 'e' 的 token
    tokens = head.split('_')

    snr1_db = None
    for tok in reversed(tokens):
        if ('.' in tok) or ('e' in tok) or ('E' in tok):
            try:
                snr1_db = float(tok)
                break
            except ValueError:
                continue

    if snr1_db is None:
        raise ValueError(f"Cannot parse snr1 from output filename: {output_name}")
    return snr1_db, snr2_db


def _load_lev_cache(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _lookup_lev(cache, path):
    if path in cache:
        return float(cache[path])
    # import pdb;pdb.set_trace()
    path_norm = path.replace("\\", "/")
    if path_norm in cache:
        return float(cache[path_norm])
    raise KeyError(f"Level cache missing entry for: {path}")


def _compute_scaling_factors(s1_path, s2_path, output_name, lev_cache, sample_rates, data_lens):
    snr1_db, snr2_db = _parse_mix_gains(output_name)
    w1 = 10.0 ** (snr1_db / 20.0)
    w2 = 10.0 ** (snr2_db / 20.0)

    lev1 = _lookup_lev(lev_cache, s1_path)
    lev2 = _lookup_lev(lev_cache, s2_path)
    slev1 = np.sqrt(lev1) if lev1 > 0 else 1.0
    slev2 = np.sqrt(lev2) if lev2 > 0 else 1.0

    s1_raw, fs1 = sf.read(s1_path, always_2d=False)
    s2_raw, fs2 = sf.read(s2_path, always_2d=False)
    s1_raw = _as_mono(np.asarray(s1_raw))
    s2_raw = _as_mono(np.asarray(s2_raw))
    if fs1 != fs2:
        raise ValueError(f"Sample rate mismatch: {s1_path} ({fs1}) vs {s2_path} ({fs2})")
    fs1 = int(fs1)

    scales = {}
    if '8k' in sample_rates:
        s1_8 = _resample_to(s1_raw, fs1, 8000)
        s2_8 = _resample_to(s2_raw, fs1, 8000)
        s1_8 = (w1 / slev1) * s1_8
        s2_8 = (w2 / slev2) * s2_8
        for datalen in data_lens:
            a1, a2 = _align_minmax(s1_8, s2_8, datalen)
            mix_scale = _mix_scaling(a1, a2)
            scales[('8k', datalen)] = (w1 * mix_scale / slev1, w2 * mix_scale / slev2)

    if '16k' in sample_rates:
        s1_16 = _resample_to(s1_raw, fs1, SAMPLERATE)
        s2_16 = _resample_to(s2_raw, fs1, SAMPLERATE)
        s1_16 = (w1 / slev1) * s1_16
        s2_16 = (w2 / slev2) * s2_16
        for datalen in data_lens:
            a1, a2 = _align_minmax(s1_16, s2_16, datalen)
            mix_scale = _mix_scaling(a1, a2)
            scales[('16k', datalen)] = (w1 * mix_scale / slev1, w2 * mix_scale / slev2)

    return scales


def create_wham(output_root, lev_cache_path):
    LEFT_CH_IND = 0
    if MONO:
        ch_ind = LEFT_CH_IND
    else:
        ch_ind = [0, 1]

    reverb_param_stub = os.path.join('prepared', 'metadata', 'reverb_params_{}.csv')
    lev_cache = _load_lev_cache(lev_cache_path)

    for splt in SPLITS:

        wsjmix_path = FILELIST_STUB.format(splt)
        wsjmix_df = pd.read_csv(wsjmix_path)
        wsjmix_map = {os.path.basename(row.output_filename): (row.s1_path, row.s2_path)
                      for row in wsjmix_df.itertuples(index=False)}

        reverb_param_path = reverb_param_stub.format(splt)
        reverb_param_df = pd.read_csv(reverb_param_path)

        for wav_dir in ['wav' + sr for sr in SAMPLE_RATES]:
            for datalen_dir in DATA_LEN:
                output_path = os.path.join(output_root, wav_dir, datalen_dir, splt)
                for sfx in SUFFIXES:
                    os.makedirs(os.path.join(output_path, CLEAN_DIR+sfx), exist_ok=True)
                    os.makedirs(os.path.join(output_path, SINGLE_DIR+sfx), exist_ok=True)
                    os.makedirs(os.path.join(output_path, BOTH_DIR+sfx), exist_ok=True)
                    os.makedirs(os.path.join(output_path, S1_DIR+sfx), exist_ok=True)
                    os.makedirs(os.path.join(output_path, S2_DIR+sfx), exist_ok=True)
                    os.makedirs(os.path.join(output_root, wav_dir, RIR_DIR + sfx, splt), exist_ok=True)
        utt_ids = reverb_param_df['utterance_id'].map(os.path.basename).to_numpy()

        # for i_utt, output_name in enumerate(utt_ids):
        for i_utt, output_name in tqdm(
            enumerate(utt_ids),
            total=len(utt_ids),
            desc=f"{splt} utterances",
            unit="utt"
        ):

            utt_row = reverb_param_df.iloc[i_utt]
            room = WhamRoom([utt_row['room_x'], utt_row['room_y'], utt_row['room_z']],
                            [[utt_row['micL_x'], utt_row['micL_y'], utt_row['mic_z']],
                             [utt_row['micR_x'], utt_row['micR_y'], utt_row['mic_z']]],
                            [utt_row['s1_x'], utt_row['s1_y'], utt_row['s1_z']],
                            [utt_row['s2_x'], utt_row['s2_y'], utt_row['s2_z']],
                            utt_row['T60'])
            room.generate_rirs()

            if output_name not in wsjmix_map:
                raise KeyError(f"Missing mix entry for {output_name} in {wsjmix_path}")
            s1_path, s2_path = wsjmix_map[output_name]
            # import pdb;pdb.set_trace()
            scaling_factors = _compute_scaling_factors(
                s1_path, s2_path, output_name, lev_cache, SAMPLE_RATES, DATA_LEN
            )

            # read the 16kHz unscaled speech files, but make sure to add all 'max' padding to end of utterances
            # for synthesizing all the reverb tails
            s1_temp = quantize(read_scaled_wav(s1_path, 1))
            s2_temp = quantize(read_scaled_wav(s2_path, 1))
            s1_temp, s2_temp = fix_length(s1_temp, s2_temp, 'max')

            room.add_audio(s1_temp, s2_temp)

            anechoic = room.generate_audio(anechoic=True, fs=SAMPLE_RATES)
            reverberant = room.generate_audio(fs=SAMPLE_RATES)

            # room.generate_rirs() 之后，你也可以先打包一次（16k基准）
            rir_ane_16k = _pack_rir_to_multich(room.rir_anechoic, mono=MONO, left_mic=LEFT_CH_IND)
            rir_rev_16k = _pack_rir_to_multich(room.rir_reverberant, mono=MONO, left_mic=LEFT_CH_IND)

            for sr_i, sr_dir in enumerate(SAMPLE_RATES):
                wav_dir = 'wav' + sr_dir
                if sr_dir == '8k':
                    sr = 8000
                    rir_ane = resample_poly(rir_ane_16k, 8000, SAMPLERATE, axis=0)
                    rir_rev = resample_poly(rir_rev_16k, 8000, SAMPLERATE, axis=0)
                    downsample = True
                else:
                    sr = SAMPLERATE
                    rir_ane = rir_ane_16k
                    rir_rev = rir_rev_16k
                    downsample = False
                # 可选：防止极端情况下 float 写盘裁剪（通常不需要）
                def _safe_norm(x):
                    m = float(np.max(np.abs(x))) if x.size else 0.0
                    return x / (m + 1e-12) if m > 1.0 else x

                rir_ane = _safe_norm(rir_ane)
                rir_rev = _safe_norm(rir_rev)

                # 写 RIR wav（文件名沿用 output_name；目录隔离，不会和音频冲突）
                sf.write(os.path.join(output_root, wav_dir, RIR_DIR + '_anechoic', splt, output_name),
                        rir_ane, sr, subtype='FLOAT')
                sf.write(os.path.join(output_root, wav_dir, RIR_DIR + '_reverb', splt, output_name),
                        rir_rev, sr, subtype='FLOAT')

                for datalen_dir in DATA_LEN:
                    output_path = os.path.join(output_root, wav_dir, datalen_dir, splt)

                    s1_scale, s2_scale = scaling_factors[(sr_dir, datalen_dir)]
                    s1 = read_scaled_wav(s1_path, s1_scale, downsample)
                    s1 = quantize(s1)
                    s2 = read_scaled_wav(s2_path, s2_scale, downsample)
                    s2 = quantize(s2)

                    # Make relative source energy of anechoic sources same with original in mono (left channel) case
                    s1_spatial_scaling = np.sqrt(np.sum(s1 ** 2) / np.sum(anechoic[sr_i][0, LEFT_CH_IND, :] ** 2))
                    s2_spatial_scaling = np.sqrt(np.sum(s2 ** 2) / np.sum(anechoic[sr_i][1, LEFT_CH_IND, :] ** 2))

                    if datalen_dir == 'max':
                        out_len = max(len(s1), len(s2))
                    else:
                        out_len = min(len(s1), len(s2))

                    s1_anechoic, s2_anechoic = fix_length(anechoic[sr_i][0, ch_ind, :out_len].T * s1_spatial_scaling,
                                                          anechoic[sr_i][1, ch_ind, :out_len].T * s2_spatial_scaling,
                                                          datalen_dir)
                    s1_reverb, s2_reverb = fix_length(reverberant[sr_i][0, ch_ind, :out_len].T * s1_spatial_scaling,
                                                      reverberant[sr_i][1, ch_ind, :out_len].T * s2_spatial_scaling,
                                                      datalen_dir)

                    sources = [(s1_anechoic, s2_anechoic), (s1_reverb, s2_reverb)]
                    for sfx, source_pair in zip(SUFFIXES, sources):
                        s1_samples, s2_samples = source_pair
                        noise_samples = np.zeros_like(s1_samples)
                        mix_clean, mix_single, mix_both = create_wham_mixes(s1_samples, s2_samples, noise_samples)

                        # write audio
                        samps = [mix_clean, mix_single, mix_both, s1_samples, s2_samples]
                        dirs = [CLEAN_DIR, SINGLE_DIR, BOTH_DIR, S1_DIR, S2_DIR]
                        for dir, samp in zip(dirs, samps):
                            sf.write(os.path.join(output_path, dir+sfx, output_name), samp,
                                     sr, subtype='FLOAT')

            if (i_utt + 1) % 500 == 0:
                print('Completed {} of {} utterances'.format(i_utt + 1, len(utt_ids)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for writing wsj0-2mix 8 k Hz and 16 kHz datasets.')
    parser.add_argument('--lev-cache', type=str, required=True,
                        help='Path to JSON cache with precomputed active levels (abs_path -> lev_power).')
    args = parser.parse_args()
    create_wham(args.output_dir, args.lev_cache)

# Run from the repository root after preparing the metadata files expected by
# FILELIST_STUB and create_wham().
# python data/create_wham_from_scratch.py \
#   --output-dir data/hetmixr \
#   --lev-cache outputs/active_levels_test.json
