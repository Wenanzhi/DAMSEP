# @author: Pengyu Wang
# @email: wangpengyu@westlake.edu.cn
# @description: calculate RT60 and DRR given stereo RIRs.

from glob import glob
from tqdm import tqdm
import os
import torch
import numpy as np
from jsonargparse import ArgumentParser
import json
import scipy
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd


def save_config_to_file(args, file_path):
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(vars(args), json_file, indent=4, ensure_ascii=False)


def load_wav_stereo(fpath, target_sr=8000):
    """
    读取双通道音频，返回 shape = [2, T] 的 numpy.float32
    支持 wav / flac
    """
    wav, sr = sf.read(fpath, always_2d=True)   # [T, C]
    wav = wav.T                                # [C, T]

    if wav.shape[0] != 2:
        raise ValueError(f"{fpath} 不是双通道音频，实际通道数 = {wav.shape[0]}")

    if sr != target_sr:
        g = gcd(sr, target_sr)
        up = target_sr // g
        down = sr // g
        wav = np.stack(
            [resample_poly(wav[ch], up, down) for ch in range(wav.shape[0])],
            axis=0
        )

    return wav.astype(np.float32)


def cal_T60(RIR, sr, savepath=None):
    """
    对单通道 RIR 计算 RT60
    """
    eps = 1e-12

    RIR = np.asarray(RIR).astype(np.float64)
    if RIR.ndim != 1:
        raise ValueError("cal_T60 输入必须是单通道一维数组")

    peak_idx = np.argmax(np.abs(RIR))
    RIR = RIR[peak_idx:]

    if len(RIR) < 10:
        raise ValueError("RIR 太短，无法估计 RT60")

    power = np.abs(RIR) ** 2
    total_power = np.sum(power)

    if total_power < eps:
        raise ValueError("RIR 能量过小，无法估计 RT60")

    # Energy Decay Curve
    edc = np.cumsum(power[::-1])[::-1]
    EDC = 10 * np.log10(np.maximum(edc / total_power, eps))
    taxis = np.arange(EDC.shape[0]) / sr

    # 原代码是在 20ms~50ms 之间找起点，然后拟合下降 5dB 的区间
    beg_start = int(sr * 0.02)
    beg_stop = int(sr * 0.05)

    beg_start = min(beg_start, len(EDC) - 3)
    beg_stop = min(beg_stop, len(EDC) - 2)

    if beg_start >= beg_stop:
        raise ValueError("RIR 长度不足以在 20ms~50ms 范围内拟合 RT60")

    r_final = np.inf
    k_final = None
    b_final = None
    beg_sample_final = None
    end_sample_final = None

    for beg_sample in range(beg_start, beg_stop):
        end_sample = beg_sample

        # 防止越界
        while (
            end_sample + 1 < len(EDC)
            and EDC[end_sample] > EDC[beg_sample] - 5
        ):
            end_sample += 1

        if end_sample - beg_sample < 2:
            continue

        line_fit = EDC[beg_sample:end_sample]
        taxis_fit = taxis[beg_sample:end_sample]

        k, b, r, _, _ = scipy.stats.linregress(taxis_fit, line_fit)

        # 只接受衰减斜率
        if k >= 0:
            continue

        # 选择相关系数更接近 -1 的拟合
        if r < r_final:
            beg_sample_final = beg_sample
            end_sample_final = end_sample
            r_final = r
            k_final = k
            b_final = b

    if k_final is None:
        raise ValueError("未找到有效的 RT60 拟合区间")

    y = k_final * taxis[beg_sample_final:end_sample_final] + b_final

    # 修复原代码的 bug：这里应该是 k_final，不是最后一次循环的 k
    RT60 = -60.0 / k_final

    if savepath is not None:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.figure()
        plt.plot(taxis, EDC, label="EDC")
        plt.plot(taxis[beg_sample_final:end_sample_final], y, label="fit")
        plt.xlim(0, min(0.3, taxis[-1] if len(taxis) > 0 else 0.3))
        plt.ylim(-40, 1)
        plt.legend()
        plt.title(f"RT60 = {RT60:.6f} s")
        plt.savefig(savepath)
        plt.close()

    return float(RT60)


def cal_DRR(rir, sr):
    """
    对单通道 RIR 计算 DRR
    直达声窗口：峰值点前后各 2.5 ms
    """
    eps = 1e-12

    rir = np.asarray(rir).astype(np.float64)
    if rir.ndim != 1:
        raise ValueError("cal_DRR 输入必须是单通道一维数组")

    peak_idx = int(np.argmax(np.abs(rir)))

    dp_start_sample = int(max(peak_idx - 0.0025 * sr, 0))
    dp_stop_sample = int(min(peak_idx + 0.0025 * sr, len(rir)))

    rir_power = np.sum(rir ** 2)
    dp_power = np.sum(rir[dp_start_sample:dp_stop_sample] ** 2)
    rev_power = rir_power - dp_power

    rev_power = max(rev_power, eps)
    dp_power = max(dp_power, eps)

    drr = 10.0 * np.log10(dp_power / rev_power)
    return float(drr)


def compare_two_values(v1, v2, tol=1e-8):
    if abs(v1 - v2) <= tol:
        return "equal"
    elif v1 < v2:
        return "ch1"
    else:
        return "ch2"


def est_T60(rir_path: str, sr=16000):
    fpath_rir = sorted(
        glob(os.path.join(rir_path, "**", "*.flac"), recursive=True)
    ) + sorted(
        glob(os.path.join(rir_path, "**", "*.wav"), recursive=True)
    )

    T60_log = {}
    DRR_log = {}

    ch1_smaller_count = 0
    ch2_smaller_count = 0
    equal_count = 0
    total_valid_files = 0

    for fpath_input_n in tqdm(fpath_rir):
        basename_input_n = os.path.basename(fpath_input_n)

        try:
            rir_stereo = load_wav_stereo(fpath_input_n, sr)  # [2, T]
            rir_ch1 = rir_stereo[0]
            rir_ch2 = rir_stereo[1]

            # EDC 图分别保存
            fpath_out_edc_ch1 = os.path.join(
                rir_path, "edc", "ch1", basename_input_n.split(".")[0] + ".png"
            )
            fpath_out_edc_ch2 = os.path.join(
                rir_path, "edc", "ch2", basename_input_n.split(".")[0] + ".png"
            )

            # RT60
            T60_ch1 = cal_T60(rir_ch1, sr, fpath_out_edc_ch1)
            T60_ch2 = cal_T60(rir_ch2, sr, fpath_out_edc_ch2)

            T60_log[basename_input_n] = {
                "ch1_RT60": T60_ch1,
                "ch2_RT60": T60_ch2,
            }

            # DRR
            DRR_ch1 = cal_DRR(rir_ch1, sr)
            DRR_ch2 = cal_DRR(rir_ch2, sr)

            smaller_channel = compare_two_values(DRR_ch1, DRR_ch2)

            if smaller_channel == "ch1":
                ch1_smaller_count += 1
            elif smaller_channel == "ch2":
                ch2_smaller_count += 1
            else:
                equal_count += 1

            total_valid_files += 1

            DRR_log[basename_input_n] = {
                "ch1_DRR": DRR_ch1,
                "ch2_DRR": DRR_ch2,
                "smaller_channel": smaller_channel,
                "mark": f"{smaller_channel} has smaller DRR" if smaller_channel != "equal" else "equal",
            }

            print(
                f"{basename_input_n} | "
                f"T60_ch1={T60_ch1:.6f}, T60_ch2={T60_ch2:.6f}, "
                f"DRR_ch1={DRR_ch1:.6f}, DRR_ch2={DRR_ch2:.6f}, "
                f"smaller={smaller_channel}"
            )

        except Exception as e:
            print(f"[ERROR] {basename_input_n}: {e}")
            DRR_log[basename_input_n] = {
                "error": str(e)
            }
            T60_log[basename_input_n] = {
                "error": str(e)
            }

    if total_valid_files > 0:
        ch1_smaller_ratio = ch1_smaller_count / total_valid_files
    else:
        ch1_smaller_ratio = 0.0

    # 把统计信息也写入 DRR 输出文件
    DRR_output = {
        "files": DRR_log,
        "summary": {
            "total_valid_files": total_valid_files,
            "ch1_smaller_count": ch1_smaller_count,
            "ch2_smaller_count": ch2_smaller_count,
            "equal_count": equal_count,
            "ch1_smaller_ratio": ch1_smaller_ratio,
            "ch1_smaller_ratio_percent": ch1_smaller_ratio * 100.0,
        },
    }

    with open(
        os.path.join(rir_path, "_T60s.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(T60_log, f, ensure_ascii=False, indent=4)

    with open(
        os.path.join(rir_path, "_DRRs.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(DRR_output, f, ensure_ascii=False, indent=4)

    print("\n===== Summary =====")
    print(f"Total valid files      : {total_valid_files}")
    print(f"Channel 1 smaller count: {ch1_smaller_count}")
    print(f"Channel 2 smaller count: {ch2_smaller_count}")
    print(f"Equal count            : {equal_count}")
    print(f"Channel 1 smaller ratio: {ch1_smaller_ratio:.6f} ({ch1_smaller_ratio * 100:.2f}%)")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    parser = ArgumentParser(
        description="code for generating wavs by PengyuWang @ Westlake University"
    )
    parser.add_argument(
        "-i", "--input_path", required=True, type=str, help="input path"
    )
    parser.add_argument(
        "--sr", default=8000, type=int, help="target sample rate"
    )

    args = parser.parse_args()

    est_T60(args.input_path, sr=args.sr)

    """
    usage:
    python estimate_T60_DRR.py -i [RIR dirpath]
    python estimate_T60_DRR.py -i [RIR dirpath] --sr 16000
    """