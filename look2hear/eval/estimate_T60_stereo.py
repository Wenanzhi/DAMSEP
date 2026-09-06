from glob import glob
from tqdm import tqdm
import os
import torch
import numpy as np
from feature import load_wav, norm_amplitude
from jsonargparse import ArgumentParser
import json
import scipy
import matplotlib.pyplot as plt


def save_config_to_file(args, file_path):
    with open(file_path, "w") as json_file:
        json.dump(args.__dict__, json_file, indent=4)


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def ensure_stereo_rir(rir):
    """
    将输入统一整理成 [T, 2] 的双通道格式。
    兼容:
        [T, 2]
        [2, T]
        torch.Tensor
        np.ndarray
    """
    rir = to_numpy(rir)
    rir = np.squeeze(rir)

    if rir.ndim != 2:
        raise ValueError(f"期望输入是双通道音频，实际 shape={rir.shape}")

    if rir.shape[-1] == 2:
        # [T, 2]
        rir = rir
    elif rir.shape[0] == 2:
        # [2, T] -> [T, 2]
        rir = rir.T
    else:
        raise ValueError(f"无法识别双通道维度，实际 shape={rir.shape}")

    return rir.astype(np.float64)


def cal_T60_single_channel(rir, sr, savepath):
    """
    单通道 RT60 计算
    """
    rir = to_numpy(rir).astype(np.float64).squeeze()

    if rir.ndim != 1:
        raise ValueError(f"单通道 RT60 计算要求 1D 输入，实际 shape={rir.shape}")

    peak_idx = np.argmax(np.abs(rir))
    rir = rir[peak_idx:]  # 从主峰开始

    if rir.size < int(0.05 * sr) + 2:
        raise ValueError("RIR 太短，无法稳定估计 RT60。")

    power = np.abs(rir) ** 2
    total_power = power.sum()
    if total_power <= 0:
        raise ValueError("RIR 能量为 0，无法计算 RT60。")

    # Schroeder backward integration
    edc = np.cumsum(power[::-1])[::-1]
    eps = np.finfo(np.float64).eps
    EDC = 10 * np.log10(np.maximum(edc, eps) / max(total_power, eps))

    taxis = np.arange(0, EDC.shape[0], 1) / sr

    best_r = 1.0
    best_params = None

    beg_min = int(sr * 0.02)
    beg_max = min(int(sr * 0.05), EDC.shape[0] - 2)

    for beg_sample in range(beg_min, beg_max):
        end_sample = beg_sample + 1
        target_level = EDC[beg_sample] - 5

        while end_sample < EDC.shape[0] and EDC[end_sample] > target_level:
            end_sample += 1

        if end_sample - beg_sample < 2:
            continue

        line_fit = EDC[beg_sample:end_sample]
        taxis_fit = taxis[beg_sample:end_sample]
        k, b, r, _, _ = scipy.stats.linregress(taxis_fit, line_fit)

        if np.isnan(r) or np.isnan(k) or k >= 0:
            continue

        # 相关系数越接近 -1 越好
        if r < best_r:
            best_r = r
            best_params = (beg_sample, end_sample, k, b)

    if best_params is None:
        raise ValueError("没有找到合适的线性拟合区间，无法计算 RT60。")

    beg_sample_final, end_sample_final, k_final, b_final = best_params
    y = k_final * taxis[beg_sample_final:end_sample_final] + b_final

    # 修正：这里必须用 k_final，而不是原代码里的 k
    RT60 = -60.0 / k_final

    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    plt.figure()
    plt.plot(taxis, EDC)
    plt.plot(taxis[beg_sample_final:end_sample_final], y, label="fit")
    plt.xlim(0, min(0.3, taxis[-1] if len(taxis) > 0 else 0.3))
    plt.ylim(-40, 1)
    plt.legend()
    plt.title(f"RT60 = {RT60:.6f} s")
    plt.savefig(savepath)
    plt.close()

    return float(RT60)


def est_T60_stereo(rir_path: str, sr=16000):
    fpath_rir = sorted(
        glob(os.path.join(rir_path, "**", "*.flac"), recursive=True)
    ) + sorted(
        glob(os.path.join(rir_path, "**", "*.wav"), recursive=True)
    )

    per_file_results = {}
    failed_files = []

    total_valid_files = 0
    ch1_smaller_count = 0
    ch2_smaller_count = 0
    tie_count = 0

    for fpath_input_n in tqdm(fpath_rir):
        basename_input_n = os.path.basename(fpath_input_n)
        stem = os.path.splitext(basename_input_n)[0]

        fpath_out_edc_ch1 = os.path.join(rir_path, "edc", f"{stem}_ch1.png")
        fpath_out_edc_ch2 = os.path.join(rir_path, "edc", f"{stem}_ch2.png")

        try:
            rir = load_wav(fpath_input_n, sr)
            rir, _ = norm_amplitude(rir)
            rir = ensure_stereo_rir(rir)

            rir_ch1 = rir[:, 0]
            rir_ch2 = rir[:, 1]

            rt60_ch1 = cal_T60_single_channel(rir_ch1, sr, fpath_out_edc_ch1)
            rt60_ch2 = cal_T60_single_channel(rir_ch2, sr, fpath_out_edc_ch2)

            if np.isclose(rt60_ch1, rt60_ch2, atol=1e-8):
                smaller_channel = "equal"
                smaller_rt60 = float(rt60_ch1)
                tie_count += 1
            elif rt60_ch1 < rt60_ch2:
                smaller_channel = "channel_1"
                smaller_rt60 = float(rt60_ch1)
                ch1_smaller_count += 1
            else:
                smaller_channel = "channel_2"
                smaller_rt60 = float(rt60_ch2)
                ch2_smaller_count += 1

            per_file_results[basename_input_n] = {
                "channel_1_RT60": float(rt60_ch1),
                "channel_2_RT60": float(rt60_ch2),
                "smaller_channel": smaller_channel,
                "smaller_RT60": smaller_rt60,
            }

            total_valid_files += 1
            print(
                f"{basename_input_n} | "
                f"ch1 RT60={rt60_ch1:.6f}s | "
                f"ch2 RT60={rt60_ch2:.6f}s | "
                f"smaller={smaller_channel}"
            )

        except Exception as e:
            failed_files.append(
                {
                    "file": basename_input_n,
                    "error": str(e),
                }
            )
            print(f"[Failed] {basename_input_n}: {e}")

    # 第一通道 RT60 更小的占比
    # 定义为: channel_1_RT60 < channel_2_RT60 的文件数 / 有效文件总数
    ch1_smaller_ratio = (
        ch1_smaller_count / total_valid_files if total_valid_files > 0 else 0.0
    )

    output = {
        "per_file_results": per_file_results,
        "summary": {
            "total_valid_files": total_valid_files,
            "channel_1_smaller_count": ch1_smaller_count,
            "channel_2_smaller_count": ch2_smaller_count,
            "tie_count": tie_count,
            "channel_1_smaller_ratio": ch1_smaller_ratio,
            "channel_1_smaller_ratio_percent": ch1_smaller_ratio * 100.0,
            "ratio_definition": "channel_1_RT60 < channel_2_RT60 的文件数 / 有效文件总数"
        },
        "failed_files": failed_files,
    }

    with open(
        os.path.join(rir_path, "_T60_stereo_results.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

    print(
        f"\n第一通道 RT60 更小的占比: "
        f"{ch1_smaller_ratio:.6f} ({ch1_smaller_ratio * 100:.2f}%)"
    )


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    parser = ArgumentParser(
        description="Estimate stereo RT60 for two-channel RIRs."
    )
    parser.add_argument(
        "-i", "--input_path", required=True, type=str, help="input RIR directory path"
    )
    parser.add_argument(
        "--sr", default=16000, type=int, help="sample rate for loading audio"
    )

    args = parser.parse_args()
    est_T60_stereo(args["input_path"], sr=args["sr"])

    """
    usage:
        python look2hear/eval/estimate_T60_stereo.py -i [RIR dirpath] --sr 8000
    """