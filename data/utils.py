import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def read_scaled_wav(path, scaling_factor, downsample_8K=False, mono=True):
    samples, sr_orig = sf.read(path)

    if len(samples.shape) > 1 and mono:
        samples = samples[:, 0]

    if downsample_8K:
        samples = resample_poly(samples, 8000, sr_orig)
    samples *= scaling_factor
    return samples


def wavwrite_quantize(samples):
    return np.int16(np.round((2 ** 15) * samples))


def quantize(samples):
    int_samples = wavwrite_quantize(samples)
    return np.float64(int_samples) / (2 ** 15)


def wavwrite(file, samples, sr):
    """This is how the old Matlab function wavwrite() quantized to 16 bit.
    We match it here to maintain parity with the original dataset"""
    int_samples = wavwrite_quantize(samples)
    sf.write(file, int_samples, sr, subtype='PCM_16')


def append_or_truncate(s1_samples, s2_samples, noise_samples, min_or_max='max', start_samp_16k=0, downsample=False):
    if downsample:
        speech_start_sample = start_samp_16k // 2
    else:
        speech_start_sample = start_samp_16k

    if min_or_max == 'min':
        speech_end_sample = speech_start_sample + len(s1_samples)
        noise_samples = noise_samples[speech_start_sample:speech_end_sample]
    else:
        speech_end_sample = len(s1_samples) - speech_start_sample
        s1_append = np.zeros_like(noise_samples)
        s2_append = np.zeros_like(noise_samples)
        s1_append[speech_start_sample:len(s1_samples)] = s1_samples[0:speech_end_sample]
        s2_append[speech_start_sample:len(s1_samples)] = s2_samples[0:speech_end_sample]
        s1_samples = s1_append
        s2_samples = s2_append

    return s1_samples, s2_samples, noise_samples


def append_zeros(samples, desired_length):
    samples_to_add = desired_length - len(samples)
    if len(samples.shape) == 1:
        new_zeros = np.zeros(samples_to_add)
    elif len(samples.shape) == 2:
        new_zeros = np.zeros((samples_to_add, 2))
    return np.append(samples, new_zeros, axis=0)


def fix_length(s1, s2, min_or_max='max'):
    # Fix length
    if min_or_max == 'min':
        utt_len = np.minimum(len(s1), len(s2))
        s1 = s1[:utt_len]
        s2 = s2[:utt_len]
    else:  # max
        utt_len = np.maximum(len(s1), len(s2))
        s1 = append_zeros(s1, utt_len)
        s2 = append_zeros(s2, utt_len)
    return s1, s2


def create_wham_mixes(s1_samples, s2_samples, noise_samples):
    mix_clean = s1_samples + s2_samples
    mix_single = noise_samples + s1_samples
    mix_both = noise_samples + s1_samples + s2_samples
    return mix_clean, mix_single, mix_both