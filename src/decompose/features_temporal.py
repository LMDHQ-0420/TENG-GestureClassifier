"""src/decompose/features_temporal.py
时序特征提取：时间分段剖面 + IMF时序 + 包络特征

这三类特征专门捕获手势的时间动态信息：
- temporal_profile：把片段均分为4段，捕获能量随时间的演化
- imf_temporal：VMD各模态的能量重心、前后比例（区分sc双峰结构）
- envelope_features：Hilbert包络的峰值数量和间距（区分复合手势）

与 features_enhanced 的 232 维静态特征组合后达到 349 维。
"""

from typing import List

import numpy as np
from scipy.signal import find_peaks, hilbert
from scipy.stats import kurtosis, skew

from .vmd import vmd_decompose_3ch


def temporal_profile(segment_3ch: np.ndarray, n_seg: int = 4, fs: int = 1000) -> List[float]:
    """把片段均分为 n_seg 段，每段提取 RMS/MAV/Peak/WL，加上段间 RMS 变化率"""
    T = len(segment_3ch)
    feats = []
    for i in range(n_seg):
        s = int(i * T / n_seg); e = int((i + 1) * T / n_seg)
        chunk = segment_3ch[s:e]
        for ch in range(3):
            x = chunk[:, ch]
            feats += [
                float(np.mean(np.abs(x))),
                float(np.sqrt(np.mean(x ** 2))),
                float(np.max(np.abs(x))),
                float(np.sum(np.abs(np.diff(x))) / (len(x) / fs)),
            ]
    # 段间 RMS 变化率（反映能量演化趋势）
    rms_seg = []
    for i in range(n_seg):
        s = int(i * T / n_seg); e = int((i + 1) * T / n_seg)
        for ch in range(3):
            rms_seg.append(float(np.sqrt(np.mean(segment_3ch[s:e, ch] ** 2))))
    rms_arr = np.array(rms_seg).reshape(n_seg, 3)
    for ch in range(3):
        feats += np.diff(rms_arr[:, ch]).tolist()
    return feats


def imf_temporal_features(segment_3ch: np.ndarray) -> List[float]:
    """VMD 各 IMF 的时间能量分布特征：重心、前后比、时间方差"""
    imfs = vmd_decompose_3ch(segment_3ch)  # [3, K, T]
    feats = []
    for ch in range(imfs.shape[0]):
        for k in range(imfs.shape[1]):
            imf = imfs[ch, k]
            T = len(imf)
            t = np.arange(T)
            energy = imf ** 2
            total_e = energy.sum() + 1e-12
            centroid = float(np.sum(t * energy) / total_e / T)
            front_ratio = float(energy[:T // 2].sum() / total_e)
            t_norm = t / T
            e_norm = energy / total_e
            e_mean = np.sum(t_norm * e_norm)
            e_var = np.sum((t_norm - e_mean) ** 2 * e_norm)
            feats += [centroid, front_ratio, float(np.sqrt(e_var))]
    return feats


def envelope_features(segment_3ch: np.ndarray, fs: int = 1000) -> List[float]:
    """Hilbert 包络的峰值统计特征——专门捕获 sc 等复合手势的双峰结构"""
    feats = []
    for ch in range(segment_3ch.shape[1]):
        sig = segment_3ch[:, ch] - segment_3ch[:, ch].mean()
        env = np.abs(hilbert(sig))
        env_s = np.convolve(env, np.ones(50) / 50, mode='same')
        T = len(env_s)
        t = np.arange(T) / fs

        peaks, _ = find_peaks(env_s, height=env_s.mean() * 1.2, distance=100)
        n_peaks = len(peaks)
        mean_iv = float(np.diff(peaks / fs).mean()) if n_peaks >= 2 else 0.0
        std_iv = float(np.diff(peaks / fs).std()) if n_peaks >= 3 else 0.0

        total_e = np.sum(env_s ** 2) + 1e-12
        centroid = float(np.sum(t * (env_s ** 2)) / total_e)
        half = T // 2
        rise = float(env_s[half:].mean() - env_s[:half].mean())

        feats += [
            float(n_peaks), mean_iv, std_iv,
            centroid / max(t), rise,
            float(kurtosis(env_s)), float(skew(env_s)),
            float(env_s.max() / (env_s.mean() + 1e-8)),
        ]
    return feats


def extract_temporal_features(segment_3ch: np.ndarray, fs: int = 1000) -> np.ndarray:
    """合并三类时序特征 → shape [117]"""
    f1 = temporal_profile(segment_3ch, n_seg=4, fs=fs)
    f2 = imf_temporal_features(segment_3ch)
    f3 = envelope_features(segment_3ch, fs=fs)
    return np.nan_to_num(np.array(f1 + f2 + f3, dtype=float))


TEMPORAL_FEATURE_DIM = (4 * 3 * 4) + (4 * 3) + (3 * 4) + (3 * 8)  # 48+12+12+24 = 96+21 = 117
