"""多域特征提取 — 从 VMD 分解的各 IMF 提取时域+频域特征

对每个通道的每个 IMF 提取 9 个特征（6 时域 + 3 频域），
3 通道 × 4 IMF × 9 = 108 维，加上 3 维通道间 RMS 占比 = 111 维。
相比原始 9 维特征，多域分解特征能更细致地刻画不同频率成分的差异。
"""

from typing import List

import numpy as np
from scipy.stats import kurtosis, skew

from .vmd import vmd_decompose_3ch, VMD_PARAMS

# 每个 IMF 的特征名
_IMF_FEATURE_TYPES = [
    "MAV",      # 平均绝对值
    "RMS",      # 均方根
    "WL",       # 波形长度
    "ZCR",      # 过零率
    "KURT",     # 峰度
    "SKEW",     # 偏度
    "FC",       # 质心频率
    "BW",       # 频带带宽
    "PF",       # 峰值频率
]

CH_NAMES = ["CH1", "CH3", "CH5"]


def _build_feature_names(n_imf: int) -> List[str]:
    names = []
    for ch in CH_NAMES:
        for k in range(n_imf):
            for ft in _IMF_FEATURE_TYPES:
                names.append(f"{ch}_IMF{k+1}_{ft}")
    # 通道间 RMS 占比
    names += [f"Ratio_{ch}" for ch in CH_NAMES]
    return names


# K=4 时的完整特征名（111 维）
RICH_FEATURE_NAMES = _build_feature_names(VMD_PARAMS["K"])


def _imf_features(imf: np.ndarray, fs: int = 1000) -> List[float]:
    """对单个 IMF 提取 9 个特征"""
    n = len(imf)
    if n < 2:
        return [0.0] * 9

    # --- 时域 ---
    mav = float(np.mean(np.abs(imf)))
    rms = float(np.sqrt(np.mean(imf ** 2)))
    wl = float(np.sum(np.abs(np.diff(imf))) / (n / fs))
    zcr = float(np.sum(np.abs(np.diff(np.sign(imf))) > 0) / n)
    kurt = float(kurtosis(imf)) if np.std(imf) > 0 else 0.0
    sk = float(skew(imf)) if np.std(imf) > 0 else 0.0

    # --- 频域 ---
    spectrum = np.abs(np.fft.rfft(imf))
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    spec_sum = np.sum(spectrum)
    if spec_sum > 0:
        fc = float(np.sum(freqs * spectrum) / spec_sum)          # 质心频率
        bw = float(np.sqrt(np.sum(((freqs - fc) ** 2) * spectrum) / spec_sum))  # 带宽
        pf = float(freqs[np.argmax(spectrum)])                   # 峰值频率
    else:
        fc, bw, pf = 0.0, 0.0, 0.0

    return [mav, rms, wl, zcr, kurt, sk, fc, bw, pf]


def extract_rich_features(
    segment_3ch: np.ndarray, fs: int = 1000, vmd_params: dict = None
) -> np.ndarray:
    """对一个清洗后片段提取 111 维多域特征

    Parameters
    ----------
    segment_3ch : shape [T, 3]，清洗后的三通道信号
    fs : 采样率
    vmd_params : VMD 参数

    Returns
    -------
    np.ndarray, shape [111]
    """
    if vmd_params is None:
        vmd_params = VMD_PARAMS

    # VMD 分解：[3, K, T']
    imfs = vmd_decompose_3ch(segment_3ch, vmd_params)
    n_ch, n_imf, _ = imfs.shape

    features = []
    for ch in range(n_ch):
        for k in range(n_imf):
            features.extend(_imf_features(imfs[ch, k], fs))

    # 通道间 RMS 占比（空间分布）
    rms = np.sqrt(np.mean(segment_3ch ** 2, axis=0))
    total = np.sum(rms)
    ratio = rms / total if total > 0 else np.zeros(n_ch)
    features.extend(ratio.tolist())

    return np.array(features, dtype=float)
