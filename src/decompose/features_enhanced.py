"""增强多域特征提取（路线A 的扩展版）

在 features_rich 的基础上增加：
- Hjorth 参数（mobility, complexity）
- Shannon 能量熵
- Hilbert 包络统计
- 小波包子带能量（频率细分）
- 整通道时域统计（峰值因子等）
- 通道间相关系数
- 片段时长

实测单片段准确率约 80%，是当前可解释特征工程的上限。
"""

from typing import List

import numpy as np
import pywt
from scipy.stats import kurtosis, skew
from scipy.signal import hilbert

from .vmd import vmd_decompose_3ch, VMD_PARAMS

CH_NAMES = ["CH1", "CH3", "CH5"]

# 每个 IMF 的 15 个特征
_IMF_FEATURES = [
    "MAV", "RMS", "WL", "ZCR", "KURT", "SKEW",
    "FC", "BW", "PF", "HMOB", "HCOMP", "ENT", "ENERGY", "ENVM", "ENVS",
]
# 整通道原始信号的 7 个统计特征
_RAW_FEATURES = ["MAV", "RMS", "KURT", "SKEW", "CREST", "HMOB", "HCOMP"]
# 小波包层数与子带数
_WPD_LEVEL = 3
_WPD_BANDS = 2 ** _WPD_LEVEL  # 8


def _hjorth(x: np.ndarray):
    dx = np.diff(x)
    ddx = np.diff(dx)
    v0 = np.var(x) + 1e-12
    v1 = np.var(dx) + 1e-12
    v2 = np.var(ddx) + 1e-12
    mob = np.sqrt(v1 / v0)
    comp = np.sqrt(v2 / v1) / (mob + 1e-12)
    return float(mob), float(comp)


def _shannon_energy(x: np.ndarray) -> float:
    p = x ** 2
    p = p / (p.sum() + 1e-12)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def _imf_features(imf: np.ndarray, fs: int = 1000) -> List[float]:
    n = len(imf)
    if n < 4:
        return [0.0] * 15
    mav = float(np.mean(np.abs(imf)))
    rms = float(np.sqrt(np.mean(imf ** 2)))
    wl = float(np.sum(np.abs(np.diff(imf))) / (n / fs))
    zcr = float(np.sum(np.abs(np.diff(np.sign(imf))) > 0) / n)
    kt = float(kurtosis(imf)) if np.std(imf) > 0 else 0.0
    sk = float(skew(imf)) if np.std(imf) > 0 else 0.0
    spectrum = np.abs(np.fft.rfft(imf))
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    ss = spectrum.sum() + 1e-12
    fc = float(np.sum(freqs * spectrum) / ss)
    bw = float(np.sqrt(np.sum((freqs - fc) ** 2 * spectrum) / ss))
    pf = float(freqs[np.argmax(spectrum)])
    mob, comp = _hjorth(imf)
    ent = _shannon_energy(imf)
    energy = float(np.sum(imf ** 2))
    env = np.abs(hilbert(imf))
    return [mav, rms, wl, zcr, kt, sk, fc, bw, pf, mob, comp, ent, energy,
            float(env.mean()), float(env.std())]


def _wpd_energy(sig: np.ndarray, wavelet: str = "db4", level: int = _WPD_LEVEL) -> List[float]:
    wp = pywt.WaveletPacket(sig, wavelet, maxlevel=level)
    nodes = [n.path for n in wp.get_level(level, "natural")]
    e = np.array([np.sum(wp[p].data ** 2) for p in nodes])
    e = e / (e.sum() + 1e-12)
    return e.tolist()


def _raw_channel_features(x: np.ndarray) -> List[float]:
    rms = float(np.sqrt(np.mean(x ** 2)))
    peak = float(np.max(np.abs(x)))
    kt = float(kurtosis(x)) if np.std(x) > 0 else 0.0
    sk = float(skew(x)) if np.std(x) > 0 else 0.0
    crest = peak / (rms + 1e-12)
    mob, comp = _hjorth(x)
    return [float(np.mean(np.abs(x))), rms, kt, sk, crest, mob, comp]


def build_enhanced_feature_names(n_imf: int = None) -> List[str]:
    if n_imf is None:
        n_imf = VMD_PARAMS["K"]
    names = []
    # IMF 特征
    for ch in CH_NAMES:
        for k in range(n_imf):
            for ft in _IMF_FEATURES:
                names.append(f"{ch}_IMF{k+1}_{ft}")
    # 小波包子带能量
    for ch in CH_NAMES:
        for b in range(_WPD_BANDS):
            names.append(f"{ch}_WPD{b+1}")
    # 整通道时域统计
    for ch in CH_NAMES:
        for ft in _RAW_FEATURES:
            names.append(f"{ch}_RAW_{ft}")
    # 通道间相关
    names += ["CORR_13", "CORR_15", "CORR_35"]
    # 通道间 RMS 占比
    names += [f"Ratio_{ch}" for ch in CH_NAMES]
    # 片段时长
    names += ["DURATION_S"]
    return names


ENHANCED_FEATURE_NAMES = build_enhanced_feature_names()


def extract_enhanced_features(segment_3ch: np.ndarray, fs: int = 1000,
                              vmd_params: dict = None) -> np.ndarray:
    """提取增强多域特征向量

    Parameters
    ----------
    segment_3ch : shape [T, 3]，清洗后的三通道信号
    fs : 采样率

    Returns
    -------
    np.ndarray, 长度 = len(ENHANCED_FEATURE_NAMES)
    """
    if vmd_params is None:
        vmd_params = VMD_PARAMS

    f = []
    imfs = vmd_decompose_3ch(segment_3ch, vmd_params)  # [3, K, T]
    n_ch, n_imf, _ = imfs.shape

    for ch in range(n_ch):
        for k in range(n_imf):
            f.extend(_imf_features(imfs[ch, k], fs))
    for ch in range(n_ch):
        f.extend(_wpd_energy(segment_3ch[:, ch]))
    for ch in range(n_ch):
        f.extend(_raw_channel_features(segment_3ch[:, ch]))

    c = segment_3ch - segment_3ch.mean(axis=0)
    f += [
        float(np.corrcoef(c[:, 0], c[:, 1])[0, 1]),
        float(np.corrcoef(c[:, 0], c[:, 2])[0, 1]),
        float(np.corrcoef(c[:, 1], c[:, 2])[0, 1]),
    ]
    rms = np.sqrt(np.mean(segment_3ch ** 2, axis=0))
    ratio = rms / (rms.sum() + 1e-12)
    f += ratio.tolist()
    f += [len(segment_3ch) / fs]

    return np.nan_to_num(np.array(f, dtype=float))
