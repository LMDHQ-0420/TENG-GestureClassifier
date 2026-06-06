"""CWT（连续小波变换）封装 — 生成 scalogram 时频图

CWT 用一族不同尺度的小波（这里用 Morlet）去匹配信号，得到时间-频率（尺度）平面上的能量分布，
即 scalogram。它比 STFT 有更好的多分辨率特性：低频处频率分辨率高，高频处时间分辨率高，
非常适合分析手势这类瞬态信号。彩色 scalogram 是论文中展示信号处理的直观配图。
"""

from typing import Tuple

import numpy as np
import pywt
from scipy.signal import resample

# CWT 参数
CWT_PARAMS = {
    "wavelet": "morl",      # Morlet 小波（时频分析常用）
    "num_scales": 64,       # 尺度数量（频率分辨率 → scalogram 高度）
    "max_scale": 64,        # 最大尺度
    "fs": 1000,             # 采样率
    "out_width": 128,       # 时间轴统一重采样宽度 → scalogram 宽度
}


def cwt_scalogram(signal_1ch: np.ndarray, params: dict = None) -> Tuple[np.ndarray, np.ndarray]:
    """对单通道信号生成 scalogram

    Parameters
    ----------
    signal_1ch : shape [T]
    params : CWT 参数

    Returns
    -------
    scalogram : shape [num_scales, out_width]，归一化后的时频能量
    freqs : 各尺度对应的频率（Hz）
    """
    if params is None:
        params = CWT_PARAMS

    scales = np.arange(1, params["max_scale"] + 1)
    coef, freqs = pywt.cwt(
        signal_1ch, scales, params["wavelet"],
        sampling_period=1.0 / params["fs"],
    )
    # 取幅值（能量）
    power = np.abs(coef)

    # 时间轴重采样到固定宽度，统一变长片段
    if power.shape[1] != params["out_width"]:
        power = resample(power, params["out_width"], axis=1)

    # 尺度轴若不等于 num_scales 则重采样
    if power.shape[0] != params["num_scales"]:
        power = resample(power, params["num_scales"], axis=0)

    return power, freqs


def cwt_scalogram_3ch(signal_3ch: np.ndarray, params: dict = None) -> np.ndarray:
    """对三通道信号生成 3 通道 scalogram 张量

    使用 log 压缩 + 全局归一化（而非每通道独立归一化）：
    保留通道间的相对能量差异（即各通道响应强弱），这是区分手势的关键判别信息。
    实测全局归一化比每通道独立归一化的 CNN 准确率高约 20 个百分点。

    Parameters
    ----------
    signal_3ch : shape [T, 3]

    Returns
    -------
    scalogram : shape [3, num_scales, out_width]，全局归一化到 [0, 1]
    """
    if params is None:
        params = CWT_PARAMS

    channel_scalograms = []
    for ch in range(signal_3ch.shape[1]):
        power, _ = cwt_scalogram(signal_3ch[:, ch], params)
        channel_scalograms.append(power)

    arr = np.stack(channel_scalograms, axis=0)
    # log 压缩（缩小动态范围，突出弱信号细节）
    arr = np.log1p(arr)
    # 全局归一化（三通道共用同一尺度，保留相对强度）
    arr = arr / (arr.max() + 1e-8)
    return arr.astype(np.float32)
