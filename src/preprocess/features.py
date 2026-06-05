"""9D 特征提取模块

来源：PDMA.py 的 calculate_9d_vector
输出：[MAV_CH1, MAV_CH3, MAV_CH5, WL_CH1, WL_CH3, WL_CH5, Ratio_CH1, Ratio_CH3, Ratio_CH5]
"""

from typing import List

import numpy as np

FEATURE_NAMES: List[str] = [
    "MAV_CH1", "MAV_CH3", "MAV_CH5",
    "WL_CH1", "WL_CH3", "WL_CH5",
    "Ratio_CH1", "Ratio_CH3", "Ratio_CH5",
]


def calculate_9d_vector(segment_3ch: np.ndarray, fs: int = 1000) -> np.ndarray:
    """提取 9 维手势特征向量

    Parameters
    ----------
    segment_3ch : shape [T, 3]，清洗后的三通道信号
    fs : 采样率

    Returns
    -------
    np.ndarray, shape [9]
        [MAV×3, WL×3, Ratio×3]

    特征说明
    --------
    MAV  : 平均绝对值 mean(|x|)
    WL   : 波形长度 sum(|diff(x)|) / duration
    Ratio: 各通道 RMS 占总 RMS 的比例
    """
    mav = np.mean(np.abs(segment_3ch), axis=0)

    duration = len(segment_3ch) / fs
    wl = np.sum(np.abs(np.diff(segment_3ch, axis=0)), axis=0) / duration

    rms = np.sqrt(np.mean(segment_3ch ** 2, axis=0))
    total_rms = np.sum(rms)
    ratio = rms / total_rms if total_rms > 0 else np.zeros(3)

    return np.concatenate([mav, wl, ratio])
