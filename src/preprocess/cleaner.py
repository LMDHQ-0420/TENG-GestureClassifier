"""单片段信号清洗模块

来源：PDMA.py 的滤波流程 + spectral_subtraction + detect_artifact_boundary
用途：对单个切出的子动作片段做去噪和稳定段裁切
"""

from typing import Optional, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, stft, istft


def build_filters(fs: int = 1000) -> dict:
    """构造带通和陷波滤波器系数（构造一次复用）

    Returns
    -------
    dict with keys: b_band, a_band, b_notch, a_notch
    """
    nyq = 0.5 * fs
    b_band, a_band = butter(4, [20 / nyq, 450 / nyq], btype="band")
    b_notch, a_notch = iirnotch(50 / nyq, 30)
    return {
        "b_band": b_band, "a_band": a_band,
        "b_notch": b_notch, "a_notch": a_notch,
    }


def spectral_subtraction(signal: np.ndarray, fs: int, nperseg: int = 256) -> np.ndarray:
    """谱减法降噪

    通过 STFT 估计噪声功率谱，从信号功率谱中减去噪声后 ISTFT 重建。
    """
    f, t_spec, Zxx = stft(signal, fs=fs, nperseg=nperseg)
    magnitude = np.abs(Zxx)
    phase = np.angle(Zxx)

    energy = np.sum(magnitude ** 2, axis=0)
    noise_idx = max(1, int(len(t_spec) * 0.1))
    noise_frames = np.argsort(energy)[:noise_idx]
    noise_mu = np.mean(magnitude[:, noise_frames], axis=1, keepdims=True)

    alpha, beta = 2.0, 0.02
    subtracted_mag = np.maximum(
        magnitude ** 2 - alpha * noise_mu ** 2,
        (beta * noise_mu) ** 2,
    ) ** 0.5

    Zxx_new = subtracted_mag * np.exp(1j * phase)
    _, signal_clean = istft(Zxx_new, fs=fs, nperseg=nperseg)

    if len(signal_clean) < len(signal):
        signal_clean = np.pad(signal_clean, (0, len(signal) - len(signal_clean)))
    else:
        signal_clean = signal_clean[: len(signal)]
    return signal_clean


def detect_artifact_boundary(
    signal: np.ndarray, fs: int, search_range_ms: int = 500
) -> Tuple[int, int]:
    """检测信号首尾伪迹边界，返回稳定段起止索引"""
    envelope = np.abs(signal)
    search_samples = int((search_range_ms / 1000) * fs)

    head_segment = envelope[:search_samples]
    head_idx = np.argmin(head_segment) + 1 if len(head_segment) > 0 else 0

    tail_segment = envelope[-search_samples:]
    tail_idx = (
        len(signal) - (len(tail_segment) - np.argmin(tail_segment)) - 1
        if len(tail_segment) > 0
        else len(signal)
    )

    return head_idx, tail_idx


def clean_segment(
    segment_3ch: np.ndarray,
    fs: int = 1000,
    filters: Optional[dict] = None,
) -> np.ndarray:
    """对单个子片段做完整信号清洗

    流程：去直流 → 带通 → 陷波 → 谱减法 → 边界裁切（三通道联合取最保守边界）

    Parameters
    ----------
    segment_3ch : shape [T, 3]
    fs : 采样率
    filters : build_filters() 的返回值，None 则内部构造

    Returns
    -------
    np.ndarray, shape [T', 3]（裁切后可能更短）
    """
    if filters is None:
        filters = build_filters(fs)

    b_band, a_band = filters["b_band"], filters["a_band"]
    b_notch, a_notch = filters["b_notch"], filters["a_notch"]

    n_channels = segment_3ch.shape[1]
    processed = []
    head_indices, tail_indices = [], []

    for ch in range(n_channels):
        col = segment_3ch[:, ch].copy()
        col -= np.mean(col)
        col = filtfilt(b_band, a_band, col)
        col = filtfilt(b_notch, a_notch, col)
        col = spectral_subtraction(col, fs)
        processed.append(col)

        h, t = detect_artifact_boundary(col, fs)
        head_indices.append(h)
        tail_indices.append(t)

    final_head = max(head_indices)
    final_tail = min(tail_indices)

    if final_tail <= final_head:
        final_head = 0
        final_tail = segment_3ch.shape[0]

    cleaned = np.column_stack(processed)
    return cleaned[final_head:final_tail]
