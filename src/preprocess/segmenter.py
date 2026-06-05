"""多动作信号切分模块

来源：SNR.py 的 analyze_channel + choose_representative_joint_action
关键改动：返回所有通过纯净度自检的联合动作段（不只选一个代表）
"""

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


@dataclass
class SegParams:
    """切分参数集（对应 SNR.py 的参数设置区）"""
    fs: int = 1000
    rms_window_s: float = 0.200        # 滑动 RMS 窗口
    min_active_s: float = 0.100        # 最短激活持续时间
    merge_gap_s: float = 0.200         # 单通道相邻激活合并间隔（0.05→0.20，适配sc等复合手势）
    baseline_low_percentile: float = 0.20  # 取 RMS 最低 20% 估计基线
    threshold_k: float = 3.0           # 阈值 = baseline_mean + K * baseline_std
    joint_merge_gap_s: float = 0.080   # 跨通道联合动作合并间隔
    joint_edge_extension_s: float = 0.250  # 联合动作窗口两侧最大扩展量
    max_joint_active_s: float = 2.800  # 单个联合动作最长持续时间
    joint_min_active_channels: int = 2 # 联合动作至少需要激活的通道数
    pre_post_pad_s: float = 0.050      # 切分时两侧额外保留的时间


# ---- 基础工具函数 ----

def moving_rms(x: np.ndarray, window: int) -> np.ndarray:
    """滑动 RMS"""
    window = max(int(window), 1)
    kernel = np.ones(window, dtype=float) / window
    return np.sqrt(np.convolve(np.nan_to_num(x, nan=0.0) ** 2, kernel, mode="same"))


def contiguous_segments(mask: np.ndarray, min_len: int) -> List[Tuple[int, int]]:
    """从布尔掩码中提取连续 True 段"""
    segs, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif (not v) and start is not None:
            if i - start >= min_len:
                segs.append((start, i))
            start = None
    if start is not None and len(mask) - start >= min_len:
        segs.append((start, len(mask)))
    return segs


def merge_close_segments(segs: List[Tuple[int, int]], max_gap: int) -> List[Tuple[int, int]]:
    """合并间隔小于 max_gap 的相邻段"""
    if not segs:
        return []
    merged = [list(segs[0])]
    for s, e in segs[1:]:
        if s - merged[-1][1] <= max_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


# ---- 单通道激活检测 ----

def analyze_single_channel(signal_1ch: np.ndarray, params: SegParams) -> dict:
    """检测单通道的所有激活段

    Returns
    -------
    dict with keys: active_segments, active_mask, rms_env, threshold
    """
    fs = params.fs
    centered = signal_1ch - np.nanmedian(signal_1ch)

    rms_window = max(1, int(round(fs * params.rms_window_s)))
    min_active = max(1, int(round(fs * params.min_active_s)))
    merge_gap = max(0, int(round(fs * params.merge_gap_s)))

    rms_env = moving_rms(centered, rms_window)

    low_cutoff = np.quantile(rms_env, params.baseline_low_percentile)
    baseline_mask = rms_env <= low_cutoff
    baseline_mean = float(np.mean(rms_env[baseline_mask]))
    baseline_std = float(np.std(rms_env[baseline_mask]))
    threshold = baseline_mean + params.threshold_k * baseline_std

    active_mask = rms_env > threshold
    initial_segs = contiguous_segments(active_mask, min_len=min_active)
    active_segments = merge_close_segments(initial_segs, max_gap=merge_gap)

    return {
        "active_segments": active_segments,
        "active_mask": active_mask,
        "rms_env": rms_env,
        "threshold": threshold,
        "baseline_mean": baseline_mean,
    }


# ---- 跨通道联合动作检测 ----

def detect_joint_actions(
    signal_3ch: np.ndarray, params: SegParams
) -> List[Tuple[int, int]]:
    """检测所有通过纯净度自检的跨通道联合动作段

    Parameters
    ----------
    signal_3ch : shape [N, 3]
    params : 切分参数

    Returns
    -------
    List of (start_sample, end_sample) tuples
    """
    fs = params.fs
    n_samples = signal_3ch.shape[0]
    min_active = max(1, int(round(fs * params.min_active_s)))
    core_merge_gap = max(0, int(round(fs * params.joint_merge_gap_s)))
    edge_extension = max(0, int(round(fs * params.joint_edge_extension_s)))
    max_joint_len = max(min_active, int(round(fs * params.max_joint_active_s)))
    min_channels = min(params.joint_min_active_channels, signal_3ch.shape[1])

    # 逐通道分析
    channel_analyses = []
    for ch in range(signal_3ch.shape[1]):
        analysis = analyze_single_channel(signal_3ch[:, ch], params)
        channel_analyses.append(analysis)

    # 统计每个时刻有多少通道同时激活
    active_count = np.zeros(n_samples, dtype=int)
    for analysis in channel_analyses:
        for a, b in analysis["active_segments"]:
            active_count[a:b] += 1

    # 找至少 min_channels 个通道同时激活的核心段
    core_mask = active_count >= min_channels
    core_segments = contiguous_segments(core_mask, min_len=min_active)
    core_segments = merge_close_segments(core_segments, max_gap=core_merge_gap)

    # 向两侧扩展，纳入边缘激活
    joint_segments = []
    for core_a, core_b in core_segments:
        a, b = core_a, core_b
        for analysis in channel_analyses:
            for s, e in analysis["active_segments"]:
                if s <= core_b + edge_extension and e >= core_a - edge_extension:
                    a = min(a, max(s, core_a - edge_extension))
                    b = max(b, min(e, core_b + edge_extension))
        if b > a and (b - a) <= max_joint_len:
            joint_segments.append((a, b))

    joint_segments = list(dict.fromkeys(joint_segments))

    # 纯净度自检：任一通道在窗口内出现多次激活则丢弃
    clean_segments = []
    for joint_a, joint_b in joint_segments:
        is_clean = True
        for analysis in channel_analyses:
            bursts_in_window = sum(
                1 for s, e in analysis["active_segments"]
                if max(joint_a, s) < min(joint_b, e)
            )
            if bursts_in_window > 1:
                is_clean = False
                break
        if is_clean:
            clean_segments.append((joint_a, joint_b))

    return clean_segments


def segment_file(
    signal_3ch: np.ndarray, params: SegParams = None
) -> List[np.ndarray]:
    """切分一个多动作文件为子片段列表

    Parameters
    ----------
    signal_3ch : shape [N, 3]
    params : 切分参数，None 使用默认值

    Returns
    -------
    List[np.ndarray]，每个元素 shape [T_i, 3]
    """
    if params is None:
        params = SegParams()

    joint_actions = detect_joint_actions(signal_3ch, params)
    pad = int(round(params.fs * params.pre_post_pad_s))

    segments = []
    for a, b in joint_actions:
        start = max(0, a - pad)
        end = min(signal_3ch.shape[0], b + pad)
        segments.append(signal_3ch[start:end].copy())

    return segments
