# -*- coding: utf-8 -*-
"""
EMG SNR figure-data exporter (Unified Plot Axis + Independent Activation Mode)

用途：
将 EMG local paired SNR 分析结果按论文示意图的小图结构导出。
1. 识别跨通道联合动作，并在宏观上统一所有通道导出 CSV 的 X 轴时间（保证 Origin 画图完美对齐）。
2. 在统一的 X 轴画幅内，严格根据各个通道真实的独立肌电越线点去标记 active_flag 和计算独立 SNR。
3. 包含“纯净度自检”，剔除包含多次波峰干扰的脏动作，只保留最干净的单次信号。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd

# =========================
# 参数设置区
# =========================
DEFAULT_FS = 1000.0              # 无时间列时默认采样率，Hz
RMS_WINDOW_S = 0.200             # 滑动 RMS 窗口
REST_WINDOW_S = 0.200            # burst 前后局部静息窗口
MIN_ACTIVE_S = 0.100             # 最短激活持续时间 (防碎片)
MERGE_GAP_S = 0.050              # 单通道相邻激活合并间隔 (强行拆分连绵的多次发力)
JOINT_MERGE_GAP_S = 0.080        # 跨通道核心动作合并间隔
JOINT_EDGE_EXTENSION_S = 0.250   # 共同动作窗口两侧最大扩展量
MAX_JOINT_ACTIVE_S = 2.800       # 单个代表动作 active 窗口最长持续时间
JOINT_MIN_ACTIVE_CHANNELS = 2    # 联合动作至少需要多少个通道出现激活
BASELINE_LOW_PERCENTILE = 0.20   # 取 RMS 最低 20% 估计静息基线
THRESHOLD_K = 3.0                # 阈值 = baseline_mean + K * baseline_std
MAX_ACTIVE_FRACTION_IN_REST = 0.10
REPRESENTATIVE_MODE = "median"   # "median" (选接近中位SNR的代表)；"max" (选最高SNR)
OUTPUT_ENCODING = "utf-8-sig"

ANALYSIS_START_S = None
ANALYSIS_END_S = 13.0


# =========================
# 文件选择与读取
# =========================
def select_files_gui() -> List[Path]:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        paths = filedialog.askopenfilenames(
            title="选择一个或多个 CSV 文件",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        root.destroy()
        return [Path(p) for p in paths]
    except Exception:
        print("未能打开文件选择窗口。")
        raw = input("请输入 CSV 文件路径，多个文件用英文分号 ; 分隔：").strip()
        if not raw:
            return []
        return [Path(p.strip().strip('"')) for p in raw.split(";") if p.strip()]


def can_be_float(value) -> bool:
    try:
        float(str(value))
        return True
    except Exception:
        return False


def read_csv_smart(file_path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "gbk", "gb18030", "latin1"]
    last_error = None
    for enc in encodings:
        try:
            df_header = pd.read_csv(file_path, encoding=enc)
            numeric_header_fraction = np.mean([can_be_float(c) for c in df_header.columns])
            if numeric_header_fraction > 0.6:
                df = pd.read_csv(file_path, encoding=enc, header=None)
                df.columns = [f"CH{i+1}" for i in range(df.shape[1])]
                return df
            return df_header
        except Exception as e:
            last_error = e
    raise RuntimeError(f"无法读取文件：{file_path}\n最后错误：{last_error}")


def find_time_column(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        name = str(col).strip().lower()
        if name in ["time", "t", "time(s)", "time (s)", "times", "时间", "时间(s)", "时间（s）"]:
            vals = pd.to_numeric(df[col], errors="coerce").to_numpy()
            vals = vals[np.isfinite(vals)]
            if len(vals) >= 3 and np.mean(np.diff(vals) > 0) > 0.8:
                return col
    for col in df.columns:
        name = str(col).strip().lower()
        if "time" in name or "时间" in name:
            vals = pd.to_numeric(df[col], errors="coerce").to_numpy()
            vals = vals[np.isfinite(vals)]
            if len(vals) >= 3 and np.mean(np.diff(vals) > 0) > 0.8:
                return col
    return None


def estimate_fs(time_s: np.ndarray) -> float:
    diffs = np.diff(time_s)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(diffs) == 0:
        return DEFAULT_FS
    return float(1.0 / np.median(diffs))


def apply_analysis_time_window(
    df: pd.DataFrame, time_s: np.ndarray, start_s: Optional[float], end_s: Optional[float]
) -> Tuple[pd.DataFrame, np.ndarray]:
    mask = np.ones(len(time_s), dtype=bool)
    if start_s is not None: mask &= time_s >= float(start_s)
    if end_s is not None: mask &= time_s <= float(end_s)
    if mask.sum() < 3: raise RuntimeError("有效分析时间窗内数据点太少。")
    return df.loc[mask].reset_index(drop=True), time_s[mask]


def numeric_signal_columns(df: pd.DataFrame, time_col: Optional[str]) -> List[str]:
    cols = []
    for col in df.columns:
        if time_col is not None and col == time_col: continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().mean() > 0.5: cols.append(str(col))
    return cols


# =========================
# 信号处理函数
# =========================
def interpolate_nan(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    valid = np.isfinite(x)
    if valid.sum() == 0 or valid.sum() == len(x): return x
    idx = np.arange(len(x))
    return np.interp(idx, idx[valid], x[valid])

def moving_rms(x: np.ndarray, window: int) -> np.ndarray:
    window = max(int(window), 1)
    kernel = np.ones(window, dtype=float) / window
    return np.sqrt(np.convolve(np.nan_to_num(x, nan=0.0) ** 2, kernel, mode="same"))

def contiguous_segments(mask: np.ndarray, min_len: int) -> List[Tuple[int, int]]:
    segs, start = [], None
    for i, v in enumerate(mask):
        if v and start is None: start = i
        elif (not v) and start is not None:
            if i - start >= min_len: segs.append((start, i))
            start = None
    if start is not None and len(mask) - start >= min_len:
        segs.append((start, len(mask)))
    return segs

def merge_close_segments(segs: List[Tuple[int, int]], max_gap: int) -> List[Tuple[int, int]]:
    if not segs: return []
    merged = [list(segs[0])]
    for s, e in segs[1:]:
        if s - merged[-1][1] <= max_gap: merged[-1][1] = e
        else: merged.append([s, e])
    return [(s, e) for s, e in merged]

def analyze_channel(time_s: np.ndarray, raw_signal: np.ndarray, fs: float) -> dict:
    raw = np.asarray(raw_signal, dtype=float)
    raw_interp = interpolate_nan(raw)
    centered = raw_interp - np.nanmedian(raw_interp)

    rms_window = max(1, int(round(fs * RMS_WINDOW_S)))
    rest_window = max(1, int(round(fs * REST_WINDOW_S)))
    min_active = max(1, int(round(fs * MIN_ACTIVE_S)))
    merge_gap = max(0, int(round(fs * MERGE_GAP_S)))

    rms_env = moving_rms(centered, rms_window)
    low_cutoff = np.quantile(rms_env, BASELINE_LOW_PERCENTILE)
    baseline_mask = rms_env <= low_cutoff
    baseline_mean = float(np.mean(rms_env[baseline_mask]))
    baseline_std = float(np.std(rms_env[baseline_mask]))
    activation_threshold = baseline_mean + THRESHOLD_K * baseline_std

    active_mask = rms_env > activation_threshold
    initial_segments = contiguous_segments(active_mask, min_len=min_active)
    active_segments = merge_close_segments(initial_segments, max_gap=merge_gap)

    bursts = []
    n = len(centered)
    for seg_id, (a, b) in enumerate(active_segments, start=1):
        pre0, pre1 = a - rest_window, a
        post0, post1 = b, b + rest_window
        if pre0 < 0 or post1 > n: continue
        if active_mask[pre0:pre1].mean() > MAX_ACTIVE_FRACTION_IN_REST: continue
        if active_mask[post0:post1].mean() > MAX_ACTIVE_FRACTION_IN_REST: continue

        rest_sig = np.concatenate([centered[pre0:pre1], centered[post0:post1]])
        active_sig = centered[a:b]
        rest_sig = rest_sig - np.mean(rest_sig)
        active_sig = active_sig - np.mean(active_sig)

        rms_rest = float(np.sqrt(np.mean(rest_sig ** 2)))
        rms_active = float(np.sqrt(np.mean(active_sig ** 2)))
        p_noise = rms_rest ** 2
        p_active = rms_active ** 2
        p_emg = p_active - p_noise

        if p_noise <= 0 or p_emg <= 0: continue
        snr_db = float(10.0 * np.log10(p_emg / p_noise))

        bursts.append({
            "burst_id": len(bursts) + 1,
            "a": a, "b": b,
            "SNR_dB": snr_db,
        })

    return {
        "raw": raw,
        "centered": centered,
        "RMS_envelope": rms_env,
        "baseline_mean_RMS": baseline_mean,
        "baseline_std_RMS": baseline_std,
        "activation_threshold_RMS": activation_threshold,
        "active_mask": active_mask,
        "active_segments": active_segments,
        "bursts": bursts,
    }


def build_combined_active_mask(analyses_by_channel: dict, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    active_count = np.zeros(n_samples, dtype=int)
    for analysis in analyses_by_channel.values():
        for a, b in analysis["active_segments"]:
            active_count[a:b] += 1
    combined_mask = active_count > 0
    return combined_mask, active_count


def compute_channel_specific_snr(
    time_s: np.ndarray, analysis: dict, joint_a: int, joint_b: int, action_id: int, fs: float
) -> Optional[dict]:
    # 找到本通道独立的实际越线点
    best_overlap = 0
    ch_a, ch_b = None, None
    for s, e in analysis["active_segments"]:
        overlap = min(joint_b, e) - max(joint_a, s)
        if overlap > best_overlap:
            best_overlap = overlap
            ch_a, ch_b = s, e

    if ch_a is None or ch_b is None:
        return None

    active_mask = analysis["active_mask"]
    local_true = np.flatnonzero(active_mask[ch_a:ch_b])
    if len(local_true) > 0:
        ch_a = ch_a + int(local_true[0])
        ch_b = ch_a + int(local_true[-1]) + 1

    centered = analysis["centered"]
    n_samples = len(centered)
    rest_window = max(1, int(round(fs * REST_WINDOW_S)))
    
    ch_pre0 = ch_a - rest_window
    ch_pre1 = ch_a
    ch_post0 = ch_b
    ch_post1 = ch_b + rest_window

    if ch_pre0 < 0 or ch_post1 > n_samples:
        return None

    rest_sig = np.concatenate([centered[ch_pre0:ch_pre1], centered[ch_post0:ch_post1]])
    active_sig = centered[ch_a:ch_b]

    if len(rest_sig) == 0 or len(active_sig) == 0:
        return None

    rest_sig = rest_sig - np.mean(rest_sig)
    active_sig = active_sig - np.mean(active_sig)

    rms_rest = float(np.sqrt(np.mean(rest_sig ** 2)))
    rms_active = float(np.sqrt(np.mean(active_sig ** 2)))
    p_noise = rms_rest ** 2
    p_active = rms_active ** 2
    p_emg = p_active - p_noise

    if p_noise <= 0 or p_emg <= 0:
        return None

    snr_db = float(10.0 * np.log10(p_emg / p_noise))

    return {
        "joint_action_id": action_id,
        "ch_pre0": int(ch_pre0),
        "ch_pre1": int(ch_pre1),
        "ch_a": int(ch_a),
        "ch_b": int(ch_b),
        "ch_post0": int(ch_post0),
        "ch_post1": int(ch_post1),
        
        "channel_pre_rest_start_s": float(time_s[ch_pre0]),
        "channel_pre_rest_end_s": float(time_s[ch_pre1 - 1]),
        "channel_active_start_s": float(time_s[ch_a]),
        "channel_active_end_s": float(time_s[ch_b - 1]),
        "channel_post_rest_start_s": float(time_s[ch_post0]),
        "channel_post_rest_end_s": float(time_s[ch_post1 - 1]),
        
        "RMS_rest": rms_rest,
        "RMS_active": rms_active,
        "P_noise": float(p_noise),
        "P_active": float(p_active),
        "P_EMG": float(p_emg),
        "SNR_dB": snr_db,
    }


def choose_representative_joint_action(
    time_s: np.ndarray, analyses_by_channel: dict, fs: float
) -> Tuple[Optional[dict], dict]:
    if not analyses_by_channel:
        return None, {}

    n = len(time_s)
    min_active = max(1, int(round(fs * MIN_ACTIVE_S)))
    core_merge_gap = max(0, int(round(fs * JOINT_MERGE_GAP_S)))
    edge_extension = max(0, int(round(fs * JOINT_EDGE_EXTENSION_S)))
    max_joint_len = max(min_active, int(round(fs * MAX_JOINT_ACTIVE_S)))
    min_required_channels = max(1, min(JOINT_MIN_ACTIVE_CHANNELS, len(analyses_by_channel)))

    _, active_count = build_combined_active_mask(analyses_by_channel, n)

    core_mask = active_count >= min_required_channels
    core_segments = contiguous_segments(core_mask, min_len=min_active)
    core_segments = merge_close_segments(core_segments, max_gap=core_merge_gap)

    joint_segments = []
    for core_a, core_b in core_segments:
        a, b = core_a, core_b
        for analysis in analyses_by_channel.values():
            for s, e in analysis["active_segments"]:
                if (s <= core_b + edge_extension) and (e >= core_a - edge_extension):
                    a = min(a, max(s, core_a - edge_extension))
                    b = max(b, min(e, core_b + edge_extension))
        if b > a and (b - a) <= max_joint_len:
            joint_segments.append((a, b))

    joint_segments = list(dict.fromkeys(joint_segments))
    candidates = []
    reps_by_action = {}

    for action_id, (joint_a, joint_b) in enumerate(joint_segments, start=1):
        is_clean_action = True
        
        # 纯净度自检防火墙：只要有一个通道出现多次发力信号，整段抛弃
        for channel, analysis in analyses_by_channel.items():
            bursts_in_this_window = 0
            for s, e in analysis["active_segments"]:
                if max(joint_a, s) < min(joint_b, e):
                    bursts_in_this_window += 1
            if bursts_in_this_window > 1:
                is_clean_action = False
                break
                
        if not is_clean_action:
            print(f"  [过滤] 抛弃时段 {time_s[joint_a]:.2f}s-{time_s[joint_b-1]:.2f}s: 内部包含断裂的多次重复信号，非干净动作。")
            continue

        channel_reps = {}
        action_snrs = []
        for channel, analysis in analyses_by_channel.items():
            rep = compute_channel_specific_snr(time_s, analysis, joint_a, joint_b, action_id, fs)
            if rep:
                channel_reps[channel] = rep
                action_snrs.append(rep["SNR_dB"])

        if len(action_snrs) >= min_required_channels:
            # 核心修正：锁定宏观绘图的 X 轴并集时间，让所有通道拥有完全相同的导出时间段长度
            global_pre0 = min(r["ch_pre0"] for r in channel_reps.values())
            global_post1 = max(r["ch_post1"] for r in channel_reps.values())

            for r in channel_reps.values():
                r["plot_pre0"] = global_pre0
                r["plot_post1"] = global_post1

            aggregate_snr = float(np.mean(action_snrs))
            candidates.append({
                "joint_action_id": action_id,
                "aggregate_SNR_dB": aggregate_snr,
                "joint_a": joint_a,
                "joint_b": joint_b
            })
            reps_by_action[action_id] = channel_reps

    if not candidates: return None, {}

    if REPRESENTATIVE_MODE == "max":
        chosen = max(candidates, key=lambda c: c["aggregate_SNR_dB"])
    else:
        aggregate_snrs = np.array([c["aggregate_SNR_dB"] for c in candidates], dtype=float)
        median_snr = float(np.median(aggregate_snrs))
        chosen = min(candidates, key=lambda c: abs(c["aggregate_SNR_dB"] - median_snr))

    return chosen, reps_by_action[chosen["joint_action_id"]]


# =========================
# 导出 A-D 小图 CSV
# =========================
def export_panel_csvs(
    out_dir: Path, channel: str, time_s: np.ndarray, analysis: dict, rep: dict, fs: float
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    centered = analysis["centered"]
    rms_env = analysis["RMS_envelope"]
    baseline = analysis["baseline_mean_RMS"]
    threshold = analysis["activation_threshold_RMS"]

    # 导出 CSV 截取范围：所有人共用 plot_pre0 和 plot_post1
    plot_pre0 = rep["plot_pre0"]
    plot_post1 = rep["plot_post1"]
    
    pad = int(round(0.08 * fs))
    i0 = max(0, plot_pre0 - pad)
    i1 = min(len(time_s), plot_post1 + pad)

    sliced_time = time_s[i0:i1]
    local_label = np.full(i1 - i0, "none", dtype=object)
    channel_active_flag = np.zeros(i1 - i0, dtype=int)
    above_threshold_flag = analysis["active_mask"][i0:i1].astype(int)

    # 画图标记范围：使用各通道自己独立算出来的起止点去打 tag
    ch_pre0, ch_pre1 = rep["ch_pre0"], rep["ch_pre1"]
    ch_a, ch_b = rep["ch_a"], rep["ch_b"]
    ch_post0, ch_post1 = rep["ch_post0"], rep["ch_post1"]

    def mark_range(start, end, name):
        s, e = max(start, i0) - i0, min(end, i1) - i0
        if s < e: local_label[s:e] = name

    mark_range(ch_pre0, ch_pre1, "channel_pre_rest")
    mark_range(ch_a, ch_b, "channel_active_window")
    mark_range(ch_post0, ch_post1, "channel_post_rest")

    s, e = max(ch_a, i0) - i0, min(ch_b, i1) - i0
    if s < e: channel_active_flag[s:e] = 1

    pd.DataFrame({
        "time_s": sliced_time,
        "centered_signal": centered[i0:i1],
        "local_label": local_label,
        "channel_active_flag": channel_active_flag,
    }).to_csv(out_dir / "A_raw_segment.csv", index=False, encoding=OUTPUT_ENCODING)

    pd.DataFrame({
        "time_s": sliced_time,
        "RMS_envelope": rms_env[i0:i1],
        "local_label": local_label,
        "channel_active_flag": channel_active_flag,
        "above_threshold_flag": above_threshold_flag,
    }).to_csv(out_dir / "B_rms_envelope.csv", index=False, encoding=OUTPUT_ENCODING)

    pd.DataFrame([{
        "baseline_mean_RMS": baseline,
        "baseline_std_RMS": analysis["baseline_std_RMS"],
        "threshold_k": THRESHOLD_K,
        "activation_threshold_RMS": threshold,
        "plot_window_start_s": float(time_s[i0]),      # 显示共用的绘图X轴起止时间
        "plot_window_end_s": float(time_s[i1-1] if i1 > 0 else 0),
        "channel_pre_rest_start_s": rep["channel_pre_rest_start_s"], # 显示通道各自特异的真实时间
        "channel_pre_rest_end_s": rep["channel_pre_rest_end_s"],
        "channel_active_start_s": rep["channel_active_start_s"],
        "channel_active_end_s": rep["channel_active_end_s"],
        "channel_post_rest_start_s": rep["channel_post_rest_start_s"],
        "channel_post_rest_end_s": rep["channel_post_rest_end_s"],
    }]).to_csv(out_dir / "B_reference_lines_and_times.csv", index=False, encoding=OUTPUT_ENCODING)

    pd.DataFrame({
        "category": ["Noise power (local rests)", "Total power (active burst)", "Effective EMG power"],
        "power": [rep["P_noise"], rep["P_active"], rep["P_EMG"]],
        "SNR_dB": rep["SNR_dB"],
        "RMS_rest": rep["RMS_rest"],
        "RMS_active": rep["RMS_active"]
    }).to_csv(out_dir / "C_power_comparison.csv", index=False, encoding=OUTPUT_ENCODING)

    bursts = analysis["bursts"]
    snrs = np.array([b["SNR_dB"] for b in bursts], dtype=float)
    pd.DataFrame({
        "burst_id": [b["burst_id"] for b in bursts],
        "SNR_dB": [b["SNR_dB"] for b in bursts],
        "SNR_mean_dB": float(np.mean(snrs)) if len(snrs) > 0 else 0.0,
        "SNR_std_dB": float(np.std(snrs, ddof=1)) if len(snrs) > 1 else 0.0,
    }).to_csv(out_dir / "D_burstwise_snr.csv", index=False, encoding=OUTPUT_ENCODING)


# =========================
# 批量执行主函数
# =========================
def process_files(files: List[Path]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_out = files[0].parent / f"EMG_Figure_Data_{timestamp}"
    root_out.mkdir(parents=True, exist_ok=False)

    exported_records = []

    for file_path in files:
        print(f"\n正在处理: {file_path.name}")
        df = read_csv_smart(file_path)

        time_col = find_time_column(df)
        if time_col is not None:
            time_s = pd.to_numeric(df[time_col], errors="coerce").to_numpy()
            valid = np.isfinite(time_s)
            df = df.loc[valid].reset_index(drop=True)
            time_s = time_s[valid]
            fs = estimate_fs(time_s)
        else:
            fs = DEFAULT_FS
            time_s = np.arange(len(df), dtype=float) / fs

        if ANALYSIS_START_S is not None or ANALYSIS_END_S is not None:
            df, time_s = apply_analysis_time_window(df, time_s, ANALYSIS_START_S, ANALYSIS_END_S)
            fs = estimate_fs(time_s)

        sig_cols = numeric_signal_columns(df, time_col)
        analyses_by_channel = {}

        for col in sig_cols:
            raw = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            analysis = analyze_channel(time_s, raw, fs)
            if analysis["bursts"]:
                analyses_by_channel[str(col)] = analysis

        # 重点：自检并寻找干净的单次宏观联合动作
        chosen_joint_action, reps_by_channel = choose_representative_joint_action(
            time_s=time_s, analyses_by_channel=analyses_by_channel, fs=fs
        )

        if chosen_joint_action is None:
            print("  未能找到合格的、干净的单次联合发力动作，跳过该文件。")
            continue

        print(f"  --> 成功锁定绝对干净的第 {chosen_joint_action['joint_action_id']} 次联合动作作为全通道图表代表！")

        # 为了打印提示用的共同X轴
        plot_s_start = time_s[max(0, list(reps_by_channel.values())[0]["plot_pre0"] - int(round(0.08 * fs)))]
        plot_s_end = time_s[min(len(time_s)-1, list(reps_by_channel.values())[0]["plot_post1"] + int(round(0.08 * fs)))]

        for col, analysis in analyses_by_channel.items():
            rep = reps_by_channel.get(col)
            if rep is None: continue

            safe_channel = str(col).replace("/", "_").replace("\\", "_").replace(":", "_")
            channel_out = root_out / file_path.stem / safe_channel
            
            export_panel_csvs(channel_out, col, time_s, analysis, rep, fs)

            print(f"      [{col}] CSV的X轴全通道统一锁定为 ({plot_s_start:.3f}s-{plot_s_end:.3f}s)。")
            print(f"      [{col}] 独立特征运算起止点实际为 ({rep['channel_active_start_s']:.3f}s-{rep['channel_active_end_s']:.3f}s)。")

            exported_records.append({
                "source_file": file_path.name,
                "channel": col,
                "joint_action_id": rep["joint_action_id"],
                "active_start_s": rep["channel_active_start_s"],
                "active_end_s": rep["channel_active_end_s"],
                "folder": str(channel_out)
            })

    if exported_records:
        pd.DataFrame(exported_records).to_csv(root_out / "figure_data_index.csv", index=False, encoding=OUTPUT_ENCODING)

    return root_out


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    files = select_files_gui()
    files = [p for p in files if p.exists()]
    if not files:
        print("未选择文件，程序退出。")
        return

    root_out = process_files(files)
    print(f"\n全部完美处理完成，绘图X轴已绝对统一，信号检测各不干扰！\n结果保存在：{root_out}")
    input("按 Enter 键退出...")

if __name__ == "__main__":
    main()