"""预处理流水线：串联 segmenter → cleaner → features

批量处理 data/raw/ 下所有环境的 CSV 文件，输出到 data/processed/
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from .io import load_raw_csv, get_gesture_label, get_gesture_name, GESTURE_NAMES
from .segmenter import segment_file, SegParams
from .cleaner import clean_segment, build_filters
from .features import calculate_9d_vector, FEATURE_NAMES


def process_env(
    env_dir: Path,
    env_name: str,
    out_segments_dir: Path,
    out_features_path: Path,
    fs: int = 1000,
    params: Optional[SegParams] = None,
) -> pd.DataFrame:
    """处理一个环境目录下的所有 CSV 文件

    Parameters
    ----------
    env_dir : data/raw/{env_name}/
    env_name : 环境标识（base/wind_noise/uv_radiation）
    out_segments_dir : 切分片段保存目录
    out_features_path : 特征 CSV 保存路径
    fs : 采样率
    params : 切分参数

    Returns
    -------
    pd.DataFrame : manifest（每行一个子片段的元信息 + 9D 特征）
    """
    if params is None:
        params = SegParams(fs=fs)

    out_segments_dir.mkdir(parents=True, exist_ok=True)
    filters = build_filters(fs)

    records = []
    csv_files = sorted(env_dir.glob("*.csv"))

    for csv_path in tqdm(csv_files, desc=f"  {env_name}", unit="file"):
        signal = load_raw_csv(csv_path)
        gesture_name = get_gesture_name(csv_path.name)
        label = get_gesture_label(csv_path.name)

        segments = segment_file(signal, params)

        for seg_idx, seg_raw in enumerate(segments):
            seg_id = f"{env_name}_{csv_path.stem}_seg{seg_idx:03d}"
            npy_name = f"{csv_path.stem}_seg{seg_idx:03d}.npy"
            npy_path = out_segments_dir / npy_name

            # 清洗
            seg_clean = clean_segment(seg_raw, fs=fs, filters=filters)
            if seg_clean.shape[0] < 10:
                continue

            # 保存片段
            np.save(npy_path, seg_clean)

            # 提取特征
            features = calculate_9d_vector(seg_clean, fs=fs)

            record = {
                "seg_id": seg_id,
                "env": env_name,
                "source_file": csv_path.name,
                "gesture_name": gesture_name,
                "label": label,
                "duration_ms": round(seg_clean.shape[0] / fs * 1000, 1),
                "npy_path": str(npy_path.relative_to(npy_path.parents[3])),
            }
            for fname, fval in zip(FEATURE_NAMES, features):
                record[fname] = float(fval)
            records.append(record)

        if not segments:
            print(f"    警告：{csv_path.name} 未检测到有效动作段")

    df = pd.DataFrame(records)
    if not df.empty:
        out_features_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_features_path, index=False)
    return df


def run_all(
    raw_root: Path,
    processed_root: Path,
    fs: int = 1000,
    params: Optional[SegParams] = None,
) -> pd.DataFrame:
    """处理所有环境，输出统计

    Parameters
    ----------
    raw_root : data/raw/
    processed_root : data/processed/

    Returns
    -------
    pd.DataFrame : 合并后的全部特征 DataFrame
    """
    all_dfs = []

    for env_name in ["base", "wind_noise", "uv_radiation"]:
        env_dir = raw_root / env_name
        if not env_dir.exists():
            print(f"  跳过：{env_dir} 不存在")
            continue

        print(f"\n处理环境：{env_name}")
        seg_dir = processed_root / "segments" / env_name
        feat_path = processed_root / "features" / f"{env_name}_features.csv"

        df = process_env(env_dir, env_name, seg_dir, feat_path, fs=fs, params=params)
        all_dfs.append(df)
        print(f"  → 切出 {len(df)} 个有效片段")

    all_features = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    # 保存合并特征
    if not all_features.empty:
        all_features.to_csv(processed_root / "features" / "all_features.csv", index=False)

    # 输出统计
    stats_dir = processed_root / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    if not all_features.empty:
        counts = all_features.pivot_table(
            index="gesture_name", columns="env", values="seg_id",
            aggfunc="count", fill_value=0,
        )
        counts["total"] = counts.sum(axis=1)
        counts.to_csv(stats_dir / "segment_counts.csv")
        print(f"\n{'='*60}")
        print("切分统计（各环境 × 各手势）：")
        print(counts.to_string())
        print(f"{'='*60}")

    return all_features
