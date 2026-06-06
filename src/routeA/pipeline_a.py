"""路线A 特征流水线：对已切分的片段做 VMD 分解 + 多域特征提取

复用 data/processed/features/all_features.csv 中的 seg_id/env/label/npy_path 元信息，
对每个 .npy 片段提取 111 维 rich features，输出 rich_features.csv。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.decompose.features_rich import extract_rich_features, RICH_FEATURE_NAMES

DATA_ROOT = PROJECT_ROOT / "data"
META_PATH = DATA_ROOT / "processed" / "features" / "all_features.csv"
OUT_PATH = DATA_ROOT / "processed" / "features" / "rich_features.csv"

# 元信息列（从 all_features.csv 继承，不含 9 维旧特征）
META_COLS = ["seg_id", "env", "source_file", "gesture_name", "label", "duration_ms", "npy_path"]


def run(fs: int = 1000):
    meta = pd.read_csv(META_PATH)
    records = []

    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="VMD features"):
        npy_path = DATA_ROOT / row["npy_path"]
        if not npy_path.exists():
            continue
        seg = np.load(npy_path)

        try:
            feat = extract_rich_features(seg, fs=fs)
        except Exception as e:
            print(f"  跳过 {row['seg_id']}: {e}")
            continue

        record = {col: row[col] for col in META_COLS}
        for name, val in zip(RICH_FEATURE_NAMES, feat):
            record[name] = float(val)
        records.append(record)

    df = pd.DataFrame(records)
    df.to_csv(OUT_PATH, index=False)
    print(f"\n生成 {len(df)} 个样本 × {len(RICH_FEATURE_NAMES)} 维特征")
    print(f"保存到: {OUT_PATH}")
    return df


if __name__ == "__main__":
    run()
