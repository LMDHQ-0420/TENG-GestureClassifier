"""路线B 数据集：CWT scalogram + 数据增强

每个片段通过 CWT 转为 [3, 64, 128] 的 scalogram 张量，
配合数据增强缓解小样本过拟合。
"""

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.decompose.cwt import cwt_scalogram_3ch

DATA_ROOT = PROJECT_ROOT / "data"
META_PATH = DATA_ROOT / "processed" / "features" / "all_features.csv"
CACHE_DIR = DATA_ROOT / "processed" / "scalograms"


def precompute_scalograms():
    """预计算所有片段的 scalogram 并缓存为 .npy，加速训练"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(META_PATH)
    paths = []
    for _, row in meta.iterrows():
        npy_path = DATA_ROOT / row["npy_path"]
        out_path = CACHE_DIR / f"{row['seg_id']}.npy"
        if not out_path.exists():
            seg = np.load(npy_path)
            scal = cwt_scalogram_3ch(seg)
            np.save(out_path, scal)
        paths.append(str(out_path.relative_to(DATA_ROOT)))
    meta["scal_path"] = paths
    meta.to_csv(CACHE_DIR / "scalogram_index.csv", index=False)
    print(f"预计算 {len(meta)} 个 scalogram → {CACHE_DIR}")
    return meta


def make_split(test_ratio: float = 0.2, val_ratio: float = 0.15, random_state: int = 42):
    """分层划分 train/val/test（与基线/路线A同 random_state，test 集一致）"""
    df = pd.read_csv(CACHE_DIR / "scalogram_index.csv")
    df["stratify_key"] = df["env"] + "_" + df["gesture_name"]

    counts = df["stratify_key"].value_counts()
    singleton = df["stratify_key"].isin(counts[counts < 2].index)
    df_main = df[~singleton]

    train_idx, test_idx = train_test_split(
        df_main.index, test_size=test_ratio,
        stratify=df_main["stratify_key"], random_state=random_state,
    )
    df["split"] = "train"
    df.loc[test_idx, "split"] = "test"

    # 从 train 再切出 val
    train_pool = df[df["split"] == "train"]
    tp_counts = train_pool["stratify_key"].value_counts()
    tp_main = train_pool[train_pool["stratify_key"].isin(tp_counts[tp_counts >= 2].index)]
    tr_idx, val_idx = train_test_split(
        tp_main.index, test_size=val_ratio,
        stratify=tp_main["stratify_key"], random_state=random_state,
    )
    df.loc[val_idx, "split"] = "val"
    return df


class ScalogramDataset(Dataset):
    """scalogram 数据集，支持训练时数据增强"""

    def __init__(self, df_split: pd.DataFrame, augment: bool = False):
        self.paths = [DATA_ROOT / p for p in df_split["scal_path"].values]
        self.labels = df_split["label"].values
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def _augment(self, x: np.ndarray) -> np.ndarray:
        """数据增强：时间平移 + 幅度缩放 + 加噪 + 频带遮挡(SpecAugment)

        增强强度经实验调校：过强会导致欠拟合（CNN 学不到信号），
        当前配置在缓解过拟合和保留信号信息之间取得平衡。
        """
        # 时间平移（沿宽度轴循环移位）
        x = np.roll(x, np.random.randint(-8, 9), axis=2)

        # 幅度缩放 + 加高斯噪声
        x = x * np.random.uniform(0.9, 1.1)
        x = x + np.random.normal(0, 0.02, x.shape).astype(np.float32)

        # SpecAugment：随机遮挡一个频带
        if np.random.rand() < 0.4:
            f0 = np.random.randint(0, x.shape[1] - 8)
            x[:, f0:f0 + 8, :] = 0

        return np.clip(x, 0, 1)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        x = np.load(self.paths[idx]).astype(np.float32)
        if self.augment:
            x = self._augment(x)
        return torch.from_numpy(x), int(self.labels[idx])
