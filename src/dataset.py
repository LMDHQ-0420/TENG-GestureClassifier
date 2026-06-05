"""数据集加载与 train/test 划分

按每个 (环境, 手势) 组合分层抽样，每组 ~20% 作为测试集，至少保证 1 个测试样本。
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .preprocess.features import FEATURE_NAMES


def load_and_split(
    csv_path: Path,
    test_ratio: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, pd.DataFrame]:
    """从 all_features.csv 加载数据并按环境分层划分 train/test

    对每个环境内部，按手势分层抽样 test_ratio 比例作为测试集。

    Returns
    -------
    X_train, X_test, y_train, y_test, scaler, df_with_split
    """
    df = pd.read_csv(csv_path)

    # 构造分层 key：环境 + 手势
    df["stratify_key"] = df["env"] + "_" + df["gesture_name"]

    # 检查每组样本数，少于 2 个的不能 stratify split，单独处理
    group_counts = df["stratify_key"].value_counts()
    singleton_mask = df["stratify_key"].isin(group_counts[group_counts < 2].index)

    df_main = df[~singleton_mask].copy()
    df_singleton = df[singleton_mask].copy()

    # 分层划分主体
    train_idx, test_idx = train_test_split(
        df_main.index,
        test_size=test_ratio,
        stratify=df_main["stratify_key"],
        random_state=random_state,
    )

    # 单样本组全部放入训练集
    df.loc[:, "split"] = ""
    df.loc[train_idx, "split"] = "train"
    df.loc[test_idx, "split"] = "test"
    if len(df_singleton) > 0:
        df.loc[df_singleton.index, "split"] = "train"

    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    # 标准化（基于训练集 fit）
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[FEATURE_NAMES].values)
    X_test = scaler.transform(test_df[FEATURE_NAMES].values)
    y_train = train_df["label"].values
    y_test = test_df["label"].values

    print(f"Train: {len(train_df)} samples")
    print(f"Test:  {len(test_df)} samples")
    print(f"\nPer-environment split:")
    for env in ["base", "wind_noise", "uv_radiation"]:
        n_train = len(train_df[train_df["env"] == env])
        n_test = len(test_df[test_df["env"] == env])
        print(f"  {env:15s}: train={n_train}, test={n_test}")

    return X_train, X_test, y_train, y_test, scaler, df
