"""数据集加载与 train/test 划分

按手势标签分层抽样（stratify by label），20% 测试集。
与基线保持相同接口，供 notebook 调用。
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .decompose.features_enhanced import ENHANCED_FEATURE_NAMES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENHANCED_PATH = PROJECT_ROOT / "data" / "processed" / "features" / "enhanced_features.csv"


def load_and_split(
    csv_path: Path = None,
    test_ratio: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, pd.DataFrame]:
    """从增强特征 CSV 加载并按标签分层划分 train/test

    Returns
    -------
    X_train, X_test, y_train, y_test, scaler, df_with_split
    """
    if csv_path is None:
        csv_path = ENHANCED_PATH

    df = pd.read_csv(csv_path)
    X = df[ENHANCED_FEATURE_NAMES].values
    y = df["label"].values

    itr, ite = train_test_split(
        np.arange(len(y)), test_size=test_ratio,
        stratify=y, random_state=random_state,
    )

    df["split"] = "train"
    df.loc[df.index[ite], "split"] = "test"

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[itr])
    X_test = scaler.transform(X[ite])
    y_train, y_test = y[itr], y[ite]

    print(f"Train: {len(y_train)} samples")
    print(f"Test:  {len(y_test)} samples")
    print(f"\nPer-environment split:")
    for env in ["base", "wind_noise", "uv_radiation"]:
        n_tr = len(df[(df["split"] == "train") & (df["env"] == env)])
        n_te = len(df[(df["split"] == "test") & (df["env"] == env)])
        print(f"  {env:15s}: train={n_tr}, test={n_te}")

    return X_train, X_test, y_train, y_test, scaler, df
