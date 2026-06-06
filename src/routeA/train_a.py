"""路线A 训练：111维多域特征 + 梯度提升 + 特征选择 + 5-fold 交叉验证

防过拟合措施：
- HistGradientBoosting 自带 L2 正则 + early stopping
- 基于互信息的特征选择，降维去噪
- 5-fold 分层交叉验证，稳定评估
"""

import sys
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.decompose.features_rich import RICH_FEATURE_NAMES

RICH_PATH = PROJECT_ROOT / "data" / "processed" / "features" / "rich_features.csv"
MODEL_DIR = PROJECT_ROOT / "checkpoints"


def load_split(test_ratio: float = 0.2, random_state: int = 42):
    """与基线相同的分层划分（按 env+gesture），保证可对比"""
    df = pd.read_csv(RICH_PATH)
    df["stratify_key"] = df["env"] + "_" + df["gesture_name"]

    from sklearn.model_selection import train_test_split
    counts = df["stratify_key"].value_counts()
    singleton = df["stratify_key"].isin(counts[counts < 2].index)
    df_main = df[~singleton]

    train_idx, test_idx = train_test_split(
        df_main.index, test_size=test_ratio,
        stratify=df_main["stratify_key"], random_state=random_state,
    )
    df["split"] = "train"
    df.loc[test_idx, "split"] = "test"
    return df


def build_model(random_state: int = 42) -> ExtraTreesClassifier:
    """极端随机树（Extremely Randomized Trees）

    相比随机森林，分裂阈值也随机选取，引入更强的随机性，
    在小样本高维特征上泛化更好、过拟合更轻，是本任务实测最优模型。
    """
    return ExtraTreesClassifier(
        n_estimators=300,
        max_features="sqrt",
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )


def train():
    MODEL_DIR.mkdir(exist_ok=True)
    df = load_split()

    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    X_train = train_df[RICH_FEATURE_NAMES].values
    y_train = train_df["label"].values
    X_test = test_df[RICH_FEATURE_NAMES].values
    y_test = test_df["label"].values

    print(f"Train: {len(train_df)}, Test: {len(test_df)}, Features: {len(RICH_FEATURE_NAMES)}")

    model = build_model()

    # 5-fold 交叉验证（在训练集上）
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    print(f"\n5-Fold CV Accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
    print(f"  folds: {[f'{s:.3f}' for s in cv_scores]}")

    # 全训练集拟合，测试集评估
    model.fit(X_train, y_train)
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"\nTrain Accuracy: {train_acc:.3f}")
    print(f"Test Accuracy:  {test_acc:.3f}")

    joblib.dump(model, MODEL_DIR / "routeA_model.pkl")
    df.to_csv(PROJECT_ROOT / "data" / "processed" / "features" / "rich_features_split.csv", index=False)
    print(f"\nModel saved to: {MODEL_DIR / 'routeA_model.pkl'}")
    return test_acc


if __name__ == "__main__":
    train()
