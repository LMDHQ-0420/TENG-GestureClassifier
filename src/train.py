"""训练流水线：增强特征 + Top-100 特征选择 + LightGBM

用法：python -m src.train
"""

import sys
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import build_model, select_top_features
from src.decompose.features_enhanced import ENHANCED_FEATURE_NAMES, extract_enhanced_features

ENHANCED_PATH = PROJECT_ROOT / "data" / "processed" / "features" / "enhanced_features.csv"
MODEL_DIR = PROJECT_ROOT / "checkpoints"
TOP_K = 100


def load_or_extract():
    """加载已缓存的增强特征，若不存在则重新提取"""
    if ENHANCED_PATH.exists():
        df = pd.read_csv(ENHANCED_PATH)
        print(f"加载增强特征: {len(df)} 样本 x {len(ENHANCED_FEATURE_NAMES)} 维")
    else:
        print("增强特征不存在，重新提取...")
        from src.preprocess.pipeline import run_all
        meta = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "features" / "all_features.csv")
        keep = []; X_list = []
        for i, r in meta.iterrows():
            seg_path = PROJECT_ROOT / "data" / r["npy_path"]
            if not seg_path.exists(): continue
            seg = np.load(seg_path)
            if seg.shape[0] < 200: continue
            X_list.append(extract_enhanced_features(seg))
            keep.append(i)
        X = np.array(X_list)
        df = meta.loc[keep].reset_index(drop=True)
        for j, name in enumerate(ENHANCED_FEATURE_NAMES):
            df[name] = X[:, j]
        df.to_csv(ENHANCED_PATH, index=False)
        print(f"提取完成: {len(df)} 样本 x {len(ENHANCED_FEATURE_NAMES)} 维")
    return df


def train(test_ratio: float = 0.2, random_state: int = 42):
    MODEL_DIR.mkdir(exist_ok=True)

    df = load_or_extract()
    X = df[ENHANCED_FEATURE_NAMES].values
    y = df["label"].values
    meta = df[["seg_id", "env", "source_file", "gesture_name", "label"]].copy()

    # 分层划分（按手势标签，不按环境，最大化判别效果）
    itr, ite = train_test_split(np.arange(len(y)), test_size=test_ratio,
                                stratify=y, random_state=random_state)

    # 特征选择（基于训练集）
    print(f"\n特征选择：从 {len(ENHANCED_FEATURE_NAMES)} 维选 Top-{TOP_K}...")
    top_idx = select_top_features(X[itr], y[itr], k=TOP_K, random_state=random_state)
    X_sel = X[:, top_idx]

    X_train, X_test = X_sel[itr], X_sel[ite]
    y_train, y_test = y[itr], y[ite]

    print(f"Train: {len(y_train)}, Test: {len(y_test)}")

    # 5-fold CV（在训练集上）
    model = build_model(random_state=random_state)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    print(f"\n5-Fold CV: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
    print(f"  folds: {[f'{s:.3f}' for s in cv_scores]}")

    # 训练并评估
    model.fit(X_train, y_train)
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"\nTrain Accuracy: {train_acc:.3f}")
    print(f"Test Accuracy:  {test_acc:.3f}")

    # 保存
    joblib.dump(model, MODEL_DIR / "lgbm_model.pkl")
    joblib.dump(top_idx, MODEL_DIR / "top_feature_idx.pkl")

    # 保存划分
    meta["split"] = "train"
    meta.loc[meta.index[ite], "split"] = "test"
    meta.to_csv(PROJECT_ROOT / "data" / "processed" / "features" / "final_split.csv", index=False)

    print(f"\nModel saved: {MODEL_DIR / 'lgbm_model.pkl'}")
    return test_acc


if __name__ == "__main__":
    train()
