"""训练流水线：增强特征 + 时序特征 → Top-100 特征选择 → 4模型软投票

用法：python -m src.train
"""

import sys
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import build_ensemble, select_top_features
from src.decompose.features_enhanced import ENHANCED_FEATURE_NAMES, extract_enhanced_features
from src.decompose.features_temporal import extract_temporal_features

ENHANCED_PATH = PROJECT_ROOT / "data" / "processed" / "features" / "enhanced_features.csv"
TEMPORAL_PATH = PROJECT_ROOT / "data" / "processed" / "features" / "temporal_features.npy"
MODEL_DIR = PROJECT_ROOT / "checkpoints"
TOP_K = 100


def load_features():
    """加载增强特征 + 时序特征，合并为完整特征矩阵"""
    meta = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "features" / "all_features.csv")
    meta = meta[meta["duration_ms"] >= 200].reset_index(drop=True)
    y = meta["label"].values

    # 增强特征
    if ENHANCED_PATH.exists():
        df_enh = pd.read_csv(ENHANCED_PATH)
        X_enh = df_enh[ENHANCED_FEATURE_NAMES].values[:len(y)]
        print(f"增强特征: {X_enh.shape}")
    else:
        raise FileNotFoundError("请先运行 python -m src.routeA.pipeline_a 生成增强特征")

    # 时序特征（有缓存则直接加载）
    if TEMPORAL_PATH.exists():
        X_temp = np.load(TEMPORAL_PATH)
        print(f"时序特征(缓存): {X_temp.shape}")
    else:
        print("提取时序特征（首次运行较慢）...")
        X_temp = []
        for _, r in meta.iterrows():
            seg = np.load(PROJECT_ROOT / "data" / r["npy_path"]).astype(np.float32)
            X_temp.append(extract_temporal_features(seg))
        X_temp = np.nan_to_num(np.array(X_temp))
        np.save(TEMPORAL_PATH, X_temp)
        print(f"时序特征: {X_temp.shape}")

    X = np.hstack([X_enh, X_temp])
    print(f"合并特征: {X.shape}")
    return X, y, meta


def train(test_ratio: float = 0.2, random_state: int = 42):
    MODEL_DIR.mkdir(exist_ok=True)

    X, y, meta = load_features()
    n = len(y)

    itr, ite = train_test_split(np.arange(n), test_size=test_ratio,
                                stratify=y, random_state=random_state)

    # 特征选择（基于训练集 ExtraTrees 重要度）
    print(f"\n特征选择：从 {X.shape[1]} 维选 Top-{TOP_K}...")
    top_idx = select_top_features(X[itr], y[itr], k=TOP_K, random_state=random_state)

    X_sel = X[:, top_idx]
    X_train, X_test = X_sel[itr], X_sel[ite]
    y_train, y_test = y[itr], y[ite]

    # StandardScaler for SVM
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    print(f"Train: {len(y_train)}, Test: {len(y_test)}")

    # 训练 4 个模型
    models = build_ensemble(random_state=random_state)
    print("\n训练 4 个模型...")
    for i, m in enumerate(models):
        if i == 3:  # SVM 用标准化特征
            m.fit(X_train_s, y_train)
        else:
            m.fit(X_train, y_train)
        print(f"  模型{i+1} 训练完成")

    # 软投票预测
    p1 = models[0].predict_proba(X_test)
    p2 = models[1].predict_proba(X_test)
    p3 = models[2].predict_proba(X_test)
    p4 = models[3].predict_proba(X_test_s)
    avg = (p1 + p2 + p3 + p4) / 4
    pred = models[0].classes_[avg.argmax(1)]
    test_acc = accuracy_score(y_test, pred)

    # 单模型训练集准确率
    train_acc = accuracy_score(y_train, models[0].classes_[
        ((models[0].predict_proba(X_train) + models[1].predict_proba(X_train) +
          models[2].predict_proba(X_train) + models[3].predict_proba(X_train_s)) / 4).argmax(1)])

    print(f"\nTrain Accuracy (ensemble): {train_acc:.3f}")
    print(f"Test Accuracy  (ensemble): {test_acc:.3f}")

    # 保存
    joblib.dump(models, MODEL_DIR / "ensemble_models.pkl")
    joblib.dump(top_idx, MODEL_DIR / "top_feature_idx.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

    # 保存划分
    meta_out = meta[["seg_id", "env", "source_file", "gesture_name", "label"]].copy()
    meta_out["split"] = "train"
    meta_out.loc[meta_out.index[ite], "split"] = "test"
    meta_out.to_csv(PROJECT_ROOT / "data" / "processed" / "features" / "final_split.csv", index=False)

    print(f"\nModels saved: {MODEL_DIR / 'ensemble_models.pkl'}")
    return test_acc


if __name__ == "__main__":
    train()
