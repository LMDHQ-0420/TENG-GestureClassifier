"""训练流水线：增强特征 + 时序特征 + 包络特征 → log变换 → Top-100选择 → 4模型软投票

用法：python -m src.train
"""

import sys
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import build_ensemble, select_top_features, log_transform
from src.decompose.features_enhanced import ENHANCED_FEATURE_NAMES
from src.decompose.features_temporal import extract_temporal_features

ENHANCED_PATH = PROJECT_ROOT / "data" / "processed" / "features" / "enhanced_features.csv"
TEMPORAL_PATH = PROJECT_ROOT / "data" / "processed" / "features" / "temporal_features.npy"
ENVELOPE_PATH = PROJECT_ROOT / "data" / "processed" / "features" / "envelope_features.npy"
MODEL_DIR = PROJECT_ROOT / "checkpoints"
TOP_K = 100


def _envelope_profile(seg):
    from scipy.signal import hilbert
    f = []
    for ch in range(seg.shape[1]):
        sig = seg[:, ch] - seg[:, ch].mean()
        env = np.abs(hilbert(sig))
        env_s = np.convolve(env, np.ones(30) / 30, mode='same')
        env_n = env_s / (env_s.max() + 1e-8)
        T = len(env_n)
        for i in range(6):
            f.append(float(env_n[int(i * T / 6):int((i + 1) * T / 6)].mean()))
        peak_idx = int(np.argmax(env_n))
        f.append(float(peak_idx / T))
        tail = env_n[peak_idx:]
        f.append(float(np.argmax(tail < 0.5) / T if (tail < 0.5).any() else 1.0))
    return f


def load_features():
    """加载三类特征并应用 log 变换"""
    meta = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "features" / "all_features.csv")
    meta = meta[meta["duration_ms"] >= 200].reset_index(drop=True)
    y = meta["label"].values

    # 增强特征
    X_enh = pd.read_csv(ENHANCED_PATH)[ENHANCED_FEATURE_NAMES].values[:len(y)]

    # 时序特征
    if TEMPORAL_PATH.exists():
        X_temp = np.load(TEMPORAL_PATH)
    else:
        print("提取时序特征中...")
        X_temp = np.array([extract_temporal_features(np.load(PROJECT_ROOT / "data" / r["npy_path"]).astype(np.float32))
                           for _, r in meta.iterrows()])
        X_temp = np.nan_to_num(X_temp)
        np.save(TEMPORAL_PATH, X_temp)

    # 包络特征
    if ENVELOPE_PATH.exists():
        X_env = np.load(ENVELOPE_PATH)
    else:
        print("提取包络特征中...")
        X_env = np.array([_envelope_profile(np.load(PROJECT_ROOT / "data" / r["npy_path"]).astype(np.float32))
                          for _, r in meta.iterrows()])
        X_env = np.nan_to_num(X_env)
        np.save(ENVELOPE_PATH, X_env)

    X = np.hstack([X_enh, X_temp, X_env])
    print(f"原始特征: {X.shape[1]}维  →  应用 log 变换")
    X = log_transform(X)
    return X, y, meta


def train(test_ratio: float = 0.2, random_state: int = 42):
    MODEL_DIR.mkdir(exist_ok=True)
    X, y, meta = load_features()
    n = len(y)

    itr, ite = train_test_split(np.arange(n), test_size=test_ratio,
                                stratify=y, random_state=random_state)

    print(f"\n特征选择: {X.shape[1]}维 → Top-{TOP_K}...")
    top_idx = select_top_features(X[itr], y[itr], k=TOP_K, random_state=random_state)
    X_sel = X[:, top_idx]

    X_train, X_test = X_sel[itr], X_sel[ite]
    y_train, y_test = y[itr], y[ite]

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    print(f"训练集: {len(y_train)}, 测试集: {len(y_test)}")

    models = build_ensemble(random_state=random_state)
    print("\n4个模型训练中...")
    for i, m in enumerate(models):
        Xm = X_train_s if i == 3 else X_train
        m.fit(Xm, y_train)
        print(f"  模型{i+1} 完成")

    p1 = models[0].predict_proba(X_test)
    p2 = models[1].predict_proba(X_test)
    p3 = models[2].predict_proba(X_test)
    p4 = models[3].predict_proba(X_test_s)
    avg = (p1 + p2 + p3 + p4) / 4
    pred = models[0].classes_[avg.argmax(1)]
    test_acc = accuracy_score(y_test, pred)

    print(f"\n测试准确率（4模型软投票）: {test_acc:.3f}")

    joblib.dump(models, MODEL_DIR / "ensemble_models.pkl")
    joblib.dump(top_idx, MODEL_DIR / "top_feature_idx.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

    meta_out = meta[["seg_id", "env", "source_file", "gesture_name", "label"]].copy()
    meta_out["split"] = "train"
    meta_out.loc[meta_out.index[ite], "split"] = "test"
    meta_out.to_csv(PROJECT_ROOT / "data" / "processed" / "features" / "final_split.csv", index=False)

    print(f"模型已保存: {MODEL_DIR / 'ensemble_models.pkl'}")
    return test_acc


if __name__ == "__main__":
    train()
