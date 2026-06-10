"""评估 4 模型软投票集成"""

import sys
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess.io import GESTURE_NAMES
from src.decompose.features_enhanced import ENHANCED_FEATURE_NAMES
from src.decompose.features_temporal import extract_temporal_features


def evaluate():
    model_path = PROJECT_ROOT / "checkpoints" / "ensemble_models.pkl"
    if not model_path.exists():
        print("请先运行 python -m src.train")
        return

    models = joblib.load(model_path)
    top_idx = joblib.load(PROJECT_ROOT / "checkpoints" / "top_feature_idx.pkl")
    scaler = joblib.load(PROJECT_ROOT / "checkpoints" / "scaler.pkl")

    # 加载特征
    df_enh = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "features" / "enhanced_features.csv")
    temp_path = PROJECT_ROOT / "data" / "processed" / "features" / "temporal_features.npy"
    X_temp = np.load(temp_path)
    X_full = np.hstack([df_enh[ENHANCED_FEATURE_NAMES].values, X_temp])

    split_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "features" / "final_split.csv")
    test_mask = split_df["split"] == "test"
    X_test = X_full[test_mask.values][:, top_idx]
    y_test = split_df[test_mask]["label"].values
    test_df = split_df[test_mask].copy()

    X_test_s = scaler.transform(X_test)

    # 软投票预测
    p1 = models[0].predict_proba(X_test)
    p2 = models[1].predict_proba(X_test)
    p3 = models[2].predict_proba(X_test)
    p4 = models[3].predict_proba(X_test_s)
    avg = (p1 + p2 + p3 + p4) / 4
    y_pred = models[0].classes_[avg.argmax(1)]

    overall_acc = accuracy_score(y_test, y_pred)
    print("=" * 60)
    print("Ensemble: 2×LGBM + ExtraTrees + SVM, Top-100 Features (349D)")
    print("=" * 60)
    print(f"Overall Test Accuracy: {overall_acc:.3f}\n")

    present = sorted(set(y_test) | set(y_pred))
    names = [GESTURE_NAMES[i] for i in present]
    print(classification_report(y_test, y_pred, labels=present, target_names=names))

    # 各环境
    test_df = test_df.reset_index(drop=True)
    test_df["pred"] = y_pred
    print("Per-environment Accuracy:")
    for env in ["base", "wind_noise", "uv_radiation"]:
        e = test_df[test_df["env"] == env]
        if len(e) > 0:
            print(f"  {env:15s}: {accuracy_score(e['label'], e['pred']):.3f} ({len(e)} samples)")

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred, labels=present)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=names, yticklabels=names, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Ensemble Confusion Matrix (Acc: {overall_acc:.3f})")
    plt.tight_layout()
    fig.savefig(PROJECT_ROOT / "checkpoints" / "ensemble_confusion.png", dpi=150)
    print(f"\nConfusion matrix saved.")


if __name__ == "__main__":
    evaluate()
