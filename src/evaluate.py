"""最终评估：LightGBM + Top-100 特征

用法：python -m src.evaluate
"""

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

MODEL_PATH = PROJECT_ROOT / "checkpoints" / "lgbm_model.pkl"
IDX_PATH = PROJECT_ROOT / "checkpoints" / "top_feature_idx.pkl"
SPLIT_PATH = PROJECT_ROOT / "data" / "processed" / "features" / "final_split.csv"
ENHANCED_PATH = PROJECT_ROOT / "data" / "processed" / "features" / "enhanced_features.csv"


def evaluate():
    if not MODEL_PATH.exists():
        print("请先运行 python -m src.train")
        return

    model = joblib.load(MODEL_PATH)
    top_idx = joblib.load(IDX_PATH)

    df_feat = pd.read_csv(ENHANCED_PATH)
    split_df = pd.read_csv(SPLIT_PATH)
    test_ids = set(split_df[split_df["split"] == "test"]["seg_id"])

    test_df = df_feat[df_feat["seg_id"].isin(test_ids)].copy()
    X_test = test_df[ENHANCED_FEATURE_NAMES].values[:, top_idx]
    y_test = test_df["label"].values
    y_pred = model.predict(X_test)

    overall_acc = accuracy_score(y_test, y_pred)
    print("=" * 60)
    print("Final Model: LightGBM + Top-100 Features")
    print("=" * 60)
    print(f"Overall Test Accuracy: {overall_acc:.3f}\n")

    present = sorted(set(y_test) | set(y_pred))
    names = [GESTURE_NAMES[i] for i in present]
    print(classification_report(y_test, y_pred, labels=present, target_names=names))

    # 各环境准确率
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
    ax.set_title(f"LightGBM+Top100 Confusion Matrix (Acc: {overall_acc:.3f})")
    plt.tight_layout()
    fig.savefig(PROJECT_ROOT / "checkpoints" / "final_confusion.png", dpi=150)
    print(f"\nConfusion matrix saved.")

    # 特征重要度
    importances = model.feature_importances_
    feat_names = [ENHANCED_FEATURE_NAMES[i] for i in top_idx]
    top20 = np.argsort(importances)[::-1][:20]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(range(20), importances[top20][::-1],
            color=plt.cm.viridis(np.linspace(0.3, 0.9, 20)))
    ax.set_yticks(range(20))
    ax.set_yticklabels([feat_names[i] for i in top20[::-1]], fontsize=8)
    ax.set_xlabel("Feature Importance (LightGBM)")
    ax.set_title("Top-20 Feature Importance")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    fig.savefig(PROJECT_ROOT / "checkpoints" / "final_feature_importance.png", dpi=150)
    print("Feature importance saved.")


if __name__ == "__main__":
    evaluate()
