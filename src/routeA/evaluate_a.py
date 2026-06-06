"""路线A 评估：分类报告 + 混淆矩阵 + 各环境准确率 + 特征重要度"""

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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess.io import GESTURE_NAMES
from src.decompose.features_rich import RICH_FEATURE_NAMES

SPLIT_PATH = PROJECT_ROOT / "data" / "processed" / "features" / "rich_features_split.csv"
MODEL_PATH = PROJECT_ROOT / "checkpoints" / "routeA_model.pkl"


def evaluate():
    if not MODEL_PATH.exists():
        print("请先运行 python -m src.routeA.train_a")
        return

    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(SPLIT_PATH)
    test_df = df[df["split"] == "test"].copy()

    X_test = test_df[RICH_FEATURE_NAMES].values
    y_test = test_df["label"].values
    y_pred = model.predict(X_test)

    overall_acc = accuracy_score(y_test, y_pred)
    print("=" * 60)
    print(f"Route A (VMD + Multi-domain Features + ExtraTrees)")
    print("=" * 60)
    print(f"Overall Test Accuracy: {overall_acc:.3f}\n")

    present = sorted(set(y_test) | set(y_pred))
    names = [GESTURE_NAMES[i] for i in present]
    print(classification_report(y_test, y_pred, labels=present, target_names=names))

    # 各环境准确率
    test_df["pred"] = y_pred
    print("Per-environment Accuracy:")
    for env in ["base", "wind_noise", "uv_radiation"]:
        e = test_df[test_df["env"] == env]
        if len(e) > 0:
            print(f"  {env:15s}: {accuracy_score(e['label'], e['pred']):.3f} ({len(e)} samples)")

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred, labels=present)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=names, yticklabels=names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Route A Confusion Matrix (Test Acc: {overall_acc:.3f})")
    plt.tight_layout()
    fig.savefig(PROJECT_ROOT / "checkpoints" / "routeA_confusion.png", dpi=150)
    print(f"\nConfusion matrix saved.")


if __name__ == "__main__":
    evaluate()
