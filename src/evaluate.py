"""评估随机森林模型：分类报告 + 混淆矩阵 + 特征重要度 + 各环境准确率

用法：python -m src.evaluate
"""

import sys
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess.io import GESTURE_NAMES
from src.preprocess.features import FEATURE_NAMES
from src.dataset import load_and_split


def evaluate():
    features_path = PROJECT_ROOT / "data" / "processed" / "features" / "all_features.csv"
    model_path = PROJECT_ROOT / "checkpoints" / "random_forest.pkl"
    scaler_path = PROJECT_ROOT / "checkpoints" / "scaler.pkl"

    if not model_path.exists():
        print(f"模型文件不存在：{model_path}")
        print("请先运行 python -m src.train")
        return

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    X_train, X_test, y_train, y_test, _, df = load_and_split(features_path)

    y_pred = model.predict(X_test)

    # Classification Report
    print("\n" + "=" * 60)
    print("Classification Report (Overall)")
    print("=" * 60)
    present_labels = sorted(set(y_test) | set(y_pred))
    present_names = [GESTURE_NAMES[i] for i in present_labels]
    print(classification_report(y_test, y_pred,
                                labels=present_labels,
                                target_names=present_names))

    # Per-environment accuracy
    test_df = df[df["split"] == "test"].copy()
    test_df["pred"] = y_pred
    print("=" * 60)
    print("Per-environment Accuracy")
    print("=" * 60)
    for env in ["base", "wind_noise", "uv_radiation"]:
        env_df = test_df[test_df["env"] == env]
        if len(env_df) > 0:
            acc = accuracy_score(env_df["label"], env_df["pred"])
            print(f"  {env:15s}: {acc:.3f} ({len(env_df)} samples)")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=present_labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=present_names, yticklabels=present_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    overall_acc = accuracy_score(y_test, y_pred)
    ax.set_title(f"Confusion Matrix (Test Accuracy: {overall_acc:.3f})")
    plt.tight_layout()
    fig_path = PROJECT_ROOT / "checkpoints" / "confusion_matrix.png"
    fig.savefig(fig_path, dpi=150)
    plt.show()
    print(f"\nConfusion matrix saved to: {fig_path}")

    # Feature Importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(FEATURE_NAMES)))
    ax.barh(range(len(FEATURE_NAMES)),
            importances[indices[::-1]],
            color=colors)
    ax.set_yticks(range(len(FEATURE_NAMES)))
    ax.set_yticklabels([FEATURE_NAMES[i] for i in indices[::-1]])
    ax.set_xlabel("Feature Importance (Gini)")
    ax.set_title("Random Forest Feature Importance")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    fig_path2 = PROJECT_ROOT / "checkpoints" / "feature_importance.png"
    fig.savefig(fig_path2, dpi=150)
    plt.show()
    print(f"Feature importance saved to: {fig_path2}")

    print("\nFeature ranking:")
    for i, idx in enumerate(indices):
        print(f"  {i+1}. {FEATURE_NAMES[idx]:12s} = {importances[idx]:.4f}")


if __name__ == "__main__":
    evaluate()
