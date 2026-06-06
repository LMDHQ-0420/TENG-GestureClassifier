"""路线B 评估：分类报告 + 混淆矩阵 + 各环境准确率 + 训练曲线"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess.io import GESTURE_NAMES
from src.routeB.dataset_b import ScalogramDataset, CACHE_DIR
from src.routeB.model_b import LightCNN

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
CKPT_PATH = PROJECT_ROOT / "checkpoints" / "routeB_cnn.pt"


def evaluate():
    if not CKPT_PATH.exists():
        print("请先运行 python -m src.routeB.train_b")
        return

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    model = LightCNN().to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    df = pd.read_csv(CACHE_DIR / "scalogram_split.csv")
    test_df = df[df["split"] == "test"].copy()
    test_loader = DataLoader(ScalogramDataset(test_df, augment=False), batch_size=64)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in test_loader:
            preds = model(X.to(DEVICE)).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    all_preds, all_labels = np.array(all_preds), np.array(all_labels)

    overall_acc = accuracy_score(all_labels, all_preds)
    print("=" * 60)
    print("Route B (CWT Scalogram + 2D-CNN)")
    print("=" * 60)
    print(f"Overall Test Accuracy: {overall_acc:.3f}\n")

    present = sorted(set(all_labels) | set(all_preds))
    names = [GESTURE_NAMES[i] for i in present]
    print(classification_report(all_labels, all_preds, labels=present, target_names=names))

    # 各环境准确率
    test_df = test_df.reset_index(drop=True)
    test_df["pred"] = all_preds
    print("Per-environment Accuracy:")
    for env in ["base", "wind_noise", "uv_radiation"]:
        e = test_df[test_df["env"] == env]
        if len(e) > 0:
            print(f"  {env:15s}: {accuracy_score(e['label'], e['pred']):.3f} ({len(e)} samples)")

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds, labels=present)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
                xticklabels=names, yticklabels=names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Route B Confusion Matrix (Test Acc: {overall_acc:.3f})")
    plt.tight_layout()
    fig.savefig(PROJECT_ROOT / "checkpoints" / "routeB_confusion.png", dpi=150)

    # 训练曲线
    h = ckpt["history"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(h["train_loss"], label="train")
    axes[0].plot(h["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(h["train_acc"], label="train")
    axes[1].plot(h["val_acc"], label="val")
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(PROJECT_ROOT / "checkpoints" / "routeB_curves.png", dpi=150)
    print("\nFigures saved.")


if __name__ == "__main__":
    evaluate()
