"""路线B 训练：scalogram + 2D-CNN

防过拟合：数据增强、weight decay、标签平滑、early stopping、学习率调度。
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.routeB.dataset_b import (
    precompute_scalograms, make_split, ScalogramDataset, CACHE_DIR
)
from src.routeB.model_b import LightCNN

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 150
LR = 1.5e-3
WEIGHT_DECAY = 3e-4
PATIENCE = 30
SEED = 42


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            if train:
                optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)
    return total_loss / total, correct / total


def train():
    set_seed(SEED)

    # 预计算 scalogram（首次运行）
    if not (CACHE_DIR / "scalogram_index.csv").exists():
        precompute_scalograms()

    df = make_split()
    df.to_csv(CACHE_DIR / "scalogram_split.csv", index=False)

    train_ds = ScalogramDataset(df[df["split"] == "train"], augment=True)
    val_ds = ScalogramDataset(df[df["split"] == "val"], augment=False)
    test_ds = ScalogramDataset(df[df["split"] == "test"], augment=False)

    print(f"Device: {DEVICE}")
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = LightCNN().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    best_state = None
    patience_ctr = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS}  "
                  f"train_loss={tr_loss:.3f} train_acc={tr_acc:.3f}  "
                  f"val_loss={val_loss:.3f} val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    # 用最佳模型评估测试集
    model.load_state_dict(best_state)
    test_loss, test_acc = run_epoch(model, test_loader, criterion)
    print(f"\nBest Val Accuracy:  {best_val_acc:.3f}")
    print(f"Test Accuracy:      {test_acc:.3f}")

    ckpt_dir = PROJECT_ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    torch.save({"model_state_dict": best_state, "history": history,
                "test_acc": test_acc}, ckpt_dir / "routeB_cnn.pt")
    print(f"Model saved to: {ckpt_dir / 'routeB_cnn.pt'}")
    return test_acc


if __name__ == "__main__":
    train()
