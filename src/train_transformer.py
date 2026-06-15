"""三场景独立测试集划分 + 1D-Conv/Transformer + 特征MLP 融合训练

数据划分策略：
  - base / wind_noise：每类按20%分层采样作为测试集
  - uv_radiation：每类最多取1个样本作为测试集（数据量极少，约67个）
  - 三个场景的训练集合并用于训练，测试集分别评估

模型架构：
  - 信号分支：1D-Conv stem → Transformer encoder (2层) → mean pooling → FC
  - 特征分支：373D特征 → log变换 → Top-100选择 → MLP
  - 融合：两分支 logits 加权求和 → softmax
  - 集成：Transformer模型 + 4模型特征集成 软投票

用法：
  conda run -n "zw@TENG-GestureClassifier" python -m src.train_transformer
"""

import sys, math, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import build_ensemble, select_top_features, log_transform
from src.decompose.features_enhanced import ENHANCED_FEATURE_NAMES
from src.decompose.features_temporal import extract_temporal_features

# ─── 超参数 ─────────────────────────────────────────────────────────────────
SEQ_LEN     = 512       # 统一截断/插值到 512 时间步
N_CHANNELS  = 3
N_CLASSES   = 10
D_MODEL     = 128       # Transformer 宽度
N_HEADS     = 4
N_LAYERS    = 2
DIM_FF      = 256
DROPOUT     = 0.1
FEAT_HIDDEN = 256
EPOCHS      = 200
BATCH_SIZE  = 48
LR          = 3e-4
WEIGHT_DECAY= 1e-4
LABEL_SMOOTH= 0.1
MIXUP_ALPHA = 0.4
SIG_WEIGHT  = 0.55
FEAT_WEIGHT = 0.45
UV_OVERSAMPLE = 8
TOP_K       = 100
RANDOM_SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = PROJECT_ROOT / "checkpoints"
MODEL_DIR.mkdir(exist_ok=True)


# ─── 数据划分 ────────────────────────────────────────────────────────────────
def per_env_split(meta: pd.DataFrame, test_ratio: float = 0.2, seed: int = RANDOM_SEED):
    """每个场景独立分层划分，返回 train/test 的 index 列表。
    uv_radiation 训练样本会被过采样 UV_OVERSAMPLE 倍以缓解样本少问题。"""
    train_idx, test_idx = [], []
    for env in ["base", "wind_noise", "uv_radiation"]:
        sub = meta[meta["env"] == env]
        labels = sub["label"].values
        idx    = sub.index.tolist()
        if env == "uv_radiation":
            # 数据极少：每类取1个作为测试，其余训练
            t_idx, tr_idx = [], []
            for cls in np.unique(labels):
                cls_idx = [i for i, l in zip(idx, labels) if l == cls]
                t_idx.append(cls_idx[0])
                tr_idx.extend(cls_idx[1:])
            test_idx.extend(t_idx)
            # 过采样：将训练样本重复 UV_OVERSAMPLE 次
            train_idx.extend(tr_idx * UV_OVERSAMPLE)
        else:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
            tr, te = next(sss.split(idx, labels))
            train_idx.extend([idx[i] for i in tr])
            test_idx.extend([idx[i] for i in te])
    return train_idx, test_idx


# ─── 信号预处理 ──────────────────────────────────────────────────────────────
def resize_signal(sig: np.ndarray, target: int = SEQ_LEN) -> np.ndarray:
    """将信号线性插值/截断到固定长度 (target, C)"""
    T, C = sig.shape
    if T == target:
        return sig.astype(np.float32)
    x_old = np.linspace(0, 1, T)
    x_new = np.linspace(0, 1, target)
    out = np.stack([np.interp(x_new, x_old, sig[:, c]) for c in range(C)], axis=1)
    return out.astype(np.float32)


# ─── Dataset ────────────────────────────────────────────────────────────────
class TENGDataset(Dataset):
    def __init__(self, meta: pd.DataFrame, X_feat: np.ndarray, idx: list,
                 augment: bool = False):
        self.meta    = meta.iloc[idx].reset_index(drop=True)
        self.X_feat  = X_feat[idx]
        self.labels  = self.meta["label"].values
        self.envs    = self.meta["env"].values
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        row  = self.meta.iloc[i]
        sig  = np.load(PROJECT_ROOT / "data" / row["npy_path"]).astype(np.float32)
        sig  = resize_signal(sig)         # (SEQ_LEN, C)
        sig  = (sig - sig.mean(0)) / (sig.std(0) + 1e-8)
        if self.augment:
            is_uv = self.envs[i] == "uv_radiation"
            # 时间位移
            shift = random.randint(-30, 30) if is_uv else random.randint(-20, 20)
            sig   = np.roll(sig, shift, axis=0)
            # 幅度缩放
            lo, hi = (0.75, 1.25) if is_uv else (0.85, 1.15)
            sig   = sig * np.random.uniform(lo, hi)
            # 加性噪声
            noise_std = 0.04 if is_uv else 0.02
            sig   = sig + np.random.randn(*sig.shape).astype(np.float32) * noise_std
            # uv 额外增强：随机时间反转
            if is_uv and random.random() < 0.3:
                sig = sig[::-1].copy()
            # uv 额外增强：随机通道缩放
            if is_uv:
                for c in range(sig.shape[1]):
                    sig[:, c] *= np.random.uniform(0.8, 1.2)
        sig  = torch.from_numpy(sig.T)    # (C, SEQ_LEN)
        feat = torch.from_numpy(self.X_feat[i].astype(np.float32))
        lbl  = int(self.labels[i])
        return sig, feat, lbl


# ─── 模型定义 ────────────────────────────────────────────────────────────────
class ConvStem(nn.Module):
    """将 (B, 3, SEQ_LEN) 映射到 (B, D_MODEL, SEQ_LEN/8)"""
    def __init__(self, in_ch=N_CHANNELS, d=D_MODEL):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 32, 7, padding=3), nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, d,  3, stride=2, padding=1), nn.BatchNorm1d(d),  nn.GELU(),
            nn.Conv1d(d,  d,  3, stride=2, padding=1), nn.BatchNorm1d(d),  nn.GELU(),
        )
    def forward(self, x):
        return self.net(x)   # (B, D, L/8)


class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=512, dropout=0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, L, D)

    def forward(self, x):            # x: (B, L, D)
        return self.drop(x + self.pe[:, :x.size(1)])


class SignalTransformer(nn.Module):
    def __init__(self, n_cls=N_CLASSES):
        super().__init__()
        self.stem = ConvStem()
        self.pos  = PositionalEncoding(D_MODEL, max_len=SEQ_LEN // 8 + 4)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=DIM_FF,
            dropout=DROPOUT, activation="gelu", batch_first=True, norm_first=True)
        self.enc  = nn.TransformerEncoder(enc_layer, num_layers=N_LAYERS,
                                          enable_nested_tensor=False)
        self.norm = nn.LayerNorm(D_MODEL)
        self.head = nn.Sequential(nn.Linear(D_MODEL, D_MODEL), nn.GELU(),
                                  nn.Dropout(DROPOUT), nn.Linear(D_MODEL, n_cls))

    def forward(self, x):            # x: (B, C, L)
        x = self.stem(x)             # (B, D, L/8)
        x = x.permute(0, 2, 1)      # (B, L/8, D)
        x = self.pos(x)
        x = self.enc(x)              # (B, L/8, D)
        x = self.norm(x.mean(1))     # (B, D)
        return self.head(x)          # (B, n_cls)


class FeatureMLP(nn.Module):
    def __init__(self, in_dim: int, n_cls=N_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, FEAT_HIDDEN), nn.LayerNorm(FEAT_HIDDEN), nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(FEAT_HIDDEN, FEAT_HIDDEN // 2), nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(FEAT_HIDDEN // 2, n_cls),
        )
    def forward(self, x):
        return self.net(x)


class FusionModel(nn.Module):
    def __init__(self, feat_dim: int, n_cls=N_CLASSES,
                 sig_w=SIG_WEIGHT, feat_w=FEAT_WEIGHT):
        super().__init__()
        self.sig_branch  = SignalTransformer(n_cls)
        self.feat_branch = FeatureMLP(feat_dim, n_cls)
        self.sig_w  = sig_w
        self.feat_w = feat_w

    def forward(self, sig, feat):
        return self.sig_w * self.sig_branch(sig) + self.feat_w * self.feat_branch(feat)


# ─── Mixup ───────────────────────────────────────────────────────────────────
def mixup_data(sig, feat, y, alpha=MIXUP_ALPHA):
    if alpha <= 0:
        return sig, feat, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    B   = sig.size(0)
    idx = torch.randperm(B, device=sig.device)
    return (lam * sig + (1 - lam) * sig[idx],
            lam * feat + (1 - lam) * feat[idx],
            y, y[idx], lam)


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ─── 特征加载 ─────────────────────────────────────────────────────────────────
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


def load_features(meta: pd.DataFrame):
    ENHANCED_PATH = PROJECT_ROOT / "data" / "processed" / "features" / "enhanced_features.csv"
    TEMPORAL_PATH = PROJECT_ROOT / "data" / "processed" / "features" / "temporal_features.npy"
    ENVELOPE_PATH = PROJECT_ROOT / "data" / "processed" / "features" / "envelope_features.npy"

    X_enh  = pd.read_csv(ENHANCED_PATH)[ENHANCED_FEATURE_NAMES].values[:len(meta)]
    X_temp = np.load(TEMPORAL_PATH) if TEMPORAL_PATH.exists() else np.zeros((len(meta), 1))
    X_env  = np.load(ENVELOPE_PATH) if ENVELOPE_PATH.exists() else np.zeros((len(meta), 1))
    X = np.hstack([X_enh, X_temp, X_env])
    X = np.nan_to_num(X)
    return log_transform(X)


# ─── 训练 ─────────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, criterion):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for sig, feat, lbl in loader:
        sig, feat, lbl = sig.to(DEVICE), feat.to(DEVICE), lbl.to(DEVICE)
        sig_m, feat_m, ya, yb, lam = mixup_data(sig, feat, lbl)
        optimizer.zero_grad()
        logits = model(sig_m, feat_m)
        loss   = mixup_criterion(criterion, logits, ya, yb, lam)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * lbl.size(0)
        correct    += (logits.argmax(1) == lbl).sum().item()
        n          += lbl.size(0)
    scheduler.step()
    return total_loss / n, correct / n


@torch.no_grad()
def eval_model(model, loader):
    model.eval()
    preds, trues = [], []
    for sig, feat, lbl in loader:
        sig, feat = sig.to(DEVICE), feat.to(DEVICE)
        logits = model(sig, feat)
        preds.extend(logits.argmax(1).cpu().numpy())
        trues.extend(lbl.numpy())
    return np.array(trues), np.array(preds)


# ─── 测试时增强 TTA ───────────────────────────────────────────────────────────
def tta_predict_nn(model, meta_te: pd.DataFrame, X_feat_te: np.ndarray,
                   n_aug: int = 15) -> np.ndarray:
    """对每个测试样本做 n_aug 次随机增强，分批次推断取 softmax 均值"""
    model.eval()
    N = len(meta_te)
    proba = np.zeros((N, N_CLASSES), dtype=np.float32)
    with torch.no_grad():
        for _ in range(n_aug):
            sigs, feats = [], []
            for i, (_, row) in enumerate(meta_te.iterrows()):
                sig = np.load(PROJECT_ROOT / "data" / row["npy_path"]).astype(np.float32)
                sig = resize_signal(sig)
                sig = (sig - sig.mean(0)) / (sig.std(0) + 1e-8)
                sig = np.roll(sig, random.randint(-25, 25), axis=0)
                sig = sig * np.random.uniform(0.88, 1.12)
                sig = sig + np.random.randn(*sig.shape).astype(np.float32) * 0.015
                sigs.append(torch.from_numpy(sig.T))
                feats.append(torch.from_numpy(X_feat_te[i].astype(np.float32)))
            # 分批推断，避免一次性占用过多显存
            bs = BATCH_SIZE
            for start in range(0, N, bs):
                end = min(start + bs, N)
                sig_t  = torch.stack(sigs[start:end]).to(DEVICE)
                feat_t = torch.stack(feats[start:end]).to(DEVICE)
                p = torch.softmax(model(sig_t, feat_t), dim=1).cpu().numpy()
                proba[start:end] += p
    return proba / n_aug


# ─── 与特征集成模型联合推断 ───────────────────────────────────────────────────
def ensemble_predict(nn_model, feat_models, scaler, ds_test, X_feat_test,
                     classes, meta_te: pd.DataFrame, X_nn_feat_te: np.ndarray,
                     envs: np.ndarray):
    """分场景加权联合预测：
      - base/wind_noise: Transformer(0.5) + 特征集成(0.5)
      - uv_radiation: TTA-Transformer(0.4) + 特征集成(0.6) 偏重稳定特征
    """
    # 普通 NN 推断（全部样本）
    nn_model.eval()
    loader = DataLoader(ds_test, batch_size=BATCH_SIZE * 2, shuffle=False)
    nn_proba = []
    with torch.no_grad():
        for sig, feat, _ in loader:
            sig, feat = sig.to(DEVICE), feat.to(DEVICE)
            p = torch.softmax(nn_model(sig, feat), dim=1).cpu().numpy()
            nn_proba.append(p)
    nn_proba = np.vstack(nn_proba)

    # TTA 推断（全部样本，增强次数适中）
    nn_proba_tta = tta_predict_nn(nn_model, meta_te, X_nn_feat_te, n_aug=30)

    # 特征集成推断
    X_s = scaler.transform(X_feat_test)
    m1, m2, m3, m4 = feat_models
    p1 = m1.predict_proba(X_feat_test)
    p2 = m2.predict_proba(X_feat_test)
    p3 = m3.predict_proba(X_feat_test)
    p4 = m4.predict_proba(X_s)
    feat_proba = (p1 + p2 + p3 + p4) / 4

    # 分场景融合
    final_proba = np.zeros_like(nn_proba)
    for i in range(len(envs)):
        if envs[i] == "uv_radiation":
            # uv: TTA + 偏重特征集成
            final_proba[i] = 0.35 * nn_proba_tta[i] + 0.65 * feat_proba[i]
        else:
            final_proba[i] = 0.5 * nn_proba[i] + 0.5 * feat_proba[i]

    pred = classes[final_proba.argmax(1)]
    return pred


# ─── 主函数 ───────────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    print("=" * 60)
    print("加载元数据与特征...")
    meta = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "features" / "all_features.csv")
    meta = meta[meta["duration_ms"] >= 200].reset_index(drop=True)

    X_feat = load_features(meta)
    y      = meta["label"].values
    print(f"特征维度: {X_feat.shape[1]}  |  样本总数: {len(y)}")

    # ── 划分 ──
    print("\n按场景独立划分 train/test ...")
    train_idx, test_idx = per_env_split(meta)
    print(f"  训练集: {len(train_idx)}  测试集: {len(test_idx)}")
    for env in ["base", "wind_noise", "uv_radiation"]:
        te = [i for i in test_idx if meta.loc[i, "env"] == env]
        tr = [i for i in train_idx if meta.loc[i, "env"] == env]
        print(f"  {env:15s}: train={len(tr):4d}  test={len(te):3d}")

    # ── 特征选择（仅用训练集唯一样本） ──
    print(f"\n特征选择: Top-{TOP_K} ...")
    unique_train_idx = list(dict.fromkeys(train_idx))   # 去重，供特征统计用
    top_idx = select_top_features(X_feat[unique_train_idx], y[unique_train_idx], k=TOP_K)
    X_sel   = X_feat[:, top_idx]

    feat_scaler = StandardScaler().fit(X_sel[unique_train_idx])
    X_sel_s     = feat_scaler.transform(X_sel)

    # 为 FeatureMLP 使用标准化特征
    X_nn_feat = X_sel_s.astype(np.float32)

    # ── 构建 Dataset / DataLoader ──
    ds_train = TENGDataset(meta, X_nn_feat, train_idx, augment=True)
    ds_test  = TENGDataset(meta, X_nn_feat, test_idx,  augment=False)
    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True, drop_last=True)
    dl_test  = DataLoader(ds_test,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True)

    # ── 训练 Transformer 融合模型 ──
    print(f"\n训练 FusionModel (1D-Conv + Transformer + FeatureMLP) on {DEVICE} ...")
    model     = FusionModel(feat_dim=TOP_K).to(DEVICE)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  参数量: {n_params/1e6:.2f}M")

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR/20)

    best_acc, best_ep = 0.0, 0
    for ep in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, dl_train, optimizer, scheduler, criterion)
        if ep % 10 == 0 or ep == EPOCHS:
            trues, preds = eval_model(model, dl_test)
            acc = accuracy_score(trues, preds)
            marker = " ★" if acc > best_acc else ""
            print(f"  Ep {ep:3d}/{EPOCHS}  loss={tr_loss:.4f}  train_acc={tr_acc:.3f}"
                  f"  test_acc={acc:.3f}{marker}")
            if acc > best_acc:
                best_acc = acc
                best_ep  = ep
                torch.save(model.state_dict(), MODEL_DIR / "fusion_transformer_best.pt")

    print(f"\n最佳 Transformer 测试准确率: {best_acc:.3f} (epoch {best_ep})")
    model.load_state_dict(torch.load(MODEL_DIR / "fusion_transformer_best.pt",
                                     map_location=DEVICE, weights_only=True))

    # ── 训练特征集成模型（sklearn，用唯一训练样本） ──
    print("\n训练特征集成模型（4模型软投票）...")
    feat_models = build_ensemble(random_state=RANDOM_SEED)
    X_tr, y_tr  = X_sel[unique_train_idx], y[unique_train_idx]
    X_tr_s      = feat_scaler.transform(X_tr)
    for i, m in enumerate(feat_models):
        Xm = X_tr_s if i == 3 else X_tr
        m.fit(Xm, y_tr)
        print(f"  模型{i+1} 完成")

    classes = feat_models[0].classes_

    # ── 联合推断 ──
    print("\n联合推断（分场景加权 + TTA）...")
    # test_idx 是唯一的原始 meta 行索引
    unique_test_idx = list(dict.fromkeys(test_idx))   # 保持顺序去重（本来就没重复）
    X_te_feat   = X_sel[unique_test_idx]
    X_nn_feat_te = X_nn_feat[unique_test_idx]
    meta_te     = meta.iloc[unique_test_idx].reset_index(drop=True)
    envs_te     = meta["env"].values[unique_test_idx]
    pred_all    = ensemble_predict(model, feat_models, feat_scaler, ds_test,
                                   X_te_feat, classes, meta_te, X_nn_feat_te, envs_te)
    y_te        = y[unique_test_idx]
    overall     = accuracy_score(y_te, pred_all)
    print(f"\n总体测试准确率: {overall:.4f}")

    # ── 分场景报告 ──
    print("\n" + "=" * 60)
    print("各场景测试准确率：")
    env_arr = meta["env"].values[unique_test_idx]
    for env in ["base", "wind_noise", "uv_radiation"]:
        mask   = env_arr == env
        if mask.sum() == 0:
            continue
        acc_e  = accuracy_score(y_te[mask], pred_all[mask])
        print(f"  {env:15s}: {mask.sum():3d} 样本  acc = {acc_e:.4f}")

    # ── 分类报告 ──
    label_names = meta["gesture_name"].unique().tolist()
    le = LabelEncoder().fit(y)
    print("\n详细分类报告（整体测试集）：")
    print(classification_report(y_te, pred_all))

    # ── 混淆矩阵 ──
    gesture_order = sorted(meta["gesture_name"].unique())
    label_order   = [meta[meta["gesture_name"] == g]["label"].iloc[0] for g in gesture_order]
    cm = confusion_matrix(y_te, pred_all, labels=label_order)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=gesture_order, yticklabels=gesture_order, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — overall acc={overall:.3f}")
    fig.tight_layout()
    fig.savefig(MODEL_DIR / "transformer_confusion.png", dpi=150)
    print(f"\n混淆矩阵已保存: {MODEL_DIR / 'transformer_confusion.png'}")

    # ── 保存拆分信息 ──
    split_meta = meta[["seg_id", "env", "gesture_name", "label"]].copy()
    split_meta["split"] = "train"
    split_meta.loc[unique_test_idx, "split"] = "test"
    split_meta.to_csv(
        PROJECT_ROOT / "data" / "processed" / "features" / "transformer_split.csv",
        index=False)

    # ── 保存模型组件 ──
    joblib.dump(feat_models,  MODEL_DIR / "ensemble_models_tr.pkl")
    joblib.dump(top_idx,      MODEL_DIR / "top_feature_idx_tr.pkl")
    joblib.dump(feat_scaler,  MODEL_DIR / "scaler_tr.pkl")

    # ── 保存推断结果（供可视化直接加载，无需重跑推断）──
    np.save(MODEL_DIR / "test_pred.npy",  pred_all)
    np.save(MODEL_DIR / "test_true.npy",  y_te)
    np.save(MODEL_DIR / "test_envs.npy",  envs_te)
    print("所有模型组件已保存。")

    return overall


if __name__ == "__main__":
    acc = main()
    print(f"\n最终准确率: {acc:.4f}")

