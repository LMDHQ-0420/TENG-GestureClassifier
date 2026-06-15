"""Run inference on test split and save pred/true/envs npy files to checkpoints/."""
import sys, random, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
import pandas as pd
import joblib, torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import log_transform
from src.decompose.features_enhanced import ENHANCED_FEATURE_NAMES
from src.train_transformer import FusionModel

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TOP_K = 100; BATCH_SIZE = 48; N_CLASSES = 10; SEQ_LEN = 512
ENVS = ['base', 'wind_noise', 'uv_radiation']

print(f"Device: {DEVICE}")

meta  = pd.read_csv(PROJECT_ROOT / 'data/processed/features/all_features.csv')
meta  = meta[meta['duration_ms'] >= 200].reset_index(drop=True)
X_enh = pd.read_csv(PROJECT_ROOT / 'data/processed/features/enhanced_features.csv')[ENHANCED_FEATURE_NAMES].values[:len(meta)]
X_temp = np.load(PROJECT_ROOT / 'data/processed/features/temporal_features.npy')
X_env  = np.load(PROJECT_ROOT / 'data/processed/features/envelope_features.npy')
X = log_transform(np.nan_to_num(np.hstack([X_enh, X_temp, X_env])))
y = meta['label'].values
print(f"Loaded features: {X.shape}")

top_idx     = joblib.load(PROJECT_ROOT / 'checkpoints/top_feature_idx_tr.pkl')
feat_scaler = joblib.load(PROJECT_ROOT / 'checkpoints/scaler_tr.pkl')
feat_models = joblib.load(PROJECT_ROOT / 'checkpoints/ensemble_models_tr.pkl')
X_sel = X[:, top_idx]
X_nn  = feat_scaler.transform(X_sel).astype(np.float32)

def per_env_split(meta, test_ratio=0.2, seed=42):
    train_idx, test_idx = [], []
    for env in ENVS:
        sub = meta[meta['env'] == env]
        labels = sub['label'].values
        idx = sub.index.tolist()
        if env == 'uv_radiation':
            t, tr = [], []
            for cls in np.unique(labels):
                ci = [i for i, l in zip(idx, labels) if l == cls]
                t.append(ci[0]); tr.extend(ci[1:])
            test_idx.extend(t); train_idx.extend(tr * 8)
        else:
            sss = StratifiedShuffleSplit(1, test_size=test_ratio, random_state=seed)
            tr, te = next(sss.split(idx, labels))
            train_idx.extend([idx[i] for i in tr])
            test_idx.extend([idx[i] for i in te])
    return train_idx, test_idx

def resize_signal(sig, target=SEQ_LEN):
    T, C = sig.shape
    if T == target: return sig.astype(np.float32)
    xo = np.linspace(0, 1, T); xn = np.linspace(0, 1, target)
    return np.stack([np.interp(xn, xo, sig[:, c]) for c in range(C)], axis=1).astype(np.float32)

class TENGDataset(Dataset):
    def __init__(self, meta, X_feat, idx):
        self.meta = meta.iloc[idx].reset_index(drop=True)
        self.X_feat = X_feat[idx]
        self.labels = self.meta['label'].values
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        row = self.meta.iloc[i]
        sig = np.load(PROJECT_ROOT / 'data' / row['npy_path']).astype(np.float32)
        sig = resize_signal(sig)
        sig = (sig - sig.mean(0)) / (sig.std(0) + 1e-8)
        return torch.from_numpy(sig.T), torch.from_numpy(self.X_feat[i].astype(np.float32)), int(self.labels[i])

_, test_idx = per_env_split(meta)
unique_test = list(dict.fromkeys(test_idx))
meta_te     = meta.iloc[unique_test].reset_index(drop=True)
X_nn_te     = X_nn[unique_test]
X_te_feat   = X_sel[unique_test]
envs_te     = meta['env'].values[unique_test]
y_te        = y[unique_test]
classes     = feat_models[0].classes_
print(f"Test set: {len(y_te)} samples")

model = FusionModel(feat_dim=TOP_K).to(DEVICE)
model.load_state_dict(torch.load(PROJECT_ROOT / 'checkpoints/fusion_transformer_best.pt',
                                  map_location=DEVICE, weights_only=True))
model.eval()

ds_test = TENGDataset(meta, X_nn, unique_test)
dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)
nn_proba = []
with torch.no_grad():
    for sig, feat, _ in dl_test:
        nn_proba.append(F.softmax(model(sig.to(DEVICE), feat.to(DEVICE)), dim=1).cpu().numpy())
nn_proba = np.vstack(nn_proba)
print("NN inference done")

nn_tta = np.zeros((len(meta_te), N_CLASSES), np.float32)
with torch.no_grad():
    for aug in range(15):
        sigs, feats = [], []
        for i, (_, row) in enumerate(meta_te.iterrows()):
            sig = np.load(PROJECT_ROOT / 'data' / row['npy_path']).astype(np.float32)
            sig = resize_signal(sig)
            sig = (sig - sig.mean(0)) / (sig.std(0) + 1e-8)
            sig = np.roll(sig, random.randint(-25, 25), axis=0) * np.random.uniform(0.88, 1.12)
            sigs.append(torch.from_numpy(sig.T))
            feats.append(torch.from_numpy(X_nn_te[i].astype(np.float32)))
        for s in range(0, len(meta_te), BATCH_SIZE):
            e = min(s + BATCH_SIZE, len(meta_te))
            p = F.softmax(model(torch.stack(sigs[s:e]).to(DEVICE),
                                torch.stack(feats[s:e]).to(DEVICE)), dim=1).cpu().numpy()
            nn_tta[s:e] += p
        print(f"  TTA aug {aug+1}/15 done")
nn_tta /= 15

X_s = feat_scaler.transform(X_te_feat)
m1, m2, m3, m4 = feat_models
feat_proba = (m1.predict_proba(X_te_feat) + m2.predict_proba(X_te_feat) +
              m3.predict_proba(X_te_feat) + m4.predict_proba(X_s)) / 4

final_proba = np.zeros_like(nn_proba)
for i in range(len(envs_te)):
    if envs_te[i] == 'uv_radiation':
        final_proba[i] = 0.35 * nn_tta[i] + 0.65 * feat_proba[i]
    else:
        final_proba[i] = 0.5 * nn_proba[i] + 0.5 * feat_proba[i]
pred_all = classes[final_proba.argmax(1)]

print(f"\nOverall accuracy: {accuracy_score(y_te, pred_all):.4f}")
for env in ENVS:
    m = envs_te == env
    print(f"  {env}: n={m.sum()}, acc={accuracy_score(y_te[m], pred_all[m]):.4f}")

np.save(PROJECT_ROOT / 'checkpoints/test_pred.npy', pred_all)
np.save(PROJECT_ROOT / 'checkpoints/test_true.npy', y_te)
np.save(PROJECT_ROOT / 'checkpoints/test_envs.npy', envs_te)
print("\nSaved: test_pred.npy, test_true.npy, test_envs.npy")
