# TENG Gesture Recognition

[中文版](README.zh.md)

Gesture classification system based on triboelectric nanogenerator (TENG) sensors. The sensor captures 3-channel voltage signals (CH1/CH3/CH5, 1000 Hz) and classifies 10 gestures across three environments.

## Dataset

| Environment | Segments | Test split |
|-------------|----------|------------|
| Base | 934 | 20% stratified |
| Wind noise | 174 | 20% stratified |
| UV radiation | 67 | 1 sample/class |

Gesture classes (10): `1` `2` `3` `4` `5` `go_the_way` `ok` `sc` `stop` `wave`

## Results

| Scene | Test samples | Accuracy |
|-------|-------------|----------|
| Base | 187 | 90.9% |
| Wind noise | 35 | 91.4% |
| UV radiation | 10 | 80.0% |
| **Overall** | **232** | **90.5%** |

## Approach

**Features**: VMD (K=4) IMF statistics (232D) + temporal profile (117D) + envelope (24D) = 373D, log-transformed, ExtraTrees Top-100 selection.

**Model**: FusionModel combining a 1D-Conv + TransformerEncoder branch (0.43M params) with a 4-model sklearn ensemble (2×LightGBM + ExtraTrees + SVC). UV radiation scene uses TTA×15 and ×8 oversampling to compensate for limited data.

## Structure

```
├── data/processed/
│   ├── segments/               per-gesture signal segments (.npy)
│   └── features/               feature files (.csv / .npy)
├── src/
│   ├── preprocess/             signal segmentation and filtering
│   ├── decompose/              VMD, wavelet, feature extraction
│   ├── train_transformer.py    training entry point
│   └── model.py                feature selection, sklearn ensemble
├── scripts/
│   └── save_predictions.py     regenerate inference result npys
├── checkpoints/                model weights and inference results
├── notebooks/
│   └── results_visualization.ipynb
└── svg/                        output figures
```

## Setup

```bash
conda create -n TENG-GestureClassifier python=3.11
conda activate TENG-GestureClassifier
pip install -r requirements.txt
# Install PyTorch separately: https://pytorch.org/get-started/locally/
```

## Usage

```bash
# Train
python -m src.train_transformer

# Regenerate inference results after training
python scripts/save_predictions.py
```

Open `notebooks/results_visualization.ipynb` to view results.
