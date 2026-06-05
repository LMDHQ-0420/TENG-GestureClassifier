"""训练随机森林模型

用法：python -m src.train
"""

import sys
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import build_model
from src.dataset import load_and_split


def train():
    features_path = PROJECT_ROOT / "data" / "processed" / "features" / "all_features.csv"
    model_dir = PROJECT_ROOT / "checkpoints"
    model_dir.mkdir(exist_ok=True)

    X_train, X_test, y_train, y_test, scaler, df = load_and_split(features_path)

    # 保存划分结果
    split_path = PROJECT_ROOT / "data" / "processed" / "features" / "all_features_split.csv"
    df.to_csv(split_path, index=False)
    print(f"\nSplit saved to: {split_path}")

    # 训练
    model = build_model()
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"\nTrain Accuracy: {train_acc:.3f}")
    print(f"Test Accuracy:  {test_acc:.3f}")

    # 保存模型和 scaler
    joblib.dump(model, model_dir / "random_forest.pkl")
    joblib.dump(scaler, model_dir / "scaler.pkl")
    print(f"\nModel saved to: {model_dir / 'random_forest.pkl'}")


if __name__ == "__main__":
    train()
