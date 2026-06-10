"""最终分类模型：LightGBM + Top-100 特征选择

基于实验结论：
- LightGBM 比 ExtraTrees/RF/SVM 准确率更高（0.847 vs 0.816）
- Top-100 特征（按 ExtraTrees 重要度选取）比全部 232 维更好（去除噪声维度）
- 参数：num_leaves=63, lr=0.03, n_estimators=800
"""

import numpy as np
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesClassifier


def select_top_features(X_train: np.ndarray, y_train: np.ndarray,
                        k: int = 100, random_state: int = 42) -> np.ndarray:
    """用 ExtraTrees 重要度选出 Top-k 特征索引

    Returns
    -------
    np.ndarray of shape [k], feature indices sorted by importance (descending)
    """
    et = ExtraTreesClassifier(800, class_weight='balanced',
                              random_state=random_state, n_jobs=-1)
    et.fit(X_train, y_train)
    return np.argsort(et.feature_importances_)[::-1][:k]


def build_model(random_state: int = 42) -> lgb.LGBMClassifier:
    """构建 LightGBM 分类器

    num_leaves=63, lr=0.03, n_estimators=800, min_child_samples=10
    subsample=0.8, colsample_bytree=0.8, class_weight=balanced
    """
    return lgb.LGBMClassifier(
        n_estimators=800,
        num_leaves=63,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=10,
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )
