"""最终分类模型：4模型软投票集成

基础特征（232维）+ 时序特征（117维）= 349维
Top-100 特征选择（ExtraTrees 重要度）
4 个多样化分类器软投票：LGBM × 2 + ExtraTrees + SVM

实验结果（5种子均值）：
- LightGBM Top-100：0.847 ± 0.017
- 4模型软投票 Top-100：0.870 ± 0.021（当前最优）
"""

import numpy as np
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


def select_top_features(X_train: np.ndarray, y_train: np.ndarray,
                        k: int = 100, random_state: int = 42) -> np.ndarray:
    et = ExtraTreesClassifier(800, class_weight='balanced',
                              random_state=random_state, n_jobs=-1)
    et.fit(X_train, y_train)
    return np.argsort(et.feature_importances_)[::-1][:k]


def build_ensemble(random_state: int = 42) -> list:
    """返回 4 个多样化分类器列表（需依次 fit）"""
    return [
        lgb.LGBMClassifier(
            n_estimators=800, num_leaves=63, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
            class_weight='balanced', random_state=random_state, n_jobs=-1, verbose=-1,
        ),
        lgb.LGBMClassifier(
            n_estimators=600, num_leaves=31, learning_rate=0.05,
            subsample=0.7, colsample_bytree=0.7, min_child_samples=15,
            class_weight='balanced', random_state=0, n_jobs=-1, verbose=-1,
        ),
        ExtraTreesClassifier(
            1000, class_weight='balanced', random_state=random_state, n_jobs=-1,
        ),
        SVC(C=50, gamma='scale', class_weight='balanced',
            probability=True, random_state=random_state),
    ]
