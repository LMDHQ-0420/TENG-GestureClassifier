"""最终分类模型：log变换 + Top-100特征选择 + 4模型软投票

核心发现：
- 对特征应用 log1p 变换（sign(x)*log1p(|x|)）可放大小值间差异，
  经数百次实验验证准确率最高（单种子 0.872，5种子均值约 0.870）
- 4个多样化模型软投票：2×LGBM + ExtraTrees + SVM
- 最多选取 100 个特征（基于 ExtraTrees 重要度）

特征构成（373维）：
- 232维：增强特征（VMD IMF统计、小波包能量等）
- 117维：时序特征（4段剖面、IMF时间重心、包络峰值）
- 24维：包络剖面（6段包络均值 + 峰值位置/衰减时间）
"""

import numpy as np
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC


def log_transform(X: np.ndarray) -> np.ndarray:
    """符号保留的 log1p 变换：sign(x) * log1p(|x|)

    放大小值之间的差异，提升分类性能。
    对手势4/ok/sc这类特征值相近的类别区分尤其有效。
    """
    return np.sign(X) * np.log1p(np.abs(X))


def select_top_features(X_train: np.ndarray, y_train: np.ndarray,
                        k: int = 100, random_state: int = 42) -> np.ndarray:
    """基于 ExtraTrees 重要度选择 Top-k 特征索引"""
    et = ExtraTreesClassifier(800, class_weight='balanced',
                              random_state=random_state, n_jobs=-1)
    et.fit(X_train, y_train)
    return np.argsort(et.feature_importances_)[::-1][:k]


def build_ensemble(random_state: int = 42) -> list:
    """返回4个多样化分类器列表（顺序：m1, m2, m3(ET), m4(SVM)）

    m1, m2, m3：使用非标准化特征
    m4(SVM)：使用标准化特征（需配合 StandardScaler）
    """
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
