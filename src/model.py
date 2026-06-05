"""随机森林手势分类模型

输入：9D 特征向量 [MAV×3, WL×3, Ratio×3]
输出：10 类手势
"""

from sklearn.ensemble import RandomForestClassifier


def build_model(n_estimators: int = 200, random_state: int = 42) -> RandomForestClassifier:
    """创建随机森林分类器"""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
