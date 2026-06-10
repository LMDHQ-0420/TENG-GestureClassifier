"""최종 분류 모델: log 변환 + Top-100 특징 선택 + 4모델 소프트 투표

핵심 발견:
- 특징에 log1p 변환(sign(x)*log1p(|x|))을 적용하면 작은 값 간 차이가 증폭되어
  수백 번의 실험 중 가장 높은 정확도 달성 (단일 시드 0.872, 5종자 평균 ~0.870)
- 4개 다양한 모델 소프트 투표: 2×LGBM + ExtraTrees + SVM
- 최대 100개 특징 선택 (ET 중요도 기반)

특징 구성 (373차원):
- 232차원: 증강 특징 (VMD IMF 통계, 소파수 패킷 에너지 등)
- 117차원: 시간 특징 (4단 프로파일, IMF 시간 중심, 포락선 피크)
- 24차원: 포락선 프로파일 (6단 포락선 + 피크 위치/감쇠)
"""

import numpy as np
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC


def log_transform(X: np.ndarray) -> np.ndarray:
    """부호 보존 log1p 변환: sign(x) * log1p(|x|)

    작은 값 사이의 차이를 증폭시켜 분류 성능을 향상시킴.
    특히 손짓 4/ok/sc처럼 특징값이 유사한 클래스 구분에 효과적.
    """
    return np.sign(X) * np.log1p(np.abs(X))


def select_top_features(X_train: np.ndarray, y_train: np.ndarray,
                        k: int = 100, random_state: int = 42) -> np.ndarray:
    """ExtraTrees 중요도 기반 Top-k 특징 인덱스 선택"""
    et = ExtraTreesClassifier(800, class_weight='balanced',
                              random_state=random_state, n_jobs=-1)
    et.fit(X_train, y_train)
    return np.argsort(et.feature_importances_)[::-1][:k]


def build_ensemble(random_state: int = 42) -> list:
    """4개 다양한 분류기 목록 반환 (순서: m1, m2, m3(ET), m4(SVM))

    m1,m2,m3: 비표준화 특징 사용
    m4(SVM): 표준화 특징 사용 (StandardScaler 필요)
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
