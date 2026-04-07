"""
ridge_model.py
==============
Ridge Regression 기반 RUL 예측
담당: [Ridge 담당자 이름]

입력 데이터: X_ml_train (N, 510 or 540)  ← 2D Flatten
이유: Ridge는 선형 모델로 3D 텐서 불가, W*F 차원의 flatten 사용
"""
import numpy as np
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from src.evaluate import evaluate_all

DATASETS   = ['FD001', 'FD002', 'FD003', 'FD004']
DATA_DIR   = 'data/processed'
OUTPUT_DIR = 'outputs/predictions'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data(dataset: str):
    base = os.path.join(DATA_DIR, dataset)
    return (
        np.load(f'{base}/X_ml_train.npy'), np.load(f'{base}/y_ml_train.npy'),
        np.load(f'{base}/X_ml_val.npy'),   np.load(f'{base}/y_ml_val.npy'),
        np.load(f'{base}/X_ml_test.npy'),  np.load(f'{base}/y_test.npy'),
    )


def train_ridge(dataset: str):
    print(f"\n{'='*50}\n  Ridge — {dataset}\n{'='*50}")
    X_tr, y_tr, X_vl, y_vl, X_te, y_te = load_data(dataset)

    # ── 하이퍼파라미터 탐색 ──────────────────────────────
    # TODO: alpha 범위 조정 가능
    param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
    model = GridSearchCV(Ridge(), param_grid, cv=3, scoring='neg_root_mean_squared_error')
    model.fit(X_tr, y_tr)

    best_alpha = model.best_params_['alpha']
    print(f"  최적 alpha: {best_alpha}")

    # ── val 성능 확인 (공통 evaluate_all 사용) ───────────
    val_pred = model.predict(X_vl)
    evaluate_all(y_vl, val_pred, model_name='Ridge_val', dataset=dataset, save=False)

    # ── 최종 예측 ────────────────────────────────────────
    pred = model.predict(X_te)
    pred = np.clip(pred, 0, 125)  # RUL 범위 제한

    # ── 저장 ─────────────────────────────────────────────
    out_path = os.path.join(OUTPUT_DIR, f'ridge_{dataset}.npy')
    np.save(out_path, pred)
    print(f"  예측 저장: {out_path}")

    # ── 평가 ─────────────────────────────────────────────
    evaluate_all(y_te, pred, model_name='Ridge', dataset=dataset)

    return model, pred


if __name__ == '__main__':
    for ds in DATASETS:
        train_ridge(ds)
    print("\n모든 데이터셋 Ridge 완료")