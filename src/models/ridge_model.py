"""
ridge_model.py
==============
Ridge Regression 기반 RUL 예측
담당: 김동환

입력 데이터: X_ml_train (N, 510 or 540)  ← 2D Flatten
이유: Ridge는 선형 모델로 3D 텐서 불가, W*F 차원의 flatten 사용
"""
import numpy as np
import os, sys, time
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from src.evaluate import evaluate_all

DATASETS   = ['FD001', 'FD002', 'FD003', 'FD004']
DATA_DIR   = 'data/processed'
OUTPUT_DIR = 'outputs/predictions'
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42


def load_data(dataset: str):
    base = os.path.join(DATA_DIR, dataset)
    return (
        np.load(f'{base}/X_ml_train.npy'), np.load(f'{base}/y_ml_train.npy'),
        np.load(f'{base}/X_ml_val.npy'),   np.load(f'{base}/y_ml_val.npy'),
        np.load(f'{base}/X_ml_test.npy'),  np.load(f'{base}/y_test.npy'),
    )


def count_params(model) -> int:
    """Ridge 파라미터 수: 계수 + 절편"""
    best = model.best_estimator_
    return int(best.coef_.shape[0] + 1)


def measure_inference_time(model, X_sample: np.ndarray,
                            n_repeat: int = 1000) -> float:
    """단건 추론 n_repeat회 평균 (ms)"""
    sample = X_sample[:1]
    for _ in range(10):          # 워밍업
        model.predict(sample)
    start = time.time()
    for _ in range(n_repeat):
        model.predict(sample)
    return (time.time() - start) / n_repeat * 1000


def train_ridge(dataset: str) -> tuple:
    print(f"\n{'='*50}\n  Ridge — {dataset}\n{'='*50}")
    X_tr, y_tr, X_vl, y_vl, X_te, y_te = load_data(dataset)

    # ── 하이퍼파라미터 탐색 ──────────────────────────────
    param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
    model = GridSearchCV(
        Ridge(random_state=SEED if hasattr(Ridge, 'random_state') else None),
        param_grid, cv=3,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr)

    best_alpha = model.best_params_['alpha']
    print(f"  최적 alpha: {best_alpha}")

    # ── val 성능 확인 ────────────────────────────────────
    val_pred = model.predict(X_vl)
    evaluate_all(y_vl, val_pred,
                 model_name='Ridge_val', dataset=dataset, save=False)

    # ── 추론 시간 측정 ───────────────────────────────────
    inference_ms = measure_inference_time(model, X_te)
    print(f"  추론 속도: {inference_ms:.4f} ms/sample (1000회 평균)")

    # ── 파라미터 수 ──────────────────────────────────────
    n_params = count_params(model)
    print(f"  파라미터 수: {n_params:,}")

    # ── 최종 예측 ────────────────────────────────────────
    pred = model.predict(X_te)
    pred = np.clip(pred, 0, 125)

    # ── 저장 ─────────────────────────────────────────────
    out_path = os.path.join(OUTPUT_DIR, f'ridge_{dataset}.npy')
    np.save(out_path, pred)
    print(f"  예측 저장: {out_path}")

    # ── 평가 (Ridge는 rmse_ridge=자기 자신 → 성능밀도 0) ─
    evaluate_all(y_te, pred,
                 model_name='Ridge', dataset=dataset,
                 inference_ms=inference_ms,
                 n_params=n_params,
                 rmse_ridge=None)   # Ridge 자신은 성능밀도 산출 불필요

    return model, pred


if __name__ == '__main__':
    for ds in DATASETS:
        train_ridge(ds)
    print("\n모든 데이터셋 Ridge 완료")