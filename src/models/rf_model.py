"""
rf_model.py
===========
Random Forest 기반 RUL 예측
담당: 이윤서

입력 데이터: X_ml_train (N, 510 or 540)  ← 2D Flatten
이유: sklearn 기반 앙상블, 3D 텐서 불가, W*F 차원의 flatten 사용
추가 분석: feature_importances_ → 중요 센서 분석 (보고서 활용)
"""
import numpy as np
import os, sys, time
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from sklearn.ensemble import RandomForestRegressor
from src.evaluate import evaluate_all

DATASETS   = ['FD001', 'FD002', 'FD003', 'FD004']
DATA_DIR   = 'data/processed'
OUTPUT_DIR = 'outputs/predictions'
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42

# ── Ridge RMSE 기준값 (scores.json에서 로드, 없으면 None) ────────────────────
def load_ridge_rmse(dataset: str) -> float | None:
    import json
    path = os.path.join('results', 'scores.json')
    if not os.path.isfile(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        data = json.loads(content) if content else []
    for d in data:
        if d['model'] == 'Ridge' and d['dataset'] == dataset:
            return d['RMSE']
    return None


def load_data(dataset: str):
    base = os.path.join(DATA_DIR, dataset)
    return (
        np.load(f'{base}/X_ml_train.npy'), np.load(f'{base}/y_ml_train.npy'),
        np.load(f'{base}/X_ml_val.npy'),   np.load(f'{base}/y_ml_val.npy'),
        np.load(f'{base}/X_ml_test.npy'),  np.load(f'{base}/y_test.npy'),
    )


def count_params(model) -> int:
    """RF 파라미터 수: 전체 트리 노드 수 합산"""
    return int(sum(tree.tree_.node_count for tree in model.estimators_))


def measure_inference_time(model, X_sample: np.ndarray,
                            n_repeat: int = 1000) -> float:
    """단건 추론 n_repeat회 평균 (ms)"""
    sample = X_sample[:1]
    for _ in range(10):
        model.predict(sample)
    start = time.time()
    for _ in range(n_repeat):
        model.predict(sample)
    return (time.time() - start) / n_repeat * 1000


def train_rf(dataset: str) -> tuple:
    print(f"\n{'='*50}\n  Random Forest — {dataset}\n{'='*50}")
    X_tr, y_tr, X_vl, y_vl, X_te, y_te = load_data(dataset)

    # ── 모델 학습 ────────────────────────────────────────
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr)

    # ── val 성능 확인 ────────────────────────────────────
    val_pred = model.predict(X_vl)
    evaluate_all(y_vl, val_pred,
                 model_name='RF_val', dataset=dataset, save=False)

    # ── 피처 중요도 (상위 10개) ──────────────────────────
    importances = model.feature_importances_
    top10 = np.argsort(importances)[::-1][:10]
    print(f"  피처 중요도 상위 10 인덱스: {top10.tolist()}")

    # ── 추론 시간 측정 ───────────────────────────────────
    inference_ms = measure_inference_time(model, X_te)
    print(f"  추론 속도: {inference_ms:.4f} ms/sample (1000회 평균)")

    # ── 파라미터 수 ──────────────────────────────────────
    n_params = count_params(model)
    print(f"  파라미터 수(노드 수): {n_params:,}")

    # ── 최종 예측 ────────────────────────────────────────
    pred = model.predict(X_te)
    pred = np.clip(pred, 0, 125)

    # ── 저장 ─────────────────────────────────────────────
    out_path = os.path.join(OUTPUT_DIR, f'rf_{dataset}.npy')
    np.save(out_path, pred)
    print(f"  예측 저장: {out_path}")

    # ── 평가 ─────────────────────────────────────────────
    rmse_ridge = load_ridge_rmse(dataset)
    if rmse_ridge is None:
        print("  ⚠ Ridge RMSE 미확인 → 성능 밀도 미산출 (Ridge 먼저 실행 필요)")

    evaluate_all(y_te, pred,
                 model_name='RF', dataset=dataset,
                 inference_ms=inference_ms,
                 n_params=n_params,
                 rmse_ridge=rmse_ridge)

    return model, pred


if __name__ == '__main__':
    for ds in DATASETS:
        train_rf(ds)
    print("\n모든 데이터셋 RF 완료")