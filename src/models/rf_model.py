"""
rf_model.py
===========
Random Forest 기반 RUL 예측
담당: [RF 담당자 이름]

입력 데이터: X_ml_train (N, 510 or 540)  ← 2D Flatten
이유: sklearn 기반 앙상블, 3D 텐서 불가, W*F 차원의 flatten 사용
추가 분석: feature_importances_ → 중요 센서 분석 가능 (보고서 활용)
"""
import numpy as np
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from sklearn.ensemble import RandomForestRegressor
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


def train_rf(dataset: str):
    print(f"\n{'='*50}\n  Random Forest — {dataset}\n{'='*50}")
    X_tr, y_tr, X_vl, y_vl, X_te, y_te = load_data(dataset)

    # ── 모델 학습 ────────────────────────────────────────
    # TODO: n_estimators, max_depth 튜닝 가능
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr)

    # ── val 성능 확인 ────────────────────────────────────
    val_pred = model.predict(X_vl)
    val_rmse = np.sqrt(np.mean((y_vl - val_pred) ** 2))
    print(f"  Val RMSE: {val_rmse:.4f}")

    # ── 피처 중요도 (상위 10개) ──────────────────────────
    importances = model.feature_importances_
    top10 = np.argsort(importances)[::-1][:10]
    print(f"  피처 중요도 상위 10 인덱스: {top10}")
    # TODO: 센서 이름 매핑 후 보고서 활용 가능

    # ── 최종 예측 ────────────────────────────────────────
    pred = model.predict(X_te)
    pred = np.clip(pred, 0, 125)

    # ── 저장 ─────────────────────────────────────────────
    out_path = os.path.join(OUTPUT_DIR, f'rf_{dataset}.npy')
    np.save(out_path, pred)
    print(f"  예측 저장: {out_path}")

    # ── 평가 ─────────────────────────────────────────────
    evaluate_all(y_te, pred, model_name='RF', dataset=dataset)

    return model, pred


if __name__ == '__main__':
    for ds in DATASETS:
        train_rf(ds)
    print("\n모든 데이터셋 RF 완료")
