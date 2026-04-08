"""
evaluate.py
===========
모든 모델이 공통으로 사용하는 평가 함수
사용법: from src.evaluate import evaluate_all
"""
import numpy as np
import json, os
from sklearn.metrics import mean_absolute_error, r2_score

RESULTS_DIR = 'results'


# ── 지표 함수 ─────────────────────────────────────────────────────────────────

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error"""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error"""
    return float(mean_absolute_error(y_true, y_pred))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² Score (참고 지표 — 시계열 회귀 특성상 음수 가능)"""
    return float(r2_score(y_true, y_pred))


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    NASA 공식 평가 지표
    - 늦은 예측(pred > true): exp(d/10) - 1  → 더 큰 패널티
    - 빠른 예측(pred < true): exp(-d/13) - 1 → 상대적으로 작은 패널티
    낮을수록 좋음
    """
    d = y_pred - y_true
    score = np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1))
    return float(score)


def performance_density(rmse_ridge: float, rmse_model: float,
                        n_params: int) -> float:
    """
    성능 밀도 = (RMSE_Ridge − RMSE_model) / 파라미터 수 × 10⁴
    값이 클수록 가볍고 정확한 모델
    Ridge 자신은 0, 파라미터 수 0이면 0 반환
    """
    if n_params <= 0:
        return 0.0
    return float((rmse_ridge - rmse_model) / n_params * 1e4)


# ── 통합 평가 함수 ────────────────────────────────────────────────────────────

def evaluate_all(y_true: np.ndarray, y_pred: np.ndarray,
                 model_name: str, dataset: str,
                 inference_ms: float = None,
                 n_params: int = None,
                 rmse_ridge: float = None,
                 save: bool = True) -> dict:
    """
    RMSE / MAE / R² / NASA Score / 추론속도 / 파라미터수 / 성능밀도 계산 및 출력
    save=True 이면 results/scores.json 에 누적 저장

    Parameters
    ----------
    y_true       : 실제 RUL (E,)
    y_pred       : 예측 RUL (E,)
    model_name   : 'Ridge' | 'RF' | 'LSTM' | 'CNN' 등
    dataset      : 'FD001' ~ 'FD004'
    inference_ms : 단건 추론 1000회 평균 (ms), None 이면 미측정
    n_params     : 학습 가능 파라미터 수, None 이면 미측정
    rmse_ridge   : Ridge RMSE 기준값 (성능 밀도 계산용), None 이면 미계산
    save         : scores.json 저장 여부
    """
    r   = rmse(y_true, y_pred)
    m   = mae(y_true, y_pred)
    r2s = r2(y_true, y_pred)
    ns  = nasa_score(y_true, y_pred)

    density = None
    if rmse_ridge is not None and n_params is not None:
        density = performance_density(rmse_ridge, r, n_params)

    print(f"┌─ [{model_name}] {dataset} {'─'*30}")
    print(f"│  RMSE         : {r:.4f}")
    print(f"│  MAE          : {m:.4f}")
    print(f"│  R² Score     : {r2s:.4f}")
    print(f"│  NASA Score   : {ns:.2f}")
    if inference_ms is not None:
        print(f"│  추론 속도    : {inference_ms:.4f} ms/sample")
    if n_params is not None:
        print(f"│  파라미터 수  : {n_params:,}")
    if density is not None:
        print(f"│  성능 밀도    : {density:.6f}")
    print(f"└{'─'*44}")

    result = {
        'model'      : model_name,
        'dataset'    : dataset,
        'RMSE'       : round(r, 4),
        'MAE'        : round(m, 4),
        'R2'         : round(r2s, 4),
        'NASA_Score' : round(ns, 2),
    }
    if inference_ms is not None:
        result['inference_ms'] = round(inference_ms, 4)
    if n_params is not None:
        result['n_params'] = n_params
    if density is not None:
        result['performance_density'] = round(density, 6)

    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = os.path.join(RESULTS_DIR, 'scores.json')
        data = []
        if os.path.isfile(path):
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                data = json.loads(content) if content else []
        # 동일 model+dataset 덮어쓰기
        data = [d for d in data
                if not (d['model'] == model_name and d['dataset'] == dataset)]
        data.append(result)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  → results/scores.json 저장 완료\n")

    return result


# ── 결과 요약 출력 ────────────────────────────────────────────────────────────

def print_summary():
    """results/scores.json 전체 결과 요약 출력"""
    path = os.path.join(RESULTS_DIR, 'scores.json')
    if not os.path.isfile(path):
        print("아직 저장된 결과가 없습니다.")
        return
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        data = json.loads(content) if content else []

    print(f"\n{'모델':<14} {'데이터셋':<8} {'RMSE':>8} {'MAE':>8} "
          f"{'R²':>7} {'NASA':>12} {'ms':>8} {'파라미터':>12} {'성능밀도':>12}")
    print("─" * 95)
    for d in sorted(data, key=lambda x: (x['dataset'], x['model'])):
        ms      = f"{d.get('inference_ms', '-'):>8}" if 'inference_ms' in d else f"{'N/A':>8}"
        params  = f"{d.get('n_params', 0):>12,}" if 'n_params' in d else f"{'N/A':>12}"
        density = f"{d.get('performance_density', 0):>12.6f}" if 'performance_density' in d else f"{'N/A':>12}"
        print(f"{d['model']:<14} {d['dataset']:<8} {d['RMSE']:>8.4f} "
              f"{d.get('MAE', 0):>8.4f} {d.get('R2', 0):>7.4f} "
              f"{d['NASA_Score']:>12.2f} {ms} {params} {density}")