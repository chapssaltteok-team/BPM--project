"""
evaluate.py
===========
모든 모델이 공통으로 사용하는 평가 함수
사용법: from src.evaluate import evaluate_all
"""
import numpy as np
import json, os

RESULTS_DIR = 'results'


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error"""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


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


def evaluate_all(y_true: np.ndarray, y_pred: np.ndarray,
                 model_name: str, dataset: str,
                 save: bool = True) -> dict:
    """
    RMSE + NASA Score 계산 및 출력
    save=True 이면 results/scores.json 에 누적 저장

    Parameters
    ----------
    y_true     : 실제 RUL 배열 (E,)
    y_pred     : 예측 RUL 배열 (E,)
    model_name : 'Ridge' | 'RF' | 'LSTM' | 'CNN'
    dataset    : 'FD001' | 'FD002' | 'FD003' | 'FD004'
    """
    r = rmse(y_true, y_pred)
    s = nasa_score(y_true, y_pred)

    print(f"┌─ [{model_name}] {dataset} ───────────────────")
    print(f"│  RMSE       : {r:.4f}")
    print(f"│  NASA Score : {s:.2f}")
    print(f"└{'─' * 36}")

    result = {
        'model'      : model_name,
        'dataset'    : dataset,
        'RMSE'       : round(r, 4),
        'NASA_Score' : round(s, 2),
    }

    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = os.path.join(RESULTS_DIR, 'scores.json')
        data = []
        if os.path.isfile(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        # 동일 model+dataset 이면 덮어쓰기
        data = [d for d in data
                if not (d['model'] == model_name and d['dataset'] == dataset)]
        data.append(result)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  → results/scores.json 저장 완료")

    return result


def print_summary():
    """results/scores.json 전체 결과 요약 출력"""
    path = os.path.join(RESULTS_DIR, 'scores.json')
    if not os.path.isfile(path):
        print("아직 저장된 결과가 없습니다.")
        return
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"\n{'모델':<10} {'데이터셋':<10} {'RMSE':>10} {'NASA Score':>12}")
    print("─" * 45)
    for d in sorted(data, key=lambda x: (x['dataset'], x['model'])):
        print(f"{d['model']:<10} {d['dataset']:<10} "
              f"{d['RMSE']:>10.4f} {d['NASA_Score']:>12.2f}")
