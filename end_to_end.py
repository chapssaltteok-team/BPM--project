"""
end_to_end.py
=============
4개 모델 앙상블 End-to-End RUL 예측
raw CMAPSSData/*.txt → 전처리 → Ridge·RF·LSTM·CNN → 가중 앙상블 → RUL

앙상블 가중치: 각 모델의 val RMSE 역수 비율 (성능 좋을수록 높은 가중치)
              w_i = (1 / RMSE_val_i) / sum(1 / RMSE_val_j)

실행: python end_to_end.py
"""
import numpy as np
import os, sys, time, random, json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

from src.preprocessing import CMAPSSPreprocessor
from src.evaluate import evaluate_all

# ── 설정 ──────────────────────────────────────────────────────────────────────
DATASETS   = ['FD001', 'FD002', 'FD003', 'FD004']
OUTPUT_DIR = 'outputs/predictions'
RESULT_DIR = 'results'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

SEED = 42

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


# ── Seed 고정 ─────────────────────────────────────────────────────────────────
def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ══════════════════════════════════════════════════════════════════════════════
# 모델 정의
# ══════════════════════════════════════════════════════════════════════════════

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm    = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                               batch_first=True,
                               dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1])).squeeze(-1)


class CNNModel(nn.Module):
    def __init__(self, in_channels, num_filters=64, dropout=0.2):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(num_filters, num_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_filters * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc  = nn.Linear(num_filters * 2, 1)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(x).squeeze(-1)


def count_params(model) -> int:
    if isinstance(model, nn.Module):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    if hasattr(model, 'estimators_'):   # RF
        return int(sum(t.tree_.node_count for t in model.estimators_))
    if hasattr(model, 'best_estimator_'):  # Ridge GridSearchCV
        return int(model.best_estimator_.coef_.shape[0] + 1)
    return 0


# ══════════════════════════════════════════════════════════════════════════════
# 추론 시간 측정 공통
# ══════════════════════════════════════════════════════════════════════════════

def measure_inference_time_sklearn(model, X_sample, n=1000):
    s = X_sample[:1]
    for _ in range(10): model.predict(s)
    t0 = time.time()
    for _ in range(n): model.predict(s)
    return (time.time() - t0) / n * 1000


def measure_inference_time_torch(model, X_tensor, n=1000):
    s = X_tensor[:1].to(DEVICE)
    model.eval()
    with torch.no_grad():
        for _ in range(10): model(s)
        t0 = time.time()
        for _ in range(n): model(s)
    return (time.time() - t0) / n * 1000


# ══════════════════════════════════════════════════════════════════════════════
# 개별 모델 학습 함수
# ══════════════════════════════════════════════════════════════════════════════

def train_ridge(X_tr, y_tr, X_vl, y_vl):
    model = GridSearchCV(
        Ridge(), {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]},
        cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1)
    model.fit(X_tr, y_tr)
    val_pred = model.predict(X_vl)
    val_rmse = float(np.sqrt(np.mean((y_vl - val_pred) ** 2)))
    print(f"    [Ridge] best_alpha={model.best_params_['alpha']}  "
          f"val_RMSE={val_rmse:.4f}")
    return model, val_rmse


def train_rf(X_tr, y_tr, X_vl, y_vl):
    model = RandomForestRegressor(
        n_estimators=100, random_state=SEED, n_jobs=-1)
    model.fit(X_tr, y_tr)
    val_pred = model.predict(X_vl)
    val_rmse = float(np.sqrt(np.mean((y_vl - val_pred) ** 2)))
    print(f"    [RF]    n_estimators=100  val_RMSE={val_rmse:.4f}")
    return model, val_rmse


def train_lstm(X_tr, y_tr, X_vl, y_vl,
               hidden_size=64, epochs=50, batch_size=256,
               lr=1e-3, patience=10):
    set_seed(SEED)
    Xtr_t = torch.tensor(X_tr, dtype=torch.float32)
    ytr_t = torch.tensor(y_tr, dtype=torch.float32)
    Xvl_t = torch.tensor(X_vl, dtype=torch.float32)
    yvl_t = torch.tensor(y_vl, dtype=torch.float32)

    loader_tr = DataLoader(TensorDataset(Xtr_t, ytr_t),
                           batch_size=batch_size, shuffle=True)
    loader_vl = DataLoader(TensorDataset(Xvl_t, yvl_t), batch_size=batch_size)

    model     = LSTMModel(X_tr.shape[2], hidden_size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val, patience_cnt, best_state = float('inf'), 0, None

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader_tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

        model.eval()
        vl = []
        with torch.no_grad():
            for xb, yb in loader_vl:
                vl.append(criterion(model(xb.to(DEVICE)),
                                    yb.to(DEVICE)).item())
        val_loss = float(np.mean(vl))

        if val_loss < best_val:
            best_val     = val_loss
            patience_cnt = 0
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"    [LSTM]  Early Stopping epoch={epoch}  "
                      f"val_loss={best_val:.4f}")
                break

    model.load_state_dict(best_state)
    val_rmse = float(np.sqrt(best_val))
    print(f"    [LSTM]  hidden={hidden_size}  val_RMSE={val_rmse:.4f}")
    return model, Xvl_t, val_rmse


def train_cnn(X_tr, y_tr, X_vl, y_vl,
              num_filters=64, epochs=100, batch_size=256,
              lr=1e-3, patience=15):
    set_seed(SEED)
    # CNN 입력: (N, F, W)
    Xtr_t = torch.tensor(X_tr.transpose(0, 2, 1), dtype=torch.float32)
    ytr_t = torch.tensor(y_tr, dtype=torch.float32)
    Xvl_t = torch.tensor(X_vl.transpose(0, 2, 1), dtype=torch.float32)
    yvl_t = torch.tensor(y_vl, dtype=torch.float32)

    loader_tr = DataLoader(TensorDataset(Xtr_t, ytr_t),
                           batch_size=batch_size, shuffle=True)
    loader_vl = DataLoader(TensorDataset(Xvl_t, yvl_t), batch_size=batch_size)

    model     = CNNModel(X_tr.shape[2], num_filters).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    best_val, patience_cnt, best_state = float('inf'), 0, None

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader_tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

        model.eval()
        vl = []
        with torch.no_grad():
            for xb, yb in loader_vl:
                vl.append(criterion(model(xb.to(DEVICE)),
                                    yb.to(DEVICE)).item())
        val_loss = float(np.mean(vl))
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val     = val_loss
            patience_cnt = 0
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"    [CNN]   Early Stopping epoch={epoch}  "
                      f"val_loss={best_val:.4f}")
                break

    model.load_state_dict(best_state)
    val_rmse = float(np.sqrt(best_val))
    print(f"    [CNN]   filters={num_filters}  val_RMSE={val_rmse:.4f}")
    return model, Xvl_t, val_rmse


# ══════════════════════════════════════════════════════════════════════════════
# 앙상블 가중치 계산
# ══════════════════════════════════════════════════════════════════════════════

def calc_weights(val_rmses: dict) -> dict:
    """
    가중치 = val RMSE 역수 비율
    성능 좋은 모델(낮은 RMSE)에 더 높은 가중치 부여
    """
    inv = {k: 1.0 / v for k, v in val_rmses.items()}
    total = sum(inv.values())
    weights = {k: v / total for k, v in inv.items()}
    print("\n  앙상블 가중치 (val RMSE 역수 비율)")
    for k, w in weights.items():
        print(f"    {k:<8}: {w:.4f}  (val_RMSE={val_rmses[k]:.4f})")
    return weights


# ══════════════════════════════════════════════════════════════════════════════
# End-to-End 메인 클래스
# ══════════════════════════════════════════════════════════════════════════════

class EnsembleRULPredictor:
    """
    raw CMAPSSData → 전처리 → 4모델 학습 → 앙상블 → RUL 예측

    사용법
    ------
    predictor = EnsembleRULPredictor('FD001')
    predictor.fit()                     # 전처리 + 학습
    pred, weights = predictor.predict() # 앙상블 예측
    predictor.evaluate()                # 전체 지표 출력
    """

    def __init__(self, dataset: str):
        assert dataset in ['FD001', 'FD002', 'FD003', 'FD004']
        self.dataset = dataset
        self.prep    = CMAPSSPreprocessor(dataset)

        # 모델 컨테이너
        self.ridge = None
        self.rf    = None
        self.lstm  = None
        self.cnn   = None

        # 텐서 (predict에서 재사용)
        self._Xte_lstm_t = None
        self._Xte_cnn_t  = None
        self._y_te       = None

        # 가중치·메타
        self.weights   = {}
        self.val_rmses = {}
        self.n_params  = {}
        self.infer_ms  = {}

    # ── 학습 ──────────────────────────────────────────────────────────────────
    def fit(self):
        set_seed(SEED)
        ds = self.dataset
        print(f"\n{'='*55}")
        print(f"  EnsembleRULPredictor.fit()  [{ds}]")
        print(f"{'='*55}")

        # 전처리 (내부에서 직접 수행 — .npy 저장 없음)
        print("\n  [1/5] 전처리 중...")
        X_tr_seq, y_tr = self.prep.get_train()   # (N, W, F)
        X_vl_seq, y_vl = self.prep.get_val()
        X_te_seq, y_te = self.prep.get_test()
        self._y_te = y_te

        # ML용 2D flatten
        X_tr_ml = X_tr_seq.reshape(X_tr_seq.shape[0], -1)
        X_vl_ml = X_vl_seq.reshape(X_vl_seq.shape[0], -1)
        X_te_ml = X_te_seq.reshape(X_te_seq.shape[0], -1)
        print(f"    train: {X_tr_seq.shape}  val: {X_vl_seq.shape}  "
              f"test: {X_te_seq.shape}")

        # 텐서 저장 (predict에서 사용)
        self._Xte_lstm_t = torch.tensor(X_te_seq, dtype=torch.float32)
        self._Xte_cnn_t  = torch.tensor(
            X_te_seq.transpose(0, 2, 1), dtype=torch.float32)
        self._Xte_ml     = X_te_ml

        # ── Ridge ─────────────────────────────────────────
        print("\n  [2/5] Ridge 학습...")
        self.ridge, vr = train_ridge(X_tr_ml, y_tr, X_vl_ml, y_vl)
        self.val_rmses['Ridge'] = vr
        self.n_params['Ridge']  = count_params(self.ridge)

        # ── RF ────────────────────────────────────────────
        print("\n  [3/5] Random Forest 학습...")
        self.rf, vr = train_rf(X_tr_ml, y_tr, X_vl_ml, y_vl)
        self.val_rmses['RF']   = vr
        self.n_params['RF']    = count_params(self.rf)

        # ── LSTM ──────────────────────────────────────────
        print("\n  [4/5] LSTM 학습...")
        self.lstm, _Xvl_lstm_t, vr = train_lstm(
            X_tr_seq, y_tr, X_vl_seq, y_vl)
        self.val_rmses['LSTM'] = vr
        self.n_params['LSTM']  = count_params(self.lstm)

        # ── CNN ───────────────────────────────────────────
        print("\n  [5/5] 1D-CNN 학습...")
        self.cnn, _Xvl_cnn_t, vr = train_cnn(
            X_tr_seq, y_tr, X_vl_seq, y_vl)
        self.val_rmses['CNN']  = vr
        self.n_params['CNN']   = count_params(self.cnn)

        # ── 가중치 계산 ───────────────────────────────────
        self.weights = calc_weights(self.val_rmses)

        print(f"\n  fit() 완료 [{ds}]")
        return self

    # ── 예측 ──────────────────────────────────────────────────────────────────
    def predict(self) -> tuple:
        """
        4개 모델 예측 → 가중 앙상블
        Returns: (앙상블 예측값, 가중치 dict)
        """
        print(f"\n  앙상블 예측 [{self.dataset}]")

        # 개별 예측
        p_ridge = np.clip(self.ridge.predict(self._Xte_ml), 0, 125)
        p_rf    = np.clip(self.rf.predict(self._Xte_ml),    0, 125)

        self.lstm.eval()
        self.cnn.eval()
        with torch.no_grad():
            p_lstm = np.clip(
                self.lstm(self._Xte_lstm_t.to(DEVICE)).cpu().numpy(), 0, 125)
            p_cnn  = np.clip(
                self.cnn(self._Xte_cnn_t.to(DEVICE)).cpu().numpy(),  0, 125)

        # 가중 평균
        w = self.weights
        pred_ensemble = (
            w['Ridge'] * p_ridge +
            w['RF']    * p_rf    +
            w['LSTM']  * p_lstm  +
            w['CNN']   * p_cnn
        )
        pred_ensemble = np.clip(pred_ensemble, 0, 125)

        # 개별 예측값 저장
        self._preds = {
            'Ridge': p_ridge, 'RF': p_rf,
            'LSTM': p_lstm,   'CNN': p_cnn,
            'Ensemble': pred_ensemble,
        }

        # npy 저장
        np.save(os.path.join(OUTPUT_DIR,
                f'ensemble_{self.dataset}.npy'), pred_ensemble)
        print(f"  예측 저장: outputs/predictions/ensemble_{self.dataset}.npy")

        return pred_ensemble, self.weights

    # ── 추론 시간 측정 ────────────────────────────────────────────────────────
    def _measure_all_inference(self):
        self.infer_ms['Ridge'] = measure_inference_time_sklearn(
            self.ridge, self._Xte_ml)
        self.infer_ms['RF']    = measure_inference_time_sklearn(
            self.rf,    self._Xte_ml)
        self.infer_ms['LSTM']  = measure_inference_time_torch(
            self.lstm, self._Xte_lstm_t)
        self.infer_ms['CNN']   = measure_inference_time_torch(
            self.cnn,  self._Xte_cnn_t)

        # 앙상블 추론 시간 = 4개 합산
        self.infer_ms['Ensemble'] = sum(self.infer_ms[m]
                                        for m in ['Ridge','RF','LSTM','CNN'])

    # ── 평가 ──────────────────────────────────────────────────────────────────
    def evaluate(self):
        """개별 모델 + 앙상블 전체 지표 출력 및 scores.json 저장"""
        if not hasattr(self, '_preds'):
            raise RuntimeError("predict() 먼저 호출하세요.")

        print(f"\n{'='*55}")
        print(f"  전체 평가 [{self.dataset}]")
        print(f"{'='*55}")

        self._measure_all_inference()

        # Ridge RMSE (성능밀도 기준)
        from src.evaluate import rmse as calc_rmse
        rmse_ridge = float(calc_rmse(self._y_te, self._preds['Ridge']))

        results = {}
        for m, pred in self._preds.items():
            n_p = self.n_params.get(m, 0)
            ms  = self.infer_ms.get(m)
            results[m] = evaluate_all(
                self._y_te, pred,
                model_name=f'E2E_{m}',
                dataset=self.dataset,
                inference_ms=ms,
                n_params=n_p if n_p > 0 else None,
                rmse_ridge=rmse_ridge if m != 'Ridge' else None,
            )

        # 앙상블 vs 개별 최고 비교
        print(f"\n  {'모델':<12} {'RMSE':>8} {'MAE':>8}")
        print(f"  {'-'*30}")
        for m, r in results.items():
            best = ' ← 최고' if r['RMSE'] == min(
                v['RMSE'] for v in results.values()) else ''
            print(f"  {'E2E_'+m:<12} {r['RMSE']:>8.4f} "
                  f"{r.get('MAE', 0):>8.4f}{best}")

        return results


# ══════════════════════════════════════════════════════════════════════════════
# 실행
# ══════════════════════════════════════════════════════════════════════════════

def run_all():
    """FD001 ~ FD004 전체 앙상블 실행"""
    all_results = {}

    for ds in DATASETS:
        predictor = EnsembleRULPredictor(ds)
        predictor.fit()
        predictor.predict()
        results = predictor.evaluate()
        all_results[ds] = results

    # ── 최종 요약 ────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  최종 앙상블 결과 요약")
    print(f"{'='*70}")
    print(f"  {'모델':<14} {'FD001':>8} {'FD002':>8} {'FD003':>8} {'FD004':>8}")
    print(f"  {'-'*50}")

    model_keys = ['E2E_Ridge','E2E_RF','E2E_LSTM','E2E_CNN','E2E_Ensemble']
    for mk in model_keys:
        row = f"  {mk:<14}"
        for ds in DATASETS:
            r = all_results[ds].get(mk.replace('E2E_',''), {})
            rmse = r.get('RMSE', 0) if r else 0
            row += f" {rmse:>8.4f}"
        print(row)

    print(f"\n  → outputs/predictions/ensemble_FDxxx.npy 저장 완료")
    print(f"  → results/scores.json 업데이트 완료")


if __name__ == '__main__':
    run_all()