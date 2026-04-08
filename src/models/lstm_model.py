"""
lstm_model.py
=============
LSTM 기반 RUL 예측
담당: 유범준

입력 데이터: X_train_seq (N, W, F)  ← 3D 시퀀스 그대로 사용
이유: LSTM은 시간 순서를 순차적으로 처리하는 RNN 계열
     (N=샘플수, W=타임스텝, F=피처수) 형태가 batch_first=True와 일치

실험:
    Exp-1  W=30 고정, 4개 모델 비교 (H2, H3)
    Exp-2  W = 10 / 20 / 30 / 50 변경 실험 (H1 검증)
"""
import numpy as np
import os, sys, time, random
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.evaluate import evaluate_all

DATASETS   = ['FD001', 'FD002', 'FD003', 'FD004']
DATA_DIR   = 'data/processed'
OUTPUT_DIR = 'outputs/predictions'
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42

# ── Seed 고정 ─────────────────────────────────────────────────────────────────
def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ── 디바이스 설정 ─────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
print(f"Using device: {DEVICE}")


# ── Ridge RMSE 로드 ───────────────────────────────────────────────────────────
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


# ── 모델 정의 ─────────────────────────────────────────────────────────────────
class LSTMModel(nn.Module):
    """
    단층 LSTM → Dropout → FC
    input : (N, W, F)  [batch_first=True]
    output: (N,)
    """
    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out    = self.dropout(out[:, -1])
        return self.fc(out).squeeze(-1)


def count_params(model: nn.Module) -> int:
    """학습 가능 파라미터 수"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── 데이터 로드 ───────────────────────────────────────────────────────────────
def load_data(dataset: str):
    base = os.path.join(DATA_DIR, dataset)
    return (
        np.load(f'{base}/X_train_seq.npy'),
        np.load(f'{base}/y_train_seq.npy'),
        np.load(f'{base}/X_val_seq.npy'),
        np.load(f'{base}/y_val_seq.npy'),
        np.load(f'{base}/X_test_seq.npy'),
        np.load(f'{base}/y_test.npy'),
    )


def to_tensor(*arrays):
    return [torch.tensor(a, dtype=torch.float32) for a in arrays]


# ── 학습 ──────────────────────────────────────────────────────────────────────
def train_lstm(dataset: str,
               window_size: int = 30,
               tag: str        = '',
               hidden_size: int = 64,
               num_layers: int  = 1,
               epochs: int      = 50,
               batch_size: int  = 256,
               lr: float        = 1e-3,
               patience: int    = 10) -> tuple:

    set_seed(SEED)
    print(f"\n{'='*50}\n  LSTM — {dataset} (W={window_size})\n{'='*50}")

    X_tr, y_tr, X_vl, y_vl, X_te, y_te = load_data(dataset)
    X_tr_t, y_tr_t, X_vl_t, y_vl_t, X_te_t = to_tensor(
        X_tr, y_tr, X_vl, y_vl, X_te)

    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                              batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_vl_t, y_vl_t),
                              batch_size=batch_size)

    model     = LSTMModel(input_size=X_tr.shape[2],
                          hidden_size=hidden_size,
                          num_layers=num_layers).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    n_params = count_params(model)
    print(f"  파라미터 수: {n_params:,}")

    best_val, patience_cnt, best_state = float('inf'), 0, None

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_losses.append(criterion(model(xb), yb).item())
        val_loss = np.mean(val_losses)

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val:
            best_val     = val_loss
            patience_cnt = 0
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"  Early Stopping at epoch {epoch}")
                break

    # ── 추론 시간 측정 ───────────────────────────────────
    model.load_state_dict(best_state)
    model.eval()
    single = X_te_t[:1].to(DEVICE)
    n_repeat = 1000

    with torch.no_grad():
        for _ in range(10):
            _ = model(single)
        start = time.time()
        for _ in range(n_repeat):
            _ = model(single)
        inference_ms = (time.time() - start) / n_repeat * 1000

    print(f"  추론 속도: {inference_ms:.4f} ms/sample ({n_repeat}회 평균)")

    # ── 최종 예측 ────────────────────────────────────────
    with torch.no_grad():
        pred = model(X_te_t.to(DEVICE)).cpu().numpy()
    pred = np.clip(pred, 0, 125)

    # ── 저장 ─────────────────────────────────────────────
    suffix   = f'_{tag}' if tag else ''
    out_path = os.path.join(OUTPUT_DIR,
                            f'lstm_W{window_size}{suffix}_{dataset}.npy')
    np.save(out_path, pred)
    print(f"  예측 저장: {out_path}")

    # ── 평가 ─────────────────────────────────────────────
    model_name = f'LSTM_W{window_size}' if tag == '' else f'LSTM_W{window_size}_{tag}'
    rmse_ridge = load_ridge_rmse(dataset)
    if rmse_ridge is None:
        print("  ⚠ Ridge RMSE 미확인 → 성능 밀도 미산출")

    evaluate_all(y_te, pred,
                 model_name=model_name,
                 dataset=dataset,
                 inference_ms=inference_ms,
                 n_params=n_params,
                 rmse_ridge=rmse_ridge)

    return model, pred


if __name__ == '__main__':
    # ── Exp-1: 기본 실험 W=30, 4개 데이터셋 ─────────────
    print("\n" + "★"*20 + "  Exp-1: 기본 실험 (W=30)  " + "★"*20)
    for ds in DATASETS:
        train_lstm(ds, window_size=30)

    # ── Exp-2: H1 검증 — W 크기 실험 (FD001 고정) ────────
    print("\n" + "★"*20 + "  Exp-2: 슬라이딩 윈도우 실험 (FD001)  " + "★"*20)
    for w in [10, 20, 30, 50]:
        train_lstm('FD001', window_size=w, tag='Exp2')

    print("\n모든 LSTM 실험 완료")