"""
lstm_model.py
=============
LSTM 기반 RUL 예측
담당: [LSTM 담당자 이름]

입력 데이터: X_train_seq (N, W, F)  ← 3D 시퀀스 그대로 사용
이유: LSTM은 시간 순서를 순차적으로 처리하는 RNN 계열
     (N=샘플수, W=타임스텝, F=피처수) 형태가 batch_first=True와 일치
"""

import numpy as np
import os, sys, time

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.evaluate import evaluate_all

DATASETS = ["FD001", "FD002", "FD003", "FD004"]
DATA_DIR = "data/processed"
OUTPUT_DIR = "outputs/predictions"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Seed 고정 (팀 규칙: seed=42) ─────────────────────────────────────────────
SEED = 42


def set_seed(seed: int = SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# 맥 GPU 가속용
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")


# ── 모델 정의 ─────────────────────────────────────────────────────────────────
class LSTMModel(nn.Module):
    """
    단층 LSTM → Dropout → FC
    input : (N, W, F)  [batch_first=True]
    output: (N, 1)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):  # x: (N, W, F)
        out, _ = self.lstm(x)  # (N, W, hidden)
        out = self.dropout(out[:, -1])  # 마지막 타임스텝
        return self.fc(out).squeeze(-1)  # (N,)


# ── 데이터 로드 ───────────────────────────────────────────────────────────────
def load_data(dataset: str):
    base = os.path.join(DATA_DIR, dataset)
    return (
        np.load(f"{base}/X_train_seq.npy"),
        np.load(f"{base}/y_train_seq.npy"),
        np.load(f"{base}/X_val_seq.npy"),
        np.load(f"{base}/y_val_seq.npy"),
        np.load(f"{base}/X_test_seq.npy"),
        np.load(f"{base}/y_test.npy"),
    )


def to_tensor(*arrays):
    return [torch.tensor(a, dtype=torch.float32) for a in arrays]


# ── 학습 ──────────────────────────────────────────────────────────────────────
def train_lstm(
    dataset: str,
    window_size: int = 30,
    hidden_size: int = 64,
    num_layers: int = 1,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 10,
):

    set_seed(SEED)

    print(f"\n{'='*50}\n  LSTM — {dataset} (W={window_size})\n{'='*50}")
    X_tr, y_tr, X_vl, y_vl, X_te, y_te = load_data(dataset)

    X_tr_t, y_tr_t, X_vl_t, y_vl_t, X_te_t = to_tensor(X_tr, y_tr, X_vl, y_vl, X_te)

    train_loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(TensorDataset(X_vl_t, y_vl_t), batch_size=batch_size)

    model = LSTMModel(
        input_size=X_tr.shape[2], hidden_size=hidden_size, num_layers=num_layers
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val, patience_cnt, best_state = float("inf"), 0, None

    for epoch in range(1, epochs + 1):
        # train
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        # validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_losses.append(criterion(model(xb), yb).item())
        val_loss = np.mean(val_losses)

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | Val Loss: {val_loss:.4f}")

        # Early Stopping
        if val_loss < best_val:
            best_val = val_loss
            patience_cnt = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"  Early Stopping at epoch {epoch}")
                break

    # ── 추론 시간 측정 (단건 1,000회 평균, 수행계획서 기준) ───
    model.load_state_dict(best_state)
    model.eval()

    single_sample = X_te_t[:1].to(DEVICE)
    n_repeat = 1000

    with torch.no_grad():
        for _ in range(10):
            _ = model(single_sample)

        start_time = time.time()
        for _ in range(n_repeat):
            _ = model(single_sample)
        total_time = time.time() - start_time

    inference_ms = (total_time / n_repeat) * 1000
    print(f"  추론 속도: {inference_ms:.4f} ms/sample ({n_repeat}회 평균)")

    # ── 전체 예측 ────────────────────────────────────────
    with torch.no_grad():
        pred = model(X_te_t.to(DEVICE)).cpu().numpy()
    pred = np.clip(pred, 0, 125)

    # ── 저장 (W값 포함 파일명) ────────────────────────────
    out_path = os.path.join(OUTPUT_DIR, f"lstm_W{window_size}_{dataset}.npy")
    np.save(out_path, pred)
    print(f"  예측 저장: {out_path}")

    # ── 평가 ─────────────────────────────────────────────
    evaluate_all(y_te, pred, model_name="LSTM", dataset=dataset)

    return model, pred


if __name__ == "__main__":
    for ds in DATASETS:
        train_lstm(ds)
    print("\n모든 데이터셋 LSTM 완료")
