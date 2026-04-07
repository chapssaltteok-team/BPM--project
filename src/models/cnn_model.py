"""
cnn_model.py
============
1D-CNN 기반 RUL 예측
담당: 안성민

입력 데이터: X_train_seq (N, 30, F) → transpose → (N, F, 30)
이유: PyTorch Conv1d 입력 형식 = (N, 채널, 길이)
     F(피처)를 채널로, 30(타임스텝)을 길이로 해석
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

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Seed 고정 (팀 규칙: seed=42) ─────────────────────────────────────────────
SEED = 42

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── 모델 정의 ─────────────────────────────────────────────────────────────────
class CNNModel(nn.Module):
    """
    1D-CNN: Conv1d → BN → ReLU → Conv1d → BN → ReLU → GAP → FC
    input : (N, F, 30)  [channels_first]
    output: (N,)
    """
    def __init__(self, in_channels: int, num_filters: int = 64,
                 dropout: float = 0.2):
        super().__init__()
        self.conv_block = nn.Sequential(
            # 1차 합성곱
            nn.Conv1d(in_channels, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
            nn.Dropout(dropout),
            # 2차 합성곱
            nn.Conv1d(num_filters, num_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_filters * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # Global Average Pooling → 타임스텝 차원 제거
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc  = nn.Linear(num_filters * 2, 1)

    def forward(self, x):                    # x: (N, F, 30)
        x = self.conv_block(x)               # (N, num_filters*2, 30)
        x = self.gap(x).squeeze(-1)          # (N, num_filters*2)
        return self.fc(x).squeeze(-1)        # (N,)


# ── 데이터 로드 + CNN 형식 변환 ───────────────────────────────────────────────
def load_data(dataset: str):
    base = os.path.join(DATA_DIR, dataset)
    X_tr = np.load(f'{base}/X_train_seq.npy').transpose(0, 2, 1)  # (N,F,30)
    X_vl = np.load(f'{base}/X_val_seq.npy').transpose(0, 2, 1)
    X_te = np.load(f'{base}/X_test_seq.npy').transpose(0, 2, 1)
    y_tr = np.load(f'{base}/y_train_seq.npy')
    y_vl = np.load(f'{base}/y_val_seq.npy')
    y_te = np.load(f'{base}/y_test.npy')
    return X_tr, y_tr, X_vl, y_vl, X_te, y_te


def to_tensor(*arrays):
    return [torch.tensor(a, dtype=torch.float32) for a in arrays]


# ── 학습 ──────────────────────────────────────────────────────────────────────
def train_cnn(dataset: str,
              num_filters: int = 64,
              epochs: int      = 100,
              batch_size: int  = 256,
              lr: float        = 1e-3,
              dropout: float   = 0.2,
              patience: int    = 15):

    set_seed(SEED)  # ← seed 고정

    print(f"\n{'='*50}\n  1D-CNN — {dataset}\n{'='*50}")
    X_tr, y_tr, X_vl, y_vl, X_te, y_te = load_data(dataset)

    X_tr_t, y_tr_t, X_vl_t, y_vl_t, X_te_t = to_tensor(X_tr, y_tr, X_vl, y_vl, X_te)

    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                              batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_vl_t, y_vl_t),
                              batch_size=batch_size)

    in_channels = X_tr.shape[1]  # F
    model       = CNNModel(in_channels=in_channels,
                           num_filters=num_filters,
                           dropout=dropout).to(DEVICE)
    optimizer   = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(
                      optimizer, patience=5, factor=0.5)  # verbose=True 제거
    criterion   = nn.MSELoss()

    best_val, patience_cnt, best_state = float('inf'), 0, None

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
        scheduler.step(val_loss)

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | Val Loss: {val_loss:.4f}")

        # Early Stopping
        if val_loss < best_val:
            best_val     = val_loss
            patience_cnt = 0
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
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
        for _ in range(10):  # 워밍업
            _ = model(single_sample)

        start_time = time.time()
        for _ in range(n_repeat):
            _ = model(single_sample)
        total_time = time.time() - start_time

    inference_ms = (total_time / n_repeat) * 1000
    print(f"  추론 속도: {inference_ms:.4f} ms/sample ({n_repeat}회 평균)")

    # ── 최종 예측 ────────────────────────────────────────
    with torch.no_grad():
        pred = model(X_te_t.to(DEVICE)).cpu().numpy()
    pred = np.clip(pred, 0, 125)

    # ── 저장 ─────────────────────────────────────────────
    out_path = os.path.join(OUTPUT_DIR, f'cnn_{dataset}.npy')
    np.save(out_path, pred)
    print(f"  예측 저장: {out_path}")

    # ── 평가 ─────────────────────────────────────────────
    evaluate_all(y_te, pred, model_name='CNN', dataset=dataset)

    return model, pred


if __name__ == '__main__':
    for ds in DATASETS:
        train_cnn(ds)
    print("\n모든 데이터셋 1D-CNN 완료")