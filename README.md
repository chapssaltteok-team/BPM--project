# NASA CMAPSS RUL Prediction

> CMAPSS 데이터셋 기반 항공 엔진 **잔존 수명(RUL)** 예측 모델 비교  
> Ridge / Random Forest / LSTM / 1D-CNN

---

## 📁 프로젝트 구조

```
BPM-project/
├── CMAPSSData/              # 원본 데이터 (Git 제외 - 별도 공유)
├── data/processed/          # 전처리 결과 .npy (Git 제외 - 별도 공유)
│   ├── FD001/
│   ├── FD002/
│   ├── FD003/
│   └── FD004/
├── src/
│   ├── eda.py               # 탐색적 데이터 분석 ✅
│   ├── preprocessing.py     # 전처리 파이프라인 ✅
│   ├── evaluate.py          # 공통 평가 함수 ✅
│   └── models/
│       ├── ridge_model.py   # Ridge Regression
│       ├── rf_model.py      # Random Forest
│       ├── lstm_model.py    # LSTM
│       └── cnn_model.py     # 1D-CNN
├── outputs/
│   └── predictions/         # 모델별 예측 결과 .npy
├── results/
│   ├── figures/             # EDA 그래프 PNG
│   ├── eda_summary.json
│   └── valid_sensors.json
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ 환경 세팅

```bash
pip install -r requirements.txt
```

---

## 📦 데이터 준비

원본 데이터(`CMAPSSData/`)와 전처리 결과(`data/processed/`)는  
용량 문제로 Git에 포함되지 않습니다. **팀장에게 별도로 받으세요.**

받은 후 아래 구조로 배치:
```
CMAPSSData/
├── train_FD001.txt
├── test_FD001.txt
├── RUL_FD001.txt
├── train_FD002.txt
... (FD002~FD004 동일)
```

전처리를 직접 실행하려면:
```bash
python src/eda.py            # 1단계: EDA (results/ 생성)
python src/preprocessing.py  # 2단계: 전처리 (data/processed/ 생성)
```

---

## 🔢 전처리 파라미터 요약

| 파라미터 | 값 | 이유 |
|---|---|---|
| 윈도우 크기 W | 30 | 엔진 최단 수명 128사이클 → 충분히 작게 |
| RUL 클리핑 | 125 | 초기 안정기 노이즈 제거 |
| 제거 센서 | sensor_1,5,10,16,18,19 | 분산=0, 학습에 무의미 |
| train:val | 80:20 (엔진 단위) | 데이터 누수 방지 |

### 데이터셋별 피처 수
| 데이터셋 | 피처 수 | 특이사항 |
|---|---|---|
| FD001 | 17 | op_3 제거 (분산=0) |
| FD002 | 18 | op_3 유효 |
| FD003 | 17 | op_3 제거 (분산=0) |
| FD004 | 18 | op_3 유효 |

---

## 📊 처리된 데이터 shape

| 파일 | Shape | 사용 모델 |
|---|---|---|
| X_train_seq.npy | (N, 30, F) | LSTM, 1D-CNN |
| X_ml_train.npy | (N, 30×F) | Ridge, RF |
| X_test_seq.npy | (엔진수, 30, F) | LSTM, 1D-CNN |
| X_ml_test.npy | (엔진수, 30×F) | Ridge, RF |

---

## 🤖 모델별 담당자 & 데이터 로드

### 공통 로드 코드
```python
import numpy as np

dataset = 'FD001'  # FD001~FD004

# DL용 (LSTM, 1D-CNN)
X_train = np.load(f'data/processed/{dataset}/X_train_seq.npy')  # (N, 30, F)
y_train = np.load(f'data/processed/{dataset}/y_train_seq.npy')
X_val   = np.load(f'data/processed/{dataset}/X_val_seq.npy')
y_val   = np.load(f'data/processed/{dataset}/y_val_seq.npy')
X_test  = np.load(f'data/processed/{dataset}/X_test_seq.npy')
y_test  = np.load(f'data/processed/{dataset}/y_test.npy')

# ML용 (Ridge, RF)
X_ml_train = np.load(f'data/processed/{dataset}/X_ml_train.npy')  # (N, 30*F)
X_ml_val   = np.load(f'data/processed/{dataset}/X_ml_val.npy')
X_ml_test  = np.load(f'data/processed/{dataset}/X_ml_test.npy')
```

### 담당자별 실행
```bash
python src/models/ridge_model.py   # Ridge 담당자
python src/models/rf_model.py      # RF 담당자
python src/models/lstm_model.py    # LSTM 담당자
python src/models/cnn_model.py     # CNN 담당자 (팀장)
```

---

## 📐 평가 지표

```python
from src.evaluate import evaluate_all
evaluate_all(y_test, pred, model_name='Ridge', dataset='FD001')
```

| 지표 | 설명 | 목표 |
|---|---|---|
| RMSE | 예측 오차 평균 | ≤ 16 |
| NASA Score | 늦은 예측에 큰 패널티 | 낮을수록 좋음 |

---

## 🔀 브랜치 전략

```
main       ← 최종본 (직접 push 금지)
dev        ← 통합 브랜치
├── feat/ridge
├── feat/rf
├── feat/lstm
└── feat/cnn
```

**작업 순서:** `feat/본인모델` 브랜치에서 작업 → PR → `dev` merge

---

## 👥 팀원 역할

| 이름 | 담당 모델 | 입력 데이터 |
|---|---|---|
| 안성민 | 1D-CNN / 레포 관리 | (N, F, 30) |
| - | Ridge | (N, 510 or 540) |
| - | Random Forest | (N, 510 or 540) |
| - | LSTM | (N, 30, F) |
