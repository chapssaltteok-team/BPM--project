"""
EDA.py
======
NASA CMAPSS 데이터셋 탐색적 데이터 분석
실행: python eda.py
결과: results/figures/ 폴더에 PNG 저장
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지
# ── 경로 설정 ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, '..', 'CMAPSSData')
FIG_DIR    = os.path.join(BASE_DIR, 'results', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ── 컬럼명 정의 ────────────────────────────────────────────────────────────────
COLS = (
    ['engine_id', 'cycle'] +
    [f'op_{i}'     for i in range(1, 4)] +
    [f'sensor_{i}' for i in range(1, 22)]
)

# ── 데이터 로드 함수 ───────────────────────────────────────────────────────────
def load_data(subset='FD001'):
    train = pd.read_csv(
        os.path.join(DATA_DIR, f'train_{subset}.txt'),
        sep=r'\s+', header=None, names=COLS
    ).dropna(axis=1, how='all')   # 마지막 빈 열 제거

    test = pd.read_csv(
        os.path.join(DATA_DIR, f'test_{subset}.txt'),
        sep=r'\s+', header=None, names=COLS
    ).dropna(axis=1, how='all')

    rul = pd.read_csv(
        os.path.join(DATA_DIR, f'RUL_{subset}.txt'),
        header=None, names=['RUL']
    )
    return train, test, rul

# ══════════════════════════════════════════════════════════════════════════════
# EDA 실행 (FD001 기준)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  NASA CMAPSS EDA — FD001")
print("=" * 60)

train, test, rul_test = load_data('FD001')

SENSOR_COLS = [f'sensor_{i}' for i in range(1, 22)]
# 실제 존재하는 센서 열만 추출 (빈 열 제거 후 달라질 수 있음)
SENSOR_COLS = [c for c in SENSOR_COLS if c in train.columns]

# ── 1. 기본 정보 출력 ──────────────────────────────────────────────────────────
print("\n[1] 기본 정보")
print(f"  train shape : {train.shape}")
print(f"  test  shape : {test.shape}")
print(f"  RUL   shape : {rul_test.shape}")
print(f"  엔진 수 (train): {train['engine_id'].nunique()}")
print(f"  엔진 수 (test) : {test['engine_id'].nunique()}")
print(f"  결측치 합계    : {train.isnull().sum().sum()}")
print(f"  중복 행        : {train.duplicated().sum()}")

# ── 2. 엔진별 수명 분포 ────────────────────────────────────────────────────────
print("\n[2] 엔진별 수명 분포")
life = train.groupby('engine_id')['cycle'].max()
print(f"  최단 수명 : {life.min()} 사이클")
print(f"  최장 수명 : {life.max()} 사이클")
print(f"  평균 수명 : {life.mean():.1f} 사이클")
print(f"  ※ W=30 슬라이딩 윈도우 시 최단 수명 엔진 샘플 수: "
      f"{max(0, life.min() - 30)}개")

fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(life, bins=25, color='steelblue', edgecolor='black')
ax.axvline(life.mean(), color='red', linestyle='--',
           label=f'평균: {life.mean():.1f}')
ax.axvline(30, color='orange', linestyle=':', linewidth=2,
           label='W=30 기준선')
ax.set_xlabel('수명 (사이클)')
ax.set_ylabel('엔진 수')
ax.set_title('엔진별 수명 분포 (FD001 train)')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'eda_01_lifecycle.png'), dpi=150)
plt.close()
print("  → eda_01_lifecycle.png 저장")

# ── 3. 분산 0 센서 확인 ────────────────────────────────────────────────────────
print("\n[3] 센서별 표준편차")
std_vals = train[SENSOR_COLS].std().sort_values()
DROP_SENSORS = std_vals[std_vals < 1e-6].index.tolist()
VALID_SENSORS = [c for c in SENSOR_COLS if c not in DROP_SENSORS]

print(f"  제거 대상 센서 ({len(DROP_SENSORS)}개): {DROP_SENSORS}")
print(f"  유효 센서 수   : {len(VALID_SENSORS)}개")
print(f"  유효 센서 목록 : {VALID_SENSORS}")

fig, ax = plt.subplots(figsize=(13, 4))
colors = ['red' if c in DROP_SENSORS else 'steelblue'
          for c in std_vals.index]
std_vals.plot(kind='bar', ax=ax, color=colors)
ax.axhline(1e-6, color='red', linestyle='--', label='제거 기준 (std < 1e-6)')
ax.set_title('센서별 표준편차 (빨간색 = 제거 대상)')
ax.set_ylabel('표준편차')
ax.set_xlabel('센서')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'eda_02_sensor_std.png'), dpi=150)
plt.close()
print("  → eda_02_sensor_std.png 저장")

# ── 4. 유효 센서 시계열 추이 (엔진 1번) ───────────────────────────────────────
print("\n[4] 유효 센서 시계열 추이 (엔진 1번)")
eng1 = train[train['engine_id'] == 1].reset_index(drop=True)

n = len(VALID_SENSORS)
cols_plot = 4
rows_plot = (n + cols_plot - 1) // cols_plot

fig, axes = plt.subplots(rows_plot, cols_plot,
                          figsize=(16, rows_plot * 3))
axes = axes.flatten()

for i, col in enumerate(VALID_SENSORS):
    axes[i].plot(eng1['cycle'], eng1[col],
                 color='steelblue', linewidth=0.9)
    axes[i].set_title(col, fontsize=9)
    axes[i].set_xlabel('cycle', fontsize=7)

for j in range(n, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('엔진 1번 — 유효 센서 시계열 추이', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'eda_03_sensor_trends.png'), dpi=150)
plt.close()
print("  → eda_03_sensor_trends.png 저장")

# ── 5. RUL 분포 (클리핑 전후) ─────────────────────────────────────────────────
print("\n[5] RUL 분포 확인")
max_cycle  = train.groupby('engine_id')['cycle'].transform('max')
rul_before = max_cycle - train['cycle']
rul_after  = rul_before.clip(upper=125)

print(f"  클리핑 전  max : {rul_before.max():.0f}")
print(f"  클리핑 후  max : {rul_after.max():.0f}  (기대값: 125)")
print(f"  RUL = 0  행 수 : {(rul_after == 0).sum()}  (고장 시점)")
print(f"  RUL ≤ 30 행 수 : {(rul_after <= 30).sum()}  (위험 구간)")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(rul_before, bins=40, color='orange', edgecolor='black')
axes[0].set_title('클리핑 전 RUL 분포')
axes[0].set_xlabel('RUL')
axes[0].set_ylabel('행 수')

axes[1].hist(rul_after, bins=40, color='steelblue', edgecolor='black')
axes[1].axvline(30, color='red', linestyle='--', label='위험 구간 기준 (RUL=30)')
axes[1].set_title('클리핑 후 RUL 분포 (upper=125)')
axes[1].set_xlabel('RUL')
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'eda_04_rul_distribution.png'), dpi=150)
plt.close()
print("  → eda_04_rul_distribution.png 저장")

# ── 6. 센서-RUL 상관관계 ───────────────────────────────────────────────────────
print("\n[6] 유효 센서 - RUL 상관관계")
tmp = train[VALID_SENSORS].copy()
tmp['RUL'] = rul_after

corr_rul = tmp.corr()['RUL'].drop('RUL').sort_values()
print("  상위 5 (양의 상관):")
print(corr_rul.tail(5).to_string())
print("  하위 5 (음의 상관):")
print(corr_rul.head(5).to_string())

fig, ax = plt.subplots(figsize=(12, 4))
colors = ['tomato' if v < 0 else 'steelblue' for v in corr_rul]
corr_rul.plot(kind='bar', ax=ax, color=colors)
ax.axhline(0,    color='black',  linewidth=0.8)
ax.axhline( 0.5, color='green',  linestyle='--', linewidth=0.8,
            label='+0.5 기준선')
ax.axhline(-0.5, color='green',  linestyle='--', linewidth=0.8)
ax.set_title('유효 센서 — RUL 피어슨 상관계수')
ax.set_ylabel('상관계수')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'eda_05_correlation.png'), dpi=150)
plt.close()
print("  → eda_05_correlation.png 저장")

# ── 7. 전체 서브셋 규모 요약 ───────────────────────────────────────────────────
print("\n[7] 서브셋별 데이터 규모 요약")
for subset in ['FD001', 'FD002', 'FD003', 'FD004']:
    tr, te, ru = load_data(subset)
    print(f"  {subset} | train: {tr.shape[0]:>6}행 "
          f"/ test: {te.shape[0]:>6}행 "
          f"/ 엔진(train): {tr['engine_id'].nunique():>3}개 "
          f"/ 엔진(test): {te['engine_id'].nunique():>3}개")

# ── 8. EDA 결과 요약 저장 ──────────────────────────────────────────────────────
import json
eda_summary = {
    'drop_sensors'  : DROP_SENSORS,
    'valid_sensors' : VALID_SENSORS,
    'clip_upper'    : 125,
    'n_engines_train': int(train['engine_id'].nunique()),
    'n_engines_test' : int(test['engine_id'].nunique()),
    'life_min'      : int(life.min()),
    'life_max'      : int(life.max()),
    'life_mean'     : round(float(life.mean()), 1),
    'rul_le30_count': int((rul_after <= 30).sum()),
    'train_shape'   : list(train.shape),
    'test_shape'    : list(test.shape),
}

summary_path = os.path.join(BASE_DIR, 'results', 'eda_summary.json')
with open(summary_path, 'w', encoding='utf-8') as f:
    json.dump(eda_summary, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 60)
print("  EDA 완료")
print(f"  그래프 5장 → results/figures/")
print(f"  요약 JSON  → results/eda_summary.json")
print("=" * 60)
