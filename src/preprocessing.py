import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler#StandardScaler
from sklearn.model_selection import train_test_split

# numba 있으면 적용(속도업됨)
try:
    from numba import njit
    _NUMBA = True
except ImportError:
    _NUMBA = False
    warnings.warn(
        "numba 없음 → numpy stride_tricks 사용\n"
        "속도 향상 원하면: pip install numba",
        stacklevel=2,
    )

#경로 
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# src/preprocessing.py 기준 → 두 단계 위 = BPM--PROJECT/
data_dir = os.path.join(BASE_DIR, 'CMAPSSData')
RESULTS_DIR = 'results'

#EDA 확정 파라미터 
W        = 30
CLIP_RUL = 125

DROP_SENSORS = frozenset({
    'sensor_1', 'sensor_5', 'sensor_10',
    'sensor_16', 'sensor_18', 'sensor_19',
})

ALL_COLS    = (['engine_id', 'cycle']
               + [f'op_{i}' for i in range(1, 4)]
               + [f'sensor_{i}' for i in range(1, 22)])
SENSOR_COLS = [f'sensor_{i}' for i in range(1, 22)]
OP_COLS     = [f'op_{i}' for i in range(1, 4)]


#슬라이딩 윈도우 구현 2종류~
if _NUMBA:
    @njit(cache=True)
    def _build_windows_numba(feat, rul, W):
        """
        numba JIT: 엔진 1개 (T,F) → (n,W,F) 윈도우
        cache=True → 첫 컴파일 후 디스크 캐시, 재실행 시 즉시 로드
        n = T - W + 1  (윈도우 마지막 시점 RUL 사용)
        """
        T, F = feat.shape
        n = T - W + 1
        X = np.empty((n, W, F), dtype=np.float32)
        y = np.empty(n, dtype=np.float32)
        for i in range(n):
            X[i] = feat[i: i + W]
            y[i] = rul[i + W - 1]
        return X, y


def _build_windows_numpy(feat, rul, W):
    """
    stride_tricks 기반 슬라이딩 윈도우
    · as_strided로 (n,W,F) view 생성 → ascontiguousarray로 copy 1회
    · 중간 임시 배열 없음
    · n = T - W + 1  (윈도우 마지막 시점 RUL 사용)
    """
    T, F = feat.shape
    n    = T - W + 1
    if n <= 0:
        return (np.empty((0, W, F), dtype=np.float32),
                np.empty(0, dtype=np.float32))
    shape   = (n, W, F)
    strides = (feat.strides[0], feat.strides[0], feat.strides[1])
    X_view  = np.lib.stride_tricks.as_strided(feat, shape=shape, strides=strides)
    X       = np.ascontiguousarray(X_view, dtype=np.float32)
    y       = rul[W - 1:].astype(np.float32)
    return X, y


_window_fn = _build_windows_numba if _NUMBA else _build_windows_numpy


#전처리기 통합
class CMAPSSPreprocessor:
    VALID_DS = ('FD001', 'FD002', 'FD003', 'FD004')

    def __init__(self, dataset='FD001', w=W, clip_rul=CLIP_RUL,
                 val_ratio=0.2, seed=42, extra_drop=None):
        if dataset not in self.VALID_DS:
            raise ValueError(f"dataset은 {self.VALID_DS} 중 하나여야 합니다.")
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(
                f"'{data_dir}/' 폴더가 없습니다. "
                "CMAPSSData 폴더를 프로젝트 루트에 놓으세요."
            )

        self.dataset   = dataset
        self.W         = w
        self.clip_rul  = clip_rul
        self.val_ratio = val_ratio
        self.seed      = seed
        self.scaler    = MinMaxScaler()#StandardScaler()
        self._fitted   = False

        #피처 확정
        _raw       = self._load_raw('train')
        _candidate = SENSOR_COLS + OP_COLS
        _extra     = set(extra_drop or [])
        # 2차 자동 탐지:실제  std < 1e-6
        _auto      = {c for c in _candidate if _raw[c].std() < 1e-6}
        _new_auto  = _auto - DROP_SENSORS - _extra
        if _new_auto:
            print(f"[{dataset}] 2차 자동 제거: {sorted(_new_auto)}")

        _dropped          = DROP_SENSORS | _extra | _auto
        self.drop_sensors = sorted(_dropped)
        self.feature_cols = [c for c in _candidate if c not in _dropped]
        self.n_features   = len(self.feature_cols)

        # ── 엔진 단위 train/val 분할 인덱스 사전 계산 ────────
        # 시계열 특성상 엔진 단위로 분할 (행 단위 shuffle=False와 동일 효과)
        _engine_ids  = _raw['engine_id'].unique()
        _n_val       = max(1, int(len(_engine_ids) * val_ratio))
        # 뒤쪽 엔진을 검증용으로 (시계열 순서 유지)
        self._train_engines = set(_engine_ids[:-_n_val])
        self._val_engines   = set(_engine_ids[-_n_val:])

        os.makedirs(RESULTS_DIR, exist_ok=True)
        self._save_meta()

    # 내부 유틸
    def _load_raw(self, split: str) -> pd.DataFrame:
        path = os.path.join(data_dir, f'{split}_{self.dataset}.txt')
        return pd.read_csv(
            path, sep=r'\s+', header=None, names=ALL_COLS,
            dtype={c: 'float32' for c in ALL_COLS if c != 'engine_id'},
        )

    def _compute_rul(self, df: pd.DataFrame) -> np.ndarray:
        max_cyc = df.groupby('engine_id')['cycle'].transform('max').to_numpy(np.float32)
        return np.clip(max_cyc - df['cycle'].to_numpy(np.float32), 0, self.clip_rul)

    def _make_windows(self, df: pd.DataFrame, rul: np.ndarray):

        feat_np = df[self.feature_cols].to_numpy(np.float32)   # (총행, F)
        eng_ids = df['engine_id'].to_numpy()
        _, counts     = np.unique(eng_ids, return_counts=True)
        boundaries    = np.concatenate([[0], np.cumsum(counts)])

        X_parts, y_parts = [], []
        for k in range(len(counts)):
            s, e     = int(boundaries[k]), int(boundaries[k + 1])
            feat_eng = feat_np[s:e]    # view, 복사 없음
            rul_eng  = rul[s:e]

            T = e - s
            if T < self.W:
                # 수명 < W: 앞 0-패딩으로 샘플 1개 보장
                pad_f    = np.zeros((self.W - T, self.n_features), np.float32)
                feat_eng = np.vstack([pad_f, feat_eng])
                pad_r    = np.zeros(self.W - T, np.float32)
                rul_eng  = np.concatenate([pad_r, rul_eng])

            Xk, yk = _window_fn(feat_eng, rul_eng, self.W)
            X_parts.append(Xk)
            y_parts.append(yk)

        return (np.concatenate(X_parts, axis=0),
                np.concatenate(y_parts, axis=0))

    def _make_windows_by_engine_set(self, engine_set: set):
        df  = self._load_raw('train')
        df  = df[df['engine_id'].isin(engine_set)].reset_index(drop=True)
        rul = self._compute_rul(df)
        return self._make_windows(df, rul)

    def _scale(self, X: np.ndarray, fit: bool) -> np.ndarray:
        N, Ww, F = X.shape
        flat = X.reshape(-1, F)
        if fit:
            scaled = self.scaler.fit_transform(flat)
            self._fitted = True
        else:
            if not self._fitted:
                raise RuntimeError("get_train()을 먼저 호출하세요 (scaler 미학습).")
            scaled = self.scaler.transform(flat)
        return scaled.reshape(N, Ww, F).astype(np.float32)

    def _save_meta(self):
        path = os.path.join(RESULTS_DIR, 'valid_sensors.json')
        data = {}
        if os.path.isfile(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        data[self.dataset] = {
            'feature_cols': self.feature_cols,
            'n_features'  : self.n_features,
            'drop_sensors': self.drop_sensors,
            'W'           : self.W,
            'clip_rul'    : self.clip_rul,
            'val_ratio'   : self.val_ratio,
            'seed'        : self.seed,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # 공개 api
    def get_train(self):
        X, y = self._make_windows_by_engine_set(self._train_engines)
        return self._scale(X, fit=True), y

    def get_val(self): #검증용 윈도우 생성
        if not self._fitted:
            raise RuntimeError("get_train()을 먼저 호출하세요.")
        X, y = self._make_windows_by_engine_set(self._val_engines)
        return self._scale(X, fit=False), y

    def get_test(self):
        if not self._fitted:
            raise RuntimeError("get_train()을 먼저 호출하세요.")

        df_test  = self._load_raw('test')
        rul_path = os.path.join(data_dir, f'RUL_{self.dataset}.txt')
        true_rul = (
            pd.read_csv(rul_path, header=None, names=['RUL'])['RUL']
            .clip(upper=self.clip_rul)
            .to_numpy(np.float32)
        )

        feat_np    = df_test[self.feature_cols].to_numpy(np.float32)
        eng_ids    = df_test['engine_id'].to_numpy()
        _, counts  = np.unique(eng_ids, return_counts=True)
        boundaries = np.concatenate([[0], np.cumsum(counts)])
        n_engines  = len(counts)

        X_test = np.zeros((n_engines, self.W, self.n_features), np.float32)
        for k in range(n_engines):
            s, e     = int(boundaries[k]), int(boundaries[k + 1])
            feat_eng = feat_np[s:e]
            T        = e - s
            if T >= self.W:
                X_test[k] = feat_eng[-self.W:]
            else:
                X_test[k, self.W - T:] = feat_eng   # 앞 0-패딩

        return self._scale(X_test, fit=False), true_rul

    # .npy 일괄 저장
    def save_all(self, out_dir: str = 'data/processed'):
        """
        get_train / get_val / get_test 결과를 .npy로 저장
        기존 preprocessing.py 환경과 호환되는 파일명 사용

        출력 파일
        ─────────
        X_train_seq.npy  y_train_seq.npy
        X_val_seq.npy    y_val_seq.npy
        X_test_seq.npy   y_test.npy
        X_ml_train.npy   y_ml_train.npy
        X_ml_val.npy     y_ml_val.npy
        X_ml_test.npy    y_ml_test.npy
        """
        os.makedirs(out_dir, exist_ok=True)

        print(f"[save_all] {self.dataset} → {out_dir}")

        X_tr, y_tr = self.get_train()
        X_vl, y_vl = self.get_val()
        X_te, y_te = self.get_test()

        saves = {
            # DL용 3D
            'X_train_seq.npy' : X_tr,
            'y_train_seq.npy' : y_tr,
            'X_val_seq.npy'   : X_vl,
            'y_val_seq.npy'   : y_vl,
            'X_test_seq.npy'  : X_te,
            'y_test.npy'      : y_te,
            # ML용 2D
            'X_ml_train.npy'  : self.to_ml(X_tr),
            'y_ml_train.npy'  : y_tr,
            'X_ml_val.npy'    : self.to_ml(X_vl),
            'y_ml_val.npy'    : y_vl,
            'X_ml_test.npy'   : self.to_ml(X_te),
            'y_ml_test.npy'   : y_te,
        }

        for fname, arr in saves.items():
            np.save(os.path.join(out_dir, fname), arr)
            mb = arr.nbytes / 1024 / 1024
            print(f"  {fname:<25} {str(arr.shape):<25} {mb:.1f} MB")

        print("[save_all] 완료")

    #모델별 변환 (static, view 반환)
    @staticmethod
    def to_cnn(X: np.ndarray) -> np.ndarray:
        """(N, W, F)<-현재 ////(N, F, W)  PyTorch Conv1d(주석)"""
        return X #np.ascontiguousarray(X.transpose(0, 2, 1)) #주석은 pytorch전용

    @staticmethod
    def to_ml(X: np.ndarray) -> np.ndarray:
        """(N, W, F) → (N, W*F)  Ridge / RandomForest 형식"""
        return X.reshape(X.shape[0], -1)

    #요약 출력
    def summary(self):
        accel = 'numba JIT (cache=True)' if _NUMBA else 'numpy stride_tricks'
        n_tr  = len(self._train_engines)
        n_vl  = len(self._val_engines)
        print(f"""
┌──────────────────────────────────────────────────────┐
│  CMAPSSPreprocessor  [{self.dataset}]
├──────────────────────────────────────────────────────┤
│  슬라이딩 윈도우  W  = {self.W}
│  RUL 클리핑 상한    = {self.clip_rul}
│  슬라이딩 구현      = {accel}
├──────────────────────────────────────────────────────┤
│  제거 ({len(self.drop_sensors):2d}개): {self.drop_sensors}
│  유효 ({self.n_features:2d}개): {self.feature_cols}
├──────────────────────────────────────────────────────┤
│  엔진 분할 (val_ratio={self.val_ratio})
│    train 엔진: {n_tr}개   val 엔진: {n_vl}개
├──────────────────────────────────────────────────────┤
│  get_train() → X:(N,   {self.W}, {self.n_features})  y:(N,)      float32
│  get_val()   → X:(Nv,  {self.W}, {self.n_features})  y:(Nv,)     float32
│  get_test()  → X:(E,   {self.W}, {self.n_features})  y:(E,)      float32
│  to_cnn(X)   → (N, {self.W}, {self.n_features})                  1D-CNN(Keras)                   Conv1d
│  to_ml(X)    → (N, {self.W * self.n_features})                   Ridge/RF
│  save_all()  → .npy 일괄 저장 (기존 환경 호환)
└──────────────────────────────────────────────────────┘""")
#328번줄 pytorch 변경시 (N, {self.n_features}, {self.W})                  Conv1d

#단독 실행 시: FD001~FD004 전체 저장
if __name__ == '__main__':
    for ds in ('FD001', 'FD002', 'FD003', 'FD004'):
        print(f"\n{'='*55}")
        print(f"  {ds} 전처리 시작")
        print(f"{'='*55}")
        prep = CMAPSSPreprocessor(ds)
        prep.summary()
        prep.save_all(out_dir=f'data/processed/{ds}')
    print("\n모든 데이터셋 전처리 완료")