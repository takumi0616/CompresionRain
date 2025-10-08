#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ratio-model (Swin-UNet)
# 目的:
# - 既知の3時間積算降水量 S=Prec_4_6h_sum を拘束条件とし、t+4/t+5/t+6 の配分比率 w=(w4,w5,w6) を各画素ごとに予測
# - 予測1時間値は pred_1h = softmax(logits) * S で復元（非負・和=1）。水収支（Σ pred_1h = S）を厳密に満たす
# 主要点:
# - モデル出力はlogits(3ch)。チャネル方向softmaxで比率化
# - 損失は量のMSE（強度重み付き; v5の設定を踏襲）＋任意の比率KL（S>0のみ）
# - 既存の評価/可視化系は維持（equal-splitとの比較など）。動画化は削除しシンプル化
# - 数値安定化: TIME_VARS の NaN/Inf→0 補正、softmax 前の logits/tau の NaN/Inf 除去、softmax 出力の再正規化

import os
import glob
import warnings
import matplotlib.pyplot as plt
# hdf5plugin は存在しない環境があるため任意依存にする
try:
    import hdf5plugin  # noqa: F401  # register external HDF5 filters for h5netcdf
    HDF5PLUGIN_AVAILABLE = True
except Exception as e:
    HDF5PLUGIN_AVAILABLE = False
    warnings.warn(f"hdf5plugin が見つからないため、NetCDFの一部圧縮フィルタが利用できない可能性があります: {e}")
from matplotlib.colors import LogNorm
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import xarray as xr
from swin_unet import SwinTransformerSys
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import logging
import sys
import random
from bisect import bisect_right
# Cartopy は任意依存（無い環境では地図付き可視化をスキップ）
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    CARTOPY_AVAILABLE = True
except Exception as e:
    CARTOPY_AVAILABLE = False
    ccrs = None
    cfeature = None
    warnings.warn(f"Cartopy が見つからないため、地図付き可視化をスキップします: {e}")

warnings.filterwarnings("ignore", category=UserWarning)
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
os.environ['HDF5_DISABLE_VERSION_CHECK'] = '1'

# ==============================================================================
# 1. 設定 & ハイパーパラメータ
# ==============================================================================

# --- データパス設定 ---
DATA_DIR = './output_nc'
TRAIN_YEARS = [2018, 2019, 2020, 2021]
VALID_YEARS = [2022]

# --- 結果の保存先ディレクトリ（ratio） ---
RESULT_DIR = 'swin-unet_ratio_result'

# モデル、プロット、ログの保存先パス（ratio）
MODEL_SAVE_PATH = os.path.join(RESULT_DIR, 'best_swin_unet_model_ratio.pth')
PLOT_SAVE_PATH = os.path.join(RESULT_DIR, 'loss_curve_ratio.png')
RESULT_IMG_DIR = os.path.join(RESULT_DIR, 'result_images_ratio')
MAIN_LOG_PATH = os.path.join(RESULT_DIR, 'main_ratio.log')
EXEC_LOG_PATH = os.path.join(RESULT_DIR, 'execution_ratio.log')
EVALUATION_LOG_PATH = os.path.join(RESULT_DIR, 'evaluation_ratio.log')

# 追加: エポックメトリクスの図出力ディレクトリ（ratio）
EPOCH_METRIC_PLOT_DIR = os.path.join(RESULT_DIR, 'epoch_metrics_plots_ratio')

NUM_WORKERS = 4

# --- 学習パラメータ ---
BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4

# --- 再現性確保のための乱数シード設定 ---
RANDOM_SEED = 1

# --- モデルパラメータ ---
IMG_SIZE = 480
PATCH_SIZE = 4
NUM_CLASSES = 3

# 温度付きSoftmaxの設定
# 'none'（従来どおり）, 'global'（学習可能なスカラー温度）, 'dynamic'（空間的に変化する温度を推定）
TEMPERATURE_MODE = 'dynamic'
TAU_INIT = 1.0
TAU_MIN = 0.05

# --- 変数定義 ---
INPUT_VARS_COMMON = [
    'Prmsl', 'U10m', 'V10m', 'T2m', 'U975', 'V975', 'T975',
    'U950', 'V950', 'T950', 'U925', 'V925', 'T925', 'R925',
    'U850', 'V850', 'T850', 'R850', 'GH500', 'T500', 'R500',
    'GH300', 'U300', 'V300'
]
INPUT_VARS_PREC = ['Prec_ft3']
TIME_VARS = ['dayofyear_sin', 'dayofyear_cos', 'hour_sin', 'hour_cos']
TARGET_VARS_1H = ['Prec_Target_ft4', 'Prec_Target_ft5', 'Prec_Target_ft6']
TARGET_VAR_SUM = 'Prec_4_6h_sum'
IN_CHANS = len(INPUT_VARS_COMMON) * 2 + len(INPUT_VARS_PREC) + len(TIME_VARS)

# --- 評価系の追加設定 ---
# 2値判定のしきい値（例: 0.0, 0.1, 1, 5, 10, 20, 30, 50 mm）
# 1時間と3時間積算で別閾値（sumは1hの2倍スケール）
BINARY_THRESHOLDS_MM_1H = [0.0, 0.1, 1.0, 5.0, 10.0, 20.0, 30.0, 50.0]
BINARY_THRESHOLDS_MM_SUM = [0.0, 0.2, 2.0, 10.0, 20.0, 40.0, 60.0, 100.0]
# カテゴリ分類のビン（sumは1hの2倍スケール）
CATEGORY_BINS_MM_1H = [0.0, 5.0, 10.0, 20.0, 30.0, 50.0, float('inf')]
CATEGORY_BINS_MM_SUM = [0.0, 10.0, 20.0, 40.0, 60.0, 100.0, float('inf')]
# 「降水あり時間数」の判定しきい値（>= 0.1mm を降水ありと判定）
WETHOUR_THRESHOLD_MM = 0.1

# 可視化（対数表示）の最小値（これ未満は透明扱い）
LOGNORM_VMIN_MM = 0.1

# --- v5: 強度重み付けロスの設定（1hとsumで方針分離） ---
ENABLE_INTENSITY_WEIGHTED_LOSS = True  # Trueで有効化
# 1時間降水の重み（ビン境界と各ビンの重み）
# 境界は [1, 5, 10, 20, 30, 50]（右開区間）。重みは len(boundaries)+1 個。
INTENSITY_WEIGHT_BINS_1H = [1.0, 5.0, 10.0, 20.0, 30.0, 50.0]  # mm/h
INTENSITY_WEIGHT_VALUES_1H = [1.0, 1.1, 1.3, 2.2, 9.7, 25.7, 100.0]

# 3時間積算（sum）は1hの2倍スケールのビン境界＋段階的な整数重み（頻度不均衡の緩和）
# 例: 1hの50mmに対してsumは100mmまで（2倍スケール）
INTENSITY_WEIGHT_BINS_SUM = [2.0, 10.0, 20.0, 40.0, 60.0, 100.0]  # mm/3h
# 不均衡を考慮した穏やかな昇順の整数重み（過度に大きな係数は避けつつ、強雨ほど重視）
INTENSITY_WEIGHT_VALUES_SUM = [1.0, 1.1, 1.3, 2.2, 9.7, 25.7, 100.0]
# 比率モデル用の追加ハイパラ
RATIO_LOSS_WEIGHT = 0.2  # 比率KLの重み（0で無効）
EPS = 1e-6

# ==============================================================================
# 1.5. 再現性確保のための関数
# ==============================================================================
def set_seed(seed):
    """
    概要:
        乱数シードを固定して、学習・評価の再現性を確保する。
    引数:
        seed (int): 固定する乱数シード値。
    処理:
        - Python 標準 random、NumPy、PyTorch (CPU/GPU) の乱数生成器に同じ seed を設定。
        - cuDNN の挙動を deterministic/banchmark=False に設定し、計算の再現性を高める。
    戻り値:
        なし（副作用として各ライブラリの乱数状態が設定される）
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_monthly_files(data_dir, years, logger=None):
    """
    概要:
        指定した年リストに対応する月次NetCDFファイル群を data_dir から収集する。
    引数:
        data_dir (str): 入力NetCDFディレクトリ（例: './output_nc'）。
        years (List[int]): 収集対象の年のリスト。
        logger (logging.Logger | None): ロガー。None の場合は print を使用。
    処理:
        - 各年についてパターン '{year}*.nc' でファイルを glob 検索し、昇順ソートして結合。
        - 見つからない場合は警告ログ、見つかった場合は件数を情報ログに出力。
    戻り値:
        List[str]: 見つかったファイルパスのリスト（ソート済み、年を跨いで連結）。
    """
    files = []
    log_func = logger.info if logger else print
    for year in years:
        pattern = os.path.join(data_dir, f'{year}*.nc')
        found_files = sorted(glob.glob(pattern))
        files.extend(found_files)
        if not found_files:
            log_func(f"警告: {year}年のファイルが'{data_dir}'に見つかりませんでした。")
        else:
            log_func(f"情報: {year}年のファイル {len(found_files)} 個を発見")
    return files

# ==============================================================================
# 2. ロギング設定関数
# ==============================================================================
def setup_loggers():
    """
    概要:
        メインロガーを初期化し、ファイル出力と標準出力への同時ロギングを有効化する。
    引数:
        なし
    処理:
        - ログフォーマット（時刻・レベル・メッセージ）を設定。
        - main_logger を作成し、ファイル（MAIN_LOG_PATH）とコンソール双方にハンドラを登録。
        - 実行ログ/評価ログ用のファイルを初期化。
    戻り値:
        Tuple[logging.Logger, str]:
            - メインロガー
            - 実行ログパス（EXEC_LOG_PATH）
    """
    main_logger = logging.getLogger('main')
    main_logger.setLevel(logging.INFO)
    if main_logger.hasHandlers():
        main_logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    fh_main = logging.FileHandler(MAIN_LOG_PATH, mode='w', encoding='utf-8')
    fh_main.setFormatter(formatter)
    main_logger.addHandler(fh_main)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    main_logger.addHandler(ch)

    with open(EXEC_LOG_PATH, 'w', encoding='utf-8') as f:
        f.write("Execution log started.\n")

    with open(EVALUATION_LOG_PATH, 'w', encoding='utf-8') as f:
        f.write("Evaluation log started.\n\n")
    return main_logger, EXEC_LOG_PATH

# ==============================================================================
# 3. DDPセットアップ / クリーンアップ関数
# ==============================================================================
def setup_ddp(rank, world_size):
    """
    概要:
        PyTorch DistributedDataParallel(DDP) を初期化する。
    引数:
        rank (int): このプロセスのGPUランク（0..world_size-1）。
        world_size (int): 使用するGPU（プロセス）数。
    処理:
        - 環境変数 MASTER_ADDR/MASTER_PORT を設定（単一ノード想定）。
        - NCCL バックエンドでプロセスグループを初期化。
    戻り値:
        なし（副作用としてDDPが有効化される）
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_ddp():
    """
    概要:
        DDP のプロセスグループを安全に破棄して終了処理を行う。
    引数:
        なし
    処理:
        - torch.distributed.destroy_process_group を呼び出す。
    戻り値:
        なし
    """
    dist.destroy_process_group()

# ==============================================================================
# 4. データセットクラスの定義
# ==============================================================================
def open_dataset_robust(path, logger):
    """
    概要:
        NetCDF を堅牢に開くユーティリティ。h5netcdf(+hdf5plugin) を優先し、失敗時は netcdf4 にフォールバックする。
    引数:
        path (str): NetCDF ファイルパス。
        logger (logging.Logger | None): ロガー。警告/エラー出力に使用。None なら無言で例外を投げる。
    処理:
        - engine='h5netcdf' で open_dataset を試行。
        - 失敗した場合、engine='netcdf4' + lock=False に切替えて再試行。
        - 双方失敗時はエラーログを出して例外を再送出。
    戻り値:
        xarray.Dataset: 開かれたデータセット。呼び出し側で close すること。
    例外:
        いずれのエンジンでも開けない場合は例外を送出。
    """
    try:
        ds = xr.open_dataset(path, engine='h5netcdf')
        return ds
    except Exception as e1:
        if logger:
            logger.warning(f"h5netcdfで開けませんでした。netcdf4にフォールバックします: {path} ({e1})")
        try:
            ds = xr.open_dataset(path, engine='netcdf4', lock=False)
            return ds
        except Exception as e2:
            if logger:
                logger.error(f"netCDFを開けませんでした: {path} ({e2})")
            raise

class NetCDFDataset(Dataset):
    """
    概要:
        大規模な月次NetCDF群を対象に、RAM 常駐を避けつつ SSD からオンデマンドで1ステップずつ読み込むデータセット。
    入力:
        - file_paths (List[str | Path]): 月次NetCDFのパス群。全ファイルで (lat, lon) 形状が同一である前提。
    主要処理:
        - 先頭ファイルから空間次元（lat, lon）と座標値を取得。
        - 各ファイルの time 長のみメタデータとして取得し、累積カウントを持つことで
          グローバルidx -> (ファイル, ローカルidx) を二分探索で即時変換。
        - 直近1ファイルの Dataset ハンドルをキャッシュし、ファイル跨ぎで安全にクローズ・切替。
        - __getitem__ では必要スライスのみを読み出し、入力/出力テンソルを構築（チャネル結合）。
    出力:
        - __len__(): 全期間の総 time ステップ数。
        - __getitem__(idx): dict {'input': Tensor[C,H,W], 'target_1h': Tensor[3,H,W], 'target_sum': Tensor[1,H,W]}。
    注意:
        - .load() は呼ばず遅延ロードを維持（必要スライスのみI/O）。
        - TIME_VARS はスカラーを HxW に複製してチャネル化（メモリ影響は小）。
    """
    def __init__(self, file_paths, logger=None):
        """
        概要:
            データセットを初期化し、空間次元・座標・各ファイルの time 長を読み取ってインデックスを構築する。
        引数:
            file_paths (List[str | Path]): 入力NetCDFファイル群（存在確認あり）。
            logger (logging.Logger | None): ロガー（None の場合は main ロガーを利用）。
        処理:
            - ファイル存在チェック。
            - 先頭ファイルから (lat, lon) サイズと座標値を取得。
            - 各ファイルの time サイズのみを取得して累積長配列（cum_counts）を作成。
            - キャッシュ用のパス/ハンドルを初期化。
        戻り値:
            なし
        例外:
            - file_paths が空のとき ValueError。
            - ファイルが見つからない場合 FileNotFoundError。
        """
        super().__init__()
        self.logger = logger if logger else logging.getLogger('main')
        if not file_paths:
            raise ValueError("ファイルパスのリストが空です")
        for fp in file_paths:
            if not os.path.exists(fp):
                raise FileNotFoundError(f"ファイルが見つかりません: {fp}")

        self.files = list(file_paths)
        self._cache_path = None
        self._cache_ds = None

        # 画像サイズは先頭ファイルから取得（lat/lon次元は全ファイルで同一前提）
        with open_dataset_robust(self.files[0], self.logger) as ds0:
            self.img_dims = (ds0.sizes['lat'], ds0.sizes['lon'])
            self.lon = ds0['lon'].values
            self.lat = ds0['lat'].values

        # 各ファイルの time 長を取得（メタデータのみ）
        self.file_time_lens = []
        for fp in self.files:
            with open_dataset_robust(fp, self.logger) as ds_meta:
                self.file_time_lens.append(int(ds_meta.sizes['time']))

        # 累積長（2分探索で (idx -> file) を即時変換）
        self.cum_counts = np.cumsum(self.file_time_lens).tolist()
        self.time_len = self.cum_counts[-1]

        self.logger.info(f"NetCDFDataset: {len(self.files)} 個のファイルをインデックス化（オンデマンド読み込み）。")
        self.logger.info(f"データセット初期化完了: 総時間ステップ数={self.time_len}, 画像サイズ={self.img_dims}")

    def __del__(self):
        """
        概要:
            キャッシュ済みの Dataset ハンドルが存在する場合に安全にクローズするデストラクタ。
        引数:
            なし
        処理:
            - 例外に寛容に、close() を試みる。
        戻り値:
            なし
        """
        try:
            if self._cache_ds is not None:
                self._cache_ds.close()
        except Exception:
            pass

    def _get_ds(self, path):
        """
        概要:
            直近で開いた1ファイルのみをキャッシュし、同一パスなら再利用、異なる場合は前ハンドルをクローズして切替。
        引数:
            path (str | Path): 開きたいNetCDFファイルパス。
        処理:
            - キャッシュが同一パスならそのまま返す。
            - 異なるパスなら既存キャッシュを close し、open_dataset_robust で新規に開いてキャッシュ更新。
        戻り値:
            xarray.Dataset: 開いた Dataset ハンドル。
        """
        if self._cache_path == path and self._cache_ds is not None:
            return self._cache_ds
        # 切り替え時に前のDSを閉じる
        if self._cache_ds is not None:
            try:
                self._cache_ds.close()
            except Exception:
                pass
            self._cache_ds = None
            self._cache_path = None
        self._cache_ds = open_dataset_robust(path, self.logger)
        self._cache_path = path
        return self._cache_ds

    def __len__(self):
        """
        概要:
            全ファイルを通した総時間ステップ数を返す。
        引数:
            なし
        戻り値:
            int: 総サンプル数（= 全 time の合計）。
        """
        return self.time_len

    def global_to_file_index(self, idx: int):
        """
        概要:
            グローバルインデックスを (対象ファイルパス, ファイル内ローカルインデックス, ファイル番号) に変換する。
        引数:
            idx (int): 0..time_len-1 のグローバル時刻インデックス。負数は Python の負インデックス規則で末尾から。
        処理:
            - cum_counts に対して二分探索して所属ファイルを同定。
            - 累積長からローカル idx を算出。
        戻り値:
            Tuple[str, int, int]: (ファイルパス, ローカルidx, file_idx)
        例外:
            IndexError: 範囲外 idx のとき。
        """
        if idx < 0:
            idx += self.time_len
        if idx < 0 or idx >= self.time_len:
            raise IndexError("Index out of range")
        file_idx = bisect_right(self.cum_counts, idx)
        prev_cum = self.cum_counts[file_idx - 1] if file_idx > 0 else 0
        local_idx = idx - prev_cum
        return self.files[file_idx], local_idx, file_idx

    def get_time(self, idx: int):
        """
        概要:
            指定グローバル idx の観測時刻（numpy.datetime64）を取得する。
        引数:
            idx (int): 0..time_len-1 の時刻インデックス（負数可）。
        処理:
            - global_to_file_index で (ファイル, ローカルidx) に変換し、その時刻成分のみ取り出す。
        戻り値:
            numpy.datetime64: 対応する時刻。
        """
        fpath, local_idx, _ = self.global_to_file_index(idx)
        ds = self._get_ds(fpath)
        return ds['time'].isel(time=local_idx).values

    def __getitem__(self, idx):
        """
        概要:
            グローバル idx に対応する1サンプル（入力とターゲット）をオンデマンドで構築して返す。
        引数:
            idx (int): 取得したいサンプルのグローバルインデックス（負数可）。
        処理:
            - 対象ファイルとローカルidxを二分探索で特定し、データセットをキャッシュ経由で取得。
            - 入力チャネル: 各物理変数の ft3/ft6（2chずつ）＋ 降水（Prec_ft3）＋ 時間特徴（HxW複製）。
            - ターゲット: 1時間降水(3ch: ft+4/5/6)、3時間積算(1ch)。
            - NaN は 0 に置換。float32 テンソルで返却。
        戻り値:
            Dict[str, torch.Tensor]:
                - 'input': Tensor[IN_CHANS, H, W]
                - 'target_1h': Tensor[3, H, W]
                - 'target_sum': Tensor[1, H, W]
        例外:
            IndexError: idx が範囲外のとき。
        """
        # マイナスidx対応
        if idx < 0:
            idx += self.time_len
        if idx < 0 or idx >= self.time_len:
            raise IndexError("Index out of range")

        # どのファイルに属するか（二分探索）
        file_idx = bisect_right(self.cum_counts, idx)
        prev_cum = self.cum_counts[file_idx - 1] if file_idx > 0 else 0
        local_idx = idx - prev_cum
        fpath = self.files[file_idx]

        ds = self._get_ds(fpath)
        sample = ds.isel(time=local_idx)  # .load()は呼ばない（必要スライスだけを読む）

        # 入力チャネルをオンデマンドに読み出し
        input_channels = []
        for var in INPUT_VARS_COMMON:
            v3 = np.nan_to_num(sample[f'{var}_ft3'].values).astype(np.float32, copy=False)
            v6 = np.nan_to_num(sample[f'{var}_ft6'].values).astype(np.float32, copy=False)
            input_channels.append(v3)
            input_channels.append(v6)
        for var in INPUT_VARS_PREC:
            vv = np.nan_to_num(sample[var].values).astype(np.float32, copy=False)
            input_channels.append(vv)

        # 時間特徴はスカラーを HxW に複製（RAM使用は小さい）
        h, w = self.img_dims
        for tvar in TIME_VARS:
            raw_val = sample[tvar].values.item()
            # 時間特徴の堅牢化: NaN/Inf/欠損は 0.0 に置換してチャネル化
            try:
                val = float(raw_val)
            except Exception:
                val = 0.0
            if not np.isfinite(val):
                val = 0.0
            channel = np.full((h, w), val, dtype=np.float32)
            input_channels.append(channel)

        input_tensor = torch.from_numpy(np.stack(input_channels, axis=0))

        # ターゲット（1h×3 + 3h積算）
        target_1h_channels = []
        for var in TARGET_VARS_1H:
            tt = np.nan_to_num(sample[var].values).astype(np.float32, copy=False)
            target_1h_channels.append(tt)
        target_1h_tensor = torch.from_numpy(np.stack(target_1h_channels, axis=0))
        target_sum_tensor = torch.from_numpy(
            np.nan_to_num(sample[TARGET_VAR_SUM].values).astype(np.float32, copy=False)
        ).unsqueeze(0)

        return {"input": input_tensor, "target_1h": target_1h_tensor, "target_sum": target_sum_tensor}

# ==============================================================================
# 5. 損失関数、学習・検証関数
# ==============================================================================

def _get_weight_map_from_bins(target_tensor, bin_edges, weight_values):
    """
    概要:
        値域ビンに基づく画素ごとの重みマップを生成する。頻度不均衡な降水強度に対して重み付けを行うために使用。
    引数:
        target_tensor (torch.Tensor): mm単位のターゲット値（任意形状）。
        bin_edges (List[float]): ビン境界（右開区間、例: [1,5,10,20,30,50]）。
        weight_values (List[float]): 各ビンの重み。長さは len(bin_edges)+1。
    処理:
        - torch.bucketize で各要素のビンIDを求め、対応する重みを選択。
    戻り値:
        torch.Tensor: target_tensor と同形状の重みマップ。
    """
    device = target_tensor.device
    dtype = target_tensor.dtype
    boundaries_t = torch.tensor(bin_edges, device=device, dtype=dtype)
    idx = torch.bucketize(target_tensor, boundaries_t, right=False)  # 0..len(boundaries)
    wvals_t = torch.tensor(weight_values, device=device, dtype=dtype)  # len = len(boundaries)+1
    weights = wvals_t[idx]
    return weights

def _weighted_mse(pred, target, bin_edges, weight_values, eps=1e-8):
    """
    概要:
        重みマップに基づく加重MSEを計算する。
    引数:
        pred (torch.Tensor): 予測値テンソル。
        target (torch.Tensor): 正解値テンソル（mm）。
        bin_edges (List[float]): ビン境界。
        weight_values (List[float]): 各ビンの重み。
        eps (float): 0除算回避の微小値。
    処理:
        - _get_weight_map_from_bins で w を作成。
        - MSE を w で加重し、sum(w*(e^2)) / (sum(w)+eps) を返す。
    戻り値:
        torch.Tensor: スカラー損失（加重MSE）。
    """
    weights = _get_weight_map_from_bins(target, bin_edges, weight_values)
    se = (pred - target) ** 2
    num = (weights * se).sum()
    den = weights.sum() + eps
    return num / den

def custom_loss_function(output, targets):
    """
    概要:
        ratio-model 用の複合損失を計算する。モデル出力の logits(3ch) を softmax(+温度) で比率 w に変換し、
        既知の3h積算 S でスケーリングして 1h 予測量を復元。量の加重MSE + （任意）比率のKLを合算。
    引数:
        output (Tensor | Tuple[Tensor, Tensor]):
            - logits: 形状 (B,3,H,W)。temperature_mode が 'dynamic' の場合は (logits, tau_map)。
        targets (Dict[str, Tensor]):
            - 'target_1h': (B,3,H,W) の1時間降水実値。
            - 'target_sum': (B,1,H,W) の3時間積算（拘束条件）。
    処理:
        - 温度付き softmax で w を得て、pred_1h = w * S、pred_sum = Σ pred_1h を復元（数値安定の clamp/再正規化あり）。
        - 加重MSE（1h, sum）を計算（ENABLE_INTENSITY_WEIGHTED_LOSS に従う）。
        - RATIO_LOSS_WEIGHT>0 のとき、S>0 にマスクした領域で KL(y||w) を追加（y は target_1h を S で正規化）。
        - ログ用に未加重RMSE（1h, sum）も計算して返す。
    戻り値:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - total_loss: 合計損失（スカラー）
            - rmse_1h: 1h RMSE（未加重）
            - rmse_sum: 3h積算 RMSE（未加重）
    """
    eps = EPS

    # 重要: ここからfloat32で安定計算
    if isinstance(output, (tuple, list)) and len(output) == 2:
        logits, tau_map = output
        logits = logits.float()
        tau_map = tau_map.float().clamp_min(eps)
    else:
        logits = output.float()
        tau_map = None
    S = targets['target_sum'].float()      # (B,1,H,W)
    target_1h = targets['target_1h'].float()  # (B,3,H,W)

    # softmax -> 比率（微小値を下限clampしてから再正規化）
    # softmax 前の数値安定化: logits/tau の NaN/Inf を除去
    logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
    if tau_map is not None:
        tau_map = torch.nan_to_num(tau_map, nan=1.0, posinf=1.0, neginf=1.0)
    logits_scaled = logits / tau_map if tau_map is not None else logits
    ratios = torch.softmax(logits_scaled, dim=1)                   # (B,3,H,W)
    # softmax 後の数値安定化: NaN/Inf は一様比率で置換し、再正規化
    ratios = torch.nan_to_num(ratios, nan=1.0/NUM_CLASSES, posinf=1.0/NUM_CLASSES, neginf=1.0/NUM_CLASSES)
    ratios = torch.clamp(ratios, min=eps)                          # avoid log(0)
    ratios = ratios / ratios.sum(dim=1, keepdim=True).clamp_min(eps)

    # 比率×既知Sで1時間値を復元
    pred_1h = ratios * S                                            # (B,3,H,W)
    pred_sum = pred_1h.sum(dim=1, keepdim=True)                     # (B,1,H,W)

    # 非加重MSE（RMSE計算用）
    unweighted_mse_1h = nn.functional.mse_loss(pred_1h, target_1h)
    unweighted_mse_sum = nn.functional.mse_loss(pred_sum, S)

    # 強度重み付きMSE（v5設定を流用）
    if ENABLE_INTENSITY_WEIGHTED_LOSS:
        loss_1h_mse = _weighted_mse(pred_1h, target_1h, INTENSITY_WEIGHT_BINS_1H, INTENSITY_WEIGHT_VALUES_1H)
        loss_sum_mse = _weighted_mse(pred_sum, S, INTENSITY_WEIGHT_BINS_SUM, INTENSITY_WEIGHT_VALUES_SUM)
    else:
        loss_1h_mse = unweighted_mse_1h
        loss_sum_mse = unweighted_mse_sum

    # 数値安定化（NaN/Inf防御）
    loss_1h_mse = torch.nan_to_num(loss_1h_mse, nan=0.0, posinf=1e6, neginf=0.0)
    loss_sum_mse = torch.nan_to_num(loss_sum_mse, nan=0.0, posinf=1e6, neginf=0.0)

    total_loss = loss_1h_mse + loss_sum_mse

    # 比率のKL（S>0の画素のみ）。y_ratioはチャネル方向で正規化して安定化
    if RATIO_LOSS_WEIGHT > 0:
        with torch.no_grad():
            mask = (S > 0.0).float()                                # (B,1,H,W)
        # まず粗い比率を作ってからチャネル方向で正規化
        y_ratio = target_1h / (S + eps)                             # (B,3,H,W)
        y_ratio = torch.clamp(y_ratio, min=0.0)                     # 負の漏れ対策
        y_den = y_ratio.sum(dim=1, keepdim=True).clamp_min(eps)     # (B,1,H,W)
        y_ratio = y_ratio / y_den                                   # チャネル合計=1に正規化
        # KL(y||w) = Σ y * log(y/w)
        kl_map = (y_ratio * (torch.log(y_ratio + eps) - torch.log(ratios + eps))).sum(dim=1, keepdim=True)
        # マスク平均
        loss_ratio = (kl_map * mask).sum() / (mask.sum() + eps)
        # NaN/Inf防御
        loss_ratio = torch.nan_to_num(loss_ratio, nan=0.0, posinf=0.0, neginf=0.0)
        total_loss = total_loss + RATIO_LOSS_WEIGHT * loss_ratio
        # 総損失の数値安定化
        total_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=1e6, neginf=0.0)

    # ログ用RMSE
    with torch.no_grad():
        rmse_1h = torch.sqrt(unweighted_mse_1h)
        rmse_sum = torch.sqrt(unweighted_mse_sum)

    return total_loss, rmse_1h, rmse_sum

def train_one_epoch(rank, model, dataloader, optimizer, scaler, epoch, exec_log_path):
    """
    概要:
        1エポック分の学習を実行し、DDP環境での平均化済み損失メトリクスを返す。
    引数:
        rank (int): GPU ランク。
        model (DDP nn.Module): DDP でラップされたモデル。
        dataloader (DataLoader): 学習用データローダ（DistributedSampler 使用）。
        optimizer (torch.optim.Optimizer): 最適化器。
        scaler (torch.amp.GradScaler): AMP 用スケーラ。
        epoch (int): 現在のエポック番号（0始まり）。
        exec_log_path (str): 進捗ログを追記保存するファイルパス。
    処理:
        - autocast(fp16) で順伝播、custom_loss_function で損失算出。
        - AMP + 勾配クリッピング + optimizer step + 学習率スケジューラ更新。
        - rank0 で進捗表示（tqdm をファイルに出力）。
        - 全プロセスで平均化（all_reduce(Average)）。
    戻り値:
        np.ndarray: [avg_total_loss, avg_1h_rmse, avg_sum_rmse]
    """
    model.train()
    total_loss, total_1h_rmse, total_sum_rmse = 0.0, 0.0, 0.0
    dataloader.sampler.set_epoch(epoch)
    
    with open(exec_log_path, 'a', encoding='utf-8') as log_file:
        progress_bar = tqdm(
            dataloader, desc=f"Train Epoch {epoch+1}", disable=(rank != 0), 
            leave=True, file=log_file,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
        )
        for batch_idx, batch in enumerate(progress_bar):
            inputs = batch['input'].to(rank, non_blocking=True)
            targets = {k: v.to(rank, non_blocking=True) for k, v in batch.items() if k != 'input'}
            optimizer.zero_grad(set_to_none=True)

            # 初回バッチのデータ範囲をデバッグ出力（データ不具合の早期検知）
            if epoch == 0 and batch_idx == 0 and rank == 0:
                progress_bar.write(f"[DEBUG] Input range: [{inputs.min().item():.4f}, {inputs.max().item():.4f}]")
                progress_bar.write(f"[DEBUG] Target_1h range: [{targets['target_1h'].min().item():.4f}, {targets['target_1h'].max().item():.4f}]")
                progress_bar.write(f"[DEBUG] Target_sum range: [{targets['target_sum'].min().item():.4f}, {targets['target_sum'].max().item():.4f}]")

            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(inputs)
                loss, loss_1h, loss_sum = custom_loss_function(outputs, targets)

            # 非有限損失の検出とスキップ（数値暴走時の安定化）
            if not torch.isfinite(loss):
                if rank == 0:
                    progress_bar.write(f"Warning: non-finite loss detected at batch {batch_idx}. loss={loss.item()}, rmse_1h={loss_1h.item()}, rmse_sum={loss_sum.item()}")
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            # 勾配クリッピング（数値安定化）
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            total_1h_rmse += loss_1h.item()
            total_sum_rmse += loss_sum.item()
            if rank == 0:
                progress_bar.set_postfix(loss=loss.item(), rmse_1h=loss_1h.item(), rmse_sum=loss_sum.item())
            
    avg_loss = torch.tensor([total_loss, total_1h_rmse, total_sum_rmse], device=rank) / len(dataloader)
    dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
    return avg_loss.cpu().numpy()

# -------------------- 評価ユーティリティ --------------------
def _tally_binary(pred, true, thr):
    """
    概要:
        2値判定（しきい値 thr 超え）に基づく TP/TN/FP/FN を集計する。
    引数:
        pred (np.ndarray | torch.Tensor): 予測（mm）。任意形状、true と同形状。
        true (np.ndarray | torch.Tensor): 正解（mm）。
        thr (float): 判定しきい値（mm）。
    処理:
        - thr を超えるか否かで2値化し、論理演算で TP/TN/FP/FN を算出。
    戻り値:
        Tuple[int, int, int, int]: (tp, tn, fp, fn)
    """
    pred_bin = pred > thr
    true_bin = true > thr
    tp = np.logical_and(pred_bin, true_bin).sum()
    tn = np.logical_and(~pred_bin, ~true_bin).sum()
    fp = np.logical_and(pred_bin, ~true_bin).sum()
    fn = np.logical_and(~pred_bin, true_bin).sum()
    return tp, tn, fp, fn

def _digitize_bins(x, bins):
    """
    概要:
        連続値をカテゴリビン（右開区間）に割り当てる（np.digitize 準拠）。
    引数:
        x (np.ndarray): 連続値配列。
        bins (List[float]): ビン境界（右開）。
    戻り値:
        np.ndarray: 0 始まりのカテゴリID配列。
    """
    return np.digitize(x, bins, right=False) - 1  # 0始まりのカテゴリID

@torch.no_grad()
def validate_one_epoch(rank, model, dataloader, epoch, exec_log_path):
    """
    概要:
        1エポック分の検証を行い、加重損失と拡張評価指標（バイナリ、カテゴリ、RMSE、Wet-hourパターン）を集計する。
    引数:
        rank (int): GPU ランク。
        model (DDP nn.Module): 評価モードで使用するモデル。
        dataloader (DataLoader): 検証用データローダ。
        epoch (int): エポック番号（0始まり）。
        exec_log_path (str): 進捗ログ出力先。
    処理:
        - autocast(fp16) で順伝播、custom_loss_function で損失算出。
        - 追加で、温度付き softmax 比率×S から各種評価量を導出。
        - DDP all_reduce(SUM/Average) で全プロセス集計。
        - rank0 で閾値別の Binary 指標、カテゴリ精度/適合率/再現率、RMSE、Wet-hour パターン一致を計算。
    戻り値:
        Tuple[np.ndarray, Optional[Dict]]:
            - avg_losses: np.array([total_loss, rmse_1h, rmse_sum])（全プロセス平均）
            - epoch_metrics: dict（rank==0 のみ辞書、他ランクは None）
    """
    model.eval()
    total_loss, total_1h_rmse, total_sum_rmse = 0.0, 0.0, 0.0

    # 追加: メトリクス集計用（ローカルバッファ）
    T_h = len(BINARY_THRESHOLDS_MM_1H)
    T_s = len(BINARY_THRESHOLDS_MM_SUM)
    K_h = len(CATEGORY_BINS_MM_1H) - 1
    K_s = len(CATEGORY_BINS_MM_SUM) - 1

    # バイナリ指標（時刻別・積算）のカウンタ: shape (T*, 4) for [tp, tn, fp, fn]
    bin_counts_hourly = np.zeros((T_h, 4), dtype=np.int64)
    bin_counts_sum = np.zeros((T_s, 4), dtype=np.int64)

    # カテゴリ混同行列（時刻別・積算）
    confmat_hourly = np.zeros((K_h, K_h), dtype=np.int64)
    confmat_sum = np.zeros((K_s, K_s), dtype=np.int64)

    # RMSE関連（ベースライン比較・積算）
    sse_model_1h = 0.0
    sse_base_1h = 0.0
    count_1h = 0
    sse_sum = 0.0
    count_sum = 0

    # Wet-hourパターン（0..3）混同行列
    conf_wethours = np.zeros((4, 4), dtype=np.int64)

    with open(exec_log_path, 'a', encoding='utf-8') as log_file:
        progress_bar = tqdm(
            dataloader, desc=f"Valid Epoch {epoch+1}", disable=(rank != 0),
            leave=True, file=log_file,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
        )
        for batch in progress_bar:
            inputs = batch['input'].to(rank, non_blocking=True)
            targets = {k: v.to(rank, non_blocking=True) for k, v in batch.items() if k != 'input'}

            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(inputs)
                loss, loss_1h, loss_sum = custom_loss_function(outputs, targets)

            total_loss += loss.item()
            total_1h_rmse += loss_1h.item()
            total_sum_rmse += loss_sum.item()

            # 追加: 詳細メトリクスのためのデータ（温度付きsoftmax比率×既知S）
            if isinstance(outputs, (tuple, list)) and len(outputs) == 2:
                logits, tau_map = outputs
                logits_scaled = logits / tau_map.clamp_min(EPS)
            else:
                logits = outputs
                logits_scaled = logits
            ratios = torch.softmax(logits_scaled, dim=1)
            ratios = torch.clamp(ratios, min=EPS)
            ratios = ratios / ratios.sum(dim=1, keepdim=True).clamp_min(EPS)
            S = targets['target_sum']
            pred_1h_t = ratios * S
            pred_sum_t = pred_1h_t.sum(dim=1, keepdim=True)
            true_1h_t = targets['target_1h']
            # detach to numpy
            pred_1h = pred_1h_t.detach().cpu().numpy()       # (B,3,H,W)
            pred_sum = pred_sum_t.detach().cpu().numpy()      # (B,1,H,W)
            true_1h = true_1h_t.detach().cpu().numpy()
            true_sum = S.detach().cpu().numpy()
            B, _, H, W = logits.shape

            # バイナリ（時刻別・積算）
            for i, thr in enumerate(BINARY_THRESHOLDS_MM_1H):
                tp, tn, fp, fn = _tally_binary(pred_1h, true_1h, thr)
                bin_counts_hourly[i, 0] += int(tp)
                bin_counts_hourly[i, 1] += int(tn)
                bin_counts_hourly[i, 2] += int(fp)
                bin_counts_hourly[i, 3] += int(fn)
            for i, thr in enumerate(BINARY_THRESHOLDS_MM_SUM):
                tp, tn, fp, fn = _tally_binary(pred_sum, true_sum, thr)
                bin_counts_sum[i, 0] += int(tp)
                bin_counts_sum[i, 1] += int(tn)
                bin_counts_sum[i, 2] += int(fp)
                bin_counts_sum[i, 3] += int(fn)

            # カテゴリ（時刻別）
            for t in range(3):
                true_bins = _digitize_bins(true_1h[:, t, :, :].ravel(), CATEGORY_BINS_MM_1H)
                pred_bins = _digitize_bins(pred_1h[:, t, :, :].ravel(), CATEGORY_BINS_MM_1H)
                idx = (true_bins >= 0) & (true_bins < K_h) & (pred_bins >= 0) & (pred_bins < K_h)
                tb = true_bins[idx]; pb = pred_bins[idx]
                np.add.at(confmat_hourly, (tb, pb), 1)

            # カテゴリ（積算）
            true_bins_sum = _digitize_bins(true_sum.ravel(), CATEGORY_BINS_MM_SUM)
            pred_bins_sum = _digitize_bins(pred_sum.ravel(), CATEGORY_BINS_MM_SUM)
            idx2 = (true_bins_sum >= 0) & (true_bins_sum < K_s) & (pred_bins_sum >= 0) & (pred_bins_sum < K_s)
            tb2 = true_bins_sum[idx2]; pb2 = pred_bins_sum[idx2]
            np.add.at(confmat_sum, (tb2, pb2), 1)

            # equal-splitベースライン vs モデルの1hRMSE
            base_1h = np.repeat(pred_sum / 3.0, repeats=3, axis=1)
            diff_model = (pred_1h - true_1h)
            diff_base = (base_1h - true_1h)
            sse_model_1h += float(np.sum(diff_model**2))
            sse_base_1h += float(np.sum(diff_base**2))
            count_1h += int(B * 3 * H * W)

            # 積算RMSE
            diff_sum = (pred_sum - true_sum)
            sse_sum += float(np.sum(diff_sum**2))
            count_sum += int(B * H * W)

            # Wet-hourパターン（0..3）
            true_wet = (true_1h >= WETHOUR_THRESHOLD_MM)
            pred_wet = (pred_1h >= WETHOUR_THRESHOLD_MM)
            true_count = np.sum(true_wet, axis=1)
            pred_count = np.sum(pred_wet, axis=1)
            true_count = np.clip(true_count, 0, 3).astype(np.int64).ravel()
            pred_count = np.clip(pred_count, 0, 3).astype(np.int64).ravel()
            np.add.at(conf_wethours, (true_count, pred_count), 1)

            if rank == 0:
                progress_bar.set_postfix(loss=loss.item())

    # ここから分散集計（all_reduce）
    device = torch.device(f"cuda:{rank}")
    avg_loss = torch.tensor([total_loss, total_1h_rmse, total_sum_rmse], device=device, dtype=torch.float64) / len(dataloader)

    # CPU集計 → Torchテンソル化
    t_bin_hourly = torch.tensor(bin_counts_hourly, device=device, dtype=torch.float64)
    t_bin_sum = torch.tensor(bin_counts_sum, device=device, dtype=torch.float64)
    t_conf_hourly = torch.tensor(confmat_hourly, device=device, dtype=torch.float64)
    t_conf_sum = torch.tensor(confmat_sum, device=device, dtype=torch.float64)
    t_wh_conf = torch.tensor(conf_wethours, device=device, dtype=torch.float64)
    t_sse_model_1h = torch.tensor(sse_model_1h, device=device, dtype=torch.float64)
    t_sse_base_1h = torch.tensor(sse_base_1h, device=device, dtype=torch.float64)
    t_count_1h = torch.tensor(count_1h, device=device, dtype=torch.float64)
    t_sse_sum = torch.tensor(sse_sum, device=device, dtype=torch.float64)
    t_count_sum = torch.tensor(count_sum, device=device, dtype=torch.float64)

    # all_reduce (SUM)
    dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
    for t in [t_bin_hourly, t_bin_sum, t_conf_hourly, t_conf_sum, t_wh_conf,
              t_sse_model_1h, t_sse_base_1h, t_count_1h, t_sse_sum, t_count_sum]:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)

    # rank0でメトリクス導出
    epoch_metrics = None
    if rank == 0:
        epsilon = 1e-6
        # バイナリ指標
        def compute_binary_metrics_from_counts(counts, thresholds):
            out = {}
            counts_np = counts.cpu().numpy()
            for i, thr in enumerate(thresholds):
                tp, tn, fp, fn = counts_np[i]
                acc = (tp + tn) / (tp + tn + fp + fn + epsilon)
                prec = tp / (tp + fp + epsilon)
                rec = tp / (tp + fn + epsilon)
                f1 = 2 * (prec * rec) / (prec + rec + epsilon)
                csi = tp / (tp + fp + fn + epsilon)
                out[thr] = {'accuracy': float(acc), 'precision': float(prec), 'recall': float(rec),
                            'f1': float(f1), 'csi': float(csi)}
            return out

        metrics_hourly = compute_binary_metrics_from_counts(t_bin_hourly, BINARY_THRESHOLDS_MM_1H)
        metrics_sum = compute_binary_metrics_from_counts(t_bin_sum, BINARY_THRESHOLDS_MM_SUM)

        # カテゴリ指標
        def compute_categorical_metrics(confmat_t):
            conf = confmat_t.cpu().numpy()
            total = conf.sum()
            diag = np.trace(conf)
            acc = diag / (total + epsilon)
            K_local = conf.shape[0]
            precisions = np.zeros(K_local, dtype=np.float64)
            recalls = np.zeros(K_local, dtype=np.float64)
            for k in range(K_local):
                col_sum = np.sum(conf[:, k])
                row_sum = np.sum(conf[k, :])
                tp_k = conf[k, k]
                precisions[k] = tp_k / (col_sum + epsilon)
                recalls[k] = tp_k / (row_sum + epsilon)
            return float(acc), precisions, recalls

        cat_acc_hourly, cat_prec_hourly, cat_rec_hourly = compute_categorical_metrics(t_conf_hourly)
        cat_acc_sum, cat_prec_sum, cat_rec_sum = compute_categorical_metrics(t_conf_sum)

        # RMSE
        rmse_model_1h = float(torch.sqrt(t_sse_model_1h / (t_count_1h + epsilon)).item())
        rmse_base_1h = float(torch.sqrt(t_sse_base_1h / (t_count_1h + epsilon)).item())
        rmse_sum = float(torch.sqrt(t_sse_sum / (t_count_sum + epsilon)).item())

        # Wet-hourパターン一致
        wh_conf = t_wh_conf.cpu().numpy()
        total_wh = wh_conf.sum()
        diag_wh = np.trace(wh_conf)
        acc_wethour_pattern = float(diag_wh / (total_wh + epsilon))

        epoch_metrics = {
            'binary_hourly': metrics_hourly,
            'binary_sum': metrics_sum,
            'categorical_hourly': {
                'overall_acc': cat_acc_hourly,
                'per_class_precision': cat_prec_hourly,
                'per_class_recall': cat_rec_hourly
            },
            'categorical_sum': {
                'overall_acc': cat_acc_sum,
                'per_class_precision': cat_prec_sum,
                'per_class_recall': cat_rec_sum
            },
            'rmse': {
                'model_1h': rmse_model_1h,
                'baseline_1h': rmse_base_1h,
                'sum': rmse_sum
            },
            'wet_hour_pattern': {
                'accuracy': acc_wethour_pattern
            }
        }

    return avg_loss.cpu().numpy(), epoch_metrics

# ==============================================================================
# 6. 可視化 & プロット & 評価関数
# ==============================================================================
@torch.no_grad()
def visualize_final_results(rank, world_size, valid_dataset, best_model_path, result_img_dir, main_logger, exec_log_path):
    """
    概要:
        最良モデルを用いて検証データ全体の可視化画像を生成する。2x4 パネル（予測:積算/ft4/5/6、正解:同）で保存。
    引数:
        rank (int): GPU ランク。
        world_size (int): 使用GPU数（rank0 のみ実処理）。
        valid_dataset (NetCDFDataset): 検証データセット（get_time, lon/lat を使用）。
        best_model_path (str): 学習済み最良モデルのパス（.pth）。
        result_img_dir (str): 画像出力ディレクトリ。
        main_logger (logging.Logger | None): ログ出力用。
        exec_log_path (str): 進捗ログ出力先。
    処理:
        - rank0 のみ実行（Cartopy の同時アクセス回避）。
        - 各サンプルで logits→比率→1h/積算を復元し、LogNorm で視認性を高めてプロット。
        - Cartopy 失敗時は通常Axesにフォールバック。
    戻り値:
        なし（副作用として result_img_dir にPNG保存）
    """
    # DDP安全化: 可視化はrank 0のみで実行（Cartopyの同時アクセスを回避）
    if rank != 0:
        return
    if not CARTOPY_AVAILABLE:
        if rank == 0:
            main_logger.warning("Cartopyが見つからないため、最終的な可視化をスキップします。")
        return

    device = torch.device(f"cuda:{rank}")
    model = SwinTransformerSys(
        img_size=IMG_SIZE, patch_size=PATCH_SIZE, in_chans=IN_CHANS, num_classes=NUM_CLASSES,
        embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=15,
        mlp_ratio=4., qkv_bias=True, norm_layer=nn.LayerNorm, use_checkpoint=False,
        temperature_mode=TEMPERATURE_MODE, tau_init=TAU_INIT, tau_min=TAU_MIN
    ).to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    if rank == 0:
        os.makedirs(result_img_dir, exist_ok=True)
        main_logger.info(f"[ratio] 検証データ全体に対する可視化を開始 (モデル: {os.path.basename(best_model_path)})")
    
    # rank 0のみで全インデックスを処理
    proc_indices = list(range(len(valid_dataset)))

    lon = valid_dataset.lon
    lat = valid_dataset.lat
    map_extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    
    with open(exec_log_path, 'a', encoding='utf-8') as log_file:
        progress_bar = tqdm(
            proc_indices, desc=f"Visualizing on Rank {rank}", 
            disable=(rank!=0), file=log_file,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
        )

        for idx in progress_bar:
            sample = valid_dataset[idx]
            inputs = sample['input'].unsqueeze(0).to(device)
            target_1h = sample['target_1h']  # (3, H, W)
            target_sum = target_1h.sum(dim=0, keepdim=False)  # (H, W)

            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(inputs)
            if isinstance(outputs, (tuple, list)) and len(outputs) == 2:
                logits, tau_map = outputs
                logits_scaled = logits / tau_map.clamp_min(EPS)
            else:
                logits = outputs
                logits_scaled = logits
            ratios = torch.softmax(logits_scaled, dim=1)
            ratios = torch.clamp(ratios, min=EPS)
            ratios = ratios / ratios.sum(dim=1, keepdim=True).clamp_min(EPS)
            S = sample['target_sum'].unsqueeze(0).to(device)  # (1,1,H,W)
            pred_1h = (ratios * S).squeeze(0).detach().cpu()  # (3, H, W)
            # 安全化: チャネル数が3未満でも可視化が落ちないようにゼロ埋めで3chに揃える
            # t: (C,H,W) or (H,W) -> (3,H,W)
            def _ensure_three_channels(t: torch.Tensor) -> torch.Tensor:
                if t.dim() == 2:
                    t = t.unsqueeze(0)
                C = t.shape[0]
                if C >= 3:
                    return t[:3]
                pads = [t.new_zeros((1, *t.shape[1:])) for _ in range(3 - C)]
                return torch.cat([t, *pads], dim=0)

            pred_3 = _ensure_three_channels(pred_1h)     # (3,H,W)
            tgt_3  = _ensure_three_channels(target_1h)   # (3,H,W)
            pred_sum = pred_3.sum(dim=0)  # (H, W)

            # 対数色スケール設定 (LogNorm)
            cmap = plt.get_cmap("Blues")
            cmap.set_under(alpha=0)
            vmin_val = max(LOGNORM_VMIN_MM, 1e-6)

            # vmaxは外れ値に影響されにくいように8枚の99パーセンタイルで決定
            stack_vals = torch.stack([
                pred_sum,
                target_sum,
                pred_3[0], pred_3[1], pred_3[2],
                tgt_3[0], tgt_3[1], tgt_3[2]
            ], dim=0).numpy()
            flat = stack_vals.reshape(8, -1)
            flat_pos = flat[flat > 0.0]
            vmax_val = np.percentile(flat_pos, 99) if flat_pos.size > 0 else vmin_val * 10.0
            if vmax_val <= vmin_val:
                vmax_val = vmin_val * 10.0

            norm = LogNorm(vmin=vmin_val, vmax=vmax_val)

            # 2行4列の図: 上段(予測: 積算, ft+4, ft+5, ft+6) 下段(正解: 積算, ft+4, ft+5, ft+6)
            fig, axes = plt.subplots(2, 4, figsize=(24, 12), subplot_kw={'projection': ccrs.PlateCarree()})
            time_val = valid_dataset.get_time(idx)
            fig.suptitle(f"Validation Time: {np.datetime_as_string(time_val, unit='m')}", fontsize=16)

            # 可視化配列（pred/gt とも安全に3ch化したテンソルを使用）
            plot_data = [
                pred_sum.numpy(), pred_3[0].numpy(), pred_3[1].numpy(), pred_3[2].numpy(),
                target_sum.numpy(), tgt_3[0].numpy(), tgt_3[1].numpy(), tgt_3[2].numpy()
            ]
            titles = [
                'Prediction Accum (4-6h sum)', 'Prediction FT+4', 'Prediction FT+5', 'Prediction FT+6',
                'Ground Truth Accum (4-6h sum)', 'Ground Truth FT+4', 'Ground Truth FT+5', 'Ground Truth FT+6'
            ]

            for i, ax in enumerate(axes.flat):
                ax.set_extent(map_extent, crs=ccrs.PlateCarree())
                ax.add_feature(cfeature.COASTLINE, edgecolor='black')
                ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='black')
                gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, color='gray', alpha=0.5)
                gl.top_labels = False
                gl.right_labels = False
                
                im = ax.imshow(plot_data[i], extent=map_extent, origin='upper', cmap=cmap, norm=norm)
                ax.set_title(titles[i])
                fig.colorbar(im, ax=ax, shrink=0.7, label='Precipitation (mm) [Log scale]')

            plt.tight_layout(rect=[0, 0.03, 1, 0.96])
            time_str = np.datetime_as_string(time_val, unit='h').replace(':', '-').replace('T', '_')
            save_path = os.path.join(result_img_dir, f'validation_{time_str}.png')
            try:
                plt.savefig(save_path, dpi=150)
            except Exception as e:
                if main_logger:
                    main_logger.warning(f"Cartopy描画でエラーが発生したため、地図レイヤを外して保存を再試行します: {e}")
                plt.close(fig)
                # フォールバック: Cartopyを使わない通常のAxesで再描画
                fig2, axes2 = plt.subplots(2, 4, figsize=(24, 12))
                fig2.suptitle(f"Validation Time: {np.datetime_as_string(time_val, unit='m')}", fontsize=16)
                for i, ax2 in enumerate(axes2.flat):
                    im2 = ax2.imshow(plot_data[i], origin='upper', cmap=cmap, norm=norm)
                    ax2.set_title(titles[i])
                    fig2.colorbar(im2, ax=ax2, shrink=0.7, label='Precipitation (mm) [Log scale]')
                plt.tight_layout(rect=[0, 0.03, 1, 0.96])
                plt.savefig(save_path, dpi=150)
                plt.close(fig2)
            else:
                plt.close(fig)

def plot_loss_curve(history, save_path, logger, best_epoch):
    """
    概要:
        学習/検証の損失と RMSE 推移を図に保存する。
    引数:
        history (Dict[str, List[float]]): 'train_loss','val_loss','train_1h_rmse','val_1h_rmse','train_sum_rmse','val_sum_rmse'。
        save_path (str): 保存先パス。
        logger (logging.Logger): ロガー。
        best_epoch (int): 最良エポック（1始まり、未設定は -1）。
    戻り値:
        なし（副作用として画像ファイルを保存）
    """
    plt.figure(figsize=(12, 8))
    epochs_range = range(1, len(history['train_loss']) + 1)
    
    plt.subplot(2, 1, 1)
    plt.plot(epochs_range, history['train_loss'], 'bo-', label='Training Total Loss')
    plt.plot(epochs_range, history['val_loss'], 'ro-', label='Validation Total Loss')
    
    if best_epoch != -1:
        best_loss_val = history['val_loss'][best_epoch-1]
        plt.axvline(x=best_epoch, color='k', linestyle='--', label=f'Best Epoch: {best_epoch} (Loss: {best_loss_val:.4f})')

    plt.title('Total Loss Curve (weighted MSE)')
    plt.ylabel('Loss')
    plt.legend(); plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(epochs_range, history['train_1h_rmse'], 'b.-', label='Train 1h-RMSE')
    plt.plot(epochs_range, history['val_1h_rmse'], 'r.-', label='Val 1h-RMSE')
    plt.title('1-hour RMSE'); plt.xlabel('Epoch'); plt.ylabel('RMSE')
    plt.legend(); plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(epochs_range, history['train_sum_rmse'], 'b.--', label='Train Sum-RMSE')
    plt.plot(epochs_range, history['val_sum_rmse'], 'r.--', label='Val Sum-RMSE')
    plt.title('Sum RMSE'); plt.xlabel('Epoch'); plt.ylabel('RMSE')
    plt.legend(); plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Loss curve saved to '{save_path}'.")

def _class_labels_from_bins(bins):
    """
    概要:
        連続ビン境界から可読なクラスラベル（例: '0-5', '5-10', '10+'）を生成する。
    引数:
        bins (List[float]): ビン境界（最後は +∞ を暗黙的に想定）。
    戻り値:
        List[str]: クラスラベルのリスト。
    """
    labels = []
    for i in range(len(bins)-1):
        left = bins[i]
        right = bins[i+1]
        if np.isinf(right):
            labels.append(f"{left:.0f}+")
        else:
            labels.append(f"{left:.0f}-{right:.0f}")
    return labels

def plot_epoch_metrics(metric_history, out_dir, logger):
    """
    概要:
        各種評価指標（Binary, Categorical, RMSE, Wet-hour）のエポック推移をまとめて可視化して保存する。
    引数:
        metric_history (Dict): validate_one_epoch から収集した各指標のエポック配列。
        out_dir (str): 画像出力ディレクトリ。
        logger (logging.Logger): ロガー。
    処理:
        - 閾値別 Binary 指標（accuracy/precision/recall/F1/CSI）をエポック曲線で保存（hourly/sum）。
        - Categorical 指標（overall acc, per-class precision/recall）を保存（hourly/sum）。
        - RMSE のモデル vs equal-split 比較、sum RMSE、Wet-hour 一致率の推移を保存。
    戻り値:
        なし（副作用として out_dir に複数PNGを保存）
    """
    os.makedirs(out_dir, exist_ok=True)
    epochs = np.arange(1, metric_history['num_epochs'] + 1)

    # 1) Binary metrics (hourly/sum)
    def plot_binary(kind):
        # kind: 'binary_hourly' or 'binary_sum'
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        axes = axes.flat
        metrics_list = ['accuracy', 'precision', 'recall', 'f1', 'csi']
        titles = ['Accuracy', 'Precision', 'Recall', 'F1', 'CSI']
        for m_idx, metric in enumerate(metrics_list):
            ax = axes[m_idx]
            thr_list = BINARY_THRESHOLDS_MM_1H if kind == 'binary_hourly' else BINARY_THRESHOLDS_MM_SUM
            for thr in thr_list:
                ys = metric_history[kind][thr][metric]
                ax.plot(epochs, ys, label=f">{thr}mm")
            ax.set_title(f"{kind.replace('_', ' ').title()} - {titles[m_idx]}")
            ax.set_xlabel("Epoch"); ax.set_ylabel(metric.upper())
            ax.grid(True)
            ax.legend(ncol=2, fontsize=8)
        # 余白埋め
        if len(metrics_list) < len(axes):
            axes[-1].axis('off')
        plt.tight_layout()
        save_path = os.path.join(out_dir, f"{kind}_metrics_over_epochs.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        logger.info(f"Saved: {save_path}")

    plot_binary('binary_hourly')
    plot_binary('binary_sum')

    # 2) Categorical metrics (hourly/sum)
    class_labels_hourly = _class_labels_from_bins(CATEGORY_BINS_MM_1H)
    class_labels_sum = _class_labels_from_bins(CATEGORY_BINS_MM_SUM)

    def plot_categorical(kind, labels):
        # overall acc
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, metric_history[kind]['overall_acc'], 'b-o', label='Overall categorical accuracy')
        plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title(f"{kind.replace('_', ' ').title()} - Overall categorical accuracy")
        plt.grid(True); plt.legend()
        save_path = os.path.join(out_dir, f"{kind}_overall_accuracy_over_epochs.png")
        plt.savefig(save_path, dpi=150); plt.close()
        logger.info(f"Saved: {save_path}")

        # per-class precision
        prec = np.stack(metric_history[kind]['per_class_precision'], axis=0)  # [E, K]
        plt.figure(figsize=(12, 6))
        for k in range(prec.shape[1]):
            plt.plot(epochs, prec[:, k], label=f"class {labels[k]}")
        plt.xlabel("Epoch"); plt.ylabel("Precision"); plt.title(f"{kind.replace('_', ' ').title()} - Per-class precision")
        plt.grid(True); plt.legend(ncol=3, fontsize=8)
        save_path = os.path.join(out_dir, f"{kind}_perclass_precision_over_epochs.png")
        plt.savefig(save_path, dpi=150); plt.close()
        logger.info(f"Saved: {save_path}")

        # per-class recall
        rec = np.stack(metric_history[kind]['per_class_recall'], axis=0)  # [E, K]
        plt.figure(figsize=(12, 6))
        for k in range(rec.shape[1]):
            plt.plot(epochs, rec[:, k], label=f"class {labels[k]}")
        plt.xlabel("Epoch"); plt.ylabel("Recall"); plt.title(f"{kind.replace('_', ' ').title()} - Per-class recall")
        plt.grid(True); plt.legend(ncol=3, fontsize=8)
        save_path = os.path.join(out_dir, f"{kind}_perclass_recall_over_epochs.png")
        plt.savefig(save_path, dpi=150); plt.close()
        logger.info(f"Saved: {save_path}")

    plot_categorical('categorical_hourly', class_labels_hourly)
    plot_categorical('categorical_sum', class_labels_sum)

    # 3) RMSE and baseline comparison
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, metric_history['rmse']['model_1h'], 'b-o', label='Model 1h RMSE')
    plt.plot(epochs, metric_history['rmse']['baseline_1h'], 'r--o', label='Equal-split baseline 1h RMSE')
    plt.xlabel("Epoch"); plt.ylabel("RMSE"); plt.title("1h RMSE vs Equal-split baseline over epochs")
    plt.grid(True); plt.legend()
    save_path = os.path.join(out_dir, "rmse_1h_baseline_over_epochs.png")
    plt.savefig(save_path, dpi=150); plt.close()
    logger.info(f"Saved: {save_path}")

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, metric_history['rmse']['sum'], 'g-o', label='Sum RMSE')
    plt.xlabel("Epoch"); plt.ylabel("RMSE"); plt.title("Sum RMSE over epochs")
    plt.grid(True); plt.legend()
    save_path = os.path.join(out_dir, "rmse_sum_over_epochs.png")
    plt.savefig(save_path, dpi=150); plt.close()
    logger.info(f"Saved: {save_path}")

    # 4) Wet-hour pattern accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, metric_history['wet_hour_pattern']['accuracy'], 'm-o', label='Wet-hour pattern accuracy (0/1/2/3)')
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Wet-hour pattern accuracy over epochs")
    plt.grid(True); plt.legend()
    save_path = os.path.join(out_dir, "wet_hour_pattern_accuracy_over_epochs.png")
    plt.savefig(save_path, dpi=150); plt.close()
    logger.info(f"Saved: {save_path}")

@torch.no_grad()
def evaluate_model(model_path, valid_dataset, device, eval_log_path, main_logger):
    """
    概要:
        最終モデルで検証データ全体を評価し、閾値別 Binary 指標、カテゴリ指標、RMSE、Wet-hour パターン一致をテキストに出力。
    引数:
        model_path (str): 学習済みモデル（.pth）。
        valid_dataset (NetCDFDataset): 検証データセット。
        device (torch.device): 評価に使用するデバイス。
        eval_log_path (str): 評価結果のテキスト出力先。
        main_logger (logging.Logger): ロガー。
    処理:
        - DataLoader で一括評価し、numpy に落として全量を逐次集計。
        - equal-split ベースラインとの 1h RMSE 比較も算出。
        - 結果は eval_log_path に詳細出力。
    戻り値:
        なし（副作用としてログファイルを出力）
    """
    main_logger.info("Starting final model evaluation (ratio)...")
    
    model = SwinTransformerSys(
        img_size=IMG_SIZE, patch_size=PATCH_SIZE, in_chans=IN_CHANS, num_classes=NUM_CLASSES,
        embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=15,
        mlp_ratio=4., qkv_bias=True, norm_layer=nn.LayerNorm, use_checkpoint=False,
        temperature_mode=TEMPERATURE_MODE, tau_init=TAU_INIT, tau_min=TAU_MIN
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    eval_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    bin_stats_hourly = {thr: {'tp':0, 'tn':0, 'fp':0, 'fn':0} for thr in BINARY_THRESHOLDS_MM_1H}
    bin_stats_sum    = {thr: {'tp':0, 'tn':0, 'fp':0, 'fn':0} for thr in BINARY_THRESHOLDS_MM_SUM}

    K_h = len(CATEGORY_BINS_MM_1H) - 1
    K_s = len(CATEGORY_BINS_MM_SUM) - 1
    confmat_hourly = np.zeros((K_h, K_h), dtype=np.int64)
    confmat_sum    = np.zeros((K_s, K_s), dtype=np.int64)

    sse_model_1h = 0.0
    sse_base_1h  = 0.0
    count_1h     = 0

    sse_sum = 0.0
    count_sum = 0

    conf_wethours = np.zeros((4, 4), dtype=np.int64)

    progress_bar = tqdm(eval_loader, desc="Evaluating Model (ratio)", leave=False)
    for batch in progress_bar:
        inputs = batch['input'].to(device)
        targ_1h = batch['target_1h'].to(device)
        targ_sum = batch['target_sum'].to(device)

        with autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(inputs)
        if isinstance(outputs, (tuple, list)) and len(outputs) == 2:
            logits, tau_map = outputs
            logits_scaled = logits / tau_map.clamp_min(EPS)
        else:
            logits = outputs
            logits_scaled = logits
        ratios = torch.softmax(logits_scaled, dim=1)
        ratios = torch.clamp(ratios, min=EPS)
        ratios = ratios / ratios.sum(dim=1, keepdim=True).clamp_min(EPS)
        S = targ_sum
        pred_1h_t = ratios * S
        pred_sum_t = pred_1h_t.sum(dim=1, keepdim=True)

        pred_1h = pred_1h_t.detach().cpu().numpy()
        true_1h = targ_1h.detach().cpu().numpy()
        pred_sum = pred_sum_t.detach().cpu().numpy()
        true_sum = targ_sum.detach().cpu().numpy()
        B, _, H, W = logits.shape

        for thr in BINARY_THRESHOLDS_MM_1H:
            tp, tn, fp, fn = _tally_binary(pred_1h, true_1h, thr)
            bin_stats_hourly[thr]['tp'] += int(tp)
            bin_stats_hourly[thr]['tn'] += int(tn)
            bin_stats_hourly[thr]['fp'] += int(fp)
            bin_stats_hourly[thr]['fn'] += int(fn)

        for thr in BINARY_THRESHOLDS_MM_SUM:
            tp, tn, fp, fn = _tally_binary(pred_sum, true_sum, thr)
            bin_stats_sum[thr]['tp'] += int(tp)
            bin_stats_sum[thr]['tn'] += int(tn)
            bin_stats_sum[thr]['fp'] += int(fp)
            bin_stats_sum[thr]['fn'] += int(fn)

        for t in range(3):
            true_bins = _digitize_bins(true_1h[:, t, :, :].ravel(), CATEGORY_BINS_MM_1H)
            pred_bins = _digitize_bins(pred_1h[:, t, :, :].ravel(), CATEGORY_BINS_MM_1H)
            idx = (true_bins >= 0) & (true_bins < K_h) & (pred_bins >= 0) & (pred_bins < K_h)
            tb = true_bins[idx]; pb = pred_bins[idx]
            np.add.at(confmat_hourly, (tb, pb), 1)

        true_bins_sum = _digitize_bins(true_sum.ravel(), CATEGORY_BINS_MM_SUM)
        pred_bins_sum = _digitize_bins(pred_sum.ravel(), CATEGORY_BINS_MM_SUM)
        idx2 = (true_bins_sum >= 0) & (true_bins_sum < K_s) & (pred_bins_sum >= 0) & (pred_bins_sum < K_s)
        tb2 = true_bins_sum[idx2]; pb2 = pred_bins_sum[idx2]
        np.add.at(confmat_sum, (tb2, pb2), 1)

        base_1h = np.repeat(pred_sum / 3.0, repeats=3, axis=1)
        diff_model = (pred_1h - true_1h)
        diff_base  = (base_1h - true_1h)
        sse_model_1h += float(np.sum(diff_model**2))
        sse_base_1h  += float(np.sum(diff_base**2))
        count_1h     += int(B * 3 * H * W)

        diff_sum = (pred_sum - true_sum)
        sse_sum += float(np.sum(diff_sum**2))
        count_sum += int(B * H * W)

        true_wet = (true_1h >= WETHOUR_THRESHOLD_MM)
        pred_wet = (pred_1h >= WETHOUR_THRESHOLD_MM)
        true_count = np.sum(true_wet, axis=1)
        pred_count = np.sum(pred_wet, axis=1)
        true_count = np.clip(true_count, 0, 3).astype(np.int64).ravel()
        pred_count = np.clip(pred_count, 0, 3).astype(np.int64).ravel()
        np.add.at(conf_wethours, (true_count, pred_count), 1)

    epsilon = 1e-6

    def compute_binary_metrics(stats):
        out = {}
        for thr, d in stats.items():
            tp, tn, fp, fn = d['tp'], d['tn'], d['fp'], d['fn']
            acc = (tp + tn) / (tp + tn + fp + fn + epsilon)
            prec = tp / (tp + fp + epsilon)
            rec = tp / (tp + fn + epsilon)
            f1 = 2 * (prec * rec) / (prec + rec + epsilon)
            csi = tp / (tp + fp + fn + epsilon)
            out[thr] = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'csi': csi,
                        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}
        return out

    metrics_hourly = compute_binary_metrics(bin_stats_hourly)
    metrics_sum    = compute_binary_metrics(bin_stats_sum)

    def compute_categorical_metrics(confmat):
        total = confmat.sum()
        diag = np.trace(confmat)
        acc = diag / (total + epsilon)
        precisions = []
        recalls = []
        for k in range(confmat.shape[0]):
            col_sum = np.sum(confmat[:, k])
            row_sum = np.sum(confmat[k, :])
            tp_k = confmat[k, k]
            p_k = tp_k / (col_sum + epsilon)
            r_k = tp_k / (row_sum + epsilon)
            precisions.append(p_k)
            recalls.append(r_k)
        return acc, np.array(precisions), np.array(recalls)

    cat_acc_hourly, cat_prec_hourly, cat_rec_hourly = compute_categorical_metrics(confmat_hourly)
    cat_acc_sum,    cat_prec_sum,    cat_rec_sum    = compute_categorical_metrics(confmat_sum)

    rmse_model_1h = np.sqrt(sse_model_1h / (count_1h + epsilon))
    rmse_base_1h  = np.sqrt(sse_base_1h  / (count_1h + epsilon))
    rmse_sum      = np.sqrt(sse_sum      / (count_sum + epsilon))

    total_wh = conf_wethours.sum()
    diag_wh = np.trace(conf_wethours)
    acc_wethour_pattern = diag_wh / (total_wh + epsilon)

    with open(eval_log_path, 'a', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("Final Model Evaluation (ratio) with extended metrics\n")
        f.write("="*70 + "\n\n")

        f.write("[Binary Metrics - Hourly (per-pixel over all 3 hours)]\n")
        for thr in BINARY_THRESHOLDS_MM_1H:
            m = metrics_hourly[thr]
            f.write(f"Threshold > {thr} mm: Acc={m['accuracy']:.4f}, Prec={m['precision']:.4f}, "
                    f"Rec={m['recall']:.4f}, F1={m['f1']:.4f}, CSI={m['csi']:.4f} "
                    f"(TP={m['tp']}, TN={m['tn']}, FP={m['fp']}, FN={m['fn']})\n")
        f.write("\n")

        f.write("[Binary Metrics - Accumulation (sum over 3 hours)]\n")
        for thr in BINARY_THRESHOLDS_MM_SUM:
            m = metrics_sum[thr]
            f.write(f"Threshold > {thr} mm: Acc={m['accuracy']:.4f}, Prec={m['precision']:.4f}, "
                    f"Rec={m['recall']:.4f}, F1={m['f1']:.4f}, CSI={m['csi']:.4f} "
                    f"(TP={m['tp']}, TN={m['tn']}, FP={m['fp']}, FN={m['fn']})\n")
        f.write("\n")

        f.write("[Categorical Metrics - Hourly]\n")
        f.write(f"Bins (mm/h): {CATEGORY_BINS_MM_1H}\n")
        f.write(f"Overall categorical accuracy: {cat_acc_hourly:.4f}\n")
        f.write("Per-class precision:\n")
        f.write(" ".join(f"{x:.4f}" for x in cat_prec_hourly) + "\n")
        f.write("Per-class recall:\n")
        f.write(" ".join(f"{x:.4f}" for x in cat_rec_hourly) + "\n\n")

        f.write("[Categorical Metrics - Accumulation]\n")
        f.write(f"Bins (mm/3h): {CATEGORY_BINS_MM_SUM}\n")
        f.write(f"Overall categorical accuracy: {cat_acc_sum:.4f}\n")
        f.write("Per-class precision:\n")
        f.write(" ".join(f"{x:.4f}" for x in cat_prec_sum) + "\n")
        f.write("Per-class recall:\n")
        f.write(" ".join(f"{x:.4f}" for x in cat_rec_sum) + "\n\n")

        f.write("[Equal-split Baseline Comparison - 1h RMSE]\n")
        f.write("Baseline definition: split model's predicted accumulation equally into 3 hours.\n")
        f.write(f"Model 1h RMSE: {rmse_model_1h:.6f}\n")
        f.write(f"Equal-split Baseline 1h RMSE: {rmse_base_1h:.6f}\n")
        better = "BETTER than" if rmse_model_1h < rmse_base_1h else "NOT better than"
        f.write(f"Result: Model is {better} the equal-split baseline (lower is better).\n\n")

        f.write("[Accumulation RMSE]\n")
        f.write(f"Sum RMSE (model sum vs. true sum): {rmse_sum:.6f}\n\n")

        f.write("[Wet-hour count pattern (0/1/2/3) - Consistency]\n")
        f.write(f"Overall pattern accuracy: {acc_wethour_pattern:.4f}\n")

    main_logger.info(f"Evaluation (ratio) finished. Results saved to '{eval_log_path}'")


# ==============================================================================
# 7. メインワーカ関数
# ==============================================================================
def main_worker(rank, world_size, train_files, valid_files):
    """
    概要:
        単一プロセス（=単一GPU）のワーカ。DDP 初期化・データセット/ローダ構築・学習/検証ループ・
        可視化/プロット/評価・後始末までを担当する。
    引数:
        rank (int): GPU ランク。
        world_size (int): 総GPU数。
        train_files (List[str]): 学習用NetCDFファイル群。
        valid_files (List[str]): 検証用NetCDFファイル群。
    処理:
        - 乱数固定、DDP 初期化。
        - NetCDFDataset/DistributedSampler/DataLoader 構築。
        - SwinTransformerSys モデル/DDP/最適化器/AMP スケーラ/スケジューラ設定。
        - NUM_EPOCHS だけ学習・検証、最良モデル保存とメトリクス履歴の蓄積。
        - rank0 で可視化・プロット・最終評価を実行。
        - DDP 後始末。
    戻り値:
        なし
    """
    set_seed(RANDOM_SEED)
    setup_ddp(rank, world_size)
    
    main_log = None
    exec_log_path = EXEC_LOG_PATH
    if rank == 0:
        main_log = logging.getLogger('main')
        main_log.info(f"DDP on {world_size} GPUs. Total batch size: {BATCH_SIZE * world_size}")
        main_log.info(f"Input channels: {IN_CHANS}")
        main_log.info(f"RANDOM_SEED set to: {RANDOM_SEED} for reproducibility.")
        # ratio: 重み付き損失の設定を表示
        if ENABLE_INTENSITY_WEIGHTED_LOSS:
            main_log.info(f"[ratio] Intensity-weighted loss ENABLED.")
            main_log.info(f"[ratio] 1h bins={INTENSITY_WEIGHT_BINS_1H}, weights={INTENSITY_WEIGHT_VALUES_1H}")
            main_log.info(f"[ratio] Sum bins={INTENSITY_WEIGHT_BINS_SUM}, weights={INTENSITY_WEIGHT_VALUES_SUM}")
        else:
            main_log.info(f"[ratio] Intensity-weighted loss DISABLED (using plain MSE).")
    
    train_dataset = NetCDFDataset(train_files, logger=(main_log if rank == 0 else None))
    valid_dataset = NetCDFDataset(valid_files, logger=(main_log if rank == 0 else None))
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, sampler=train_sampler)
    
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, sampler=valid_sampler)
    
    model = SwinTransformerSys(
        img_size=IMG_SIZE, patch_size=PATCH_SIZE, in_chans=IN_CHANS, num_classes=NUM_CLASSES,
        embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=15,
        mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
        norm_layer=nn.LayerNorm, ape=False, patch_norm=True, use_checkpoint=False,
        temperature_mode=TEMPERATURE_MODE, tau_init=TAU_INIT, tau_min=TAU_MIN
    ).to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE * world_size)
    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    if rank == 0:
        main_log.info("Starting training...")
    
    best_val_loss = float('inf')
    best_epoch = -1
    loss_history = {
        'train_loss': [], 'train_1h_rmse': [], 'train_sum_rmse': [],
        'val_loss': [], 'val_1h_rmse': [], 'val_sum_rmse': []
    }

    # 追加: エポックごとのメトリクス履歴
    metric_history = {
        'num_epochs': NUM_EPOCHS,
        'binary_hourly': {thr: {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'csi': []} for thr in BINARY_THRESHOLDS_MM_1H},
        'binary_sum': {thr: {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'csi': []} for thr in BINARY_THRESHOLDS_MM_SUM},
        'categorical_hourly': {'overall_acc': [], 'per_class_precision': [], 'per_class_recall': []},
        'categorical_sum': {'overall_acc': [], 'per_class_precision': [], 'per_class_recall': []},
        'rmse': {'model_1h': [], 'baseline_1h': [], 'sum': []},
        'wet_hour_pattern': {'accuracy': []}
    }
    
    for epoch in range(NUM_EPOCHS):
        train_losses = train_one_epoch(rank, model, train_loader, optimizer, scaler, epoch, exec_log_path)
        val_losses, val_metrics = validate_one_epoch(rank, model, valid_loader, epoch, exec_log_path)
        scheduler.step()
        
        loss_history['train_loss'].append(train_losses[0]); loss_history['train_1h_rmse'].append(train_losses[1]); loss_history['train_sum_rmse'].append(train_losses[2])
        loss_history['val_loss'].append(val_losses[0]); loss_history['val_1h_rmse'].append(val_losses[1]); loss_history['val_sum_rmse'].append(val_losses[2])
        
        if rank == 0:
            main_log.info(f"--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
            main_log.info(f"Train Loss (weighted): {train_losses[0]:.4f} (1h_rmse: {train_losses[1]:.4f}, sum_rmse: {train_losses[2]:.4f})")
            main_log.info(f"Valid Loss (weighted): {val_losses[0]:.4f} (1h_rmse: {val_losses[1]:.4f}, sum_rmse: {val_losses[2]:.4f})")

            # 追加: エポックごとの拡張メトリクスの記録
            if val_metrics is not None:
                for thr in BINARY_THRESHOLDS_MM_1H:
                    for mname in ['accuracy', 'precision', 'recall', 'f1', 'csi']:
                        metric_history['binary_hourly'][thr][mname].append(val_metrics['binary_hourly'][thr][mname])
                for thr in BINARY_THRESHOLDS_MM_SUM:
                    for mname in ['accuracy', 'precision', 'recall', 'f1', 'csi']:
                        metric_history['binary_sum'][thr][mname].append(val_metrics['binary_sum'][thr][mname])
                # カテゴリ（時刻別/積算）
                metric_history['categorical_hourly']['overall_acc'].append(val_metrics['categorical_hourly']['overall_acc'])
                metric_history['categorical_hourly']['per_class_precision'].append(val_metrics['categorical_hourly']['per_class_precision'])
                metric_history['categorical_hourly']['per_class_recall'].append(val_metrics['categorical_hourly']['per_class_recall'])
                metric_history['categorical_sum']['overall_acc'].append(val_metrics['categorical_sum']['overall_acc'])
                metric_history['categorical_sum']['per_class_precision'].append(val_metrics['categorical_sum']['per_class_precision'])
                metric_history['categorical_sum']['per_class_recall'].append(val_metrics['categorical_sum']['per_class_recall'])
                # RMSEとWet-hour
                metric_history['rmse']['model_1h'].append(val_metrics['rmse']['model_1h'])
                metric_history['rmse']['baseline_1h'].append(val_metrics['rmse']['baseline_1h'])
                metric_history['rmse']['sum'].append(val_metrics['rmse']['sum'])
                metric_history['wet_hour_pattern']['accuracy'].append(val_metrics['wet_hour_pattern']['accuracy'])
            
            if val_losses[0] < best_val_loss:
                best_val_loss = val_losses[0]
                best_epoch = epoch + 1
                torch.save(model.module.state_dict(), MODEL_SAVE_PATH)
                main_log.info(f"Saved best model at epoch {best_epoch} with validation (weighted) loss: {best_val_loss:.4f}")
    
    dist.barrier()

    if rank == 0:
        main_log.info("\nTraining finished.")
        
    # 可視化は rank0 のみ実行。例外時も学習終了処理まで到達できるように防御
    try:
        visualize_final_results(rank, world_size, valid_dataset, MODEL_SAVE_PATH, RESULT_IMG_DIR, main_log, exec_log_path)
    except Exception as e:
        if rank == 0 and main_log:
            main_log.error(f"[ratio] visualize_final_results で例外が発生しました。可視化をスキップして後続処理を継続します: {e}", exc_info=True)
    
    dist.barrier()

    if rank == 0:
        # 既存の損失曲線
        plot_loss_curve(loss_history, PLOT_SAVE_PATH, main_log, best_epoch)
        # 追加: エポックメトリクスの図
        plot_epoch_metrics(metric_history, EPOCH_METRIC_PLOT_DIR, main_log)
        # v5: 拡張評価（最終モデル）
        evaluate_model(MODEL_SAVE_PATH, valid_dataset, torch.device(f"cuda:{rank}"), EVALUATION_LOG_PATH, main_log)
        main_log.info(f"Result images saved in '{RESULT_IMG_DIR}/'.")
        main_log.info(f"Epoch metrics plots saved in '{EPOCH_METRIC_PLOT_DIR}/'.")
        main_log.info("All processes finished successfully.")
    
    cleanup_ddp()

# ==============================================================================
# 8. メイン実行ブロック
# ==============================================================================
if __name__ == '__main__':
    set_seed(RANDOM_SEED)

    os.makedirs(RESULT_DIR, exist_ok=True)
    
    main_logger, _ = setup_loggers()

    if CARTOPY_AVAILABLE:
        main_logger.info("Cartopyの地図データを事前にダウンロードします...")
        try:
            # ダミーのプロットを作成して、COASTLINEとBORDERSのデータをダウンロードさせる
            # これにより、並列プロセスが同時にダウンロードするのを防ぐ
            fig_pre = plt.figure()
            ax_pre = plt.axes(projection=ccrs.PlateCarree())
            ax_pre.add_feature(cfeature.COASTLINE)
            ax_pre.add_feature(cfeature.BORDERS)
            plt.close(fig_pre)
            main_logger.info("Cartopyのデータダウンロードが完了しました。")
        except Exception as e:
            main_logger.warning(f"Cartopyのデータ事前ダウンロード中にエラーが発生しました: {e}")
    else:
        main_logger.warning("Cartopyが利用できないため、地図データの事前ダウンロードと地図付き可視化をスキップします。")

    main_logger.info(f"データディレクトリ: {DATA_DIR}")
    main_logger.info(f"現在の作業ディレクトリ: {os.getcwd()}")
    
    train_files = get_monthly_files(DATA_DIR, TRAIN_YEARS, main_logger)
    valid_files = get_monthly_files(DATA_DIR, VALID_YEARS, main_logger)
    
    if not train_files or not valid_files:
        main_logger.error("学習または検証ファイルが見つかりません。DATA_DIRとYEARSの設定を確認してください。")
    else:
        main_logger.info(f"学習ファイル数: {len(train_files)}")
        main_logger.info(f"検証ファイル数: {len(valid_files)}")
        
        world_size = torch.cuda.device_count()
        if world_size > 1:
            main_logger.info(f"{world_size}個のGPUを検出。DDPを開始します。")
            mp.spawn(main_worker,
                     args=(world_size, train_files, valid_files),
                     nprocs=world_size,
                     join=True)
        elif world_size == 1:
            main_logger.info("1個のGPUを検出。シングルGPUモードで実行します。")
            main_worker(0, 1, train_files, valid_files)
        else:
            main_logger.error("GPUが見つかりません。このスクリプトはGPUが必要です。")
