#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# swinunet_main_v5.py
# 変更点(v5):
# - 評価・可視化の「1h」と「3h積算(sum)」で閾値ビンを分離（sumは1hの2倍スケール）
# - 損失関数の重み付けを「1h」と「sum」で分離
#   * 1h（3チャネル）は強度重み付きMSE（従来通りのビンと重み）
#   * sum（3時間積算）はビン境界を1hの2倍に設定し、頻度不均衡を緩和するための段階的な整数重みを導入
# - v3の機能（DDP、詳細メトリクス、可視化、動画化など）はすべて踏襲
# - 保存先ディレクトリやログのバージョン表記をv5に更新

import os
import glob
import warnings
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import xarray as xr
from swin_unet import SwinTransformerSys
from torch.amp import GradScaler, autocast
from swinunet_main_v5_config import (
    CFG, SEED,
    DATA_DIR, TRAIN_YEARS, VALID_YEARS,
    RESULT_DIR, MODEL_SAVE_PATH, PLOT_SAVE_PATH, RESULT_IMG_DIR,
    MAIN_LOG_PATH, EXEC_LOG_PATH, EVALUATION_LOG_PATH, EPOCH_METRIC_PLOT_DIR,
    VIDEO_OUTPUT_PATH, VIDEO_FPS,
    NUM_WORKERS, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,
    IMG_SIZE, PATCH_SIZE, NUM_CLASSES, IN_CHANS,
    INPUT_VARS_COMMON, INPUT_VARS_PREC, TIME_VARS,
    TARGET_VARS_1H, TARGET_VAR_SUM,
    ENABLE_INTENSITY_WEIGHTED_LOSS,
    INTENSITY_WEIGHT_BINS_1H, INTENSITY_WEIGHT_VALUES_1H,
    INTENSITY_WEIGHT_BINS_SUM, INTENSITY_WEIGHT_VALUES_SUM,
    BINARY_THRESHOLDS_MM_1H, BINARY_THRESHOLDS_MM_SUM,
    CATEGORY_BINS_MM_1H, CATEGORY_BINS_MM_SUM,
    WETHOUR_THRESHOLD_MM, LOGNORM_VMIN_MM,
    MODEL_ARGS, AMP_DTYPE
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import logging
import sys
import random
import json
from pathlib import Path
from bisect import bisect_right
# 任意依存関係（存在しない場合は機能をスキップ）
try:
    import hdf5plugin  # noqa: F401
    HDF5PLUGIN_AVAILABLE = True
except Exception as e:
    HDF5PLUGIN_AVAILABLE = False
    warnings.warn(f"hdf5plugin が見つからないため、NetCDFの一部圧縮フィルタが利用できない可能性があります: {e}")

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    CARTOPY_AVAILABLE = True
except Exception as e:
    CARTOPY_AVAILABLE = False
    ccrs = None
    cfeature = None
    warnings.warn(f"Cartopy が見つからないため、地図付き可視化をスキップします: {e}")

# 画像→動画化用（imageioが未インストールの場合はスキップ）
try:
    import imageio.v2 as imageio
    IMAGEIO_AVAILABLE = True
except Exception as e:
    IMAGEIO_AVAILABLE = False
    imageio = None
    warnings.warn(f"imageio が見つからないため、動画作成をスキップします: {e}")

warnings.filterwarnings("ignore", category=UserWarning)
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
os.environ['HDF5_DISABLE_VERSION_CHECK'] = '1'

# AMP dtype mapping from config ("fp16" or "bf16")
AMP_TORCH_DTYPE = torch.float16 if AMP_DTYPE == 'fp16' else torch.bfloat16

# ==============================================================================
# Min-Max 逆変換ユーティリティ（scaler_groups.json を使用）
# ==============================================================================
SCALER_JSON_PATH = os.path.join(os.path.dirname(__file__), "scaler_groups.json")
ENFORCE_SUM_CONSISTENCY = True  # 逆変換後に3時間積算（和）が一致するようにスケーリングを適用する

_group_minmax_cache = None
def load_group_minmax(path: str = SCALER_JSON_PATH):
    """
    概要:
        scaler_groups.json からグループ単位の min/max（グローバルスカラー）を読み込み、プロセス内でキャッシュする。
    引数:
        path (str): メタ情報JSONファイル（scaler_groups.json）のパス。
    処理:
        - 既に読み込み済みの場合はキャッシュを返す。
        - JSON を読み込み、'group_minmax' キーの辞書を取り出す（なければ空辞書）。
    戻り値:
        Dict[str, Dict[str, float]]:
            例: {"precip": {"min": 0.0, "max": 120.5}, ...}
    """
    global _group_minmax_cache
    if _group_minmax_cache is not None:
        return _group_minmax_cache
    try:
        with open(path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        _group_minmax_cache = meta.get('group_minmax', {}) or {}
    except Exception:
        _group_minmax_cache = {}
    return _group_minmax_cache

def inverse_precip_tensor(t: torch.Tensor, group_minmax: dict) -> torch.Tensor:
    """
    概要:
        降水グループ 'precip' に対して min-max 正規化の逆変換を適用し、mm 単位へ復元する。
    引数:
        t (torch.Tensor): 任意shape（..., H, W）。チャネル数は 1ch または 3ch を想定。
        group_minmax (dict): group_minmax の辞書（load_group_minmax の戻り値）。
    処理:
        - group_minmax['precip'] の min/max を取得し、x = t * (max - min) + min を適用。
        - スケーラが存在しない場合は入力をそのまま返す。
    戻り値:
        torch.Tensor: 入力と同形状の mm スケールテンソル。
    """
    if not isinstance(t, torch.Tensor):
        return t
    gm = group_minmax.get('precip')
    if not gm:
        return t  # スケーラが無い場合はそのまま
    gmin = float(gm.get('min', 0.0))
    gmax = float(gm.get('max', 1.0))
    scale = (gmax - gmin)
    if scale <= 1e-12:
        scale = 1.0
    return t * scale + gmin

def enforce_sum_constraint_tensor(pred_1h_mm: torch.Tensor, true_sum_mm: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    概要:
        予測1時間値（B,3,H,W）のチャネル和が、与えられた3時間積算（B,1,H,W）に各画素ごと一致するよう一様スケール係数で補正する。
    引数:
        pred_1h_mm (Tensor): 予測1時間値（mm）。
        true_sum_mm (Tensor): 既知の3時間積算（mm）。
        eps (float): 0除算回避の微小値。
    処理:
        - denom = Σ_t pred_1h_mm を計算し、factor = true_sum_mm / clamp_min(denom, eps) を求める。
        - pred_1h_mm * factor を返す。
    戻り値:
        Tensor: 補正後の 1h 予測（mm, B×3×H×W）。
    """
    denom = pred_1h_mm.sum(dim=1, keepdim=True).clamp_min(eps)
    factor = true_sum_mm / denom
    return pred_1h_mm * factor

# ==============================================================================
# 1. 設定 & ハイパーパラメータ
# ==============================================================================

# --- 再現性確保のための乱数シード設定 ---
RANDOM_SEED = SEED

# ==============================================================================
# 1.5. 再現性確保のための関数
# ==============================================================================
def set_seed(seed):
    """
    概要:
        乱数シードを固定して再現性を確保する。
    引数:
        seed (int): 固定するシード値。
    処理:
        - Python, NumPy, PyTorch(CPU/GPU) の乱数状態を初期化。
        - cuDNN の deterministic/banchmark を適切に設定。
    戻り値:
        なし
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
        指定年の NetCDF 月次ファイル（{year}*.nc）を data_dir から収集する。
    引数:
        data_dir (str): 入力ディレクトリパス。
        years (List[int]): 対象年のリスト。
        logger (logging.Logger|None): ロガー（None のときは print）。
    処理:
        - 各年についてパターン検索し、昇順ソートして全一覧に追加。
        - 見つからない年は警告、見つかった年は件数を出力。
    戻り値:
        List[str]: 見つかったファイルのパス（ソート済み）。
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
        ログ出力（ファイル＋標準出力）を初期化し、実行ログ・評価ログのヘッダを書き出す。
    引数:
        なし
    処理:
        - main ロガーを INFO レベルで作成し、FileHandler と StreamHandler を設定。
        - EXEC_LOG_PATH/EVALUATION_LOG_PATH に初期メッセージを書き込む。
    戻り値:
        Tuple[logging.Logger, str]: (main_logger, EXEC_LOG_PATH)
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
        単一ノード想定で NCCL バックエンドにより DDP を初期化する。
    引数:
        rank (int): このプロセスの GPU ランク。
        world_size (int): 総プロセス（GPU）数。
    処理:
        - MASTER_ADDR/MASTER_PORT を設定し、init_process_group を呼び出す。
    戻り値:
        なし
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_ddp():
    """
    概要:
        DDP のプロセスグループを安全に破棄する。
    引数:
        なし
    戻り値:
        なし
    """
    dist.destroy_process_group()

# ==============================================================================
# 4. データセットクラスの定義
# ==============================================================================
class NetCDFDataset(Dataset):
    """
    メモリ常駐を避け、SSD上のNetCDFから必要な時刻スライスだけをオンデマンドで読み込む実装。
    - 全ファイルをまとめて open_mfdataset せず、各ファイルの time 長だけをメタデータとして保持
    - __getitem__ では (グローバルidx) -> (対象ファイル, ローカルidx) に変換し、そのスライスのみを読み込み
    - 直近の1ファイルだけを開いたままキャッシュし、ファイルを跨いだら前のハンドルを閉じる
    これにより、巨大配列をRAMに展開せず、SSDからの逐次I/O中心で動作します。
    """
    def __init__(self, file_paths, logger=None):
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
        with xr.open_dataset(self.files[0], engine='netcdf4', lock=False) as ds0:
            self.img_dims = (ds0.sizes['lat'], ds0.sizes['lon'])
            self.lon = ds0['lon'].values
            self.lat = ds0['lat'].values

        # 各ファイルの time 長を取得（メタデータのみ）
        self.file_time_lens = []
        for fp in self.files:
            with xr.open_dataset(fp, engine='netcdf4', lock=False) as ds_meta:
                self.file_time_lens.append(int(ds_meta.sizes['time']))

        # 累積長（2分探索で (idx -> file) を即時変換）
        self.cum_counts = np.cumsum(self.file_time_lens).tolist()
        self.time_len = self.cum_counts[-1]

        self.logger.info(f"NetCDFDataset: {len(self.files)} 個のファイルをインデックス化（オンデマンド読み込み）。")
        self.logger.info(f"データセット初期化完了: 総時間ステップ数={self.time_len}, 画像サイズ={self.img_dims}")

    def __del__(self):
        try:
            if self._cache_ds is not None:
                self._cache_ds.close()
        except Exception:
            pass

    def _get_ds(self, path):
        """直近1ファイルのみ開いたままにする簡易キャッシュ"""
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
        self._cache_ds = xr.open_dataset(path, engine='netcdf4', lock=False)
        self._cache_path = path
        return self._cache_ds

    def __len__(self):
        return self.time_len

    def global_to_file_index(self, idx: int):
        """グローバルidx -> (ファイルパス, ローカルidx, ファイル番号) を返す"""
        if idx < 0:
            idx += self.time_len
        if idx < 0 or idx >= self.time_len:
            raise IndexError("Index out of range")
        file_idx = bisect_right(self.cum_counts, idx)
        prev_cum = self.cum_counts[file_idx - 1] if file_idx > 0 else 0
        local_idx = idx - prev_cum
        return self.files[file_idx], local_idx, file_idx

    def get_time(self, idx: int):
        """グローバルidxの time 値（numpy.datetime64）を返す"""
        fpath, local_idx, _ = self.global_to_file_index(idx)
        ds = self._get_ds(fpath)
        return ds['time'].isel(time=local_idx).values

    def __getitem__(self, idx):
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
            val = float(sample[tvar].values.item())
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
        連続値（mm）をビン境界で区分し、各画素に対応する重みを返す。
    引数:
        target_tensor (Tensor): 任意shapeの mm 値テンソル。
        bin_edges (List[float]): 右開区間のビン境界（例: [1,5,10,20,30,50]）。
        weight_values (List[float]): 各ビンの重み（len = len(bin_edges)+1）。
    処理:
        - torch.bucketize でビンIDを求め、weight_values から該当重みを選択。
        - 数値安定性のため float32 で計算。
    戻り値:
        Tensor: target_tensor と同shapeの重みテンソル（float32）。
    """
    # 数値安定化のため、ビニングと重み計算は常に float32 で実施（AMP下のhalfでのオーバーフロー回避）
    device = target_tensor.device
    t = target_tensor.float()
    boundaries_t = torch.tensor(bin_edges, device=device, dtype=torch.float32)
    idx = torch.bucketize(t, boundaries_t, right=False)  # 0..len(boundaries)
    wvals_t = torch.tensor(weight_values, device=device, dtype=torch.float32)  # len = len(boundaries)+1
    weights = wvals_t[idx]
    return weights

def _weighted_mse(pred, target, bin_edges, weight_values, eps=1e-8):
    """
    概要:
        重みマップに基づく加重MSEを計算する。
    引数:
        pred (Tensor): 予測テンソル。
        target (Tensor): 正解テンソル（mm）。
        bin_edges (List[float]): ビン境界。
        weight_values (List[float]): 各ビン重み。
        eps (float): 0除算回避の微小値。
    処理:
        - _get_weight_map_from_bins で weights を作成（float32）。
        - num = Σ weights * (pred-target)^2、den = Σ weights + eps を計算し num/den を返す。
    戻り値:
        Tensor: スカラー損失（float32）。
    注意:
        AMP(fp16/bf16) 下でも安定するよう、計算は float32 に揃える。
    """
    pred_f = pred.float()
    target_f = target.float()
    weights = _get_weight_map_from_bins(target_f, bin_edges, weight_values)  # float32
    se = (pred_f - target_f) ** 2
    num = (weights * se).sum(dtype=torch.float32)
    den = weights.sum(dtype=torch.float32) + torch.tensor(eps, device=weights.device, dtype=torch.float32)
    return num / den

def custom_loss_function(output, targets, eps=1e-8):
    """
    概要:
        v5 損失。1h（3ch）は強度重み付きMSE、sum（3h積算）も2倍スケールのビン＋重みで強度重み付きMSE。
        ログ用の RMSE は非加重MSEから算出。
    引数:
        output (Tensor): モデル出力（B,3,H,W）非負想定（ReLUで整形）。
        targets (Dict[str, Tensor]): {'target_1h':(B,3,H,W), 'target_sum':(B,1,H,W)}。
        eps (float): RMSE 計算の数値安定用微小値。
    処理:
        - 出力/ターゲットの NaN/Inf を除去し、極端値をクリップ。
        - predicted_sum = Σ output、未加重MSE を計算。
        - ENABLE_INTENSITY_WEIGHTED_LOSS に応じて、_weighted_mse を1h/sumに適用。
        - total_loss, rmse_1h, rmse_sum を返す。
    戻り値:
        Tuple[Tensor, Tensor, Tensor]: (total_loss, rmse_1h, rmse_sum)
    """
    # 物理量制約: 負の降水は0に、NaN/Infは安全値に置換
    output = torch.nan_to_num(output, nan=0.0, posinf=1e6, neginf=0.0)
    output = torch.relu(output).float()
    target_1h = torch.nan_to_num(targets['target_1h'], nan=0.0, posinf=1e6, neginf=0.0).float()
    target_sum = torch.nan_to_num(targets['target_sum'], nan=0.0, posinf=1e6, neginf=0.0).float()
    # 物理的上限のクリッピング（極端値による勾配爆発を回避）
    output = torch.clamp(output, min=0.0, max=500.0)
    target_1h = torch.clamp(target_1h, min=0.0, max=500.0)
    target_sum = torch.clamp(target_sum, min=0.0, max=1500.0)

    predicted_sum = torch.sum(output, dim=1, keepdim=True)

    # 非加重MSE（RMSE計算用, float32）
    unweighted_mse_1h = nn.functional.mse_loss(output, target_1h)
    unweighted_mse_sum = nn.functional.mse_loss(predicted_sum, target_sum)

    if ENABLE_INTENSITY_WEIGHTED_LOSS:
        # 1時間（3ch）は強度重み付きMSE（計算はfloat32）
        loss_1h_mse = _weighted_mse(output, target_1h, INTENSITY_WEIGHT_BINS_1H, INTENSITY_WEIGHT_VALUES_1H)
        # sumも強度重み付きMSE（または非加重にするなら値を全て1.0に設定）
        loss_sum_mse = _weighted_mse(predicted_sum, target_sum, INTENSITY_WEIGHT_BINS_SUM, INTENSITY_WEIGHT_VALUES_SUM)
    else:
        loss_1h_mse = unweighted_mse_1h
        loss_sum_mse = unweighted_mse_sum

    total_loss = (loss_1h_mse + loss_sum_mse).float()

    with torch.no_grad():
        rmse_1h = torch.sqrt(torch.clamp(unweighted_mse_1h, min=eps)).float()
        rmse_sum = torch.sqrt(torch.clamp(unweighted_mse_sum, min=eps)).float()

    return total_loss, rmse_1h, rmse_sum

def train_one_epoch(rank, model, dataloader, optimizer, scaler, epoch, exec_log_path):
    """
    概要:
        1エポック分の学習を実行し、DDP 全プロセスで平均化した損失/指標を返す。
    引数:
        rank (int): GPU ランク。
        model (DDP Module): 学習対象モデル（DDP 包装済み）。
        dataloader (DataLoader): 学習データローダ（DistributedSampler 必須）。
        optimizer (Optimizer): 最適化器（AdamW 等）。
        scaler (GradScaler): AMP 用スケーラ（bf16 時は disabled 可）。
        epoch (int): エポック番号（0始まり）。
        exec_log_path (str): tqdm の出力ファイルパス。
    処理:
        - autocast(AMP) で forward、custom_loss_function で損失算出。
        - 非有限損失はスキップ。勾配クリップ後に optimizer step / scaler update。
        - 進捗は rank0 のみファイルに表示。
        - all_reduce(Average) で [total, 1h_rmse, sum_rmse] を集計。
    戻り値:
        np.ndarray: [avg_total_loss, avg_1h_rmse, avg_sum_rmse]
    """
    model.train()
    total_loss, total_1h_rmse, total_sum_rmse = 0.0, 0.0, 0.0
    skipped_batches = 0
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
            
            # 入力データの健全性チェック（初期エポックのみ）
            if epoch == 0 and batch_idx == 0 and rank == 0:
                progress_bar.write(f"[DEBUG] Input range: [{inputs.min().item():.4f}, {inputs.max().item():.4f}]")
                progress_bar.write(f"[DEBUG] Target_1h range: [{targets['target_1h'].min().item():.4f}, {targets['target_1h'].max().item():.4f}]")
                progress_bar.write(f"[DEBUG] Target_sum range: [{targets['target_sum'].min().item():.4f}, {targets['target_sum'].max().item():.4f}]")
            
            with autocast(device_type='cuda', dtype=AMP_TORCH_DTYPE):
                outputs = model(inputs)
                loss, loss_1h, loss_sum = custom_loss_function(outputs, targets)
            
            # 非有限値の検出とスキップ（DDP安定のためゼログラドしてcontinue）
            if not torch.isfinite(loss):
                skipped_batches += 1
                if rank == 0:
                    progress_bar.write(f"Warning: non-finite loss detected at batch {batch_idx}. loss={loss.item()}, rmse_1h={loss_1h.item()}, rmse_sum={loss_sum.item()}")
                    progress_bar.write(f"Output range: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")
                optimizer.zero_grad(set_to_none=True)
                continue
            
            scaler.scale(loss).backward()
            # AMP下の勾配を一度unscaleしてからクリッピング（NaN/Inf対策）
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 勾配の健全性チェック（初期エポックのみ）
            if epoch == 0 and batch_idx == 0 and rank == 0:
                progress_bar.write(f"[DEBUG] Gradient norm before clipping: {grad_norm:.6f}")
            
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            total_1h_rmse += loss_1h.item()
            total_sum_rmse += loss_sum.item()
            if rank == 0:
                progress_bar.set_postfix(loss=loss.item(), rmse_1h=loss_1h.item(), rmse_sum=loss_sum.item())
        
        if rank == 0 and skipped_batches > 0:
            progress_bar.write(f"[WARNING] Epoch {epoch+1}: Skipped {skipped_batches} batches due to non-finite loss")
            
    avg_loss = torch.tensor([total_loss, total_1h_rmse, total_sum_rmse], device=rank) / len(dataloader)
    dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
    return avg_loss.cpu().numpy()

# -------------------- 評価ユーティリティ --------------------
def _tally_binary(pred, true, thr):
    """2値判定のTP/TN/FP/FNを返す（pred, trueは同形状テンソル/ndarray, mm単位）"""
    pred_bin = pred > thr
    true_bin = true > thr
    tp = np.logical_and(pred_bin, true_bin).sum()
    tn = np.logical_and(~pred_bin, ~true_bin).sum()
    fp = np.logical_and(pred_bin, ~true_bin).sum()
    fn = np.logical_and(~pred_bin, true_bin).sum()
    return tp, tn, fp, fn

def _digitize_bins(x, bins):
    """連続値をカテゴリbin(右開区間)に割り当てる（numpy.digitize準拠）"""
    return np.digitize(x, bins, right=False) - 1  # 0始まりのカテゴリID

@torch.no_grad()
def validate_one_epoch(rank, model, dataloader, epoch, exec_log_path):
    """
    概要:
        1エポック分の検証を実行し、加重損失・1h/sum RMSE に加え、閾値別 Binary 指標やカテゴリ指標などを集計する。
    引数:
        rank (int): GPU ランク。
        model (DDP Module): 評価対象モデル。
        dataloader (DataLoader): 検証データローダ。
        epoch (int): エポック番号。
        exec_log_path (str): tqdm の出力ファイルパス。
    処理:
        - autocast(AMP) で forward、custom_loss_function で損失算出。
        - min-max 逆変換＋和一致補正を適用した mm スケールで詳細メトリクスを算出。
        - DDP all_reduce(SUM/Average) で集計し、rank0 で指標辞書を構築。
    戻り値:
        Tuple[np.ndarray, Optional[Dict]]: (avg_losses, epoch_metrics or None)
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

            with autocast(device_type='cuda', dtype=AMP_TORCH_DTYPE):
                outputs = model(inputs)
                loss, loss_1h, loss_sum = custom_loss_function(outputs, targets)

            total_loss += loss.item()
            total_1h_rmse += loss_1h.item()
            total_sum_rmse += loss_sum.item()

            # 追加: 詳細メトリクスのためのデータ（min-max逆変換＋積算一致の補正を適用）
            outputs = torch.relu(outputs)
            gm = load_group_minmax()
            pred_1h_mm_t = inverse_precip_tensor(outputs.float(), gm)                             # (B,3,H,W) in mm
            true_1h_mm_t = inverse_precip_tensor(targets['target_1h'].float(), gm)               # (B,3,H,W) in mm
            true_sum_mm_t = inverse_precip_tensor(targets['target_sum'].float(), gm)             # (B,1,H,W) in mm
            if ENFORCE_SUM_CONSISTENCY:
                pred_1h_mm_t = enforce_sum_constraint_tensor(pred_1h_mm_t, true_sum_mm_t)
            pred_sum_mm_t = pred_1h_mm_t.sum(dim=1, keepdim=True)                                # (B,1,H,W)
            # detach to numpy
            pred_1h = pred_1h_mm_t.detach().cpu().numpy()       # (B,3,H,W)
            pred_sum = pred_sum_mm_t.detach().cpu().numpy()      # (B,1,H,W)
            true_1h = true_1h_mm_t.detach().cpu().numpy()
            true_sum = true_sum_mm_t.detach().cpu().numpy()
            B, _, H, W = outputs.shape

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
        最良モデルを用いて検証データ全体を可視化する。2x4 パネル（予測:積算/ft4/5/6、正解:同）で PNG を保存。
    引数:
        rank (int): GPU ランク（rank0 のみ実行）。
        world_size (int): GPU 総数。
        valid_dataset (Dataset): 検証データセット（lon/lat/get_time を利用）。
        best_model_path (str): 最良モデル（.pth）へのパス。
        result_img_dir (str): 画像出力ディレクトリ。
        main_logger (Logger|None): ロガー。
        exec_log_path (str): tqdm 出力ファイル。
    処理:
        - rank0 のみモデルをロードして推論、min-max 逆変換＋和一致補正を適用。
        - LogNorm で見やすくし、地図レイヤ付きで保存。失敗時は通常Axesへフォールバック。
    戻り値:
        なし（副作用として PNG ファイル保存）
    """
    # DDP安全化: 可視化はrank 0のみで実行（Cartopyの同時アクセスを回避）
    if rank != 0:
        return
    if not CARTOPY_AVAILABLE:
        if rank == 0:
            main_logger.warning("Cartopyが見つからないため、最終的な可視化をスキップします。")
        return

    device = torch.device(f"cuda:{rank}")
    model = SwinTransformerSys(**MODEL_ARGS).to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    if rank == 0:
        os.makedirs(result_img_dir, exist_ok=True)
        main_logger.info(f"[v5] 検証データ全体に対する可視化を開始 (モデル: {os.path.basename(best_model_path)})")
    
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

            with autocast(device_type='cuda', dtype=AMP_TORCH_DTYPE):
                output = model(inputs)
            output = torch.relu(output).squeeze(0)  # (3, H, W) on device

            # min-max逆変換 + 積算一致補正（描画はmmスケールで統一）
            gm = load_group_minmax()
            output_mm = inverse_precip_tensor(output.float().unsqueeze(0), gm).squeeze(0).cpu()           # (3,H,W)
            target_1h_mm = inverse_precip_tensor(target_1h.float().unsqueeze(0), gm).squeeze(0).cpu()     # (3,H,W)
            target_sum_mm = inverse_precip_tensor(target_sum.float().unsqueeze(0).unsqueeze(0), gm).squeeze(0).cpu()  # (H,W)
            if ENFORCE_SUM_CONSISTENCY:
                pred_1h_mm = enforce_sum_constraint_tensor(output_mm.unsqueeze(0), target_sum_mm.unsqueeze(0).unsqueeze(0)).squeeze(0)
            else:
                pred_1h_mm = output_mm
            # 安全化: チャネル数が3未満でも可視化が落ちないようにゼロ埋めで3chに揃える
            def _ensure_three_channels(t: torch.Tensor) -> torch.Tensor:
                # t: (C,H,W) or (H,W) -> (3,H,W)
                if t.dim() == 2:
                    t = t.unsqueeze(0)
                C = t.shape[0]
                if C >= 3:
                    return t[:3]
                pads = [t.new_zeros((1, *t.shape[1:])) for _ in range(3 - C)]
                return torch.cat([t, *pads], dim=0)

            pred_3 = _ensure_three_channels(pred_1h_mm)     # (3,H,W)
            tgt_3  = _ensure_three_channels(target_1h_mm)   # (3,H,W)
            pred_sum = pred_3.sum(dim=0)  # (H, W)

            # 対数色スケール設定 (LogNorm)
            cmap = plt.get_cmap("Blues")
            cmap.set_under(alpha=0)
            vmin_val = max(LOGNORM_VMIN_MM, 1e-6)

            # vmaxは外れ値に影響されにくいように8枚の99パーセンタイルで決定
            stack_vals = torch.stack([
                pred_sum,
                target_sum_mm,
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
                target_sum_mm.numpy(), tgt_3[0].numpy(), tgt_3[1].numpy(), tgt_3[2].numpy()
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
        学習/検証の合計損失と RMSE（1h/sum）のエポック推移をプロットして保存する。
    引数:
        history (Dict[str, List[float]]): 'train_loss','val_loss','train_1h_rmse','val_1h_rmse','train_sum_rmse','val_sum_rmse'。
        save_path (str): 保存先パス。
        logger (Logger): ロガー。
        best_epoch (int): 最良エポック（1始まり、無ければ -1）。
    戻り値:
        なし
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
        連続ビン境界から可読なラベル（例: '0-5','5-10','10+'）を生成する。
    引数:
        bins (List[float]): ビン境界（最後の境界は +inf を想定）。
    戻り値:
        List[str]: クラスラベル配列。
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
        各種評価指標（Binary, Categorical, RMSE, Wet-hour）のエポック推移をまとめて図に保存する。
    引数:
        metric_history (Dict): validate_one_epoch の 'epoch_metrics' を各エポックで蓄積した構造。
        out_dir (str): 出力ディレクトリ。
        logger (Logger): ロガー。
    戻り値:
        なし（複数 PNG を保存）
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
        最終モデルを用いて検証データ全体を評価し、拡張メトリクス（Binary/カテゴリ/RMSE/Wet-hour）をテキストに出力。
    引数:
        model_path (str): .pth のモデルファイル。
        valid_dataset (Dataset): 検証データセット。
        device (torch.device): 評価に使用するデバイス。
        eval_log_path (str): 結果の出力パス。
        main_logger (Logger): ロガー。
    処理:
        - DataLoader で一括推論、min-max 逆変換＋和一致補正の mm スケールで集計。
        - 等分割ベースラインとの RMSE 比較を含む。
    戻り値:
        なし（副作用として eval_log_path に詳細を出力）
    """
    main_logger.info("Starting final model evaluation (v5)...")
    
    model = SwinTransformerSys(**MODEL_ARGS).to(device)
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

    progress_bar = tqdm(eval_loader, desc="Evaluating Model (v5)", leave=False)
    for batch in progress_bar:
        inputs = batch['input'].to(device)
        targ_1h = batch['target_1h'].to(device)
        targ_sum = batch['target_sum'].to(device)

        with autocast(device_type='cuda', dtype=AMP_TORCH_DTYPE):
            outputs = model(inputs)
        outputs = torch.relu(outputs)

        # min-max逆変換 + 積算一致補正（評価用はmmスケールで実施）
        gm = load_group_minmax()
        pred_1h_mm_t = inverse_precip_tensor(outputs.float(), gm)                   # (B,3,H,W)
        true_1h_mm_t = inverse_precip_tensor(targ_1h.float(), gm)                  # (B,3,H,W)
        true_sum_mm_t = inverse_precip_tensor(targ_sum.float(), gm)                # (B,1,H,W)
        if ENFORCE_SUM_CONSISTENCY:
            pred_1h_mm_t = enforce_sum_constraint_tensor(pred_1h_mm_t, true_sum_mm_t)
        pred_sum_mm_t = pred_1h_mm_t.sum(dim=1, keepdim=True)

        # numpyへ
        pred_1h = pred_1h_mm_t.detach().cpu().numpy()
        true_1h = true_1h_mm_t.detach().cpu().numpy()
        pred_sum = pred_sum_mm_t.detach().cpu().numpy()
        true_sum = true_sum_mm_t.detach().cpu().numpy()
        B, _, H, W = outputs.shape

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
        f.write("Final Model Evaluation (v5) with extended metrics\n")
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

    main_logger.info(f"Evaluation (v5) finished. Results saved to '{eval_log_path}'")

def create_video_from_images(image_dir, output_path, fps, logger):
    """
    概要:
        検証画像（validation_*.png）を連結して MP4 動画を作成する（libx264）。
    引数:
        image_dir (str): 画像が格納されたディレクトリ。
        output_path (str): 出力動画パス（.mp4）。
        fps (int): フレームレート。
        logger (Logger): ロガー。
    処理:
        - imageio が無ければ警告してスキップ。
        - 昇順に画像を読み込み、エンコーダで逐次書き込み。
    戻り値:
        なし（副作用として MP4 を出力）
    """
    if not IMAGEIO_AVAILABLE:
        logger.warning("imageioが見つからないため、動画作成をスキップします。pip install imageio imageio-ffmpeg を検討してください。")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # 検証画像は validation_*.png で保存される
    pattern = os.path.join(image_dir, 'validation_*.png')
    files = sorted(glob.glob(pattern))

    if len(files) == 0:
        logger.warning(f"動画化対象画像が見つかりませんでした: {pattern}")
        return

    logger.info(f"動画作成を開始: フレーム数={len(files)}, fps={fps}, 出力='{output_path}'")
    try:
        writer = imageio.get_writer(output_path, fps=fps, codec='libx264', quality=8, pixelformat='yuv420p')
        for i, fpath in enumerate(files):
            frame = imageio.imread(fpath)
            writer.append_data(frame)
        writer.close()
        logger.info(f"動画作成が完了しました: '{output_path}'")
    except Exception as e:
        logger.error(f"動画作成に失敗しました: {e}")

# ==============================================================================
# 7. メインワーカ関数
# ==============================================================================
def main_worker(rank, world_size, train_files, valid_files):
    """
    概要:
        単一 GPU プロセスのワーカー。DDP 初期化から学習/検証/可視化/評価/動画化までを実行する。
    引数:
        rank (int): GPU ランク。
        world_size (int): 総GPU数。
        train_files (List[str]): 学習ファイルパス。
        valid_files (List[str]): 検証ファイルパス。
    処理:
        - 乱数固定、DDP 初期化、データローダ構築、モデル/DDP/最適化器/AMP 準備。
        - エポックループで学習/検証、最良モデル保存、メトリクス履歴蓄積。
        - rank0 で可視化・プロット・評価・動画化を実行。
        - DDP 後処理。
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
        # v5: 重み付き損失の設定を表示
        if ENABLE_INTENSITY_WEIGHTED_LOSS:
            main_log.info(f"[v5] Intensity-weighted loss ENABLED.")
            main_log.info(f"[v5] 1h bins={INTENSITY_WEIGHT_BINS_1H}, weights={INTENSITY_WEIGHT_VALUES_1H}")
            main_log.info(f"[v5] Sum bins={INTENSITY_WEIGHT_BINS_SUM}, weights={INTENSITY_WEIGHT_VALUES_SUM}")
        else:
            main_log.info(f"[v5] Intensity-weighted loss DISABLED (using plain MSE).")
    
    train_dataset = NetCDFDataset(train_files, logger=(main_log if rank == 0 else None))
    valid_dataset = NetCDFDataset(valid_files, logger=(main_log if rank == 0 else None))
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, sampler=train_sampler)
    
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, sampler=valid_sampler)
    
    model = SwinTransformerSys(**MODEL_ARGS).to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE * world_size)
    # fp16 のみ GradScaler を有効化（bf16は不要）
    scaler = GradScaler(enabled=(AMP_DTYPE == 'fp16'))
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
        
    try:
        visualize_final_results(rank, world_size, valid_dataset, MODEL_SAVE_PATH, RESULT_IMG_DIR, main_log, exec_log_path)
    except Exception as e:
        if rank == 0 and main_log:
            main_log.error(f"visualize_final_results で例外が発生しました。可視化をスキップして後続処理を継続します: {e}", exc_info=True)
    
    dist.barrier()

    if rank == 0:
        # 既存の損失曲線
        plot_loss_curve(loss_history, PLOT_SAVE_PATH, main_log, best_epoch)
        # 追加: エポックメトリクスの図
        plot_epoch_metrics(metric_history, EPOCH_METRIC_PLOT_DIR, main_log)
        # v5: 拡張評価（最終モデル）
        evaluate_model(MODEL_SAVE_PATH, valid_dataset, torch.device(f"cuda:{rank}"), EVALUATION_LOG_PATH, main_log)
        # v5: 画像→動画化
        create_video_from_images(RESULT_IMG_DIR, VIDEO_OUTPUT_PATH, VIDEO_FPS, main_log)

        main_log.info(f"Result images saved in '{RESULT_IMG_DIR}/'.")
        main_log.info(f"Epoch metrics plots saved in '{EPOCH_METRIC_PLOT_DIR}/'.")
        if IMAGEIO_AVAILABLE:
            main_log.info(f"Validation video saved to '{VIDEO_OUTPUT_PATH}'.")
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
