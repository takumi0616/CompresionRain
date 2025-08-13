#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# swinunet_main_v2.py
# 変更点:
# - カテゴリ評価(5mm,10mm...)の追加
# - 可視化: 左側に積算(予測/正解)を追加し2x4図に拡張
# - 可視化: 対数表示(LogNorm)で見やすく
# - 評価: equal-split(積算を3分割)ベースラインとの比較
# - 評価: 3時間の降水「あり時間数」(0/1/2/3)のパターン一致評価

import os
import glob
import warnings
import hdf5plugin
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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import logging
import sys
import random
import cartopy.crs as ccrs
import cartopy.feature as cfeature
CARTOPY_AVAILABLE = True

warnings.filterwarnings("ignore", category=UserWarning)
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
os.environ['HDF5_DISABLE_VERSION_CHECK'] = '1'

# ==============================================================================
# 1. 設定 & ハイパーパラメータ
# ==============================================================================

# --- データパス設定 ---
DATA_DIR = './optimization_nc'
TRAIN_YEARS = [2018, 2019, 2020, 2021]
VALID_YEARS = [2022]

# --- 結果の保存先ディレクトリ（v2用に名称を変更） ---
RESULT_DIR = 'swin-unet_main_result_v2'

# モデル、プロット、ログの保存先パス（v2）
MODEL_SAVE_PATH = os.path.join(RESULT_DIR, 'best_swin_unet_model_ddp_monthly_v2.pth')
PLOT_SAVE_PATH = os.path.join(RESULT_DIR, 'loss_curve_monthly_v2.png')
RESULT_IMG_DIR = os.path.join(RESULT_DIR, 'result_images_monthly_v2')
MAIN_LOG_PATH = os.path.join(RESULT_DIR, 'main_v2.log')
EXEC_LOG_PATH = os.path.join(RESULT_DIR, 'execution_v2.log')
EVALUATION_LOG_PATH = os.path.join(RESULT_DIR, 'evaluation_v2.log')

NUM_WORKERS = 0

# --- 学習パラメータ ---
BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4

# --- 再現性確保のための乱数シード設定 ---
RANDOM_SEED = 40

# --- モデルパラメータ ---
IMG_SIZE = 480
PATCH_SIZE = 4
NUM_CLASSES = 3

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
BINARY_THRESHOLDS_MM = [0.0, 0.1, 1.0, 5.0, 10.0, 20.0, 30.0, 50.0]
# カテゴリ分類のビン（例: 0-5, 5-10, 10-20, 20-30, 30-50, 50+）
CATEGORY_BINS_MM = [0.0, 5.0, 10.0, 20.0, 30.0, 50.0, float('inf')]
# 「降水あり時間数」の判定しきい値（>= 0.1mm を降水ありと判定）
WETHOUR_THRESHOLD_MM = 0.1

# 可視化（対数表示）の最小値（これ未満は透明扱い）
LOGNORM_VMIN_MM = 0.1

# ==============================================================================
# 1.5. 再現性確保のための関数
# ==============================================================================
def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_monthly_files(data_dir, years, logger=None):
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
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_ddp():
    dist.destroy_process_group()

# ==============================================================================
# 4. データセットクラスの定義
# ==============================================================================
class NetCDFDataset(Dataset):
    def __init__(self, file_paths, logger=None):
        super().__init__()
        self.logger = logger if logger else logging.getLogger('main')
        if not file_paths:
            raise ValueError("ファイルパスのリストが空です")
        for fp in file_paths:
            if not os.path.exists(fp):
                raise FileNotFoundError(f"ファイルが見つかりません: {fp}")
        self.logger.info(f"NetCDFDataset: {len(file_paths)} 個のファイルを読み込み中...")
        try:
            self.ds = xr.open_mfdataset(
                file_paths, chunks={}, parallel=False, combine='by_coords',
                engine='netcdf4', lock=False
            )
        except Exception as e:
            self.logger.error(f"xarray.open_mfdatasetでの読み込みに失敗: {e}")
            raise
        self.time_len = len(self.ds['time'])
        self.img_dims = (self.ds.sizes['lat'], self.ds.sizes['lon'])
        self.logger.info(f"データセット初期化完了: 時間ステップ数={self.time_len}, 画像サイズ={self.img_dims}")
    
    def __len__(self):
        return self.time_len
    
    def __getitem__(self, idx):
        sample = self.ds.isel(time=idx).load()
        input_channels = []
        for var in INPUT_VARS_COMMON:
            input_channels.append(np.nan_to_num(sample[f'{var}_ft3'].values))
            input_channels.append(np.nan_to_num(sample[f'{var}_ft6'].values))
        for var in INPUT_VARS_PREC:
            input_channels.append(np.nan_to_num(sample[var].values))
        h, w = self.img_dims
        for tvar in TIME_VARS:
            val = sample[tvar].item()
            channel = np.full((h, w), val, dtype=np.float32)
            input_channels.append(channel)
        input_tensor = torch.from_numpy(np.stack(input_channels, axis=0)).float()
        
        target_1h_channels = [np.nan_to_num(sample[var].values) for var in TARGET_VARS_1H]
        target_1h_tensor = torch.from_numpy(np.stack(target_1h_channels, axis=0)).float()
        target_sum_tensor = torch.from_numpy(np.nan_to_num(sample[TARGET_VAR_SUM].values)).float().unsqueeze(0)
        
        return {"input": input_tensor, "target_1h": target_1h_tensor, "target_sum": target_sum_tensor}

# ==============================================================================
# 5. 損失関数、学習・検証関数
# ==============================================================================
def custom_loss_function(output, targets):
    output = torch.relu(output)
    target_1h = targets['target_1h']
    loss_1h_mse = nn.functional.mse_loss(output, target_1h)
    predicted_sum = torch.sum(output, dim=1, keepdim=True)
    target_sum = targets['target_sum']
    loss_sum_mse = nn.functional.mse_loss(predicted_sum, target_sum)
    total_loss = loss_1h_mse + loss_sum_mse
    with torch.no_grad():
        rmse_1h = torch.sqrt(loss_1h_mse)
        rmse_sum = torch.sqrt(loss_sum_mse)
    return total_loss, rmse_1h, rmse_sum

def train_one_epoch(rank, model, dataloader, optimizer, scaler, epoch, exec_log_path):
    model.train()
    total_loss, total_1h_rmse, total_sum_rmse = 0.0, 0.0, 0.0
    dataloader.sampler.set_epoch(epoch)
    
    with open(exec_log_path, 'a', encoding='utf-8') as log_file:
        progress_bar = tqdm(
            dataloader, desc=f"Train Epoch {epoch+1}", disable=(rank != 0), 
            leave=True, file=log_file,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
        )
        for batch in progress_bar:
            inputs = batch['input'].to(rank, non_blocking=True)
            targets = {k: v.to(rank, non_blocking=True) for k, v in batch.items() if k != 'input'}
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(inputs)
                loss, loss_1h, loss_sum = custom_loss_function(outputs, targets)
            scaler.scale(loss).backward()
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

@torch.no_grad()
def validate_one_epoch(rank, model, dataloader, epoch, exec_log_path):
    model.eval()
    total_loss, total_1h_rmse, total_sum_rmse = 0.0, 0.0, 0.0
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
            if rank == 0:
                progress_bar.set_postfix(loss=loss.item())
    
    avg_loss = torch.tensor([total_loss, total_1h_rmse, total_sum_rmse], device=rank) / len(dataloader)
    dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
    return avg_loss.cpu().numpy()

# ==============================================================================
# 6. 可視化 & プロット & 評価関数
# ==============================================================================
@torch.no_grad()
def visualize_final_results(rank, world_size, valid_dataset, best_model_path, result_img_dir, main_logger, exec_log_path):
    """検証データ全体での可視化: 2x4配置(左に積算 予測/正解 + 各時刻 予測/正解)、LogNormで見やすく"""
    if not CARTOPY_AVAILABLE:
        if rank == 0:
            main_logger.warning("Cartopyが見つからないため、最終的な可視化をスキップします。")
        return

    device = torch.device(f"cuda:{rank}")
    model = SwinTransformerSys(
        img_size=IMG_SIZE, patch_size=PATCH_SIZE, in_chans=IN_CHANS, num_classes=NUM_CLASSES,
        embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=15,
        mlp_ratio=4., qkv_bias=True, norm_layer=nn.LayerNorm, use_checkpoint=False
    ).to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    if rank == 0:
        os.makedirs(result_img_dir, exist_ok=True)
        main_logger.info(f"検証データ全体に対する可視化を開始 (モデル: {os.path.basename(best_model_path)})")
    
    indices = list(range(len(valid_dataset)))
    proc_indices = indices[rank::world_size]

    lon = valid_dataset.ds['lon'].values
    lat = valid_dataset.ds['lat'].values
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
                output = model(inputs)
            output = torch.relu(output).squeeze(0).cpu()  # (3, H, W)
            pred_sum = output.sum(dim=0)  # (H, W)

            # 対数色スケール設定 (LogNorm)
            cmap = plt.get_cmap("Blues")
            cmap.set_under(alpha=0)
            vmin_val = max(LOGNORM_VMIN_MM, 1e-6)

            # vmaxは外れ値に影響されにくいように8枚の99パーセンタイルで決定
            stack_vals = torch.stack([
                pred_sum,
                target_sum,
                output[0], output[1], output[2],
                target_1h[0], target_1h[1], target_1h[2]
            ], dim=0).numpy()
            flat = stack_vals.reshape(8, -1)
            # 0以下は除外してパーセンタイル計算
            flat_pos = flat[flat > 0.0]
            vmax_val = np.percentile(flat_pos, 99) if flat_pos.size > 0 else vmin_val * 10.0
            if vmax_val <= vmin_val:  # 保険
                vmax_val = vmin_val * 10.0

            norm = LogNorm(vmin=vmin_val, vmax=vmax_val)

            # 2行4列の図: 上段(予測: 積算, ft+4, ft+5, ft+6) 下段(正解: 積算, ft+4, ft+5, ft+6)
            fig, axes = plt.subplots(2, 4, figsize=(24, 12), subplot_kw={'projection': ccrs.PlateCarree()})
            time_val = valid_dataset.ds.time.values[idx]
            fig.suptitle(f"Validation Time: {np.datetime_as_string(time_val, unit='m')}", fontsize=16)

            plot_data = [
                pred_sum.numpy(), output[0].numpy(), output[1].numpy(), output[2].numpy(),
                target_sum.numpy(), target_1h[0].numpy(), target_1h[1].numpy(), target_1h[2].numpy()
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
            plt.savefig(save_path, dpi=150)
            plt.close(fig)

def plot_loss_curve(history, save_path, logger, best_epoch):
    plt.figure(figsize=(12, 8))
    epochs_range = range(1, len(history['train_loss']) + 1)
    
    plt.subplot(2, 1, 1)
    plt.plot(epochs_range, history['train_loss'], 'bo-', label='Training Total Loss')
    plt.plot(epochs_range, history['val_loss'], 'ro-', label='Validation Total Loss')
    
    if best_epoch != -1:
        best_loss_val = history['val_loss'][best_epoch-1]
        plt.axvline(x=best_epoch, color='k', linestyle='--', label=f'Best Epoch: {best_epoch} (Loss: {best_loss_val:.4f})')

    plt.title('Total Loss Curve (MSE-based)')
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
    # bins: 例えば [0,5,10,20,30,50,inf]
    # np.digitizeは右側は閉区間にしないため、binsに合わせて通常利用でOK
    return np.digitize(x, bins, right=False) - 1  # 0始まりのカテゴリID

@torch.no_grad()
def evaluate_model(model_path, valid_dataset, device, eval_log_path, main_logger):
    """評価拡張版:
    - 2値判定: 複数しきい値
    - カテゴリ分類: binsごとの混同行列(時刻別/積算)
    - equal-split(積算/3)ベースラインとの比較（1h RMSE）
    - 「降水あり時間数(0/1/2/3)」のパターン一致評価
    """
    main_logger.info("Starting final model evaluation (v2)...")
    
    model = SwinTransformerSys(
        img_size=IMG_SIZE, patch_size=PATCH_SIZE, in_chans=IN_CHANS, num_classes=NUM_CLASSES,
        embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=15,
        mlp_ratio=4., qkv_bias=True, norm_layer=nn.LayerNorm, use_checkpoint=False
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    eval_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # 2値判定用カウンタ（時刻別の全ピクセルを合算）
    bin_stats_hourly = {thr: {'tp':0, 'tn':0, 'fp':0, 'fn':0} for thr in BINARY_THRESHOLDS_MM}
    bin_stats_sum    = {thr: {'tp':0, 'tn':0, 'fp':0, 'fn':0} for thr in BINARY_THRESHOLDS_MM}

    # カテゴリ分類（bins）混同行列（時刻別/積算）
    K = len(CATEGORY_BINS_MM) - 1  # カテゴリ数
    confmat_hourly = np.zeros((K, K), dtype=np.int64)  # [true, pred]
    confmat_sum    = np.zeros((K, K), dtype=np.int64)

    # equal-splitベースラインとの比較（1時間ごとRMSE）
    # モデルの1h RMSEと、積算(モデル予測sum)を3等分したベースラインの1h RMSEを比較
    sse_model_1h = 0.0
    sse_base_1h  = 0.0
    count_1h     = 0

    # 積算RMSE（モデル予測sum vs 正解sum）
    sse_sum = 0.0
    count_sum = 0

    # 3時間のうち降水「あり時間数」パターン(0/1/2/3)の一致評価
    # confusion_4x4: rows=true_count(0..3), cols=pred_count(0..3)
    conf_wethours = np.zeros((4, 4), dtype=np.int64)

    progress_bar = tqdm(eval_loader, desc="Evaluating Model (v2)", leave=False)
    for batch in progress_bar:
        inputs = batch['input'].to(device)
        targ_1h = batch['target_1h'].to(device)  # (B, 3, H, W)
        targ_sum = batch['target_sum'].to(device) # (B, 1, H, W)

        with autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(inputs)
        outputs = torch.relu(outputs)  # (B, 3, H, W)

        # CPUに移してnumpyへ
        pred_1h = outputs.detach().cpu().numpy()  # (B,3,H,W)
        true_1h = targ_1h.detach().cpu().numpy()  # (B,3,H,W)
        pred_sum = outputs.sum(dim=1, keepdim=True).detach().cpu().numpy()  # (B,1,H,W)
        true_sum = targ_sum.detach().cpu().numpy()  # (B,1,H,W)

        B, _, H, W = outputs.shape

        # 2値判定（時刻別）：(B,3,H,W)をまとめて評価
        for thr in BINARY_THRESHOLDS_MM:
            tp, tn, fp, fn = _tally_binary(pred_1h, true_1h, thr)
            bin_stats_hourly[thr]['tp'] += int(tp)
            bin_stats_hourly[thr]['tn'] += int(tn)
            bin_stats_hourly[thr]['fp'] += int(fp)
            bin_stats_hourly[thr]['fn'] += int(fn)

        # 2値判定（積算）
        for thr in BINARY_THRESHOLDS_MM:
            tp, tn, fp, fn = _tally_binary(pred_sum, true_sum, thr)
            bin_stats_sum[thr]['tp'] += int(tp)
            bin_stats_sum[thr]['tn'] += int(tn)
            bin_stats_sum[thr]['fp'] += int(fp)
            bin_stats_sum[thr]['fn'] += int(fn)

        # カテゴリ分類（時刻別）
        # flat化で大容量になりがちなので、時間(3)でループして処理
        for t in range(3):
            true_bins = _digitize_bins(true_1h[:, t, :, :].ravel(), CATEGORY_BINS_MM)
            pred_bins = _digitize_bins(pred_1h[:, t, :, :].ravel(), CATEGORY_BINS_MM)
            for tb, pb in zip(true_bins, pred_bins):
                if 0 <= tb < K and 0 <= pb < K:
                    confmat_hourly[tb, pb] += 1

        # カテゴリ分類（積算）
        true_bins_sum = _digitize_bins(true_sum.ravel(), CATEGORY_BINS_MM)
        pred_bins_sum = _digitize_bins(pred_sum.ravel(), CATEGORY_BINS_MM)
        for tb, pb in zip(true_bins_sum, pred_bins_sum):
            if 0 <= tb < K and 0 <= pb < K:
                confmat_sum[tb, pb] += 1

        # equal-splitベースライン vs モデルの1hRMSE
        # ベースライン: pred_sumを3等分して各時間へ
        base_1h = np.repeat(pred_sum / 3.0, repeats=3, axis=1)  # (B,3,H,W)
        diff_model = (pred_1h - true_1h)
        diff_base  = (base_1h - true_1h)
        sse_model_1h += float(np.sum(diff_model**2))
        sse_base_1h  += float(np.sum(diff_base**2))
        count_1h     += int(B * 3 * H * W)

        # 積算RMSE（モデルsum vs 正解sum）
        diff_sum = (pred_sum - true_sum)
        sse_sum += float(np.sum(diff_sum**2))
        count_sum += int(B * H * W)

        # 「降水あり時間数」(>=WETHOUR_THRESHOLD_MM)のパターン一致評価
        true_wet = (true_1h >= WETHOUR_THRESHOLD_MM)
        pred_wet = (pred_1h >= WETHOUR_THRESHOLD_MM)
        # 各画素ごとに wet時間数を数える：shape (B,H,W)
        true_count = np.sum(true_wet, axis=1)
        pred_count = np.sum(pred_wet, axis=1)
        # 0..3にクリップ（安全のため）
        true_count = np.clip(true_count, 0, 3).astype(np.int64).ravel()
        pred_count = np.clip(pred_count, 0, 3).astype(np.int64).ravel()
        for tc, pc in zip(true_count, pred_count):
            conf_wethours[tc, pc] += 1

    # 指標集計
    epsilon = 1e-6

    # 2値判定の指標（時刻別・積算）
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

    # カテゴリ分類の指標（単純に「カテゴリ一致率=対角和/総和」）
    def compute_categorical_metrics(confmat):
        total = confmat.sum()
        diag = np.trace(confmat)
        acc = diag / (total + epsilon)
        # クラス別の適合率/再現率(参考): 行=真、列=予測
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

    # RMSE計算
    rmse_model_1h = np.sqrt(sse_model_1h / (count_1h + epsilon))
    rmse_base_1h  = np.sqrt(sse_base_1h  / (count_1h + epsilon))
    rmse_sum      = np.sqrt(sse_sum      / (count_sum + epsilon))

    # wet-hour-countパターンの一致率（対角和/総和）
    total_wh = conf_wethours.sum()
    diag_wh = np.trace(conf_wethours)
    acc_wethour_pattern = diag_wh / (total_wh + epsilon)

    # 結果をログへ書き出し
    with open(eval_log_path, 'a', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("Final Model Evaluation (v2) with extended metrics\n")
        f.write("="*70 + "\n\n")

        # 2値判定（時刻別）
        f.write("[Binary Metrics - Hourly (per-pixel over all 3 hours)]\n")
        for thr in BINARY_THRESHOLDS_MM:
            m = metrics_hourly[thr]
            f.write(f"Threshold > {thr} mm: Acc={m['accuracy']:.4f}, Prec={m['precision']:.4f}, "
                    f"Rec={m['recall']:.4f}, F1={m['f1']:.4f}, CSI={m['csi']:.4f} "
                    f"(TP={m['tp']}, TN={m['tn']}, FP={m['fp']}, FN={m['fn']})\n")
        f.write("\n")

        # 2値判定（積算）
        f.write("[Binary Metrics - Accumulation (sum over 3 hours)]\n")
        for thr in BINARY_THRESHOLDS_MM:
            m = metrics_sum[thr]
            f.write(f"Threshold > {thr} mm: Acc={m['accuracy']:.4f}, Prec={m['precision']:.4f}, "
                    f"Rec={m['recall']:.4f}, F1={m['f1']:.4f}, CSI={m['csi']:.4f} "
                    f"(TP={m['tp']}, TN={m['tn']}, FP={m['fp']}, FN={m['fn']})\n")
        f.write("\n")

        # カテゴリ分類（時刻別）
        f.write("[Categorical Metrics - Hourly]\n")
        f.write(f"Bins (mm): {CATEGORY_BINS_MM}\n")
        f.write(f"Overall categorical accuracy: {cat_acc_hourly:.4f}\n")
        f.write("Confusion Matrix [true x pred]:\n")
        for i in range(K):
            f.write(" ".join(str(v) for v in confmat_hourly[i, :]) + "\n")
        f.write("Per-class precision:\n")
        f.write(" ".join(f"{x:.4f}" for x in cat_prec_hourly) + "\n")
        f.write("Per-class recall:\n")
        f.write(" ".join(f"{x:.4f}" for x in cat_rec_hourly) + "\n\n")

        # カテゴリ分類（積算）
        f.write("[Categorical Metrics - Accumulation]\n")
        f.write(f"Bins (mm): {CATEGORY_BINS_MM}\n")
        f.write(f"Overall categorical accuracy: {cat_acc_sum:.4f}\n")
        f.write("Confusion Matrix [true x pred]:\n")
        for i in range(K):
            f.write(" ".join(str(v) for v in confmat_sum[i, :]) + "\n")
        f.write("Per-class precision:\n")
        f.write(" ".join(f"{x:.4f}" for x in cat_prec_sum) + "\n")
        f.write("Per-class recall:\n")
        f.write(" ".join(f"{x:.4f}" for x in cat_rec_sum) + "\n\n")

        # equal-splitベースラインとの比較（1h RMSE）
        f.write("[Equal-split Baseline Comparison - 1h RMSE]\n")
        f.write("Baseline definition: split model's predicted accumulation equally into 3 hours.\n")
        f.write(f"Model 1h RMSE: {rmse_model_1h:.6f}\n")
        f.write(f"Equal-split Baseline 1h RMSE: {rmse_base_1h:.6f}\n")
        better = "BETTER than" if rmse_model_1h < rmse_base_1h else "NOT better than"
        f.write(f"Result: Model is {better} the equal-split baseline (lower is better).\n\n")

        # 積算RMSE
        f.write("[Accumulation RMSE]\n")
        f.write(f"Sum RMSE (model sum vs. true sum): {rmse_sum:.6f}\n\n")

        # 「降水あり時間数(0/1/2/3)」のパターン一致率
        f.write("[Wet-hour count pattern (0/1/2/3) - Consistency]\n")
        f.write(f"Overall pattern accuracy: {acc_wethour_pattern:.4f}\n")
        f.write("Confusion Matrix [true_count x pred_count]:\n")
        for i in range(4):
            f.write(" ".join(str(v) for v in conf_wethours[i, :]) + "\n")
        f.write("\n")

    main_logger.info(f"Evaluation (v2) finished. Results saved to '{eval_log_path}'")

# ==============================================================================
# 7. メインワーカ関数
# ==============================================================================
def main_worker(rank, world_size, train_files, valid_files):
    set_seed(RANDOM_SEED)
    setup_ddp(rank, world_size)
    
    main_log = None
    exec_log_path = EXEC_LOG_PATH
    if rank == 0:
        main_log = logging.getLogger('main')
        main_log.info(f"DDP on {world_size} GPUs. Total batch size: {BATCH_SIZE * world_size}")
        main_log.info(f"Input channels: {IN_CHANS}")
        main_log.info(f"RANDOM_SEED set to: {RANDOM_SEED} for reproducibility.")
    
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
        norm_layer=nn.LayerNorm, ape=False, patch_norm=True, use_checkpoint=False
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
    
    for epoch in range(NUM_EPOCHS):
        train_losses = train_one_epoch(rank, model, train_loader, optimizer, scaler, epoch, exec_log_path)
        val_losses = validate_one_epoch(rank, model, valid_loader, epoch, exec_log_path)
        scheduler.step()
        
        loss_history['train_loss'].append(train_losses[0]); loss_history['train_1h_rmse'].append(train_losses[1]); loss_history['train_sum_rmse'].append(train_losses[2])
        loss_history['val_loss'].append(val_losses[0]); loss_history['val_1h_rmse'].append(val_losses[1]); loss_history['val_sum_rmse'].append(val_losses[2])
        
        if rank == 0:
            main_log.info(f"--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
            main_log.info(f"Train Loss: {train_losses[0]:.4f} (1h_rmse: {train_losses[1]:.4f}, sum_rmse: {train_losses[2]:.4f})")
            main_log.info(f"Valid Loss: {val_losses[0]:.4f} (1h_rmse: {val_losses[1]:.4f}, sum_rmse: {val_losses[2]:.4f})")
            
            if val_losses[0] < best_val_loss:
                best_val_loss = val_losses[0]
                best_epoch = epoch + 1
                torch.save(model.module.state_dict(), MODEL_SAVE_PATH)
                main_log.info(f"Saved best model at epoch {best_epoch} with validation loss: {best_val_loss:.4f}")
    
    dist.barrier()

    if rank == 0:
        main_log.info("\nTraining finished.")
        
    visualize_final_results(rank, world_size, valid_dataset, MODEL_SAVE_PATH, RESULT_IMG_DIR, main_log, exec_log_path)
    
    dist.barrier()

    if rank == 0:
        plot_loss_curve(loss_history, PLOT_SAVE_PATH, main_log, best_epoch)
        # v2: 拡張評価を実施
        evaluate_model(MODEL_SAVE_PATH, valid_dataset, torch.device(f"cuda:{rank}"), EVALUATION_LOG_PATH, main_log)
        main_log.info(f"Result images saved in '{RESULT_IMG_DIR}/'.")
        main_log.info("All processes finished successfully.")
    
    cleanup_ddp()

# ==============================================================================
# 8. メイン実行ブロック
# ==============================================================================
if __name__ == '__main__':
    set_seed(RANDOM_SEED)

    os.makedirs(RESULT_DIR, exist_ok=True)
    
    main_logger, _ = setup_loggers()

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