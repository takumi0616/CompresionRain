#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# swinunet_main.py (月次データ対応・改善版・再現性対応版・評価機能追加版)
import os
import glob
import warnings
import hdf5plugin
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
DATA_DIR = './output_nc'
TRAIN_YEARS = [2018, 2019, 2020, 2021]
VALID_YEARS = [2022]

# --- 結果の保存先ディレクトリを一元管理 ---
RESULT_DIR = 'swin-unet_main_result_v1'

# モデル、プロット、ログの保存先パスを定義
MODEL_SAVE_PATH = os.path.join(RESULT_DIR, 'best_swin_unet_model_ddp_monthly.pth')
PLOT_SAVE_PATH = os.path.join(RESULT_DIR, 'loss_curve_monthly.png')
RESULT_IMG_DIR = os.path.join(RESULT_DIR, 'result_images_monthly')
MAIN_LOG_PATH = os.path.join(RESULT_DIR, 'main.log')
EXEC_LOG_PATH = os.path.join(RESULT_DIR, 'execution.log')
# 【新規】評価ログのパスを追加
EVALUATION_LOG_PATH = os.path.join(RESULT_DIR, 'evaluation.log')

NUM_WORKERS=0

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

# ==============================================================================
# 1.5. 再現性確保のための関数
# ==============================================================================
def set_seed(seed):
    """各種ライブラリの乱数シードを固定し、再現性を確保する"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    # PyTorchの決定的アルゴリズムを使用する設定
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_monthly_files(data_dir, years, logger=None):
    """指定された年の全ての月次NetCDFファイルのリストを取得する"""
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
    """ログ設定を行い、メインロガーと実行ログのパスを返す"""
    # メインロガー: INFOレベル以上の情報をmain.logとコンソールに出力
    main_logger = logging.getLogger('main')
    main_logger.setLevel(logging.INFO)
    
    # 既存のハンドラをクリア
    if main_logger.hasHandlers():
        main_logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # ファイルハンドラ (main.log)
    fh_main = logging.FileHandler(MAIN_LOG_PATH, mode='w', encoding='utf-8')
    fh_main.setFormatter(formatter)
    main_logger.addHandler(fh_main)
    
    # コンソールハンドラ
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    main_logger.addHandler(ch)

    # 実行ログ (tqdm用) ファイルを初期化
    with open(EXEC_LOG_PATH, 'w', encoding='utf-8') as f:
        f.write("Execution log started.\n")

    # 【新規】評価ログファイルを初期化
    with open(EVALUATION_LOG_PATH, 'w', encoding='utf-8') as f:
        f.write("Evaluation log started.\n\n")
            
    return main_logger, EXEC_LOG_PATH


# ==============================================================================
# 3. DDPセットアップ / クリーンアップ関数
# ==============================================================================
def setup_ddp(rank, world_size):
    """DDPのためのプロセスグループを初期化"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_ddp():
    """DDPのプロセスグループを破棄"""
    dist.destroy_process_group()

# ==============================================================================
# 4. データセットクラスの定義
# ==============================================================================
class NetCDFDataset(Dataset):
    """
    複数のNetCDFファイルを扱うカスタムデータセット。
    __getitem__で入力テンソルと、2種類の目的変数テンソルを辞書で返す。
    """
    def __init__(self, file_paths, logger=None):
        super().__init__()
        self.logger = logger if logger else logging.getLogger('main')
        
        if not file_paths: raise ValueError("ファイルパスのリストが空です")
        for fp in file_paths:
            if not os.path.exists(fp): raise FileNotFoundError(f"ファイルが見つかりません: {fp}")
        
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
    """複合損失関数"""
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
    """1エポック分の学習処理"""
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
    """1エポック分の検証処理"""
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
    """学習完了後、最適なモデルを使用して検証データ全体で高機能な可視化を行う"""
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
            target_1h = sample['target_1h']

            with autocast(device_type='cuda', dtype=torch.float16):
                output = model(inputs)
            output = torch.relu(output).squeeze(0).cpu()

            fig, axes = plt.subplots(2, 3, figsize=(20, 12), subplot_kw={'projection': ccrs.PlateCarree()})
            time_val = valid_dataset.ds.time.values[idx]
            fig.suptitle(f"Validation Time: {np.datetime_as_string(time_val, unit='m')}", fontsize=16)

            plot_data = [output[0], output[1], output[2], target_1h[0], target_1h[1], target_1h[2]]
            titles = [f'Prediction FT+{i}' for i in [4,5,6]] + [f'Ground Truth FT+{i}' for i in [4,5,6]]
            vmax = max(target_1h.max(), output.max(), 0.1)
            cmap = plt.get_cmap("Blues")
            cmap.set_under(alpha=0)
            vmin_val = 1e-4

            for i, ax in enumerate(axes.flat):
                ax.set_extent(map_extent, crs=ccrs.PlateCarree())
                ax.add_feature(cfeature.COASTLINE, edgecolor='black')
                ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='black')
                gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, color='gray', alpha=0.5)
                gl.top_labels = False
                gl.right_labels = False
                
                im = ax.imshow(plot_data[i], extent=map_extent, origin='upper', cmap=cmap, vmin=vmin_val, vmax=vmax)
                ax.set_title(titles[i])
                fig.colorbar(im, ax=ax, shrink=0.7, label='Precipitation (mm)')

            plt.tight_layout(rect=[0, 0.03, 1, 0.96])
            time_str = np.datetime_as_string(time_val, unit='h').replace(':', '-').replace('T', '_')
            save_path = os.path.join(result_img_dir, f'validation_{time_str}.png')
            plt.savefig(save_path, dpi=150)
            plt.close(fig)

def plot_loss_curve(history, save_path, logger, best_epoch):
    """損失曲線をプロットして保存し、最適なエポックに線を引く"""
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

@torch.no_grad()
def evaluate_model(model_path, valid_dataset, device, eval_log_path, main_logger):
    """最終モデルをロードし、降水の有無で二値分類評価を行う"""
    main_logger.info("Starting final model evaluation...")
    
    model = SwinTransformerSys(
        img_size=IMG_SIZE, patch_size=PATCH_SIZE, in_chans=IN_CHANS, num_classes=NUM_CLASSES,
        embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=15,
        mlp_ratio=4., qkv_bias=True, norm_layer=nn.LayerNorm, use_checkpoint=False
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 評価用にDDPサンプラーなしのデータローダーを作成
    eval_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    total_tp, total_tn, total_fp, total_fn = 0, 0, 0, 0
    
    # 降水有無の閾値
    prec_threshold = 0.0

    progress_bar = tqdm(eval_loader, desc="Evaluating Model", leave=False)
    for batch in progress_bar:
        inputs = batch['input'].to(device)
        targets_1h = batch['target_1h'].to(device)

        with autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(inputs)
        
        outputs = torch.relu(outputs)

        pred_binary = (outputs > prec_threshold).cpu().numpy()
        true_binary = (targets_1h > prec_threshold).cpu().numpy()

        total_tp += np.sum(np.logical_and(pred_binary, true_binary))
        total_tn += np.sum(np.logical_and(np.logical_not(pred_binary), np.logical_not(true_binary)))
        total_fp += np.sum(np.logical_and(pred_binary, np.logical_not(true_binary)))
        total_fn += np.sum(np.logical_and(np.logical_not(pred_binary), true_binary))

    # 指標の計算 (ゼロ除算を回避)
    epsilon = 1e-6
    accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn + epsilon)
    precision = total_tp / (total_tp + total_fp + epsilon)
    recall = total_tp / (total_tp + total_fn + epsilon)
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
    csi = total_tp / (total_tp + total_fp + total_fn + epsilon)

    # 結果をログファイルに書き込む
    with open(eval_log_path, 'a', encoding='utf-8') as f:
        f.write("="*50 + "\n")
        f.write("Final Model Evaluation (Binary Classification for Precipitation)\n")
        f.write(f"Threshold for precipitation: > {prec_threshold} mm\n")
        f.write("="*50 + "\n\n")
        f.write(f"Total Pixels: {total_tp + total_tn + total_fp + total_fn}\n")
        f.write(f"True Positives (TP - 予測も正解も降水あり): {total_tp}\n")
        f.write(f"True Negatives (TN - 予測も正解も降水なし): {total_tn}\n")
        f.write(f"False Positives (FP - 予測は降水あり、正解はなし): {total_fp}\n")
        f.write(f"False Negatives (FN - 予測は降水なし、正解はあり): {total_fn}\n\n")
        f.write("="*50 + "\n")
        f.write("Performance Metrics:\n")
        f.write("="*50 + "\n\n")
        f.write(f"Accuracy (正解率): {accuracy:.4f}\n")
        f.write("  - 説明: 全てのピクセルの中で、予測が正解と一致した割合。\n\n")
        f.write(f"Precision (適合率): {precision:.4f}\n")
        f.write("  - 説明: 「降水あり」と予測した中で、実際に降水があった割合。\n\n")
        f.write(f"Recall (再現率): {recall:.4f}\n")
        f.write("  - 説明: 実際に降水があった中で、正しく「降水あり」と予測できた割合。\n\n")
        f.write(f"F1 Score (F1値): {f1_score:.4f}\n")
        f.write("  - 説明: 適合率と再現率の調和平均。モデル性能のバランスを示す。\n\n")
        f.write(f"CSI (Critical Success Index / Threat Score): {csi:.4f}\n")
        f.write("  - 説明: 気象分野でよく使われる指標。「降水なし」の正解を除外して計算するため、降水予測性能をより的確に評価できる。\n\n")

    main_logger.info(f"Evaluation finished. Results saved to '{eval_log_path}'")


# ==============================================================================
# 7. メインワーカ関数
# ==============================================================================
def main_worker(rank, world_size, train_files, valid_files):
    """DDPの各プロセスで実行されるメインの学習処理"""
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

    if rank == 0: main_log.info("Starting training...")
    
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

    if rank == 0: main_log.info("\nTraining finished.")
        
    visualize_final_results(rank, world_size, valid_dataset, MODEL_SAVE_PATH, RESULT_IMG_DIR, main_log, exec_log_path)
    
    dist.barrier()

    if rank == 0:
        plot_loss_curve(loss_history, PLOT_SAVE_PATH, main_log, best_epoch)
        # 【新規】最終評価を実行
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