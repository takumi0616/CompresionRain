#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# gemini_main.py

import os
import warnings

import matplotlib.pyplot as plt
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

# 警告を非表示 (xarrayやpygribが出すものを抑制)
warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
# 1. 設定 & ハイパーパラメータ
# ==============================================================================
# --- データパス設定 ---
# 学習に使用するNetCDFファイル (複数年指定可能)
TRAIN_FILES = ['./output_nc/2018.nc', './output_nc/2019.nc', './output_nc/2020.nc', './output_nc/2021.nc']
# 検証に使用するNetCDFファイル
VALID_FILE = ['./output_nc/2022.nc']
# モデルやプロットの保存先
MODEL_SAVE_PATH = 'best_swin_unet_model_ddp.pth'
PLOT_SAVE_PATH = 'loss_curve.png'
RESULT_IMG_DIR = 'result_images' # 結果画像の保存ディレクトリ

# --- 学習パラメータ ---
# 各GPUあたりのバッチサイズ (合計バッチサイズは BATCH_SIZE * WORLD_SIZE)
BATCH_SIZE = 4
NUM_EPOCHS = 20
# 学習率 (DDPではGPU数に応じてスケールさせるのが一般的)
LEARNING_RATE = 1e-4

# --- モデルパラメータ ---
IMG_SIZE = 480
PATCH_SIZE = 4
# モデルの出力チャネル数 (FT=4, 5, 6 の3つ)
NUM_CLASSES = 3

# --- 変数定義 ---
# モデルの入力に使用する変数リスト (FT=3, FT=6)
# `Prec` は特別扱いするため、このリストからは除外
INPUT_VARS_COMMON = [
    'Prmsl', 'U10m', 'V10m', 'T2m', 'U975', 'V975', 'T975',
    'U950', 'V950', 'T950', 'U925', 'V925', 'T925', 'R925',
    'U850', 'V850', 'T850', 'R850', 'GH500', 'T500', 'R500',
    'GH300', 'U300', 'V300'
]
# 入力に使用する降水量変数
INPUT_VARS_PREC = ['Prec_ft3', 'Prec_4_6h_sum', 'Prec_6_9h_sum']
# 時間特徴量
TIME_VARS = ['dayofyear_sin', 'dayofyear_cos', 'hour_sin', 'hour_cos']

# 目的変数 (教師データ)
TARGET_VARS_1H = ['Prec_Target_ft4', 'Prec_Target_ft5', 'Prec_Target_ft6']
TARGET_VAR_SUM = 'Prec_4_6h_sum'

# 入力チャネル数を自動計算
IN_CHANS = len(INPUT_VARS_COMMON) * 2 + len(INPUT_VARS_PREC) + len(TIME_VARS)

# ==============================================================================
# 2. DDPセットアップ / クリーンアップ関数
# ==============================================================================
def setup_ddp(rank, world_size):
    """DDPのためのプロセスグループを初期化"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'  # 任意の空きポート
    # backendはNVIDIA GPUなら'nccl'が最速
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_ddp():
    """DDPのプロセスグループを破棄"""
    dist.destroy_process_group()

# ==============================================================================
# 3. データセットクラスの定義 ★★★ 課題解決のための修正箇所 ★★★
# ==============================================================================
class NetCDFDataset(Dataset):
    """
    複数のNetCDFファイルを扱うカスタムデータセット。
    __getitem__で入力テンソルと、2種類の目的変数テンソルを辞書で返す。
    """
    def __init__(self, file_paths):
        super().__init__()
        self.ds = xr.open_mfdataset(file_paths, chunks={}, parallel=True)
        self.time_len = len(self.ds['time'])
        self.img_dims = (self.ds.sizes['lat'], self.ds.sizes['lon'])

    def __len__(self):
        return self.time_len

    def __getitem__(self, idx):
        # 特定のタイムステップの全変数をメモリに読み込む
        sample = self.ds.isel(time=idx).load()

        # --- 入力テンソルの作成 ---
        input_channels = []
        # FT=3とFT=6の共通変数を追加
        for var in INPUT_VARS_COMMON:
            # FT=3
            input_channels.append(np.nan_to_num(sample[f'{var}_ft3'].values))
            # FT=6
            input_channels.append(np.nan_to_num(sample[f'{var}_ft6'].values))

        # 降水量変数を追加
        for var in INPUT_VARS_PREC:
            input_channels.append(np.nan_to_num(sample[var].values))

        # 時間特徴量を追加 (空間方向にブロードキャスト)
        h, w = self.img_dims
        for tvar in TIME_VARS:
            val = sample[tvar].item()
            channel = np.full((h, w), val, dtype=np.float32)
            input_channels.append(channel)

        input_tensor = torch.from_numpy(np.stack(input_channels, axis=0)).float()

        # --- 教師データテンソルの作成 ---
        # 1. 1時間ごとの降水量 (3チャネル)
        target_1h_channels = [np.nan_to_num(sample[var].values) for var in TARGET_VARS_1H]
        target_1h_tensor = torch.from_numpy(np.stack(target_1h_channels, axis=0)).float()

        # 2. 3時間積算降水量 (1チャネル)
        target_sum_tensor = torch.from_numpy(np.nan_to_num(sample[TARGET_VAR_SUM].values)).float().unsqueeze(0)

        return {
            "input": input_tensor,
            "target_1h": target_1h_tensor,
            "target_sum": target_sum_tensor
        }

# ==============================================================================
# 4. 損失関数、学習・可視化関数 ★★★ 課題解決のための修正箇所 ★★★
# ==============================================================================
def rmse_loss(pred, target):
    """単純なRMSEを計算するヘルパー関数"""
    return torch.sqrt(nn.functional.mse_loss(pred, target))

def custom_loss_function(output, targets):
    """
    課題仕様に準拠した複合損失関数。
    - L1: 1時間ごとの降水量に対するRMSE
    - L2: 3時間積算降水量に対するRMSE
    - Total Loss = L1 + L2
    """
    # モデル出力は負の値を取りうるため、物理的にありえない負の降水量を0に補正
    output = torch.relu(output)

    # --- L1: 1時間ごとのRMSE ---
    target_1h = targets['target_1h']
    loss_1h_rmse = rmse_loss(output, target_1h)

    # --- L2: 3時間積算のRMSE ---
    predicted_sum = torch.sum(output, dim=1, keepdim=True)
    target_sum = targets['target_sum']
    loss_sum_rmse = rmse_loss(predicted_sum, target_sum)

    # --- Total Loss ---
    total_loss = loss_1h_rmse + loss_sum_rmse
    
    # 損失の内訳も返すことで、学習状況のモニタリングを容易にする
    return total_loss, loss_1h_rmse, loss_sum_rmse

def train_one_epoch(rank, model, dataloader, optimizer, scaler):
    """1エポック分の学習処理"""
    model.train()
    total_loss, total_1h_rmse, total_sum_rmse = 0.0, 0.0, 0.0
    dataloader.sampler.set_epoch(epoch)
    
    progress_bar = tqdm(dataloader, desc=f"Train Epoch {epoch+1}", disable=(rank != 0), leave=False)
    
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
            
    # 全プロセスで平均損失を集計
    avg_loss = torch.tensor([total_loss, total_1h_rmse, total_sum_rmse], device=rank) / len(dataloader)
    dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
    
    return avg_loss.cpu().numpy()

@torch.no_grad()
def validate_one_epoch(rank, model, dataloader):
    """1エポック分の検証処理"""
    model.eval()
    total_loss, total_1h_rmse, total_sum_rmse = 0.0, 0.0, 0.0
    progress_bar = tqdm(dataloader, desc=f"Valid Epoch {epoch+1}", disable=(rank != 0), leave=False)
    
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

@torch.no_grad()
def visualize_results(rank, model, dataloader, epoch, num_samples=2):
    """学習結果を可視化して画像ファイルとして保存"""
    model.eval()
    # 最初のバッチを取得
    batch = next(iter(dataloader))
    inputs = batch['input'][:num_samples].to(rank)
    targets_1h = batch['target_1h'][:num_samples]
    targets_sum = batch['target_sum'][:num_samples]

    with autocast(device_type='cuda', dtype=torch.float16):
        outputs = model.module(inputs)
    outputs = torch.relu(outputs).cpu()
    
    # 3時間積算値を計算
    outputs_sum = outputs.sum(dim=1, keepdim=True)
    
    os.makedirs(RESULT_IMG_DIR, exist_ok=True)

    for i in range(num_samples):
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle(f'Epoch {epoch+1} - Sample {i+1} on GPU {rank}', fontsize=18)
        
        # 共通のカラーマップと最大値
        prec_max = max(targets_sum[i].max(), outputs_sum[i].max(), 1.0)
        
        # --- Row 0: 入力と積算値の比較 ---
        ax = axes[0, 0]
        im = ax.imshow(inputs[i, 0].cpu().numpy(), cmap='viridis') # Prmsl_ft3
        ax.set_title('Input: Prmsl_ft3')
        fig.colorbar(im, ax=ax, shrink=0.8)

        ax = axes[0, 1]
        im = ax.imshow(inputs[i, 1].cpu().numpy(), cmap='viridis') # Prmsl_ft6
        ax.set_title('Input: Prmsl_ft6')
        fig.colorbar(im, ax=ax, shrink=0.8)

        ax = axes[0, 2]
        im = ax.imshow(targets_sum[i, 0], cmap='Blues', vmin=0, vmax=prec_max)
        ax.set_title(f'Target Sum: {targets_sum[i].sum():.2f} mm')
        fig.colorbar(im, ax=ax, shrink=0.8)

        ax = axes[0, 3]
        im = ax.imshow(outputs_sum[i, 0], cmap='Blues', vmin=0, vmax=prec_max)
        ax.set_title(f'Prediction Sum: {outputs_sum[i].sum():.2f} mm')
        fig.colorbar(im, ax=ax, shrink=0.8)

        # --- Row 1 & 2: 1時間ごとの比較 ---
        for ft_idx in range(3):
            ft = ft_idx + 4
            target_img = targets_1h[i, ft_idx]
            pred_img = outputs[i, ft_idx]
            
            # Target
            ax = axes[1, ft_idx]
            im = ax.imshow(target_img, cmap='Blues', vmin=0, vmax=prec_max/2)
            ax.set_title(f'Target FT={ft} ({target_img.sum():.2f} mm)')
            fig.colorbar(im, ax=ax, shrink=0.8)
            
            # Prediction
            ax = axes[2, ft_idx]
            im = ax.imshow(pred_img, cmap='Blues', vmin=0, vmax=prec_max/2)
            ax.set_title(f'Prediction FT={ft} ({pred_img.sum():.2f} mm)')
            fig.colorbar(im, ax=ax, shrink=0.8)

        # エラーマップ
        ax = axes[2, 3]
        error_map = outputs_sum[i, 0] - targets_sum[i, 0]
        err_max = torch.abs(error_map).max()
        im = ax.imshow(error_map, cmap='coolwarm', vmin=-err_max, vmax=err_max)
        ax.set_title('Error (Pred_sum - Target_sum)')
        fig.colorbar(im, ax=ax, shrink=0.8)

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.savefig(f'{RESULT_IMG_DIR}/epoch_{epoch+1:03d}_sample_{i+1}.png')
        plt.close(fig)

def plot_loss_curve(history, save_path):
    """損失曲線をプロットして保存"""
    plt.figure(figsize=(12, 8))
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.subplot(2, 1, 1)
    plt.plot(epochs, history['train_loss'], 'bo-', label='Training Total Loss')
    plt.plot(epochs, history['val_loss'], 'ro-', label='Validation Total Loss')
    plt.title('Total Loss Curve')
    plt.ylabel('Loss (RMSE_1h + RMSE_sum)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(epochs, history['train_1h_rmse'], 'b.-', label='Train 1h-RMSE')
    plt.plot(epochs, history['val_1h_rmse'], 'r.-', label='Val 1h-RMSE')
    plt.title('1-hour RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(epochs, history['train_sum_rmse'], 'b.--', label='Train Sum-RMSE')
    plt.plot(epochs, history['val_sum_rmse'], 'r.--', label='Val Sum-RMSE')
    plt.title('Sum RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ==============================================================================
# 5. メインワーカ関数
# ==============================================================================
def main_worker(rank, world_size):
    """DDPの各プロセスで実行されるメインの学習処理"""
    setup_ddp(rank, world_size)
    
    if rank == 0:
        print(f"INFO: Running DDP on {world_size} GPUs. Total batch size: {BATCH_SIZE * world_size}")
        print(f"INFO: Input channels: {IN_CHANS}")

    # --- データセットとデータローダー ---
    train_dataset = NetCDFDataset(TRAIN_FILES)
    valid_dataset = NetCDFDataset(VALID_FILE)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True, sampler=train_sampler)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True, sampler=valid_sampler)

    # --- モデル ---
    model = SwinTransformerSys(
        img_size=IMG_SIZE, patch_size=PATCH_SIZE, in_chans=IN_CHANS, num_classes=NUM_CLASSES,
        embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7, # Window size is often 7
        mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
        norm_layer=nn.LayerNorm, ape=False, patch_norm=True, use_checkpoint=False
    ).to(rank)
    
    # torch.compile -> DDP ラップ
    model = torch.compile(model, mode="reduce-overhead")
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    
    # --- 損失関数、オプティマイザ、スケーラー ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE * world_size)
    scaler = GradScaler()
    
    # --- 学習ループ ---
    if rank == 0:
        print("INFO: Starting training...")
    best_val_loss = float('inf')
    loss_history = {
        'train_loss': [], 'train_1h_rmse': [], 'train_sum_rmse': [],
        'val_loss': [], 'val_1h_rmse': [], 'val_sum_rmse': []
    }
    
    global epoch
    for epoch in range(NUM_EPOCHS):
        
        train_losses = train_one_epoch(rank, model, train_loader, optimizer, scaler)
        val_losses = validate_one_epoch(rank, model, valid_loader)
        
        # 損失を履歴に保存
        loss_history['train_loss'].append(train_losses[0])
        loss_history['train_1h_rmse'].append(train_losses[1])
        loss_history['train_sum_rmse'].append(train_losses[2])
        loss_history['val_loss'].append(val_losses[0])
        loss_history['val_1h_rmse'].append(val_losses[1])
        loss_history['val_sum_rmse'].append(val_losses[2])

        if rank == 0:
            print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
            print(f"Train Loss: {train_losses[0]:.4f} (1h: {train_losses[1]:.4f}, sum: {train_losses[2]:.4f})")
            print(f"Valid Loss: {val_losses[0]:.4f} (1h: {val_losses[1]:.4f}, sum: {val_losses[2]:.4f})")
            
            # 最良モデルの保存
            if val_losses[0] < best_val_loss:
                best_val_loss = val_losses[0]
                torch.save(model.module.state_dict(), MODEL_SAVE_PATH)
                print(f"INFO: Saved best model with validation loss: {best_val_loss:.4f}")
            
            # 定期的に可視化を実行 (例: 5エポックごとと最終エポック)
            if (epoch + 1) % 5 == 0 or (epoch + 1) == NUM_EPOCHS:
                print("INFO: Visualizing results...")
                visualize_results(rank, model, valid_loader, epoch, num_samples=2)
    
    if rank == 0:
        print("\nINFO: Training finished.")
        print("INFO: Plotting loss curve...")
        plot_loss_curve(loss_history, PLOT_SAVE_PATH)
        print(f"INFO: Loss curve saved to '{PLOT_SAVE_PATH}'.")
        print(f"INFO: Result images saved in '{RESULT_IMG_DIR}/'.")

    cleanup_ddp()

# ==============================================================================
# 6. メイン実行ブロック
# ==============================================================================
if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    if world_size > 1:
        print(f"Found {world_size} GPUs. Launching DDP.")
        mp.spawn(main_worker,
                 args=(world_size,),
                 nprocs=world_size,
                 join=True)
    elif world_size == 1:
        print("Found 1 GPU. Running in single-GPU mode.")
        # DDPのセットアップを簡略化してシングルプロセスで実行
        main_worker(0, 1)
    else:
        print("ERROR: No GPU found. This script requires at least one GPU.")