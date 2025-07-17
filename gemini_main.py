# gemini_main.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import warnings

# DDP, AMP, compileのためのインポート
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
# torch.ampを直接使うように変更
from torch.amp import GradScaler, autocast

# swin_unet.py からモデルをインポート
from swin_unet import SwinTransformerSys

# 警告を非表示
warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
# 1. 設定/ハイパーパラメータ
# ==============================================================================
# データファイルパス
TRAIN_FILES = ['./output_nc/2018.nc', './output_nc/2019.nc']
VALID_FILE = ['./output_nc/2020.nc']

# 学習パラメータ
# バッチサイズはGPUごとのサイズ。合計バッチサイズは BATCH_SIZE * WORLD_SIZE となります。
BATCH_SIZE = 8
NUM_EPOCHS = 5
LEARNING_RATE = 1e-4

# モデルパラメータ
IMG_SIZE = 480
PATCH_SIZE = 4

# NetCDF内の変数リスト
BASE_VARS = [
    'Prmsl', 'U10m', 'V10m', 'T2m', 'U975', 'V975', 'T975',
    'U950', 'V950', 'T950', 'U925', 'V925', 'T925', 'R925',
    'U850', 'V850', 'T850', 'R850', 'GH500', 'T500', 'R500',
    'GH300', 'U300', 'V300', 'Prec'
]
TIME_VARS = ['dayofyear_sin', 'dayofyear_cos', 'hour_sin', 'hour_cos']

# 入力チャネル数
IN_CHANS = len(BASE_VARS) * 2 - 1 + len(TIME_VARS)
# 出力チャネル数
NUM_CLASSES = 3
MODEL_SAVE_PATH = 'best_swin_unet_model_ddp.pth'
PLOT_SAVE_PATH = 'loss_accuracy_plot.png'


# ==============================================================================
# 2. DDPセットアップ/クリーンアップ関数
# ==============================================================================
def setup_ddp(rank, world_size):
    """DDPのためのプロセスグループを初期化"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'  # 任意の空きポート
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_ddp():
    """DDPのプロセスグループを破棄"""
    dist.destroy_process_group()


# ==============================================================================
# 3. データセットクラスの定義 (変更なし)
# ==============================================================================
class NetCDFDataset(Dataset):
    """
    複数のNetCDFファイルを扱うためのカスタムデータセット。
    xarrayの遅延読み込み機能を利用して、メモリに乗り切らない大規模データを効率的に扱います。
    """
    def __init__(self, file_paths, base_vars, time_vars):
        super().__init__()
        # 複数のNetCDFファイルを一つのデータセットとして開く
        self.ds = xr.open_mfdataset(file_paths, chunks={})
        self.base_vars = base_vars
        self.time_vars = time_vars
        self.time_len = len(self.ds['time'])
        self.img_dims = (self.ds.sizes['lat'], self.ds.sizes['lon'])

    def __len__(self):
        return self.time_len

    def __getitem__(self, idx):
        # 特定のタイムステップのデータをスライスしてメモリに読み込む
        sample = self.ds.isel(time=idx).load()

        # --- 入力テンソルの作成 ---
        input_channels = []

        for var in self.base_vars:
            if var == 'Prec':
                ft3_var_name = f'{var}_ft3'
                if ft3_var_name in sample:
                    ft3_data = np.nan_to_num(sample[ft3_var_name].values)
                    input_channels.append(ft3_data)
            else:
                ft3_var_name = f'{var}_ft3'
                ft3_data = np.nan_to_num(sample[ft3_var_name].values)
                input_channels.append(ft3_data)
                
                ft6_var_name = f'{var}_ft6'
                ft6_data = np.nan_to_num(sample[ft6_var_name].values)
                input_channels.append(ft6_data)

        h, w = self.img_dims
        for tvar in self.time_vars:
            val = sample[tvar].item()
            channel = np.full((h, w), val, dtype=np.float32)
            input_channels.append(channel)

        input_array = np.stack(input_channels, axis=0)

        # --- 教師データテンソルの作成 ---
        target_array = np.nan_to_num(sample['Prec_ft6'].values)
        
        input_tensor = torch.from_numpy(input_array).float()
        target_tensor = torch.from_numpy(target_array).float().unsqueeze(0)

        return input_tensor, target_tensor


# ==============================================================================
# 4. 損失関数、学習・検証、可視化の各関数の定義
# ==============================================================================
def custom_loss_function(output, target):
    """指定された損失関数 (RMSE)"""
    output = torch.relu(output)
    predicted_sum = torch.sum(output, dim=1, keepdim=True)
    mse = nn.functional.mse_loss(predicted_sum, target)
    rmse = torch.sqrt(mse)
    return rmse

def train_one_epoch(rank, model, dataloader, optimizer, criterion, scaler):
    """1エポック分の学習処理 (DDP, AMP対応)"""
    model.train()
    total_loss = 0.0
    # DistributedSamplerを使用する場合、エポックごとに手動で設定する必要がある
    dataloader.sampler.set_epoch(epoch)
    
    # メインプロセス(rank 0)でのみプログレスバーを表示
    progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch+1}", disable=(rank != 0))
    
    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(rank), targets.to(rank)
        
        optimizer.zero_grad()
        
        # AMPを有効化 (推奨される記法に変更)
        with autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        # スケーラーを使って勾配を計算
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        if rank == 0:
            progress_bar.set_postfix(loss=loss.item())
            
    # 全プロセスで平均損失を計算
    avg_loss = torch.tensor(total_loss / len(dataloader), device=rank)
    dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
    
    return avg_loss.item()

def validate_one_epoch(rank, model, dataloader, criterion):
    """1エポック分の検証処理 (DDP, AMP対応)"""
    model.eval()
    total_loss = 0.0
    
    progress_bar = tqdm(dataloader, desc=f"Validating Epoch {epoch+1}", disable=(rank != 0))
    
    with torch.no_grad():
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(rank), targets.to(rank)
            
            # AMPを有効化 (推奨される記法に変更)
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
            total_loss += loss.item()
            if rank == 0:
                progress_bar.set_postfix(loss=loss.item())

    # 全プロセスで平均損失を計算
    avg_loss = torch.tensor(total_loss / len(dataloader), device=rank)
    dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
    
    return avg_loss.item()

def visualize_results(rank, model, dataloader, num_samples=2):
    """学習結果を可視化する (DDP対応)"""
    model.eval()
    
    # dataloaderからいくつかのサンプルを取得
    # DistributedSamplerはデータを分割するので、1つのバッチで十分
    inputs, targets = next(iter(dataloader))
    inputs, targets = inputs.to(rank), targets.to(rank)

    with torch.no_grad():
        # AMPを有効化 (推奨される記法に変更)
        with autocast(device_type='cuda', dtype=torch.float16):
            outputs = model.module(inputs)
        outputs = torch.relu(outputs)

    # CPUにデータを移し、Numpy配列に変換
    inputs_np = inputs.cpu().numpy()
    targets_np = targets.cpu().numpy()
    outputs_np = outputs.cpu().numpy()

    # 可視化
    for i in range(min(num_samples, len(inputs))):
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f'Sample {i+1} on GPU {rank}', fontsize=16)
        
        prmsl_ft3 = inputs_np[i, 0, :, :]
        prmsl_ft6 = inputs_np[i, 1, :, :]
        
        im1 = axes[0, 0].imshow(prmsl_ft3, cmap='viridis')
        axes[0, 0].set_title('Input: Prmsl_ft3')
        fig.colorbar(im1, ax=axes[0, 0])
        
        im2 = axes[0, 1].imshow(prmsl_ft6, cmap='viridis')
        axes[0, 1].set_title('Input: Prmsl_ft6')
        fig.colorbar(im2, ax=axes[0, 1])

        prec_max = max(targets_np[i].max(), outputs_np[i].sum(axis=0).max())
        
        target_prec_sum = targets_np[i, 0, :, :]
        im3 = axes[0, 2].imshow(target_prec_sum, cmap='Blues', vmin=0, vmax=prec_max)
        axes[0, 2].set_title('Target: Prec_ft6 (Sum)')
        fig.colorbar(im3, ax=axes[0, 2])
        
        pred_prec_sum = outputs_np[i].sum(axis=0)
        im4 = axes[0, 3].imshow(pred_prec_sum, cmap='Blues', vmin=0, vmax=prec_max)
        axes[0, 3].set_title('Prediction (Sum)')
        fig.colorbar(im4, ax=axes[0, 3])

        pred_ft4 = outputs_np[i, 0, :, :]
        pred_ft5 = outputs_np[i, 1, :, :]
        pred_ft6 = outputs_np[i, 2, :, :]
        
        im5 = axes[1, 0].imshow(pred_ft4, cmap='Blues', vmin=0)
        axes[1, 0].set_title('Prediction: Prec FT=4')
        fig.colorbar(im5, ax=axes[1, 0])
        
        im6 = axes[1, 1].imshow(pred_ft5, cmap='Blues', vmin=0)
        axes[1, 1].set_title('Prediction: Prec FT=5')
        fig.colorbar(im6, ax=axes[1, 1])

        im7 = axes[1, 2].imshow(pred_ft6, cmap='Blues', vmin=0)
        axes[1, 2].set_title('Prediction: Prec FT=6')
        fig.colorbar(im7, ax=axes[1, 2])
        
        error_map = pred_prec_sum - target_prec_sum
        err_max = np.abs(error_map).max()
        im8 = axes[1, 3].imshow(error_map, cmap='coolwarm', vmin=-err_max, vmax=err_max)
        axes[1, 3].set_title('Error (Pred_sum - Target_sum)')
        fig.colorbar(im8, ax=axes[1, 3])
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'result_sample_{i+1}.png')
        plt.close(fig)

def plot_loss_curve(train_losses, val_losses, save_path):
    """損失曲線をプロットして保存する"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (RMSE)')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

# ==============================================================================
# 5. メインワーカ関数 (各プロセスで実行される)
# ==============================================================================
def main_worker(rank, world_size):
    """DDPの各プロセスで実行されるメインの学習処理"""
    setup_ddp(rank, world_size)
    
    if rank == 0:
        print(f"Running DDP on {world_size} GPUs.")

    # --- データセットとデータローダーの準備 ---
    train_dataset = NetCDFDataset(TRAIN_FILES, BASE_VARS, TIME_VARS)
    valid_dataset = NetCDFDataset(VALID_FILE, BASE_VARS, TIME_VARS)
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True, sampler=train_sampler)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True, sampler=valid_sampler)

    # --- モデルの準備 ---
    model = SwinTransformerSys(
        img_size=IMG_SIZE, patch_size=PATCH_SIZE, in_chans=IN_CHANS, num_classes=NUM_CLASSES,
        embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=5,
        mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
        drop_path_rate=0.1, norm_layer=nn.LayerNorm, ape=False, patch_norm=True, use_checkpoint=False
    ).to(rank)

    # --- torch.compile と DDP の適用 ---
    # まずモデルをコンパイルし、その後にDDPでラップするのが推奨される
    model = torch.compile(model, mode="reduce-overhead")
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    
    if rank == 0:
        print(f"Model initialized with {IN_CHANS} input channels and {NUM_CLASSES} output classes.")
        print("torch.compile and DDP are enabled.")

    # --- 損失関数、オプティマイザ、AMPスケーラーの準備 ---
    criterion = custom_loss_function
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE * world_size)
    # GradScalerの初期化 (推奨される記法に変更)
    scaler = GradScaler(device='cuda')
    
    # --- 学習ループ ---
    if rank == 0:
        print("Starting training...")
    best_val_loss = float('inf')
    train_loss_history = []
    val_loss_history = []
    
    global epoch
    for epoch in range(NUM_EPOCHS):
        
        train_loss = train_one_epoch(rank, model, train_loader, optimizer, criterion, scaler)
        val_loss = validate_one_epoch(rank, model, valid_loader, criterion)
        
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        if rank == 0:
            print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
            print(f"Epoch {epoch+1} Training Loss: {train_loss:.4f}")
            print(f"Epoch {epoch+1} Validation Loss: {val_loss:.4f}")
            
            # 最も検証ロスが良かったモデルを保存 (rank 0 のみ)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # DDPモデルのstate_dictを保存する際は .module を付ける
                torch.save(model.module.state_dict(), MODEL_SAVE_PATH)
                print(f"Saved best model with validation loss: {best_val_loss:.4f}")
    
    if rank == 0:
        print("\nTraining finished.")
        # --- 結果の可視化 (rank 0 のみ) ---
        print("\nPlotting loss curve...")
        plot_loss_curve(train_loss_history, val_loss_history, PLOT_SAVE_PATH)
        print(f"Loss curve saved to '{PLOT_SAVE_PATH}'.")

        print("\nVisualizing results with the best model...")
        # 保存した最良モデルを読み込む
        best_model_state = torch.load(MODEL_SAVE_PATH, map_location=torch.device(rank))
        model.module.load_state_dict(best_model_state)
        visualize_results(rank, model, valid_loader, num_samples=2)
        print("Visualization complete. Check for 'result_sample_*.png' files.")

    cleanup_ddp()

# ==============================================================================
# 6. メイン実行ブロック
# ==============================================================================
if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    if world_size > 1:
        # マルチGPU環境の場合、spawnを使って各GPU用のプロセスを起動
        mp.spawn(main_worker,
                 args=(world_size,),
                 nprocs=world_size,
                 join=True)
    elif world_size == 1:
        # シングルGPU環境の場合（DDPは使わないが、コード共通化のためワーカを直接実行）
        print("Running on a single GPU.")
        main_worker(0, 1)
    else:
        print("No GPU found. This script requires at least one GPU.")