import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np

# Ensure local imports work when running from repo root
sys.path.append(str(Path(__file__).parent.resolve()))

"""
swinunet_main_v5_config.py

目的:
- src/CompresionRain/swinunet_main_v5.py 内の「ハイパーパラメータ/設定値」を一元管理し、ここだけを編集すれば調整できるようにする
- 記述スタイルは src/FrontLine/main_v3/main_v3_config.py を参考にしたネスト辞書 CFG とショートカット変数の併用

使い方（例）:
- 既存のスクリプトで定数を定義している箇所を削除し、代わりに本ファイルを import して参照する
  from swinunet_main_v5_config import (
      CFG, SEED, DATA_DIR, TRAIN_YEARS, VALID_YEARS,
      IMG_SIZE, PATCH_SIZE, NUM_CLASSES, IN_CHANS,
      BATCH_SIZE, NUM_WORKERS, NUM_EPOCHS, LEARNING_RATE,
      RESULT_DIR, MODEL_SAVE_PATH, PLOT_SAVE_PATH, RESULT_IMG_DIR,
      MAIN_LOG_PATH, EXEC_LOG_PATH, EVALUATION_LOG_PATH, EPOCH_METRIC_PLOT_DIR,
      VIDEO_OUTPUT_PATH, VIDEO_FPS,
      ENABLE_INTENSITY_WEIGHTED_LOSS,
      INTENSITY_WEIGHT_BINS_1H, INTENSITY_WEIGHT_VALUES_1H,
      INTENSITY_WEIGHT_BINS_SUM, INTENSITY_WEIGHT_VALUES_SUM,
      BINARY_THRESHOLDS_MM_1H, BINARY_THRESHOLDS_MM_SUM,
      CATEGORY_BINS_MM_1H, CATEGORY_BINS_MM_SUM, WETHOUR_THRESHOLD_MM,
      LOGNORM_VMIN_MM, MODEL_ARGS, AMP_DTYPE
  )

- モデル初期化時は:
  model = SwinTransformerSys(**MODEL_ARGS).to(device_or_rank)

- Optimizer/LR Scheduler は CFG["OPTIMIZER"], CFG["SCHEDULER"] を参照
- 追加の高度設定（ウォームアップ、EMAなど）を導入する場合は CFG にキーを足してください
"""

# ======================================
# Centralized configuration
# ======================================
CFG = {
    "SEED": 1,            # 乱数シード
    "THREADS": 8,         # 数値計算スレッド数 (必要なら使用)

    "PATHS": {
        # データ/結果ルート
        "data_dir": "./optimization_nc",
        "result_dir": "swin-unet_main_result_v5",

        # 生成ファイル（result_dir配下）
        "model_save_name": "best_swin_unet_model_ddp_monthly_v5.pth",
        "plot_save_name": "loss_curve_monthly_v5.png",
        "result_img_dir_name": "result_images_monthly_v5",
        "result_img_dir_name_rn": "result_images_monthly_RN_v5",
        "main_log_name": "main_v5.log",
        "exec_log_name": "execution_v5.log",
        "evaluation_log_name": "evaluation_v5.log",
        "epoch_metric_plot_dir_name": "epoch_metrics_plots_v5",
        "video_name": "validation_results_v5.mp4",
    },

    # 入出力/データ仕様
    "DATA": {
        "train_years": [2018, 2019, 2020, 2021],
        "valid_years": [2022],
        "img_size": 480,
        "patch_size": 4,
        "num_classes": 3,

        # 入力/ターゲット変数定義
        "input_vars_common": [
            "Prmsl", "U10m", "V10m", "T2m", "U975", "V975", "T975",
            "U950", "V950", "T950", "U925", "V925", "T925", "R925",
            "U850", "V850", "T850", "R850", "GH500", "T500", "R500",
            "GH300", "U300", "V300"
        ],
        "input_vars_prec": ["Prec_ft3"],
        "time_vars": ["dayofyear_sin", "dayofyear_cos", "hour_sin", "hour_cos"],

        "target_vars_1h": ["Prec_Target_ft4", "Prec_Target_ft5", "Prec_Target_ft6"],
        "target_var_sum": "Prec_4_6h_sum",
    },

    # 学習/Loader
    "DATALOADER": {
        "batch_size": 8,
        "num_workers": 4,
        "pin_memory": True,
    },
    "TRAINING": {
        "epochs": 60,
        "amp_dtype": "bf16",      # "fp16" もしくは "bf16" - bf16は数値安定性が高い
        "use_grad_scaler": True,
        "ddp_find_unused_parameters": False,
    },
    "OPTIMIZER": {
        "name": "AdamW",
        "lr": 5e-5,               # 実行時は lr * world_size でスケーリング - 勾配爆発防止のため半減
        "weight_decay": 0.0,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
    },
    "SCHEDULER": {
        "name": "CosineAnnealingLR",
        "t_max": "NUM_EPOCHS",    # 文字列 "NUM_EPOCHS" の場合、実行時に TRAINING.epochs を代入
        "eta_min": 1e-6,
        "warmup_epochs": 0,       # ウォームアップ未使用なら0
    },

    # モデル（Swin U-Net 系）
    "MODEL": {
        "embed_dim": 128,
        "depths": [2, 2, 12, 2],
        "num_heads": [4, 8, 16, 32],
        "window_size": 5,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "drop_rate": 0.0,
        "attn_drop_rate": 0.0,
        "drop_path_rate": 0.2,
        "norm_layer": "LayerNorm",   # "LayerNorm" 固定（本ファイル内で nn.LayerNorm に解決）
        "ape": False,
        "patch_norm": True,
        "use_checkpoint": False,
        # SwinUnetのデコーダ深さなどが必要なら追記: "depths_decoder": [1,2,2,2]
    },

    # 損失
    "LOSS": {
        "enable_intensity_weighted_loss": True,

        # 1時間降水（mm/h）- 極端な重み（100倍）を避け、最大10倍に抑えて勾配爆発を防止
        "intensity_weight_bins_1h": [1.0, 5.0, 10.0, 20.0, 30.0, 50.0],
        "intensity_weight_values_1h": [
            1.0,    # 0.0 - 1.0 mm
            5.0,    # 1.0 - 5.0 mm
            20.0,   # 5.0 - 10.0 mm
            100.0,  # 10.0 - 20.0 mm
            250.0,  # 20.0 - 30.0 mm
            500.0,  # 30.0 - 50.0 mm
            1000.0  # 50.0 mm 以上
        ],

        # 3時間積算（mm/3h）- 同様に最大10倍に抑制
        "intensity_weight_bins_sum": [2.0, 10.0, 20.0, 40.0, 60.0, 100.0],
        "intensity_weight_values_sum": [
            1.0,    # 0.0 - 2.0 mm
            5.0,    # 2.0 - 10.0 mm
            20.0,   # 10.0 - 20.0 mm
            100.0,  # 20.0 - 40.0 mm
            250.0,  # 40.0 - 60.0 mm
            500.0,  # 60.0 - 100.0 mm
            1000.0  # 100.0 mm 以上
        ],
    },

    # 評価/可視化
    "METRICS": {
        # Binary thresholds（1h と 3h積算）
        "binary_thresholds_mm_1h": [0.0, 0.1, 1.0, 5.0, 10.0, 20.0, 30.0, 50.0],
        "binary_thresholds_mm_sum": [0.0, 0.2, 2.0, 10.0, 20.0, 40.0, 60.0, 100.0],

        # Category bins（右開区間、sumは1hの2倍スケール）
        "category_bins_mm_1h": [0.0, 5.0, 10.0, 20.0, 30.0, 50.0, float("inf")],
        "category_bins_mm_sum": [0.0, 10.0, 20.0, 40.0, 60.0, 100.0, float("inf")],

        # Wet-hour 判定しきい値
        "wethour_threshold_mm": 0.1,

        # 可視化LogNorm最小値
        "lognorm_vmin_mm": 0.1,
    },

    # 動画関連
    "VIDEO": {
        "fps": 2,  # 0.5秒/枚
    },
}

# ======================================
# Derived constants and convenience exports
# ======================================

# シード/デバイス（FrontLineの流儀に合わせてここで設定）
SEED = CFG["SEED"]
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# スレッド環境（必要なら有効化）
try:
    torch.set_num_threads(CFG["THREADS"])
except Exception:
    pass

# パス類（この設定ファイルの場所を基準に絶対パスへ解決）
BASE_DIR = Path(__file__).parent.resolve()
RESULT_DIR = str((BASE_DIR / CFG["PATHS"]["result_dir"]).resolve())
MODEL_SAVE_PATH = str((Path(RESULT_DIR) / CFG["PATHS"]["model_save_name"]).resolve())
PLOT_SAVE_PATH = str((Path(RESULT_DIR) / CFG["PATHS"]["plot_save_name"]).resolve())
RESULT_IMG_DIR = str((Path(RESULT_DIR) / CFG["PATHS"]["result_img_dir_name"]).resolve())
RESULT_IMG_DIR_RN = str((Path(RESULT_DIR) / CFG["PATHS"]["result_img_dir_name_rn"]).resolve())
MAIN_LOG_PATH = str((Path(RESULT_DIR) / CFG["PATHS"]["main_log_name"]).resolve())
EXEC_LOG_PATH = str((Path(RESULT_DIR) / CFG["PATHS"]["exec_log_name"]).resolve())
EVALUATION_LOG_PATH = str((Path(RESULT_DIR) / CFG["PATHS"]["evaluation_log_name"]).resolve())
EPOCH_METRIC_PLOT_DIR = str((Path(RESULT_DIR) / CFG["PATHS"]["epoch_metric_plot_dir_name"]).resolve())
VIDEO_OUTPUT_PATH = str((Path(RESULT_DIR) / CFG["PATHS"]["video_name"]).resolve())

# データ仕様
# データディレクトリもこの設定ファイルの場所基準で解決
DATA_DIR = str((BASE_DIR / CFG["PATHS"]["data_dir"]).resolve())
TRAIN_YEARS = list(CFG["DATA"]["train_years"])
VALID_YEARS = list(CFG["DATA"]["valid_years"])
IMG_SIZE = int(CFG["DATA"]["img_size"])
PATCH_SIZE = int(CFG["DATA"]["patch_size"])
NUM_CLASSES = int(CFG["DATA"]["num_classes"])

INPUT_VARS_COMMON = list(CFG["DATA"]["input_vars_common"])
INPUT_VARS_PREC = list(CFG["DATA"]["input_vars_prec"])
TIME_VARS = list(CFG["DATA"]["time_vars"])
TARGET_VARS_1H = list(CFG["DATA"]["target_vars_1h"])
TARGET_VAR_SUM = str(CFG["DATA"]["target_var_sum"])

# 入力チャネル数: 共通×2（ft+3/ft+6）+ 降水 + 時間特徴
IN_CHANS = len(INPUT_VARS_COMMON) * 2 + len(INPUT_VARS_PREC) + len(TIME_VARS)

# Loader/Training
BATCH_SIZE = int(CFG["DATALOADER"]["batch_size"])
NUM_WORKERS = int(CFG["DATALOADER"]["num_workers"])
NUM_EPOCHS = int(CFG["TRAINING"]["epochs"])
LEARNING_RATE = float(CFG["OPTIMIZER"]["lr"])
AMP_DTYPE = CFG["TRAINING"]["amp_dtype"]  # "fp16" or "bf16"

# メトリクス/可視化
BINARY_THRESHOLDS_MM_1H = list(CFG["METRICS"]["binary_thresholds_mm_1h"])
BINARY_THRESHOLDS_MM_SUM = list(CFG["METRICS"]["binary_thresholds_mm_sum"])
CATEGORY_BINS_MM_1H = list(CFG["METRICS"]["category_bins_mm_1h"])
CATEGORY_BINS_MM_SUM = list(CFG["METRICS"]["category_bins_mm_sum"])
WETHOUR_THRESHOLD_MM = float(CFG["METRICS"]["wethour_threshold_mm"])
LOGNORM_VMIN_MM = float(CFG["METRICS"]["lognorm_vmin_mm"])
VIDEO_FPS = int(CFG["VIDEO"]["fps"])

# 損失（強度重み）
ENABLE_INTENSITY_WEIGHTED_LOSS = bool(CFG["LOSS"]["enable_intensity_weighted_loss"])
INTENSITY_WEIGHT_BINS_1H = list(CFG["LOSS"]["intensity_weight_bins_1h"])
INTENSITY_WEIGHT_VALUES_1H = list(CFG["LOSS"]["intensity_weight_values_1h"])
INTENSITY_WEIGHT_BINS_SUM = list(CFG["LOSS"]["intensity_weight_bins_sum"])
INTENSITY_WEIGHT_VALUES_SUM = list(CFG["LOSS"]["intensity_weight_values_sum"])

# モデル引数（SwinTransformerSys にそのまま渡せる dict）
_norm_layer = nn.LayerNorm if CFG["MODEL"]["norm_layer"] == "LayerNorm" else nn.LayerNorm
MODEL_ARGS = dict(
    img_size=IMG_SIZE,
    patch_size=PATCH_SIZE,
    in_chans=IN_CHANS,
    num_classes=NUM_CLASSES,
    embed_dim=int(CFG["MODEL"]["embed_dim"]),
    depths=list(CFG["MODEL"]["depths"]),
    num_heads=list(CFG["MODEL"]["num_heads"]),
    window_size=int(CFG["MODEL"]["window_size"]),
    mlp_ratio=float(CFG["MODEL"]["mlp_ratio"]),
    qkv_bias=bool(CFG["MODEL"]["qkv_bias"]),
    drop_rate=float(CFG["MODEL"]["drop_rate"]),
    attn_drop_rate=float(CFG["MODEL"]["attn_drop_rate"]),
    drop_path_rate=float(CFG["MODEL"]["drop_path_rate"]),
    norm_layer=_norm_layer,
    ape=bool(CFG["MODEL"]["ape"]),
    patch_norm=bool(CFG["MODEL"]["patch_norm"]),
    use_checkpoint=bool(CFG["MODEL"]["use_checkpoint"]),
)

def print_cfg_summary():
    """
    概要:
        本設定モジュール（CFG と派生定数）に基づく主要パラメータのサマリを標準出力へ表示する。
        実行時の設定確認・デバッグの起点として利用する想定。
    引数:
        なし（グローバルに定義された CFG および派生済みのショートカット定数を参照）
    処理:
        - シード値、データディレクトリ、学習/検証年、画像サイズ・パッチサイズ、入出力チャネル数など
          学習に影響する代表的な設定値をフォーマットして出力する。
        - モデルの構成（embed_dim、depths、num_heads、window_size）や損失の有効/無効も確認できる。
    戻り値:
        なし（副作用として標準出力へ設定サマリを印字）
    """
    print("=== SwinUNet v5 Config Summary ===")
    print(f"SEED: {SEED}")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"TRAIN_YEARS: {TRAIN_YEARS}  VALID_YEARS: {VALID_YEARS}")
    print(f"IMG_SIZE: {IMG_SIZE}  PATCH_SIZE: {PATCH_SIZE}  IN_CHANS: {IN_CHANS}  NUM_CLASSES: {NUM_CLASSES}")
    print(f"BATCH_SIZE: {BATCH_SIZE}  NUM_WORKERS: {NUM_WORKERS}  NUM_EPOCHS: {NUM_EPOCHS}  LR: {LEARNING_RATE}")
    print(f"MODEL: embed_dim={MODEL_ARGS['embed_dim']}, depths={MODEL_ARGS['depths']}, heads={MODEL_ARGS['num_heads']}, window={MODEL_ARGS['window_size']}")
    print(f"LOSS: enable_intensity_weighted_loss={ENABLE_INTENSITY_WEIGHTED_LOSS}")
    print(f"PATHS: RESULT_DIR={RESULT_DIR}")
    print("==================================")

__all__ = [
    # Master dict
    "CFG",

    # Seed/threads
    "SEED",

    # Paths (derived)
    "RESULT_DIR", "MODEL_SAVE_PATH", "PLOT_SAVE_PATH", "RESULT_IMG_DIR", "RESULT_IMG_DIR_RN",
    "MAIN_LOG_PATH", "EXEC_LOG_PATH", "EVALUATION_LOG_PATH", "EPOCH_METRIC_PLOT_DIR",
    "VIDEO_OUTPUT_PATH",

    # Data spec
    "DATA_DIR", "TRAIN_YEARS", "VALID_YEARS",
    "IMG_SIZE", "PATCH_SIZE", "NUM_CLASSES",
    "INPUT_VARS_COMMON", "INPUT_VARS_PREC", "TIME_VARS",
    "TARGET_VARS_1H", "TARGET_VAR_SUM", "IN_CHANS",

    # Loader/Training
    "BATCH_SIZE", "NUM_WORKERS", "NUM_EPOCHS", "LEARNING_RATE", "AMP_DTYPE",

    # Metrics/Visualization
    "BINARY_THRESHOLDS_MM_1H", "BINARY_THRESHOLDS_MM_SUM",
    "CATEGORY_BINS_MM_1H", "CATEGORY_BINS_MM_SUM",
    "WETHOUR_THRESHOLD_MM", "LOGNORM_VMIN_MM", "VIDEO_FPS",

    # Loss
    "ENABLE_INTENSITY_WEIGHTED_LOSS",
    "INTENSITY_WEIGHT_BINS_1H", "INTENSITY_WEIGHT_VALUES_1H",
    "INTENSITY_WEIGHT_BINS_SUM", "INTENSITY_WEIGHT_VALUES_SUM",

    # Model args
    "MODEL_ARGS",

    # Utils
    "print_cfg_summary",
]
