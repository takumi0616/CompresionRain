#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import xarray as xr
import datetime
import time
import logging
import hdf5plugin # LZ4圧縮に必要
import dask       # 💡 [修正点 1/2] daskモジュールをインポート
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
from dask.diagnostics import ProgressBar

# ==============================================================================
# --- 基本設定 (このセクションを編集して実行) ---
# ==============================================================================
# 処理対象期間
START_YEAR = 2018
END_YEAR = 2023

# 並列処理ワーカー数 (マシンのCPUコア数に合わせて調整)
# メモリ使用量とディスクI/Oを考慮し、CPUコア数より少し少なめに設定するのも良い
MAX_WORKERS = 16

# --- パス設定 ---
# このスクリプトが存在するディレクトリを基準にパスを構築
SCRIPT_DIR = Path(__file__).parent
# 変換元のNetCDFファイルが格納されているディレクトリ
INPUT_DIR = SCRIPT_DIR / "output_nc"
# 最適化されたNetCDFファイルの出力先
OUTPUT_DIR = SCRIPT_DIR / "optimization_nc"
# 正規化のための統計情報（平均・標準偏差）を保存するファイル
SCALER_FILE = SCRIPT_DIR / "scaler.nc"

# ==============================================================================
# --- ログ設定 ---
# ==============================================================================
def setup_logging():
    """ログ設定を初期化する"""
    log_dir = SCRIPT_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    log_filename = log_dir / f"optimization_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    # tqdmは標準エラー出力にのみ表示
    tqdm.pandas(file=os.sys.stderr)

# ==============================================================================
# --- 関数定義 ---
# ==============================================================================

def get_target_variables(ds):
    """
    データセットから正規化対象となる物理変数を取得する。
    座標変数や時間特徴量（sin/cos）は除外する。
    """
    coord_vars = list(ds.coords)
    time_feature_vars = ['dayofyear_sin', 'dayofyear_cos', 'hour_sin', 'hour_cos']
    return [v for v in ds.data_vars if v not in coord_vars and v not in time_feature_vars]

def calculate_and_save_stats(file_paths, scaler_path, target_variables):
    """
    全データセットに渡る各変数の平均と標準偏差を計算し、ファイルに保存する。
    Daskを使用して、全データをメモリにロードせずに計算を実行する。
    """
    logging.info("--- Starting calculation of statistics (mean and std) for normalization ---")
    logging.info(f"Target files: {len(file_paths)}")

    logging.info("Opening all files as a single virtual dataset with Dask...")
    with xr.open_mfdataset(file_paths, parallel=True, chunks={'time': 248}) as ds:

        ds_vars = ds[target_variables]

        logging.info("Defining mean and std calculation...")
        mean_ds = ds_vars.mean(dim='time', skipna=True)
        std_ds = ds_vars.std(dim='time', skipna=True)

        mean_da = mean_ds.to_dataarray(name='mean')
        std_da = std_ds.to_dataarray(name='std')

        std_da = std_da.where(std_da > 1e-6, 1.0)

        logging.info("Executing Dask computation to get statistics... (This may take a while)")
        with ProgressBar():
            # 💡 [修正点 2/2] xr.compute を dask.compute に変更
            mean_computed, std_computed = dask.compute(mean_da, std_da)

        scaler_ds = xr.Dataset({
            'mean': mean_computed,
            'std': std_computed
        })

        scaler_ds.to_netcdf(scaler_path)
        logging.info(f"✅ Statistics saved successfully to: {scaler_path}")
        return scaler_ds

def optimize_monthly_netcdf(file_path, output_path, scaler_ds, target_variables):
    """
    単一の月次NetCDFファイルを読み込み、正規化、チャンク設定、LZ4圧縮を適用して保存する。
    """
    try:
        if output_path.exists():
            logging.info(f"Skipping: {output_path} already exists.")
            return f"Skipped: {file_path.name}"

        logging.debug(f"Processing: {file_path.name}")

        with xr.open_dataset(file_path, chunks={'time': 1}) as ds:

            for var_name in target_variables:
                if var_name in ds:
                    mean = scaler_ds['mean'].sel(variable=var_name, drop=True)
                    std = scaler_ds['std'].sel(variable=var_name, drop=True)
                    ds[var_name] = (ds[var_name] - mean) / std

            encoding = {}
            for var in ds.data_vars:
                if ds[var].ndim >= 3:
                    encoding[var] = {
                        **hdf5plugin.LZ4(),
                        'chunksizes': (1,) + ds[var].shape[1:]
                    }

            write_job = ds.to_netcdf(
                output_path,
                engine='h5netcdf',
                encoding=encoding,
                mode='w',
                compute=False
            )

            with ProgressBar():
                write_job.compute()

            logging.debug(f"Successfully created: {output_path.name}")
            return f"Success: {file_path.name}"

    except Exception as e:
        logging.error(f"❌ Failed to process {file_path.name}: {e}", exc_info=True)
        return f"Failed: {file_path.name}"

# ==============================================================================
# --- メイン実行部 ---
# ==============================================================================
def main():
    """
    メイン処理。統計情報の計算後、全ファイルを並列で最適化する。
    """
    setup_logging()
    total_start_time = time.time()
    logging.info("===== NetCDF Optimization Process Start =====")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.info(f"Input directory: {INPUT_DIR}")
    logging.info(f"Output directory: {OUTPUT_DIR}")
    logging.info(f"Scaler file: {SCALER_FILE}")

    file_paths = sorted([p for p in INPUT_DIR.glob("*.nc")])
    if not file_paths:
        logging.error(f"No NetCDF files found in {INPUT_DIR}. Exiting.")
        return

    if SCALER_FILE.exists():
        logging.info(f"Found existing scaler file. Loading from: {SCALER_FILE}")
        scaler_ds = xr.open_dataset(SCALER_FILE)
        target_variables = list(scaler_ds.coords['variable'].values)
    else:
        logging.info("Scaler file not found. Calculating from scratch...")
        with xr.open_dataset(file_paths[0]) as temp_ds:
            target_variables = get_target_variables(temp_ds)
        scaler_ds = calculate_and_save_stats(file_paths, SCALER_FILE, target_variables)

    logging.info(f"Target variables for normalization ({len(target_variables)}): {target_variables}")

    logging.info(f"\n--- Starting parallel optimization of {len(file_paths)} monthly files using {MAX_WORKERS} workers ---")

    process_start_time = time.time()

    with tqdm(total=len(file_paths), desc="Optimizing files", file=os.sys.stderr) as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(
                    optimize_monthly_netcdf,
                    fp,
                    OUTPUT_DIR / fp.name,
                    scaler_ds,
                    target_variables
                ) for fp in file_paths
            ]

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                logging.debug(result)
                pbar.update(1)

    process_elapsed = time.time() - process_start_time
    logging.info(f"--- Parallel optimization finished. Time taken: {datetime.timedelta(seconds=int(process_elapsed))} ---")

    total_elapsed = time.time() - total_start_time
    logging.info(f"\n===== All processes finished. Total execution time: {datetime.timedelta(seconds=int(total_elapsed))} =====")

if __name__ == '__main__':
    # hdf5plugin, dask をpipでインストールしてください
    # pip install hdf5plugin dask
    main()