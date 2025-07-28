#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import xarray as xr
import pygrib
import datetime
import pandas as pd
import concurrent.futures
import time
import logging
from pathlib import Path
from tqdm import tqdm
from scipy.interpolate import RectBivariateSpline
from dask.diagnostics import ProgressBar

# ==============================================================================
# --- 基本設定 (このセクションを編集して実行) ---
# ==============================================================================
# 処理対象期間
START_YEAR = 2018
END_YEAR = 2018

# 並列処理ワーカー数 (マシンのCPUコア数に合わせて調整)
MAX_WORKERS = 22

# --- パス設定 ---
# このスクリプトが存在するディレクトリを基準にパスを構築
SCRIPT_DIR = Path(__file__).parent
# MSM GRIB2データが格納されているルートディレクトリ (ご自身の環境に合わせて変更してください)
MSM_DIR = Path("/mnt/gpu01/MSM/")
# 完成したNetCDFファイルの出力先
OUTPUT_DIR = SCRIPT_DIR / "output_nc"

# ==============================================================================
# --- ログ設定 ---
# ==============================================================================
def setup_logging():
    """ログ設定を初期化する"""
    log_dir = SCRIPT_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    log_filename = log_dir / f"conversion_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # nohupで実行した際にprintと同様にファイルに書き込まれるように、StreamHandlerも使用する
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    # tqdmはログファイルには出力せず、標準エラー出力にのみ表示する設定
    tqdm.pandas(file=os.sys.stderr)

# ==============================================================================
# --- 定数定義 ---
# ==============================================================================
# 抽出対象の変数リスト
MEPS_SURFACE_VARS = {'prmsl': 'Prmsl', '10u': 'U10m', '10v': 'V10m'}
MEPS_SURFACE_LVL_VARS = {('t', 2): 'T2m'}
MEPS_PRESSURE_SPEC = {
    975: ['u', 'v', 't'],
    950: ['u', 'v', 't'],
    925: ['u', 'v', 't', 'r'],
    850: ['u', 'v', 't', 'r'],
    500: ['gh', 't', 'r'],
    300: ['gh', 'u', 'v']
}
PRESSURE_LEVELS = sorted(MEPS_PRESSURE_SPEC.keys())

# --- 座標情報 ---
# 入力データの格子定義
MSM_P_LATS = 47.6 - np.arange(253) * 0.1
MSM_P_LONS = 120.0 + np.arange(241) * 0.125
MSM_S_LATS = 47.6 - np.arange(505) * 0.05
MSM_S_LONS = 120.0 + np.arange(481) * 0.0625

# 出力データの格子定義 (480x480)
OUTPUT_LATS_SLICE = slice(46.95, 23.0)
OUTPUT_LONS_SLICE = slice(120.0, 149.9375)
OUTPUT_LATS = 46.95 - np.arange(480) * 0.05
OUTPUT_LONS = 120.0 + np.arange(480) * 0.0625

# ==============================================================================
# --- 関数定義 ---
# ==============================================================================
def get_pressure_var_name(short_name, level):
    """気圧面変数名を作成する"""
    return f"{short_name.upper()}{level}"

def interpolate_grid_fast(data_values, src_lons, src_lats, target_lons, target_lats):
    """
    RectBivariateSplineを使用して高速に格子内挿を行う。
    入力緯度は北から南、出力もそれに合わせる。
    """
    # 緯度を昇順にソート（南から北へ）
    src_lats_sorted = src_lats[::-1]
    data_values_sorted = data_values[::-1, :]
    
    # 補間関数を作成
    interp_func = RectBivariateSpline(src_lats_sorted, src_lons, data_values_sorted, kx=1, ky=1, s=0)
    
    # ターゲットの緯度も昇順にして補間を実行
    target_lats_sorted = target_lats[::-1]
    interp_values_sorted = interp_func(target_lats_sorted, target_lons, grid=True)
    
    # 結果を元の順序（北から南）に戻して返す
    return interp_values_sorted[::-1, :].astype(np.float32)

def find_msm_files(base_time, msm_dir):
    """指定された初期時刻に対応するGRIB2ファイル群を検索する"""
    file_paths = {}
    year_month = base_time.strftime('%Y%m')
    date_str_with_hour = base_time.strftime('%Y%m%d%H')
    file_template = "Z__C_RJTD_{datetime}0000_MSM_GPV_Rjp_{product}_{ft_str}_grib2.bin"
    
    # Lsurf (地上) と L-pall (気圧面) ファイル
    for product_type in ['Lsurf', 'L-pall']:
        for ft in [3, 6]:
            key = f"{product_type}_ft{ft}"
            ft_str = f"FH{ft:02d}"
            path = Path(msm_dir) / year_month / file_template.format(datetime=date_str_with_hour, product=product_type, ft_str=ft_str)
            file_paths[key] = path if path.exists() else None
            
    # Prr (降水量) ファイル
    for ft_range in ["00-03", "03-06"]:
        key = f"Prr_ft{ft_range}"
        ft_str = f"FH{ft_range}"
        path = Path(msm_dir) / year_month / file_template.format(datetime=date_str_with_hour, product="Prr", ft_str=ft_str)
        file_paths[key] = path if path.exists() else None
        
    return file_paths

def process_grib_files(file_paths):
    """GRIBファイルからデータを抽出し、変数名をキーとする辞書として返す"""
    data_vars = {}
    try:
        # --- 説明変数 (予報時間 ft=3, ft=6) ---
        for ft in [3, 6]:
            # 地上データ (Lsurf)
            if (lsurf_path := file_paths.get(f"Lsurf_ft{ft}")):
                with pygrib.open(str(lsurf_path)) as grbs:
                    for grb in grbs:
                        if grb.shortName in MEPS_SURFACE_VARS:
                            var_name = MEPS_SURFACE_VARS[grb.shortName]
                            data_vars[f"{var_name}_ft{ft}"] = grb.values.astype(np.float32)
                        elif (grb.shortName, grb.level) in MEPS_SURFACE_LVL_VARS:
                            var_name = MEPS_SURFACE_LVL_VARS[(grb.shortName, grb.level)]
                            data_vars[f"{var_name}_ft{ft}"] = grb.values.astype(np.float32)
            
            # 気圧面データ (L-pall)
            if (lpall_path := file_paths.get(f"L-pall_ft{ft}")):
                with pygrib.open(str(lpall_path)) as grbs:
                    for grb in grbs:
                        if grb.level in MEPS_PRESSURE_SPEC and grb.shortName in MEPS_PRESSURE_SPEC[grb.level]:
                            var_name = get_pressure_var_name(grb.shortName, grb.level)
                            # 地上格子に内挿
                            interp_data = interpolate_grid_fast(grb.values, MSM_P_LONS, MSM_P_LATS, MSM_S_LONS, MSM_S_LATS)
                            data_vars[f"{var_name}_ft{ft}"] = interp_data
        
        # --- 降水量 (説明変数) ---
        # 0-3時間積算降水量
        if (prr_path := file_paths.get("Prr_ft00-03")):
            with pygrib.open(str(prr_path)) as grbs:
                data_vars['Prec_ft3'] = np.sum([g.values for g in grbs], axis=0).astype(np.float32)
        
        # --- 降水量 (説明変数 + 目的変数) ---
        # 3-6時間データ
        if (prr_path := file_paths.get("Prr_ft03-06")):
            with pygrib.open(str(prr_path)) as grbs:
                hourly_prec = {}
                all_values = []
                # デバッグ: 各メッセージの情報を出力
                logging.info(f"Processing precipitation file: {prr_path}")
                for i, grb in enumerate(grbs):
                    logging.info(f"Message {i}: forecastTime={grb.forecastTime}, "
                               f"startStep={grb.startStep if hasattr(grb, 'startStep') else 'N/A'}, "
                               f"endStep={grb.endStep if hasattr(grb, 'endStep') else 'N/A'}, "
                               f"stepRange={grb.stepRange if hasattr(grb, 'stepRange') else 'N/A'}")
                    
                    all_values.append(grb.values)
                    
                    # メッセージのインデックスで判断する方法
                    if i == 0:  # 最初のメッセージ = 4時間目
                        hourly_prec['Prec_Target_ft4'] = grb.values.astype(np.float32)
                    elif i == 1:  # 2番目のメッセージ = 5時間目
                        hourly_prec['Prec_Target_ft5'] = grb.values.astype(np.float32)
                    elif i == 2:  # 3番目のメッセージ = 6時間目
                        hourly_prec['Prec_Target_ft6'] = grb.values.astype(np.float32)
                
                # 説明変数として3-6時間積算降水量を保存
                if all_values:
                    data_vars['Prec_4_6h_sum'] = np.sum(all_values, axis=0).astype(np.float32)
                
                data_vars.update(hourly_prec)

    except Exception as e:
        logging.error(f"GRIB processing failed. file_paths: {file_paths}, error: {e}", exc_info=True)
        return {}
        
    return data_vars

def process_single_time_to_dataset(base_time, msm_dir):
    """単一時刻のGRIB群を処理し、メモリ上にxarray.Datasetを作成して返す"""
    file_paths = find_msm_files(base_time, msm_dir)
    
    # 課題で要求される必須ファイルが一つでも欠けていたらスキップ
    required_keys = ['Lsurf_ft3', 'Lsurf_ft6', 'L-pall_ft3', 'L-pall_ft6', 'Prr_ft00-03', 'Prr_ft03-06']
    if any(file_paths.get(key) is None for key in required_keys):
        logging.warning(f"Skipping {base_time}: Missing one or more required GRIB files.")
        return None
        
    data_vars_raw = process_grib_files(file_paths)
    if not data_vars_raw:
        logging.warning(f"Skipping {base_time}: Data processing returned empty.")
        return None

    try:
        xds_vars = {}
        # 生データから変数を抽出し、領域を切り出してxarray.DataArrayを作成
        for name, data in data_vars_raw.items():
            # 一旦フルサイズの座標でDataArrayを作成
            da_full = xr.DataArray(data, dims=['lat', 'lon'], coords={'lat': MSM_S_LATS, 'lon': MSM_S_LONS})
            # 目的の領域を切り出し
            cropped_data = da_full.sel(lat=OUTPUT_LATS_SLICE, lon=OUTPUT_LONS_SLICE).values
            # time次元を追加して格納
            xds_vars[name] = (['time', 'lat', 'lon'], np.expand_dims(cropped_data, axis=0))
        
        # この時刻のDatasetを作成
        ds = xr.Dataset(
            data_vars=xds_vars,
            coords={'time': pd.to_datetime([base_time]), 'lat': OUTPUT_LATS, 'lon': OUTPUT_LONS}
        )
        return ds
    except Exception as e:
        logging.error(f"Failed to create xarray.Dataset for {base_time}: {e}", exc_info=True)
        return None

def convert_monthly_data(year, month, msm_dir, output_dir, max_workers):
    """指定された年月のデータを変換し、月次NetCDFファイルとして保存する"""
    month_start_time = time.time()
    logging.info(f"--- Starting conversion for {year}-{month:02d} ---")
    
    output_file = output_dir / f"{year}{month:02d}.nc"
    if output_file.exists():
        logging.info(f"File {output_file} already exists. Skipping.")
        return

    # --- 1. 月内の全時刻について並列処理でデータセットを作成 ---
    logging.info("Starting parallel processing to create in-memory datasets...")
    t_start_parallel = time.time()
    
    start_date = f"{year}-{month:02d}-01"
    end_date_dt = pd.to_datetime(start_date) + pd.offsets.MonthEnd(0)
    date_range = pd.date_range(start=start_date, end=end_date_dt, freq='D')
    base_times = [d + pd.Timedelta(hours=h) for d in date_range for h in range(0, 24, 3)]
    
    datasets_in_month = []
    desc = f"Processing data for {year}-{month:02d}"
    
    with tqdm(total=len(base_times), desc=desc, file=os.sys.stderr, dynamic_ncols=True) as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_time = {executor.submit(process_single_time_to_dataset, bt, msm_dir): bt for bt in base_times}
            
            for future in concurrent.futures.as_completed(future_to_time):
                try:
                    result_ds = future.result()
                    if result_ds is not None:
                        datasets_in_month.append(result_ds)
                except Exception as e:
                    logging.error(f"A worker process failed for {future_to_time[future]}: {e}", exc_info=True)
                pbar.update(1)
                
    logging.info(f"Finished parallel processing. Time taken: {time.time() - t_start_parallel:.2f} seconds.")
    
    if not datasets_in_month:
        logging.warning(f"No data was processed for {year}-{month:02d}. Skipping file creation.")
        return

    # --- 2. データセット結合と時間特徴量追加 ---
    logging.info(f"Concatenating {len(datasets_in_month)} datasets for {year}-{month:02d}...")
    t_start_concat = time.time()
    
    # 時刻順にソートしてから結合
    datasets_in_month.sort(key=lambda ds: ds.time.values[0])
    monthly_ds = xr.concat(datasets_in_month, dim='time')
    
    logging.info(f"  - Concatenation completed. Time taken: {time.time() - t_start_concat:.2f} seconds.")

    # 時間特徴量を追加
    t_start_features = time.time()
    time_coord = monthly_ds.coords['time']
    monthly_ds['dayofyear_sin'] = np.sin(2 * np.pi * time_coord.dt.dayofyear / 366.0).astype(np.float32)
    monthly_ds['dayofyear_cos'] = np.cos(2 * np.pi * time_coord.dt.dayofyear / 366.0).astype(np.float32)
    monthly_ds['hour_sin']      = np.sin(2 * np.pi * time_coord.dt.hour / 24.0).astype(np.float32)
    monthly_ds['hour_cos']      = np.cos(2 * np.pi * time_coord.dt.hour / 24.0).astype(np.float32)
    logging.info(f"  - Adding time features completed. Time taken: {time.time() - t_start_features:.2f} seconds.")
    
    # --- 3. ファイル保存 ---
    logging.info(f"Saving final NetCDF file to {output_file}...")
    t_start_save = time.time()
    encoding = {var: {'zlib': True, 'complevel': 5} for var in monthly_ds.data_vars}
    write_job = monthly_ds.to_netcdf(output_file, encoding=encoding, mode='w', engine='h5netcdf', compute=False)
    
    with ProgressBar():
        write_job.compute()
        
    logging.info(f"  - Saving to NetCDF completed. Time taken: {time.time() - t_start_save:.2f} seconds.")
    logging.info(f"Successfully created {output_file}.")
    logging.info(f"Finished conversion for {year}-{month:02d} in {time.time() - month_start_time:.2f} seconds.")

# ==============================================================================
# --- メイン実行部 ---
# ==============================================================================
def main():
    """メイン処理"""
    setup_logging()
    total_start_time = time.time()
    logging.info("===== MSM GRIB2 to NetCDF Conversion Process Start =====")
    
    # --- ディレクトリ作成 ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.info(f"Input GRIB2 directory: {MSM_DIR}")
    logging.info(f"Output NetCDF directory: {OUTPUT_DIR}")

    # --- データ変換 (月次) ---
    logging.info(f"--- Running in [convert] mode for {START_YEAR} to {END_YEAR} ---")
    
    total_months = (END_YEAR - START_YEAR + 1) * 12
    processed_months = 0
    conversion_start_time = time.time()

    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):
            processed_months += 1
            logging.info(f"--- Processing month {processed_months} of {total_months} ({year}-{month:02d}) ---")
            
            convert_monthly_data(year, month, MSM_DIR, OUTPUT_DIR, MAX_WORKERS)
            
            # --- 進捗報告 ---
            elapsed_time = time.time() - conversion_start_time
            if processed_months > 0:
                avg_time_per_month = elapsed_time / processed_months
                remaining_months = total_months - processed_months
                estimated_time_remaining = avg_time_per_month * remaining_months
                
                logging.info(f"Progress: {processed_months}/{total_months} months complete.")
                logging.info(f"Elapsed time: {datetime.timedelta(seconds=int(elapsed_time))}")
                logging.info(f"Estimated time remaining: {datetime.timedelta(seconds=int(estimated_time_remaining))}")
            
    total_elapsed = time.time() - total_start_time
    logging.info(f"===== All processes finished. Total execution time: {datetime.timedelta(seconds=int(total_elapsed))} =====")

if __name__ == '__main__':
    main()