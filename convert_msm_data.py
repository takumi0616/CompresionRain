#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import xarray as xr
import pygrib
import datetime
import pandas as pd
import tempfile
import shutil
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
END_YEAR = 2023

# 学習データ期間 (統計量計算用)
TRAIN_START_YEAR = 2018
TRAIN_END_YEAR = 2023

# 並列処理ワーカー数 (マシンのCPUコア数に合わせて調整)
MAX_WORKERS = 22

# --- パス設定 ---
# このスクリプトが存在するディレクトリを基準にパスを構築
SCRIPT_DIR = Path(__file__).parent
MSM_DIR = SCRIPT_DIR / "MSM_data"       # MSM GRIB2データが格納されているルートディレクトリ
OUTPUT_DIR = SCRIPT_DIR / "output_nc"   # 完成したNetCDFファイルの出力先
TEMP_ROOT_DIR = OUTPUT_DIR / "temp"     # 一時ファイルの作成場所
STATS_FILE = OUTPUT_DIR / f"stats_{TRAIN_START_YEAR}-{TRAIN_END_YEAR}.nc" # 統計量ファイル

# ==============================================================================
# --- ログ設定 ---
# ==============================================================================
def setup_logging():
    """ログ設定を初期化する"""
    log_dir = SCRIPT_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    log_filename = log_dir / f"conversion_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    # tqdmとloggingを連携させる
    tqdm.pandas()
    
# ==============================================================================
# --- 定数定義 ---
# ==============================================================================
MEPS_SURFACE_VARS = {'prmsl': 'Prmsl', '10u': 'U10m', '10v': 'V10m'}
MEPS_SURFACE_LVL_VARS = {('t', 2): 'T2m'}
MEPS_PRESSURE_SPEC = {
    975: ['u', 'v', 't'], 950: ['u', 'v', 't'], 925: ['u', 'v', 't', 'r'],
    850: ['u', 'v', 't', 'r'], 500: ['gh', 't', 'r'], 300: ['gh', 'u', 'v']
}
PRESSURE_LEVELS = sorted(MEPS_PRESSURE_SPEC.keys())

# --- 座標情報 ---
MSM_P_LATS = 47.6 - np.arange(253) * 0.1
MSM_P_LONS = 120.0 + np.arange(241) * 0.125
MSM_S_LATS = 47.6 - np.arange(505) * 0.05
MSM_S_LONS = 120.0 + np.arange(481) * 0.0625
OUTPUT_LATS_SLICE = slice(46.95, 23.0)
OUTPUT_LONS_SLICE = slice(120.0, 149.9375)

# ==============================================================================
# --- 関数定義 ---
# ==============================================================================
def get_pressure_var_name(short_name, level):
    return f"{short_name.upper()}{level}"

def interpolate_grid_fast(data_values, src_lons, src_lats, target_lons, target_lats):
    src_lats_sorted = src_lats[::-1]
    data_values_sorted = data_values[::-1, :]
    interp_func = RectBivariateSpline(src_lats_sorted, src_lons, data_values_sorted, kx=1, ky=1, s=0)
    target_lats_sorted = target_lats[::-1]
    interp_values_sorted = interp_func(target_lats_sorted, target_lons, grid=True)
    return interp_values_sorted[::-1, :].astype(np.float32)

def find_msm_files(base_time, msm_dir):
    file_paths = {}
    year_month = base_time.strftime('%Y%m')
    date_str_with_hour = base_time.strftime('%Y%m%d%H')
    file_template = "Z__C_RJTD_{datetime}0000_MSM_GPV_Rjp_{product}_{ft_str}_grib2.bin"
    
    for product_type in ['Lsurf', 'L-pall']:
        for ft in [3, 6]:
            key = f"{product_type}_ft{ft}"
            ft_str = f"FH{ft:02d}"
            path = Path(msm_dir) / year_month / file_template.format(datetime=date_str_with_hour, product=product_type, ft_str=ft_str)
            file_paths[key] = path if path.exists() else None
            
    for ft_range in ["00-03", "03-06", "06-09"]:
        key = f"Prr_ft{ft_range}"
        ft_str = f"FH{ft_range}"
        path = Path(msm_dir) / year_month / file_template.format(datetime=date_str_with_hour, product="Prr", ft_str=ft_str)
        file_paths[key] = path if path.exists() else None
        
    return file_paths

def process_grib_files(file_paths):
    """GRIBファイルからデータを抽出し、辞書として返す（目的変数も含む）"""
    data_vars = {}
    try:
        # --- 説明変数 ---
        for ft in [3, 6]:
            if (lsurf_path := file_paths.get(f"Lsurf_ft{ft}")):
                with pygrib.open(str(lsurf_path)) as grbs:
                    for grb in grbs:
                        if grb.shortName in MEPS_SURFACE_VARS:
                            var_name = MEPS_SURFACE_VARS[grb.shortName]
                            data_vars[f"{var_name}_ft{ft}"] = grb.values.astype(np.float32)
                        elif (grb.shortName, grb.level) in MEPS_SURFACE_LVL_VARS:
                            var_name = MEPS_SURFACE_LVL_VARS[(grb.shortName, grb.level)]
                            data_vars[f"{var_name}_ft{ft}"] = grb.values.astype(np.float32)
            
            if (lpall_path := file_paths.get(f"L-pall_ft{ft}")):
                with pygrib.open(str(lpall_path)) as grbs:
                    for grb in grbs:
                        if grb.level in MEPS_PRESSURE_SPEC and grb.shortName in MEPS_PRESSURE_SPEC[grb.level]:
                            var_name = get_pressure_var_name(grb.shortName, grb.level)
                            interp_data = interpolate_grid_fast(grb.values, MSM_P_LONS, MSM_P_LATS, MSM_S_LONS, MSM_S_LATS)
                            data_vars[f"{var_name}_ft{ft}"] = interp_data
        
        # --- 降水量 (説明変数) ---
        if (prr_path := file_paths.get("Prr_ft00-03")):
            with pygrib.open(str(prr_path)) as grbs:
                data_vars['Prec_ft3'] = np.sum([g.values for g in grbs], axis=0).astype(np.float32)
        if (prr_path := file_paths.get("Prr_ft06-09")):
            with pygrib.open(str(prr_path)) as grbs:
                data_vars['Prec_6_9h_sum'] = np.sum([g.values for g in grbs], axis=0).astype(np.float32)

        # --- 降水量 (説明変数 + 目的変数) ---
        if (prr_path := file_paths.get("Prr_ft03-06")):
            with pygrib.open(str(prr_path)) as grbs:
                hourly_prec = {}
                all_values = []
                for grb in grbs:
                    all_values.append(grb.values)
                    if grb.forecastTime in [4, 5, 6]:
                        hourly_prec[f'Prec_Target_ft{grb.forecastTime}'] = grb.values.astype(np.float32)
                if all_values:
                    data_vars['Prec_4_6h_sum'] = np.sum(all_values, axis=0).astype(np.float32)
                data_vars.update(hourly_prec)

    except Exception as e:
        logging.error(f"GRIB processing failed for files linked to {file_paths.get('Lsurf_ft3')}: {e}", exc_info=True)
        return {}
    return data_vars

def create_temporary_dataset(base_time, msm_dir, temp_dir):
    """単一時刻のデータを処理し、一時的なNetCDFファイルとして保存する"""
    file_paths = find_msm_files(base_time, msm_dir)
    # 必須ファイルが一つでも欠けていたらスキップ
    required_keys = ['Lsurf_ft3', 'Lsurf_ft6', 'L-pall_ft3', 'L-pall_ft6', 'Prr_ft00-03', 'Prr_ft03-06']
    if any(file_paths.get(key) is None for key in required_keys):
        logging.warning(f"Skipping {base_time}: Missing one or more required GRIB files.")
        return None
        
    data_vars_raw = process_grib_files(file_paths)
    if not data_vars_raw:
        return None

    output_path = Path(temp_dir) / f"{base_time.strftime('%Y%m%d%H')}.nc"
    
    try:
        xds_vars = {}
        temp_coords_da = xr.DataArray(dims=['lat', 'lon'], coords={'lat': MSM_S_LATS, 'lon': MSM_S_LONS})
        final_lats = temp_coords_da.sel(lat=OUTPUT_LATS_SLICE).lat.values
        final_lons = temp_coords_da.sel(lon=OUTPUT_LONS_SLICE).lon.values

        for name, data in data_vars_raw.items():
            ds_temp = xr.DataArray(data, dims=['lat', 'lon'], coords={'lat': MSM_S_LATS, 'lon': MSM_S_LONS})
            cropped_data = ds_temp.sel(lat=OUTPUT_LATS_SLICE, lon=OUTPUT_LONS_SLICE).values
            xds_vars[name] = (['time', 'lat', 'lon'], np.expand_dims(cropped_data, axis=0))
        
        ds = xr.Dataset(
            data_vars=xds_vars,
            coords={'time': pd.to_datetime([base_time]), 'lat': final_lats, 'lon': final_lons}
        )
        ds.to_netcdf(output_path, engine='h5netcdf')
        return str(output_path)
    except Exception as e:
        logging.error(f"Failed to create temp NetCDF for {base_time}: {e}", exc_info=True)
        return None

def run_parallel_processing(base_times, msm_dir, temp_dir, max_workers, desc):
    """並列処理を実行し、生成された一時ファイルのリストを返す"""
    temp_files = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_time = {executor.submit(create_temporary_dataset, bt, msm_dir, temp_dir): bt for bt in base_times}
        
        with tqdm(total=len(base_times), desc=desc) as pbar:
            for future in concurrent.futures.as_completed(future_to_time):
                try:
                    result_path = future.result()
                    if result_path is not None:
                        temp_files.append(result_path)
                except Exception as e:
                    logging.error(f"A worker process failed: {e}", exc_info=True)
                pbar.update(1)
    return temp_files

def calc_stats():
    """データセット全体の平均と標準偏差を計算し、ファイルに保存する"""
    logging.info("--- Running in [calc_stats] mode ---")
    start_time = time.time()
    
    # 一時ディレクトリを作成
    temp_dir = Path(tempfile.mkdtemp(prefix="stats_calc_", dir=TEMP_ROOT_DIR))
    logging.info(f"Created temporary directory for stats calculation: {temp_dir}")
    
    try:
        start_date = f"{TRAIN_START_YEAR}-01-01"
        end_date = f"{TRAIN_END_YEAR}-12-31"
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        base_times = [d + pd.Timedelta(hours=h) for d in date_range for h in range(0, 24, 3)]

        desc = f"Creating temp files for stats ({TRAIN_START_YEAR}-{TRAIN_END_YEAR})"
        temp_files = run_parallel_processing(base_times, MSM_DIR, temp_dir, MAX_WORKERS, desc)

        if not temp_files:
            logging.error("No temporary files were created. Cannot calculate statistics.")
            return

        logging.info(f"Calculating statistics from {len(temp_files)} files...")
        with xr.open_mfdataset(sorted(temp_files), engine='h5netcdf', combine='by_coords', parallel=True) as ds:
            means = ds.mean('time').compute()
            stds = ds.std('time').compute()
            
            stats_ds = xr.Dataset()
            for var in ds.data_vars:
                stats_ds[f'{var}_mean'] = means[var]
                stats_ds[f'{var}_std'] = stds[var]
            
            STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
            stats_ds.to_netcdf(STATS_FILE)
            logging.info(f"Statistics saved to: {STATS_FILE}")
            logging.info(f"\n[Statistics Dataset Info]\n{stats_ds}")

    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            logging.info(f"Removed temporary directory: {temp_dir}")
    logging.info(f"Finished [calc_stats] mode in {time.time() - start_time:.2f} seconds.")

def convert_monthly_data(year, month, stats_ds):
    """指定された年月のデータを標準化し、月次ファイルとして保存する"""
    month_start_time = time.time()
    logging.info(f"--- Starting conversion for {year}-{month:02d} ---")
    
    output_file = OUTPUT_DIR / f"{year}{month:02d}.nc"
    if output_file.exists():
        logging.info(f"File {output_file} already exists. Skipping.")
        return
        
    temp_dir = Path(tempfile.mkdtemp(prefix=f"convert_{year}_{month:02d}_", dir=TEMP_ROOT_DIR))
    logging.info(f"Created temporary directory for conversion: {temp_dir}")
    
    try:
        start_date = f"{year}-{month:02d}-01"
        # end_dateの計算
        if month == 12:
            end_date = pd.to_datetime(f"{year}-12-31")
        else:
            end_date = pd.to_datetime(f"{year}-{month+1:02d}-01") - pd.Timedelta(days=1)
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        base_times = [d + pd.Timedelta(hours=h) for d in date_range for h in range(0, 24, 3)]

        desc = f"Processing data for {year}-{month:02d}"
        temp_files = run_parallel_processing(base_times, MSM_DIR, temp_dir, MAX_WORKERS, desc)
        
        if not temp_files:
            logging.warning(f"No data was processed for {year}-{month:02d}. Skipping file creation.")
            return

        logging.info(f"Concatenating and standardizing {len(temp_files)} files for {year}-{month:02d}...")
        with xr.open_mfdataset(sorted(temp_files), engine='h5netcdf', combine='by_coords', parallel=True, chunks={'time': 10}) as ds:
            standardized_ds = xr.Dataset(coords=ds.coords)
            for var in ds.data_vars:
                if f'{var}_mean' in stats_ds and f'{var}_std' in stats_ds:
                    mean = stats_ds[f'{var}_mean']
                    std = stats_ds[f'{var}_std']
                    standardized_ds[var] = xr.where(std > 1e-9, (ds[var] - mean) / std, 0)
                else:
                    logging.warning(f"Stats for '{var}' not found. Data will not be standardized.")
                    standardized_ds[var] = ds[var]

            time_coord = standardized_ds.coords['time']
            standardized_ds['dayofyear_sin'] = (('time',), np.sin(2 * np.pi * time_coord.dt.dayofyear / 366.0).data)
            standardized_ds['dayofyear_cos'] = (('time',), np.cos(2 * np.pi * time_coord.dt.dayofyear / 366.0).data)
            standardized_ds['hour_sin'] = (('time',), np.sin(2 * np.pi * time_coord.dt.hour / 24.0).data)
            standardized_ds['hour_cos'] = (('time',), np.cos(2 * np.pi * time_coord.dt.hour / 24.0).data)
            
            logging.info(f"Saving final NetCDF file to {output_file}...")
            encoding = {var: {'zlib': True, 'complevel': 5} for var in standardized_ds.data_vars}
            write_job = standardized_ds.to_netcdf(output_file, encoding=encoding, mode='w', engine='h5netcdf', compute=False)
            with ProgressBar():
                write_job.compute()
            logging.info(f"Successfully created {output_file}.")

    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            logging.info(f"Removed temporary directory: {temp_dir}")
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
    TEMP_ROOT_DIR.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory: {OUTPUT_DIR}")
    logging.info(f"Temporary directory: {TEMP_ROOT_DIR}")

    # --- 1. 統計量計算 ---
    if not STATS_FILE.exists():
        logging.info("Statistics file not found. Starting statistics calculation...")
        calc_stats()
    else:
        logging.info(f"Statistics file found at {STATS_FILE}. Skipping calculation.")
    
    try:
        stats_ds = xr.open_dataset(STATS_FILE)
        logging.info("Successfully loaded statistics file.")
    except Exception as e:
        logging.error(f"Could not load statistics file: {e}. Please delete it and run again.", exc_info=True)
        return

    # --- 2. データ変換 (月次) ---
    logging.info(f"--- Running in [convert] mode for {START_YEAR} to {END_YEAR} ---")
    
    total_months = (END_YEAR - START_YEAR + 1) * 12
    processed_months = 0
    conversion_start_time = time.time()

    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):
            processed_months += 1
            logging.info(f"--- Processing month {processed_months} of {total_months} ---")
            
            convert_monthly_data(year, month, stats_ds)
            
            # --- 進捗報告 ---
            elapsed_time = time.time() - conversion_start_time
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