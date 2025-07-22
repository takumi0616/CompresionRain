#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import concurrent.futures
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pygrib
import xarray as xr
from dask.diagnostics import ProgressBar
from scipy.interpolate import RectBivariateSpline
from tqdm import tqdm

# --- 定数定義 (課題仕様に準拠) ---

# MEPS仕様に含まれる地上要素
MEPS_SURFACE_VARS = {
    'prmsl': 'Prmsl', '10u': 'U10m', '10v': 'V10m'
}
MEPS_SURFACE_LVL_VARS = {
    ('t', 2): 'T2m'
}

# 気圧面ごとの要素定義
MEPS_PRESSURE_SPEC = {
    975: ['u', 'v', 't'],
    950: ['u', 'v', 't'],
    925: ['u', 'v', 't', 'r'],
    850: ['u', 'v', 't', 'r'],
    500: ['gh', 't', 'r'],
    300: ['gh', 'u', 'v']
}
PRESSURE_LEVELS = sorted(MEPS_PRESSURE_SPEC.keys())

# 座標情報
MSM_P_LATS = 47.6 - np.arange(253) * 0.1
MSM_P_LONS = 120.0 + np.arange(241) * 0.125
MSM_S_LATS = 47.6 - np.arange(505) * 0.05
MSM_S_LONS = 120.0 + np.arange(481) * 0.0625
# 最終的に切り出す格子範囲 (480x480)
# ラベルベースのスライスなので、開始・終了値を指定
OUTPUT_LATS_SLICE = slice(46.95, 23.0)
OUTPUT_LONS_SLICE = slice(120.0, 149.9375)


def get_pressure_var_name(short_name, level):
    """気圧面変数名を作成する (例: u, 850 -> U850)"""
    # 'gh'はそのまま大文字に、他は短縮名を大文字にする
    var_prefix = short_name.upper()
    return f"{var_prefix}{level}"

def interpolate_grid_fast(data_values, src_lons, src_lats, target_lons, target_lats):
    """高速な格子点補間（高層データを地上データ格子に合わせる）"""
    if data_values.ndim != 2:
        raise ValueError(f"Input data must be 2D, but got shape {data_values.shape}")
    # 緯度を昇順に並べ替えてから補間を実行
    src_lats_sorted = src_lats[::-1]
    data_values_sorted = data_values[::-1, :]
    interp_func = RectBivariateSpline(src_lats_sorted, src_lons, data_values_sorted, kx=1, ky=1, s=0)
    target_lats_sorted = target_lats[::-1]
    interp_values_sorted = interp_func(target_lats_sorted, target_lons, grid=True)
    return interp_values_sorted[::-1, :].astype(np.float32)

def find_msm_files(base_time, msm_dir):
    """指定初期時刻に対応するMSM GRIB2ファイル群のパスを検索する"""
    file_paths = {}
    year_month = base_time.strftime('%Y%m')
    date_str_with_hour = base_time.strftime('%Y%m%d%H')
    file_template = "Z__C_RJTD_{datetime}0000_MSM_GPV_Rjp_{product}_{ft_str}_grib2.bin"

    # 地上(Lsurf)・高層(L-pall)データ
    for product_type in ['Lsurf', 'L-pall']:
        for ft in [3, 6]:
            key = f"{product_type}_ft{ft}"
            ft_str = f"FH{ft:02d}"
            path = Path(msm_dir) / year_month / file_template.format(datetime=date_str_with_hour, product=product_type, ft_str=ft_str)
            file_paths[key] = path if path.exists() else None

    # 降水量(Prr)データ
    for ft_range in ["00-03", "03-06", "06-09"]:
        key = f"Prr_ft{ft_range}"
        ft_str = f"FH{ft_range}"
        path = Path(msm_dir) / year_month / file_template.format(datetime=date_str_with_hour, product="Prr", ft_str=ft_str)
        file_paths[key] = path if path.exists() else None

    return file_paths

def process_grib_files(file_paths, base_time):
    """
    GRIBファイルから仕様に準拠したデータを抽出し、辞書として返す
    """
    data_vars = {}
    try:
        # --- 予報時間(FT) 3, 6時間後でループ ---
        for ft in [3, 6]:
            # --- 地上データ処理 (Lsurf) ---
            if (lsurf_path := file_paths.get(f"Lsurf_ft{ft}")):
                with pygrib.open(str(lsurf_path)) as grbs:
                    for short_name, var_name in MEPS_SURFACE_VARS.items():
                        if (msg := grbs.select(shortName=short_name)):
                            data_vars[f"{var_name}_ft{ft}"] = msg[0].values.astype(np.float32)
                    for (short_name, level), var_name in MEPS_SURFACE_LVL_VARS.items():
                        if (msg := grbs.select(shortName=short_name, level=level)):
                            data_vars[f"{var_name}_ft{ft}"] = msg[0].values.astype(np.float32)

            # --- 高層データ処理 (L-pall) ---
            if (lpall_path := file_paths.get(f"L-pall_ft{ft}")):
                with pygrib.open(str(lpall_path)) as grbs:
                    for level in PRESSURE_LEVELS:
                        for short_name in MEPS_PRESSURE_SPEC[level]:
                            if (msg := grbs.select(shortName=short_name, level=level)):
                                var_name = get_pressure_var_name(short_name, level)
                                interp_data = interpolate_grid_fast(msg[0].values, MSM_P_LONS, MSM_P_LATS, MSM_S_LONS, MSM_S_LATS)
                                data_vars[f"{var_name}_ft{ft}"] = interp_data

        # --- 降水量データ処理 (Prr) ---
        # 0-3時間積算降水量 (説明変数)
        if (prr_path := file_paths.get("Prr_ft00-03")):
            with pygrib.open(str(prr_path)) as grbs:
                data_vars['Prec_ft3'] = np.sum([g.values for g in grbs], axis=0).astype(np.float32)

        # 3-6時間降水量 (説明変数・目的変数)
        if (prr_path := file_paths.get("Prr_ft03-06")):
            with pygrib.open(str(prr_path)) as grbs:
                msgs = list(grbs)
                # 3-6時間積算降水量 (説明変数)
                data_vars['Prec_4_6h_sum'] = np.sum([g.values for g in msgs], axis=0).astype(np.float32)
                # 各1時間値 (目的変数)
                for msg in msgs:
                    if msg.endStep == 4:
                        data_vars['Prec_Target_ft4'] = msg.values.astype(np.float32)
                    elif msg.endStep == 5:
                        data_vars['Prec_Target_ft5'] = msg.values.astype(np.float32)
                    elif msg.endStep == 6:
                        data_vars['Prec_Target_ft6'] = msg.values.astype(np.float32)

        # 6-9時間積算降水量 (説明変数)
        if (prr_path := file_paths.get("Prr_ft06-09")):
            with pygrib.open(str(prr_path)) as grbs:
                data_vars['Prec_6_9h_sum'] = np.sum([g.values for g in grbs], axis=0).astype(np.float32)

    except Exception as e:
        pid = os.getpid()
        print(f"\n[PID {pid}] ERROR: Failed during GRIB processing for base time {base_time}: {e}", flush=True)
        return None
    return data_vars

def create_temporary_dataset(base_time, msm_dir, temp_dir):
    """単一時刻のGRIBデータを処理し、一時的なNetCDFファイルとして保存する"""
    file_paths = find_msm_files(base_time, msm_dir)
    # モデルに必要な基本ファイル(地上/高層)が欠けていたらスキップ
    if file_paths['Lsurf_ft3'] is None or file_paths['L-pall_ft3'] is None:
        return None

    data_vars_raw = process_grib_files(file_paths, base_time)
    if not data_vars_raw:
        return None
        
    # 目的変数が生成されなかった場合はスキップ
    if any(key not in data_vars_raw for key in ['Prec_Target_ft4', 'Prec_Target_ft5', 'Prec_Target_ft6']):
        # print(f"WARN: Target precipitation data not found for {base_time}. Skipping.")
        return None

    output_path = Path(temp_dir) / f"{base_time.strftime('%Y%m%d%H%M')}.nc"
    try:
        ds_vars = {name: (['lat', 'lon'], data) for name, data in data_vars_raw.items()}
        ds_full = xr.Dataset(
            data_vars=ds_vars,
            coords={'lat': MSM_S_LATS, 'lon': MSM_S_LONS}
        )
        ds_cropped = ds_full.sel(lat=OUTPUT_LATS_SLICE, lon=OUTPUT_LONS_SLICE)
        ds_final = ds_cropped.expand_dims(time=pd.to_datetime([base_time]))
        
        # 欠損変数がないか最終確認
        if ds_final.isnull().any():
            # print(f"WARN: Null values detected in data for {base_time}. Skipping file creation.")
            return None
            
        ds_final.to_netcdf(output_path, engine='h5netcdf')
        return str(output_path)
    except Exception as e:
        pid = os.getpid()
        print(f"\n[PID {pid}] ERROR: Failed to create temp NetCDF for {base_time}: {e}\n", flush=True)
        return None

def run_parallel_processing(process_function, base_times, msm_dir, temp_dir, max_workers, desc):
    """並列処理を実行し、生成された一時ファイルのリストを返す"""
    temp_files = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_time = {executor.submit(process_function, bt, msm_dir, temp_dir): bt for bt in base_times}
        for future in tqdm(concurrent.futures.as_completed(future_to_time), total=len(base_times), desc=desc):
            try:
                result_path = future.result()
                if result_path is not None:
                    temp_files.append(result_path)
            except Exception as e:
                print(f"\nERROR: A worker process failed: {e}", flush=True)
    return temp_files

def get_base_times(start_date, end_date):
    """指定期間内の要件に合う初期時刻リストを生成する"""
    base_times = []
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    for date in date_range:
        for hour in [0, 6, 12, 18]:  # 要件通り 00, 06, 12, 18 UTC のみ
            base_times.append(date + pd.Timedelta(hours=hour))
    return base_times

def process_and_save(args, mode):
    """統計量計算またはデータ変換の主処理"""
    is_calc_stats = mode == 'calc_stats'
    start_date, end_date = args.start_date, args.end_date
    temp_root_dir = Path(args.temp_dir)
    
    # 年ごとの処理か、統計用の全期間処理かを判定
    process_id = f"stats_{start_date[:4]}-{end_date[:4]}" if is_calc_stats else f"convert_{start_date[:4]}"
    temp_dir = temp_root_dir / process_id
    
    try:
        # 一時ディレクトリをクリーンアップしてから作成
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        print(f"INFO: Created temporary directory: {temp_dir}")
        
        base_times = get_base_times(start_date, end_date)
        if not base_times:
            print("WARN: No valid base times found for the given period.")
            return

        desc = f"Creating temp files for {process_id}"
        temp_files = run_parallel_processing(create_temporary_dataset, base_times, args.msm_dir, str(temp_dir), args.max_workers, desc)

        if not temp_files:
            print("ERROR: No temporary files were created. Aborting.")
            return

        if is_calc_stats:
            print(f"\nINFO: Calculating statistics from {len(temp_files)} valid time steps...")
            with xr.open_mfdataset(sorted(temp_files), engine='h5netcdf', combine='by_coords', parallel=True) as ds:
                means = ds.mean('time', skipna=True).compute()
                stds = ds.std('time', skipna=True).compute()
                stats_ds = xr.merge([means.rename({v: f"{v}_mean" for v in means}), 
                                     stds.rename({v: f"{v}_std" for v in stds})])
                stats_ds.to_netcdf(args.stats_file)
                print(f"SUCCESS: Statistics saved to: {args.stats_file}")
                print("\n[Statistics Dataset Info]\n", stats_ds)
        else: # convert mode
            stats_ds = xr.open_dataset(args.stats_file)
            print(f"\nINFO: Concatenating and standardizing {len(temp_files)} files...")
            with xr.open_mfdataset(sorted(temp_files), engine='h5netcdf', combine='by_coords', parallel=True, chunks={'time': 1}) as ds:
                standardized_ds = xr.Dataset(coords=ds.coords)
                for var in ds.data_vars:
                    if f'{var}_mean' in stats_ds and f'{var}_std' in stats_ds:
                        mean = stats_ds[f'{var}_mean']
                        std = stats_ds[f'{var}_std']
                        standardized_ds[var] = xr.where(std > 1e-9, (ds[var] - mean) / std, 0)
                    else:
                        print(f"WARN: Stats for '{var}' not found. Skipping standardization.")
                        standardized_ds[var] = ds[var]

                # 時間特徴量を追加 (Sin/Cos変換)
                time_coord = standardized_ds.coords['time'].dt
                standardized_ds['dayofyear_sin'] = (('time',), np.sin(2 * np.pi * time_coord.dayofyear / 366.0).values)
                standardized_ds['dayofyear_cos'] = (('time',), np.cos(2 * np.pi * time_coord.dayofyear / 366.0).values)
                standardized_ds['hour_sin'] = (('time',), np.sin(2 * np.pi * time_coord.hour / 24.0).values)
                standardized_ds['hour_cos'] = (('time',), np.cos(2 * np.pi * time_coord.hour / 24.0).values)

                print("INFO: Saving final dataset...")
                encoding = {v: {'zlib': True, 'complevel': 5} for v in standardized_ds.data_vars}
                write_job = standardized_ds.to_netcdf(args.output_file, encoding=encoding, mode='w', engine='h5netcdf', compute=False)
                with ProgressBar():
                    write_job.compute()

                print(f"SUCCESS: Final dataset saved to: {args.output_file}")
                with xr.open_dataset(args.output_file) as final_ds:
                    print("\n[Final Dataset Info]\n", final_ds)
    finally:
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir)
            print(f"INFO: Removed temporary directory: {temp_dir}")


def main():
    parser = argparse.ArgumentParser(description='Convert MSM GRIB2 to preprocessed NetCDF for ML models.',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--mode', type=str, choices=['calc_stats', 'convert'], required=True,
                        help="Execution mode:\n'calc_stats': Calculate and save statistics.\n'convert': Convert data using pre-calculated statistics.")
    parser.add_argument('start_date', type=str, help='Start date in YYYY-MM-DD format.')
    parser.add_argument('end_date', type=str, help='End date in YYYY-MM-DD format.')
    parser.add_argument('--msm_dir', type=str, required=True, help='Root directory of MSM GRIB2 files.')
    parser.add_argument('--output_file', type=str, help="Path to the output NetCDF file (required in 'convert' mode).")
    parser.add_argument('--stats_file', type=str, required=True, help="Path to the statistics file (to save or load).")
    parser.add_argument('--temp_dir', type=str, required=True, help='Directory for temporary files on a large-capacity disk.')
    parser.add_argument('--max_workers', type=int, default=8, help='Number of parallel processes.')

    args = parser.parse_args()

    if args.mode == 'convert' and not args.output_file:
        parser.error("--output_file is required in 'convert' mode.")

    total_start_time = time.time()
    if args.mode == 'calc_stats':
        print("--- Running in [calc_stats] mode ---")
        process_and_save(args, 'calc_stats')
    elif args.mode == 'convert':
        print("--- Running in [convert] mode ---")
        if not Path(args.stats_file).exists():
            raise FileNotFoundError(f"Statistics file not found at {args.stats_file}. Please run 'calc_stats' mode first.")
        process_and_save(args, 'convert')
    
    total_elapsed = time.time() - total_start_time
    print(f"\nTotal execution time: {total_elapsed:.2f} seconds")


if __name__ == '__main__':
    main()