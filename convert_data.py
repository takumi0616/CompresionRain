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
import argparse
from scipy.interpolate import RectBivariateSpline
from pathlib import Path
from tqdm import tqdm
from dask.diagnostics import ProgressBar

# --- 定数定義 (ユーザー指定の最終仕様に厳密に準拠) ---

# MEPS仕様に含まれる地上要素。
MEPS_SURFACE_VARS = {
    # 短縮名: 出力時の変数名
    'prmsl': 'Prmsl', '10u': 'U10m', '10v': 'V10m'
}
MEPS_SURFACE_LVL_VARS = {
    # (短縮名, レベル): 出力時の変数名
    ('t', 2): 'T2m' # 地上2m気温
}

# ユーザー指定の最終仕様に基づく、気圧面ごとの要素定義。
MEPS_PRESSURE_SPEC = {
    # 気圧面レベル: [利用可能な変数の短縮名リスト]
    975: ['u', 'v', 't'],
    950: ['u', 'v', 't'],
    925: ['u', 'v', 't', 'r'],
    850: ['u', 'v', 't', 'r'],
    500: ['gh', 't', 'r'],
    300: ['gh', 'u', 'v']
}
# 上記の定義から、処理対象となる全ての気圧面レベルを自動生成。
PRESSURE_LEVELS = sorted(MEPS_PRESSURE_SPEC.keys())

# --- 座標情報 ---
MSM_P_LATS = 47.6 - np.arange(253) * 0.1
MSM_P_LONS = 120.0 + np.arange(241) * 0.125
MSM_S_LATS = 47.6 - np.arange(505) * 0.05
MSM_S_LONS = 120.0 + np.arange(481) * 0.0625
OUTPUT_LATS_SLICE = slice(46.95, 23.0)
OUTPUT_LONS_SLICE = slice(120.0, 149.9375)

# --- 関数定義 ---

def get_pressure_var_name(short_name, level):
    """気圧面変数名を作成する"""
    return f"{short_name.upper()}{level}"

def interpolate_grid_fast(data_values, src_lons, src_lats, target_lons, target_lats):
    """高速な格子点補間（高層データを地上データ格子に合わせる）"""
    if data_values.ndim != 2:
        raise ValueError(f"Input data_values must be 2D, but got shape {data_values.shape}")
    # 高速化のため、緯度を昇順に並べ替えてから補間を実行
    src_lats_sorted = src_lats[::-1]
    data_values_sorted = data_values[::-1, :]
    interp_func = RectBivariateSpline(src_lats_sorted, src_lons, data_values_sorted, kx=1, ky=1, s=0)
    target_lats_sorted = target_lats[::-1]
    interp_values_sorted = interp_func(target_lats_sorted, target_lons, grid=True)
    interp_values = interp_values_sorted[::-1, :]
    return interp_values.astype(np.float32)

def find_msm_files(base_time, msm_dir):
    """指定された初期時刻に対応するMSM GRIB2ファイル群のパスを検索する"""
    file_paths = {}
    year_month = base_time.strftime('%Y%m')
    date_str_with_hour = base_time.strftime('%Y%m%d%H')
    file_template = "Z__C_RJTD_{datetime}0000_MSM_GPV_Rjp_{product}_{ft_str}_grib2.bin"
    
    # 地上(Lsurf)・高層(L-pall)データ
    for product_type in ['Lsurf', 'L-pall']:
        # 予報時間(FT) 3, 6時間後を対象
        for ft in [3, 6]:
            key = f"{product_type}_ft{ft}"
            ft_str = f"FH{ft:02d}"
            path = Path(msm_dir) / year_month / file_template.format(datetime=date_str_with_hour, product=product_type, ft_str=ft_str)
            file_paths[key] = path if path.exists() else None
            
    # 降水量(Prr)データ
    for ft_range in ["00-03", "03-06"]:
        key = f"Prr_ft{ft_range}"
        ft_str = f"FH{ft_range}"
        path = Path(msm_dir) / year_month / file_template.format(datetime=date_str_with_hour, product="Prr", ft_str=ft_str)
        file_paths[key] = path if path.exists() else None
        
    return file_paths

def process_grib_files(file_paths):
    """GRIBファイルからMEPS仕様に準拠したデータを抽出し、辞書として返す"""
    data_vars = {}
    try:
        # 予報時間(FT) 3, 6時間後でループ
        for ft in [3, 6]:
            # --- 地上データ処理 ---
            if (lsurf_path := file_paths.get(f"Lsurf_ft{ft}")):
                with pygrib.open(str(lsurf_path)) as grbs:
                    for grb in grbs:
                        if grb.shortName in MEPS_SURFACE_VARS:
                            var_name = MEPS_SURFACE_VARS[grb.shortName]
                            data_vars[f"{var_name}_ft{ft}"] = grb.values.astype(np.float32)
                        elif (grb.shortName, grb.level) in MEPS_SURFACE_LVL_VARS:
                            var_name = MEPS_SURFACE_LVL_VARS[(grb.shortName, grb.level)]
                            data_vars[f"{var_name}_ft{ft}"] = grb.values.astype(np.float32)
            
            # --- 高層データ処理 ---
            if (lpall_path := file_paths.get(f"L-pall_ft{ft}")):
                with pygrib.open(str(lpall_path)) as grbs:
                    for grb in grbs:
                        if grb.level in MEPS_PRESSURE_SPEC and grb.shortName in MEPS_PRESSURE_SPEC[grb.level]:
                            var_name = get_pressure_var_name(grb.shortName, grb.level)
                            interp_data = interpolate_grid_fast(grb.values, MSM_P_LONS, MSM_P_LATS, MSM_S_LONS, MSM_S_LATS)
                            data_vars[f"{var_name}_ft{ft}"] = interp_data
        
        # --- 降水量データ処理 ---
        # 0-3時間積算降水量
        if (prr_path := file_paths.get("Prr_ft00-03")):
            with pygrib.open(str(prr_path)) as grbs:
                data_vars['Prec_ft3'] = np.sum([g.values for g in grbs], axis=0).astype(np.float32)
        
        # 3-6時間積算降水量
        if (prr_path := file_paths.get("Prr_ft03-06")):
            with pygrib.open(str(prr_path)) as grbs:
                data_vars['Prec_ft6'] = np.sum([g.values for g in grbs], axis=0).astype(np.float32)

    except Exception as e:
        pid = os.getpid()
        print(f"\n[PID {pid}] Error during GRIB processing for base time associated with {file_paths.get('Lsurf_ft3')}: {e}", flush=True)
        return {}
    return data_vars

def create_temporary_dataset(base_time, msm_dir, temp_dir):
    """単一時刻のGRIBデータを処理し、一時的なNetCDFファイルとして保存する"""
    file_paths = find_msm_files(base_time, msm_dir)
    if any(p is None for p in file_paths.values()):
        return None # 必要なファイルが一つでも欠けていたらスキップ
        
    data_vars_raw = process_grib_files(file_paths)
    if not data_vars_raw:
        return None

    # 出力先の一時ファイルパス
    output_path = Path(temp_dir) / f"{base_time.strftime('%Y%m%d%H%M')}.nc"
    
    try:
        xds_vars = {}
        # データを指定の緯度経度範囲に切り出す
        temp_coords_da = xr.DataArray(dims=['lat', 'lon'], coords={'lat': MSM_S_LATS, 'lon': MSM_S_LONS})
        final_lats = temp_coords_da.sel(lat=OUTPUT_LATS_SLICE).lat.values
        final_lons = temp_coords_da.sel(lon=OUTPUT_LONS_SLICE).lon.values

        for name, data in data_vars_raw.items():
            ds_temp = xr.DataArray(data, dims=['lat', 'lon'], coords={'lat': MSM_S_LATS, 'lon': MSM_S_LONS})
            cropped_data = ds_temp.sel(lat=OUTPUT_LATS_SLICE, lon=OUTPUT_LONS_SLICE).values
            # time次元を追加
            xds_vars[name] = (['time', 'lat', 'lon'], np.expand_dims(cropped_data, axis=0))
        
        ds = xr.Dataset(
            data_vars=xds_vars,
            coords={'time': pd.to_datetime([base_time]), 'lat': final_lats, 'lon': final_lons}
        )
        ds.to_netcdf(output_path, engine='h5netcdf')
        return str(output_path)
    except Exception as e:
        pid = os.getpid()
        print(f"\n[PID {pid}] Error creating temp NetCDF for {base_time}: {e}\n", flush=True)
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
                print(f"\nA worker process failed: {e}", flush=True)
    return temp_files

def calc_stats(args):
    """データセット全体の平均と標準偏差を計算し、ファイルに保存する"""
    print("--- Running in [calc_stats] mode ---")
    temp_dir = None
    try:
        # 一時ディレクトリを作成
        Path(args.temp_dir).mkdir(parents=True, exist_ok=True)
        temp_dir = tempfile.mkdtemp(prefix="stats_calc_", dir=args.temp_dir)
        print(f"Created temporary directory for stats calculation: {temp_dir}")
        
        # ③ 3時間ごとの時刻リストを生成
        base_times = []
        date_range = pd.date_range(start=args.start_date, end=args.end_date, freq='D')
        for date in date_range:
            for hour in range(0, 24, 3): # 0, 3, 6, ..., 21
                base_times.append(date + pd.Timedelta(hours=hour))

        # 並列処理で全時刻の生データ一時ファイルを作成
        desc = f"Creating temp files for stats ({args.start_date} to {args.end_date})"
        temp_files = run_parallel_processing(create_temporary_dataset, base_times, args.msm_dir, temp_dir, args.max_workers, desc)

        if not temp_files:
            print("No temporary files were created. Cannot calculate statistics.")
            return

        print(f"\nCalculating statistics from {len(temp_files)} files...")
        # Daskを使ってメモリ効率良く全一時ファイルを読み込む
        with xr.open_mfdataset(sorted(temp_files), engine='h5netcdf', combine='by_coords', parallel=True) as ds:
            # 各変数の平均と標準偏差を計算
            means = ds.mean('time').compute()
            stds = ds.std('time').compute()
            
            # 統計量データセットを作成
            stats_ds = xr.Dataset()
            for var in ds.data_vars:
                stats_ds[f'{var}_mean'] = means[var]
                stats_ds[f'{var}_std'] = stds[var]
            
            # 統計量ファイルを保存
            stats_ds.to_netcdf(args.stats_file)
            print(f"Statistics saved to: {args.stats_file}")
            print("\n[Statistics Dataset Info]")
            print(stats_ds)

    finally:
        if temp_dir and Path(temp_dir).exists():
            shutil.rmtree(temp_dir)
            print(f"Removed temporary directory: {temp_dir}")

def convert_data(args):
    """統計量を用いてデータを標準化し、時間特徴量を追加して年次ファイルとして保存する"""
    print("--- Running in [convert] mode ---")
    if not Path(args.stats_file).exists():
        print(f"Error: Statistics file not found at {args.stats_file}")
        print("Please run 'calc_stats' mode first.")
        return

    temp_dir = None
    try:
        # 統計量ファイルを読み込み
        stats_ds = xr.open_dataset(args.stats_file)
        print(f"Loaded statistics from: {args.stats_file}")

        # 一時ディレクトリを作成
        Path(args.temp_dir).mkdir(parents=True, exist_ok=True)
        temp_dir = tempfile.mkdtemp(prefix=f"convert_{args.start_date[:4]}_", dir=args.temp_dir)
        print(f"Created temporary directory for conversion: {temp_dir}")

        # ③ 3時間ごとの時刻リストを生成
        base_times = []
        date_range = pd.date_range(start=args.start_date, end=args.end_date, freq='D')
        for date in date_range:
            for hour in range(0, 24, 3): # 0, 3, 6, ..., 21
                base_times.append(date + pd.Timedelta(hours=hour))

        # 並列処理で該当年の一時ファイルを作成
        desc = f"Processing data for {args.start_date[:4]}"
        temp_files = run_parallel_processing(create_temporary_dataset, base_times, args.msm_dir, temp_dir, args.max_workers, desc)
        
        if not temp_files:
            print("No data was processed for this period. Exiting.")
            return

        print(f"\nConcatenating and standardizing {len(temp_files)} files...")
        # Daskを使って全一時ファイルを結合
        with xr.open_mfdataset(sorted(temp_files), engine='h5netcdf', combine='by_coords', parallel=True, chunks={'time': 10}) as ds:
            
            # ②-1. 各変数を標準化
            standardized_ds = xr.Dataset(coords=ds.coords)
            for var in ds.data_vars:
                if f'{var}_mean' in stats_ds and f'{var}_std' in stats_ds:
                    mean = stats_ds[f'{var}_mean']
                    std = stats_ds[f'{var}_std']
                    # 標準偏差が0の場合はゼロ除算を避ける
                    standardized_ds[var] = xr.where(std > 1e-9, (ds[var] - mean) / std, 0)
                else:
                    print(f"Warning: Statistics for '{var}' not found. Skipping standardization.")
                    standardized_ds[var] = ds[var]

            # ②-2. 時間特徴量を追加
            time_coord = standardized_ds.coords['time']
            dayofyear = time_coord.dt.dayofyear
            hour = time_coord.dt.hour
            
            # 年周期 (うるう年を考慮し366日で割る)
            dayofyear_sin_data = np.sin(2 * np.pi * dayofyear / 366.0)
            standardized_ds['dayofyear_sin'] = (('time',), dayofyear_sin_data.data) # .data を追加

            dayofyear_cos_data = np.cos(2 * np.pi * dayofyear / 366.0)
            standardized_ds['dayofyear_cos'] = (('time',), dayofyear_cos_data.data) # .data を追加

            # 1日周期
            hour_sin_data = np.sin(2 * np.pi * hour / 24.0)
            standardized_ds['hour_sin'] = (('time',), hour_sin_data.data) # .data を追加

            hour_cos_data = np.cos(2 * np.pi * hour / 24.0)
            standardized_ds['hour_cos'] = (('time',), hour_cos_data.data) # .data を追加
            
            print("Standardization and feature engineering complete. Saving to final file...")
            
            # 圧縮設定
            encoding = {var: {'zlib': True, 'complevel': 5} for var in standardized_ds.data_vars}

            # NetCDFファイルへ書き出し
            write_job = standardized_ds.to_netcdf(
                args.output_file,
                encoding=encoding,
                mode='w',
                engine='h5netcdf',
                compute=False
            )
            with ProgressBar():
                write_job.compute()
            
            print("\n--- Processing Complete ---")
            print(f"Final dataset saved to: {args.output_file}")
            with xr.open_dataset(args.output_file) as final_ds:
                print("\n[Final Dataset Info]")
                print(final_ds)

    finally:
        if temp_dir and Path(temp_dir).exists():
            shutil.rmtree(temp_dir)
            print(f"Removed temporary directory: {temp_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert MSM GRIB2 to preprocessed NetCDF for ML models.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    # --- モード選択 ---
    parser.add_argument(
        '--mode',
        type=str,
        choices=['calc_stats', 'convert'],
        required=True,
        help=(
            "Execution mode:\n"
            "'calc_stats': Calculate mean/std over a period and save them.\n"
            "'convert': Convert data, applying pre-calculated statistics."
        )
    )
    
    # --- 期間指定 ---
    parser.add_argument('start_date', type=str, help='Start date in YYYY-MM-DD format.')
    parser.add_argument('end_date', type=str, help='End date in YYYY-MM-DD format.')

    # --- パス指定 ---
    parser.add_argument('--msm_dir', type=str, default='./MSM_data/', help='Root directory of MSM GRIB2 files.')
    parser.add_argument('--output_file', type=str, help="Path to the output NetCDF file (required in 'convert' mode).")
    parser.add_argument('--stats_file', type=str, help="Path to the statistics file (required for both modes to save or load).")
    parser.add_argument(
        '--temp_dir', 
        type=str, 
        default=None, 
        help='Directory for temporary files. If not set, system default is used. \n'
             '★ It is strongly recommended to specify a directory on a large-capacity disk. ★'
    )
    
    # --- 並列処理設定 ---
    parser.add_argument('--max_workers', type=int, default=8, help='Number of parallel processes.')
    
    args = parser.parse_args()

    # ① データ保存先の指定: --temp_dir と --output_file 引数で指定されたパスを使用します。
    # これにより、実行スクリプト(run_conversion.sh)側で /home 配下のパスを指定できます。
    if args.temp_dir is None:
        # --temp_dir が指定されない場合、システムの一時ディレクトリを使用する
        # 大容量のデータを扱う場合、容量不足になる可能性があるため、明示的な指定を推奨
        temp_dir_path = tempfile.gettempdir()
        print(f"Warning: --temp_dir is not set. Using system default temp directory: {temp_dir_path}")
        print("It is recommended to use --temp_dir to specify a directory on a large disk.")
        args.temp_dir = temp_dir_path

    # モードに応じた引数チェック
    if args.mode == 'convert' and not args.output_file:
        parser.error("--output_file is required in 'convert' mode.")
    if not args.stats_file:
        parser.error("--stats_file is required.")

    # メインロジックの実行
    total_start_time = time.time()
    if args.mode == 'calc_stats':
        calc_stats(args)
    elif args.mode == 'convert':
        convert_data(args)
        
    total_elapsed = time.time() - total_start_time
    print(f"\nTotal execution time for this run: {total_elapsed:.2f} seconds")


if __name__ == '__main__':
    main()