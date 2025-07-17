#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import xarray as xr
import pygrib
import datetime
import pandas as pd
import re
from pathlib import Path
import concurrent.futures
import io
import time as time_module
from scipy.interpolate import RectBivariateSpline
import cProfile
import pstats

# MSM上空データのグリッド情報（半分の解像度）
# 緯度：北緯22.4度〜47.6度（0.1度間隔、253格子点）
# 経度：東経120度〜150度（0.125度間隔、241格子点）
MSM_P_LATS = 47.6 - np.arange(253) * 0.1  # 北緯22.4度から0.1度間隔
MSM_P_LONS = 120.0 + np.arange(241) * 0.125  # 東経120度から0.125度間隔

# 出力グリッド情報（地上データと同じ高解像度グリッド）
OUTPUT_LATS = 47.6 - np.arange(505, dtype=np.float64) * 0.05
OUTPUT_LONS = 120.0 + np.arange(481, dtype=np.float64) * 0.0625

# グリッド補間のためのメッシュグリッド（一度だけ計算）
TARGET_LON_MESH, TARGET_LAT_MESH = np.meshgrid(OUTPUT_LONS, OUTPUT_LATS)

# 圧力レベルごとの変数名マッピング
PRESSURE_LEVELS = [925, 850, 700, 500, 300]

# 変数名のマッピング（圧力レベルごとに異なる変数を作成）
VAR_BY_LEVEL = {}
for level in PRESSURE_LEVELS:
    # gribファイルの短い名前(shortName)を使用する
    VAR_BY_LEVEL[level] = {
        'u': f'U{level}',         # 東西風
        'v': f'V{level}',         # 南北風
        't': f'T{level}',         # 気温
        'r': f'H{level}',         # 相対湿度
        'gh': f'Z{level}'        # ジオポテンシャル高度
    }
    
    # 700hPaレベルには上昇流（鉛直速度）を追加
    if level == 700:
        VAR_BY_LEVEL[level]['w'] = f'W{level}'  # 上昇流（鉛直速度）

def interpolate_grid_fast(data_values, src_lons, src_lats, target_lons, target_lats):
    # 元データは降順なので、昇順に反転
    src_lats_sorted = src_lats[::-1]
    data_values_sorted = data_values[::-1, :]  # 緯度方向を反転

    # 昇順の src_lats_sorted を使って補間関数を作成
    interp_func = RectBivariateSpline(src_lats_sorted, src_lons, data_values_sorted, kx=1, ky=1)

    # target_lats 昇順に変換
    target_lats_sorted = target_lats[::-1]
    # 補間実行
    interp_values_sorted = interp_func(target_lats_sorted, target_lons, grid=True)
    # 補間結果を反転して、元の target_lats の順序に合わせる
    interp_values = interp_values_sorted[::-1, :]

    return interp_values

def create_lonlat_grid():
    """指定された512x512のグリッドを作成"""
    return OUTPUT_LONS, OUTPUT_LATS

def convert_to_jst(utc_time):
    """UTC時間をJST時間に変換（+9時間）"""
    if isinstance(utc_time, pd.Timestamp):
        return utc_time + pd.Timedelta(hours=9)
    return utc_time + datetime.timedelta(hours=9)

def extract_date_from_filename(filename):
    """ファイル名から日時情報を抽出"""
    # Z__C_RJTD_20180131210000_MSM_GPV_Rjp_L-pall_FH00-15_grib2.bin
    # 正規表現を使ってファイル名からYYYYMMDDHH部分を抽出
    pattern = r'Z__C_RJTD_(\d{10})'
    match = re.search(pattern, filename)
    if match:
        date_str = match.group(1)
        try:
            dt = datetime.datetime.strptime(date_str, '%Y%m%d%H')
            return dt
        except ValueError as e:
            print(f"Error parsing date from {date_str}: {e}")
    
    print(f"Could not extract date from filename: {os.path.basename(filename)}")
    return None

def load_grib_files_for_day(date_str, msm_dir="./MSM_grib"):
    """JST 1時〜24時の1時間値を作成するために、3時間間隔のベースGRIBファイル（FT=0,3）を読み込む"""
    # 日付から年月を取得
    year_month = date_str[:6]  # YYYYmm
    
    # JST 1時〜24時に対応するUTCの時刻を計算（JST 1時＝前日UTC16時、JST24時＝当日UTC15時）
    # ベースファイルは全てFT=0,3の3時間間隔で取得
    jst_date = datetime.datetime.strptime(date_str, '%Y%m%d')
    prev_date = jst_date - datetime.timedelta(days=1)
    
    # ファイルのリスト（JST 1時〜24時の値を生成するためのFT=0,3ベースファイル）
    files = []
    
    # 前日のUTC 15時、18時、21時のファイル（JST 0時〜9時相当のFT=3,0を取得）
    prev_date_str = prev_date.strftime('%Y%m%d')
    prev_month_str = prev_date.strftime('%Y%m')
    for hour in [15, 18, 21]:
        file_path = f"{msm_dir}/{prev_month_str}/Z__C_RJTD_{prev_date_str}{hour:02d}0000_MSM_GPV_Rjp_L-pall_FH00-15_grib2.bin"
        if os.path.exists(file_path):
            files.append(file_path)
        else:
            print(f"Warning: File not found: {file_path}")
    
    # 当日のUTC 0時、3時、6時、9時、12時のファイル（JST 9時〜21時相当のFT=0,3を取得し、JST24時は12時FT3）
    current_month_str = jst_date.strftime('%Y%m')
    
    for hour in [0, 3, 6, 9, 12]:
        file_path = f"{msm_dir}/{current_month_str}/Z__C_RJTD_{date_str}{hour:02d}0000_MSM_GPV_Rjp_L-pall_FH00-15_grib2.bin"
        if os.path.exists(file_path):
            files.append(file_path)
        else:
            print(f"Warning: File not found: {file_path}")
    
    if not files:
        print(f"No files found for JST {date_str}")
        return None
    
    print(f"Found {len(files)} files for JST {date_str} with base forecast steps 0,3 (will interpolate FT=1,2)")
    return files

def buffer_file(file_path):
    """ファイルをメモリにバッファする（NFS遅延の削減）"""
    start_time = time_module.time()
    with open(file_path, 'rb') as f:
        buffer = io.BytesIO(f.read())
    elapsed = time_module.time() - start_time
    print(f"File buffered in {elapsed:.2f}s: {os.path.basename(file_path)}")
    return buffer, file_path

def get_target_messages(grbs, step_range=(3,)):
    """処理対象のメッセージだけを抽出（メモリと処理時間の削減）"""
    target_messages = []
    
    for grb in grbs:
        step = grb.forecastTime
        if step not in step_range:
            continue
            
        short_name = grb.shortName if hasattr(grb, 'shortName') else None
        level = grb.level if hasattr(grb, 'level') else None
        level_type = grb.levelType if hasattr(grb, 'levelType') else None
        
        # 対象の圧力レベルと変数だけを処理
        if (level in PRESSURE_LEVELS and level_type == 'pl' and 
            short_name in VAR_BY_LEVEL[level]):
            target_messages.append((short_name, level, step, grb))
    
    return target_messages

def process_single_file(file_buffer_tuple, target_lons, target_lats, step_range=(0,3)):
    """1つのGRIBファイルを処理してxr.Datasetに変換する（FT=0,3取得後にFT=1,2を内挿して1時間値を作成）"""
    file_buffer, file_path = file_buffer_tuple
    
    try:
        # バッファからGRIBファイルをオープン
        start_time = time_module.time()
        file_buffer.seek(0)
        grbs = pygrib.open(file_buffer.name if hasattr(file_buffer, 'name') else file_path)
        
        # ベース時間を取得
        base_time = extract_date_from_filename(file_path)
        if base_time is None:
            print(f"Could not extract base time from filename: {file_path}")
            return None
        
        # データを格納するための辞書
        data_dict = {}
        
        # 処理対象のステップ
        processed_steps = set()
        
        # 対象のメッセージだけを抽出（メモリ効率の向上）
        target_messages = get_target_messages(grbs, step_range)
        
        # 出力グリッドの形状
        output_shape = (len(target_lats), len(target_lons))
        
        # 各メッセージを処理
        for short_name, level, step, grb in target_messages:
            # この圧力レベルでの変数マッピング
            target_name = VAR_BY_LEVEL[level][short_name]
            
            # データを取得
            data_values = grb.values
            
            # データが2次元か確認
            if data_values.ndim == 2:
                # 予報時刻を計算（ベース時間 + 予報ステップ）
                # FT=0,3なので、ベース時間から0時間および3時間後の時刻が予報対象時刻
                valid_time = base_time + datetime.timedelta(hours=step)
                
                # JSTに変換
                jst_time = convert_to_jst(valid_time)
                
                # 上空データの粗いグリッドから高解像度グリッドへの高速な線形補間
                output_data = interpolate_grid_fast(data_values, MSM_P_LONS, MSM_P_LATS, target_lons, target_lats)
                
                # float32に変換
                output_data = output_data.astype(np.float32)
                
                # 時間、変数名、予報ステップでキーを生成
                key = (jst_time, target_name, step)
                data_dict[key] = np.expand_dims(output_data, axis=0)
                
                processed_steps.add(step)
        
        # FT=0とFT=3のデータがある場合、FT=1,2を線形内挿してdata_dictに追加
        if 0 in processed_steps and 3 in processed_steps:
            # 基準JST時間を取得
            jst_t0 = convert_to_jst(base_time)
            jst_t3 = convert_to_jst(base_time + datetime.timedelta(hours=3))
            # 変数ごとに内挿
            for var_key in {key[1] for key in data_dict.keys()}:
                arr0 = data_dict.get((jst_t0, var_key, 0))
                arr3 = data_dict.get((jst_t3, var_key, 3))
                if arr0 is not None and arr3 is not None:
                    data0 = arr0[0]
                    data3 = arr3[0]
                    for h in [1,2]:
                        alpha = h / 3.0
                        new_data = (1 - alpha) * data0 + alpha * data3
                        new_time = jst_t0 + datetime.timedelta(hours=h)
                        data_dict[(new_time, var_key, h)] = np.expand_dims(new_data, axis=0)
                        processed_steps.add(h)
        
        # グリブファイルを閉じる
        grbs.close()
        elapsed = time_module.time() - start_time
        print(f"Processed file in {elapsed:.2f}s: {os.path.basename(file_path)} (steps: {processed_steps})")
        
        # データがない場合は終了
        if not data_dict:
            print(f"No valid data extracted from {os.path.basename(file_path)}")
            return None
        
        # 処理したデータから新しいxr.Datasetを作成
        # 時間ごとに整理（データセットの効率的な結合のため）
        datasets_by_time = {}
        
        for (valid_time, var_name, step), data_array in data_dict.items():
            if valid_time not in datasets_by_time:
                datasets_by_time[valid_time] = {}
            
            # データをこの時間のディクショナリに追加
            datasets_by_time[valid_time][var_name] = data_array[0]  # 3次元から2次元に
        
        # 時間ごとにデータセットを作成
        datasets = []
        for valid_time, var_dict in datasets_by_time.items():
            data_vars = {}
            for var_name, array in var_dict.items():
                # 変数データとその属性を設定
                level = int(re.search(r'\d+', var_name).group())
                var_type = var_name[0]
                
                attrs = {
                    'forecast_step': step_range[0],
                    'level': level
                }
                
                if var_type == 'U':
                    attrs.update({'units': 'm/s', 'long_name': f'U-wind Component at {level}hPa'})
                elif var_type == 'V':
                    attrs.update({'units': 'm/s', 'long_name': f'V-wind Component at {level}hPa'})
                elif var_type == 'T':
                    attrs.update({'units': 'K', 'long_name': f'Temperature at {level}hPa'})
                elif var_type == 'H':
                    attrs.update({'units': '%', 'long_name': f'Relative Humidity at {level}hPa'})
                elif var_type == 'Z':
                    attrs.update({'units': 'm', 'long_name': f'Geopotential Height at {level}hPa'})
                elif var_type == 'W':
                    attrs.update({'units': 'Pa/s', 'long_name': f'Vertical Velocity (omega) at {level}hPa'})
                
                data_vars[var_name] = (['lat', 'lon'], array, attrs)
            
            # 全変数を持つデータセットを作成
            ds = xr.Dataset(
                data_vars=data_vars,
                coords={
                    'time': [valid_time],
                    'lat': target_lats,
                    'lon': target_lons
                }
            )
            datasets.append(ds)
        
        # すべてのデータセットをマージ
        if datasets:
            merged_ds = xr.concat(datasets, dim='time')
            return merged_ds
        
        return None
            
    except Exception as e:
        print(f"Error processing {os.path.basename(file_path)}: {str(e)}")
        return None

def main():
    # コマンドライン引数を解析
    import argparse
    parser = argparse.ArgumentParser(description='Convert MSM pressure level grib files to netCDF')
    parser.add_argument('date', help='Date in YYYYMM format or YYYYMMDD format (if YYYYMMDD, only the YYYYMM part will be used)')
    parser.add_argument('--msm_dir', default='./MSM_grib', help='Directory containing MSM grib files')
    parser.add_argument('--output_dir', default='./MSM_nc', help='Output directory for netCDF files')
    parser.add_argument('--max_workers', type=int, default=8, help='Maximum number of parallel workers')
    parser.add_argument('--profile', action='store_true', help='Enable profiling')
    args = parser.parse_args()
    
    if args.profile:
        # プロファイリングを有効にする
        profiler = cProfile.Profile()
        profiler.enable()
    
    # パフォーマンス計測開始
    total_start_time = time_module.time()
    
    # 年月を取得
    year_month = args.date[:6]  # YYYYmm
    
    # 出力ディレクトリの作成（年月のサブディレクトリも含む）
    output_subdir = args.output_dir
    os.makedirs(output_subdir, exist_ok=True)
    print(f"Created output directory: {output_subdir}")
    
    # グリッドの作成
    target_lons, target_lats = create_lonlat_grid()
    
    # 月の日数を計算
    year = int(year_month[:4])
    month = int(year_month[4:6])
    if month == 12:
        next_year = year + 1
        next_month = 1
    else:
        next_year = year
        next_month = month + 1
    
    # 月の最終日を計算
    last_day = (datetime.datetime(next_year, next_month, 1) - datetime.timedelta(days=1)).day
    
    # 月内の全ての日付を処理
    all_datasets = []
    for day in range(1, last_day + 1):
        date_str = f"{year_month}{day:02d}"
        print(f"Processing date: {date_str}")
        
        # 指定した日のファイルを取得（JST 0時～21時の予報対象時刻に対応するため、初期時刻は3時間前）
        files = load_grib_files_for_day(date_str, args.msm_dir)
        if not files:
            print(f"No files found for {date_str}")
            continue
        
        # ファイルをバッファに読み込む（NFS遅延を削減）
        buffered_files = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            # ファイルのバッファリングを並列実行
            buffer_futures = [executor.submit(buffer_file, file_path) for file_path in files]
            
            # 結果を収集
            for future in concurrent.futures.as_completed(buffer_futures):
                try:
                    buffered_files.append(future.result())
                except Exception as e:
                    print(f"Error buffering file: {e}")
        
        # 各ファイルを並列処理（3時間予報のデータ（FT=3）のみを使用）
        day_datasets = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            # ファイルの処理を並列実行
            process_futures = [executor.submit(process_single_file, file_buffer, target_lons, target_lats) 
                               for file_buffer in buffered_files]
            
            # 結果を収集
            for future in concurrent.futures.as_completed(process_futures):
                try:
                    ds = future.result()
                    if ds is not None:
                        day_datasets.append(ds)
                except Exception as e:
                    print(f"Error in parallel processing: {e}")
        
        # 1日分のデータセットをマージ
        if day_datasets:
            day_ds = xr.concat(day_datasets, dim='time')
            
            # 時間でソート
            day_ds = day_ds.sortby('time')
            
            # 指定された日のJST時間を作成（1時〜24時、1時間間隔）
            target_date = datetime.datetime(year, month, day)
            target_times = [target_date + datetime.timedelta(hours=hour) for hour in range(1, 25)]
            
            # 時間に最も近いデータを選択（完全一致でなくても許容）
            filtered_day_datasets = []
            for target_time in target_times:
                # 各時刻から±30分以内のデータを検索
                time_diff = np.abs((day_ds.time.values - np.datetime64(target_time)) / np.timedelta64(1, 'h'))
                if len(time_diff) > 0:
                    closest_idx = np.argmin(time_diff)
                    
                    if time_diff[closest_idx] <= 0.5:  # 30分以内なら採用
                        filtered_ds = day_ds.isel(time=[closest_idx])
                        print(f"Found data for JST {target_time}: {day_ds.time.values[closest_idx]}")
                        filtered_day_datasets.append(filtered_ds)
                    else:
                        print(f"Warning: No data found for JST {target_time}")
                else:
                    print(f"Warning: No data found for JST {target_time}")
            
            if filtered_day_datasets:
                # フィルタリングされたデータセットをマージ
                day_filtered_ds = xr.concat(filtered_day_datasets, dim='time')
                
                # 時間でソート（念のため）
                day_filtered_ds = day_filtered_ds.sortby('time')
                
                print(f"Filtered dataset for {date_str} contains {len(day_filtered_ds.time)} time points")
                
                # 月のデータセットに追加
                all_datasets.append(day_filtered_ds)
            else:
                print(f"No valid filtered data for {date_str}")
        else:
            print(f"No valid data processed for {date_str}")
    
    # 月のすべてのデータセットをマージ
    if all_datasets:
        final_ds = xr.concat(all_datasets, dim='time')
        
        # 時間でソート
        final_ds = final_ds.sortby('time')
        
        print(f"Final dataset contains {len(final_ds.time)} time points")
        
        # 変数の属性設定
        for var in final_ds.data_vars:
            final_ds[var].attrs['_FillValue'] = np.float32(-9999.0)
        
        # グローバル属性の設定
        final_ds.attrs['title'] = 'MSM Pressure Level Data (1-hourly)'
        final_ds.attrs['source'] = 'JMA Mesoscale Model'
        final_ds.attrs['created'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        final_ds.attrs['description'] = f'Converted from JMA MSM GRIB2 files to NetCDF (1-hourly, JST 01-24) for {year_month}'
        
        # 出力ファイル名の設定（新形式: MSM1h-P481_{YYYYmm}.nc）
        output_file = f"{output_subdir}/MSM1h-P481_{year_month}.nc"
        
        # netCDFファイルとして保存（圧縮あり）
        # 時間方向のチャンキングを設定
        chunksizes = (1, 505, 481)  # (time, lat, lon)
        #comp = dict(compression="zstd",complevel=3)
        #comp = dict(compression="blosc_zstd",blosc_shuffle=2,complevel=3)
        #comp = dict(compression="blosc_zstd",blosc_shuffle=2,complevel=3,significant_digits=4,quantize_mode='BitGroom')
        comp = dict(compression="blosc_zstd",blosc_shuffle=2,complevel=3,significant_digits=5)
        encoding = {
            var: {**comp, 'dtype': 'float32', 'chunksizes': chunksizes} 
            for var in final_ds.data_vars
        }
        
        # 座標変数のエンコーディングを設定（緯度・経度はfloat64のまま）
        encoding.update({
            'time': {'compression': 'zstd', 'complevel': 3},
            'lat': {'compression': 'zstd', 'complevel': 3, 'dtype': 'float64'},
            'lon': {'compression': 'zstd', 'complevel': 3, 'dtype': 'float64'},
        })
        
        save_start_time = time_module.time()
        print(f"Saving to {output_file}")
        final_ds.to_netcdf(output_file, encoding=encoding)
        save_elapsed = time_module.time() - save_start_time
        print(f"File saved in {save_elapsed:.2f}s")
        
        total_elapsed = time_module.time() - total_start_time
        print(f"Total processing time: {total_elapsed:.2f}s")
    else:
        print("No valid data processed for the month")
    
    if args.profile:
        # プロファイリングを無効にして結果を表示
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats(30)  # 上位30の結果を表示

if __name__ == "__main__":
    main() 