#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GRIB2 to NetCDF 変換プログラム (v2: データ統合ロジック改善版)
- 複数の予報時間ステップを持つGRIB2ファイルを正しくstep次元で結合する
"""

import os
import re
import warnings
from datetime import datetime
import concurrent.futures
import math

import cfgrib
import pandas as pd
import xarray as xr

warnings.filterwarnings('ignore', category=FutureWarning)

# --- ▼ 設定項目 ▼ ---
BASE_INPUT_DIR = "MSM_data"
OUTPUT_DIR = "MSM_data_nc"
START_DATE = "2018-01-01"
END_DATE = "2023-12-31"
# --- ▲ 設定項目 ▲ ---

def get_ref_time_from_filename(filename):
    """ファイル名から基準時刻（ref_time）のみを抽出する"""
    pattern = re.search(r'Z__C_RJTD_(\d{14})_MSM_GPV_Rjp', os.path.basename(filename))
    if pattern:
        return datetime.strptime(pattern.group(1), '%Y%m%d%H%M%S')
    return None

def process_month(year, month):
    """指定された年月の全GRIB2データをデータタイプ別に1つのNetCDFに変換する"""
    input_dir = os.path.join(BASE_INPUT_DIR, f"{year}{month:02d}")
    if not os.path.exists(input_dir):
        print(f"ディレクトリが存在しません、スキップします: {input_dir}")
        return

    print(f"--- {year}年{month:02d}月の処理を開始 ---")

    # --- 1. 月内の全GRIB2ファイルをデータタイプと基準時刻でグループ化 ---
    file_groups = {}
    for filename in os.listdir(input_dir):
        if not filename.endswith('.bin'):
            continue
        
        # ファイル名からデータタイプを判定 (L-pall, Lsurf, Prr)
        fname_base = os.path.basename(filename)
        data_type = None
        if "L-pall" in fname_base: data_type = "L-pall"
        elif "Lsurf" in fname_base: data_type = "Lsurf"
        elif "Prr" in fname_base: data_type = "Prr"
        else: continue
            
        ref_time = get_ref_time_from_filename(filename)
        if ref_time is None: continue

        # (基準時刻, データタイプ) をキーとしてファイルをグループ化
        key = (ref_time, data_type)
        if key not in file_groups:
            file_groups[key] = []
        file_groups[key].append(os.path.join(input_dir, filename))

    # --- 2. 基準時刻ごと、データタイプごとに処理し、最後に月全体で結合 ---
    monthly_datasets = {'L-pall': [], 'Lsurf': [], 'Prr': []}

    # 基準時刻でソートして処理順を安定させる
    sorted_keys = sorted(file_groups.keys())

    for key in sorted_keys:
        ref_time, data_type = key
        file_list = file_groups[key]
        
        try:
            # 複数のGRIB2ファイルを一度に開き、step次元で自動的に結合させる
            # Prrファイルは複数のデータセット(1h, 2h, 3h降水量)に分かれていることがある
            list_of_ds = [xr.merge(cfgrib.open_datasets(f), compat='override') for f in file_list]
            
            # 基準時刻が同じファイル群をstep次元で結合
            combined_ds = xr.concat(list_of_ds, dim='step').sortby('step')
            
            # 基準時刻(time)を座標として追加し、time次元を付与
            combined_ds = combined_ds.assign_coords(time=ref_time).expand_dims('time')
            
            monthly_datasets[data_type].append(combined_ds)

        except Exception as e:
            print(f"  [エラー] 処理失敗 ({ref_time}, {data_type}): {e}")
            continue

    # --- 3. 月全体のデータセットを結合し、ファイルに保存 ---
    for data_type, ds_list in monthly_datasets.items():
        if not ds_list:
            continue
            
        output_filename = os.path.join(OUTPUT_DIR, f"MSM_data_{year}{month:02d}_{data_type}.nc")
        if os.path.exists(output_filename):
            print(f"  ファイルが既に存在するためスキップ: {output_filename}")
            continue

        print(f"  > {data_type} データのNetCDFファイルを作成中...")
        try:
            # 月内の全データセットをtime次元で結合
            final_monthly_ds = xr.concat(ds_list, dim='time').sortby('time')

            # 変数名と座標名の変更
            if data_type == 'L-pall':
                final_monthly_ds = final_monthly_ds.rename({'isobaricInhPa': 'level'})
            elif data_type == 'Lsurf':
                rename_map = {}
                if 't' in final_monthly_ds.data_vars: rename_map['t'] = 't2m'
                if 'r' in final_monthly_ds.data_vars: rename_map['r'] = 'r2m'
                final_monthly_ds = final_monthly_ds.rename(rename_map)
            
            # 圧縮設定
            encoding = {var: {'zlib': True, 'complevel': 5} for var in final_monthly_ds.data_vars}
            
            # ファイルに書き出し
            final_monthly_ds.to_netcdf(output_filename, mode='w', encoding=encoding)
            print(f"  保存完了: {output_filename}")

        except Exception as e:
            print(f"  [エラー] ファイル保存失敗 ({output_filename}): {e}")

    print(f"--- {year}年{month:02d}月の処理完了 ---")


def main():
    """メイン処理"""
    print("========== GRIB2 to NetCDF 変換プログラム開始 (v2) ==========")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"出力ディレクトリを作成しました: {OUTPUT_DIR}")

    try:
        cpu_count = os.cpu_count() or 1
        num_workers = max(1, math.floor(cpu_count * 0.8)) # 少し多めに設定
    except NotImplementedError:
        num_workers = 4
        
    print(f"並列処理を開始します (使用ワーカー数: {num_workers})")

    date_range = pd.date_range(start=START_DATE, end=END_DATE, freq='MS')

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_date = {executor.submit(process_month, dt.year, dt.month): f"{dt.year}-{dt.month:02d}" for dt in date_range}
        
        for future in concurrent.futures.as_completed(future_to_date):
            date_str = future_to_date[future]
            try:
                future.result()
            except Exception as exc:
                print(f"[致命的エラー] {date_str} の処理中に予期せぬ例外: {exc}")
    
    print("========== 全ての処理が完了しました ==========")

if __name__ == "__main__":
    main()