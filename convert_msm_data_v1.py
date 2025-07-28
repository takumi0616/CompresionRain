#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GRIB2 to NetCDF 変換プログラム (v5: 破損インデックス自動修復機能付き)
- cfgribの破損した.idxファイルによるEOFErrorを自動検知し、修復して処理を続行する
"""

import os
import re
import warnings
from datetime import datetime
import concurrent.futures
import math
import glob # .idxファイルの検索に使用

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

def open_grib_safely(file_path):
    """破損したインデックスファイルを処理し、安全にGRIBファイルを開く"""
    try:
        # 最初の試行
        return xr.merge(cfgrib.open_datasets(file_path), compat='override')
    except EOFError:
        # EOFErrorが発生した場合、インデックスが破損していると判断
        print(f"  [警告] 破損インデックスファイルを検出: {os.path.basename(file_path)}")
        
        # 関連する.idxファイルを全て検索して削除
        idx_pattern = f"{file_path}*.idx"
        deleted_count = 0
        for idx_file in glob.glob(idx_pattern):
            try:
                os.remove(idx_file)
                deleted_count += 1
            except OSError as e:
                print(f"  [エラー] インデックスファイルの削除に失敗: {idx_file}, {e}")
        
        if deleted_count > 0:
            print(f"  > {deleted_count}個のインデックスファイルを削除し、再試行します...")
        
        # 再度ファイルを開く（これでインデックスが再生成される）
        return xr.merge(cfgrib.open_datasets(file_path), compat='override')


def process_month(year, month):
    """指定された年月のGRIB2データをNetCDFに変換する"""
    input_dir = os.path.join(BASE_INPUT_DIR, f"{year}{month:02d}")
    if not os.path.exists(input_dir):
        print(f"ディレクトリが存在しません、スキップします: {input_dir}")
        return

    print(f"--- {year}年{month:02d}月の処理を開始 ---")

    file_groups = {}
    for filename in os.listdir(input_dir):
        if not filename.endswith('.bin'): continue
        fname_base = os.path.basename(filename)
        data_type = None
        if "L-pall" in fname_base: data_type = "L-pall"
        elif "Lsurf" in fname_base: data_type = "Lsurf"
        elif "Prr" in fname_base: data_type = "Prr"
        else: continue
        ref_time = get_ref_time_from_filename(filename)
        if ref_time is None: continue
        key = (ref_time, data_type)
        if key not in file_groups: file_groups[key] = []
        file_groups[key].append(os.path.join(input_dir, filename))

    for data_type in ['L-pall', 'Lsurf', 'Prr']:
        output_filename = os.path.join(OUTPUT_DIR, f"MSM_data_{year}{month:02d}_{data_type}.nc")
        if os.path.exists(output_filename):
            print(f"  ファイルが既に存在するためスキップ: {output_filename}")
            continue

        ref_times_for_type = sorted([key[0] for key in file_groups if key[1] == data_type])
        if not ref_times_for_type: continue

        print(f"  > {data_type} データのNetCDFファイルを作成中...")
        is_first_write = True

        for ref_time in ref_times_for_type:
            key = (ref_time, data_type)
            file_list = file_groups[key]
            
            try:
                # ★★ ここが修正点 ★★
                # 安全なファイルオープン関数を呼び出す
                list_of_ds = [open_grib_safely(f) for f in file_list]
                
                combined_ds = xr.concat(list_of_ds, dim='step').sortby('step')
                combined_ds = combined_ds.assign_coords(time=ref_time).expand_dims('time')

                if data_type == 'L-pall':
                    combined_ds = combined_ds.rename({'isobaricInhPa': 'level'})
                elif data_type == 'Lsurf':
                    rename_map = {}
                    if 't' in combined_ds.data_vars: rename_map['t'] = 't2m'
                    if 'r' in combined_ds.data_vars: rename_map['r'] = 'r2m'
                    combined_ds = combined_ds.rename(rename_map)

                encoding = {var: {'zlib': True, 'complevel': 5} for var in combined_ds.data_vars}

                if is_first_write:
                    combined_ds.to_netcdf(output_filename, mode='w', encoding=encoding)
                    is_first_write = False
                else:
                    combined_ds.to_netcdf(output_filename, mode='a', encoding=encoding)
            
            except Exception as e:
                print(f"  [致命的エラー] 処理失敗 ({ref_time}, {data_type})。この月の処理を中断します: {e}")
                if os.path.exists(output_filename): os.remove(output_filename)
                break 

        if not is_first_write:
            print(f"  保存完了: {output_filename}")

    print(f"--- {year}年{month:02d}月の処理完了 ---")


def main():
    """メイン処理"""
    print("========== GRIB2 to NetCDF 変換プログラム開始 (v5: 自動修復機能付き) ==========")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"出力ディレクトリを作成しました: {OUTPUT_DIR}")

    try:
        cpu_count = os.cpu_count() or 1
        num_workers = max(1, math.floor(cpu_count * 0.8))
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