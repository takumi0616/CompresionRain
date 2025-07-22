# analyze_all_data_single_process.py

import xarray as xr
import numpy as np
import os
import pandas as pd
import contextlib
import io
from collections import defaultdict

def analyze_single_grib_file(file_path: str) -> dict:
    """
    単一のGRIB2ファイルを詳細に解析し、構造と問題点を辞書として返す。
    この関数はメインプロセスから直接呼び出される。
    """
    if not os.path.exists(file_path):
        return {'status': 'not_found', 'path': file_path}

    # エラーログは抑制する
    devnull = open(os.devnull, 'w')
    result = {'path': file_path, 'status': 'ok'}
    datasets = None

    try:
        # この環境ではファイルパスを直接渡すのが最も確実
        with contextlib.redirect_stderr(devnull):
            ds_or_datasets = xr.open_dataset(file_path, engine='cfgrib',
                                             backend_kwargs={'filter_by_keys': {}})
        
        if isinstance(ds_or_datasets, xr.Dataset):
            datasets = [ds_or_datasets]
        else:
            datasets = ds_or_datasets

        if datasets is not None:
            all_data_vars, all_coords, has_nan, pall_schema = (set(), set(), False, defaultdict(set))
            for ds in datasets:
                all_data_vars.update(ds.data_vars.keys())
                all_coords.update(ds.coords.keys())
                for var_name in ds.data_vars:
                    if np.isnan(ds[var_name].values).any():
                        has_nan = True
                if 'isobaricInhPa' in ds.coords:
                    vars_in_ds = set(ds.data_vars.keys())
                    for level_val in ds['isobaricInhPa'].values:
                        pall_schema[int(level_val)].update(vars_in_ds)
            
            result.update({
                'data_vars': all_data_vars, 'coords': all_coords, 
                'has_nan': has_nan, 'pall_schema': dict(pall_schema)
            })
        else:
            raise RuntimeError("データセットの解析に失敗しました。")

    except Exception as e:
        result['status'] = 'error'
        result['message'] = str(e)
    
    finally:
        devnull.close()
        
    return result

def run_serial_analysis(base_dir, start_date, end_date):
    """
    ファイルパスを生成し、シングルプロセスで順番に解析を実行、結果を集計して表示する。
    """
    print("--- ステージ1: 対象ファイルパスのリストアップを開始します... ---")
    hours = ['00', '03', '06', '09', '12', '15', '18', '21']
    file_types_to_check = [
        "L-pall_FH00_grib2.bin", "L-pall_FH03_grib2.bin", "L-pall_FH06_grib2.bin",
        "Lsurf_FH00_grib2.bin", "Lsurf_FH03_grib2.bin", "Lsurf_FH06_grib2.bin",
        "Prr_FH00-03_grib2.bin", "Prr_FH03-06_grib2.bin",
    ]
    all_paths_to_scan = []
    for day in pd.date_range(start_date, end_date, freq='D'):
        for hour in hours:
            timestamp = f"{day.year}{day.month:02d}{day.day:02d}{hour}0000"
            filename_base = f"Z__C_RJTD_{timestamp}_MSM_GPV_Rjp_"
            dir_path = os.path.join(base_dir, f"{day.year}{day.month:02d}")
            for ftype in file_types_to_check:
                full_path = os.path.join(dir_path, filename_base + ftype)
                if os.path.exists(full_path):
                    all_paths_to_scan.append(full_path)

    total_files = len(all_paths_to_scan)
    if total_files == 0:
        print("対象ファイルが見つかりませんでした。")
        return
        
    print(f"リストアップ完了: {total_files} 個のファイルが対象です。")
    print(f"--- ステージ2: シングルプロセスによる全データチェックと情報集計を開始します... ---")
    
    problems = []
    schemas = defaultdict(lambda: {'data_vars': set(), 'coords': set(), 'pall_schema': defaultdict(set)})
    
    for i, file_path in enumerate(all_paths_to_scan):
        print(f"\r処理中: {i+1}/{total_files} ({(i+1)/total_files:.1%})", end="", flush=True)
        result = analyze_single_grib_file(file_path)

        if result['status'] == 'error':
            problems.append(f"解析エラー: {result['path']}\n  詳細: {result['message']}\n")
        elif result['status'] == 'ok':
            if result['has_nan']:
                problems.append(f"欠損値あり: {result['path']}\n")
            
            path = result['path']
            file_basename_type = os.path.basename(path).split('_')[-2] + '_' + os.path.basename(path).split('_')[-1]
            schemas[file_basename_type]['data_vars'].update(result['data_vars'])
            schemas[file_basename_type]['coords'].update(result['coords'])
            for level, var_set in result['pall_schema'].items():
                schemas[file_basename_type]['pall_schema'][level].update(var_set)

    print("\n\n--- ステージ3: 結果の表示 ---")
    if problems:
        print("\n--- 品質チェック完了: 問題が検出されました ---")
        for p in problems: print(p)
    else:
        print("\n--- 品質チェック完了: 問題は見つかりませんでした ---")

    print("\n--- ファイル種別ごとの共通情報サマリー ---\n")
    for ftype, content in sorted(schemas.items()):
        print(f"--- サマリー: {ftype} ---")
        print(f"  [データ変数]")
        for var in sorted(list(content['data_vars'])): print(f"    - {var}")
        print(f"\n  [座標変数]")
        for coord in sorted(list(content['coords'])): print(f"    - {coord}")
        if content['pall_schema']:
            print(f"\n  [気圧面ごとのデータ変数]")
            for level, var_set in sorted(content['pall_schema'].items(), reverse=True):
                print(f"    - {level} hPa: {', '.join(sorted(list(var_set)))}")
        print("-" * 50 + "\n")

if __name__ == '__main__':
    BASE_DIR = 'MSM_data'
    START_DATE = '2023-12-29'
    END_DATE = '2023-12-31'
    
    run_serial_analysis(BASE_DIR, START_DATE, END_DATE)
    
    print("全ての処理が終了しました。")