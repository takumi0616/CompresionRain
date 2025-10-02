#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MSM NetCDFデータチェックプログラム
作成されたNetCDFファイルが時間内挿モデルに必要なデータを含んでいるか検証する
"""

import os
import sys
import xarray as xr
import numpy as np
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# チェック対象のデータ仕様
REQUIRED_VARIABLES = {
    'L-pall': {
        # 気圧面データの必要な変数と気圧レベル
        'variables': {
            'u': [975, 950, 925, 850, 300],  # 東西風
            'v': [975, 950, 925, 850, 300],  # 南北風
            't': [975, 950, 925, 850, 500],  # 気温
            'r': [925, 850, 500],             # 相対湿度
            'gh': [500, 300]                  # ジオポテンシャル高度
        },
        'grid_size': (253, 241),
        'lat_range': (47.6, 22.4),
        'lon_range': (120.0, 150.0)
    },
    'Lsurf': {
        # 地上データの必要な変数
        'variables': {
            'prmsl': None,  # 海面更正気圧
            't2m': None,    # 地上2m気温
            'u10': None,    # 地上10m東西風
            'v10': None     # 地上10m南北風
        },
        'grid_size': (505, 481),
        'lat_range': (47.6, 22.4),
        'lon_range': (120.0, 150.0)
    },
    'Prr': {
        # 降水量データ
        'variables': {
            'unknown': None  # 降水量（変数名がunknownになっている）
        },
        'grid_size': (505, 481),
        'lat_range': (47.6, 22.4),
        'lon_range': (120.0, 150.0)
    }
}

# 必要なステップ（FT=3とFT=6が必須）
REQUIRED_STEPS = [3, 6]

def check_file_exists(filepath):
    """ファイルの存在確認"""
    if not os.path.exists(filepath):
        return False, f"ファイルが存在しません: {filepath}"
    return True, "OK"

def check_dataset_structure(ds, data_type):
    """データセットの基本構造をチェック"""
    results = []
    
    # 次元のチェック
    required_dims = ['time', 'step', 'latitude', 'longitude']
    if data_type == 'L-pall':
        required_dims.append('level')
    
    for dim in required_dims:
        if dim in ds.dims:
            results.append(f"  ✓ 次元 '{dim}' が存在 (サイズ: {ds.dims[dim]})")
        else:
            results.append(f"  ✗ 次元 '{dim}' が存在しません")
    
    return results

def check_grid_specifications(ds, expected_grid, expected_lat_range, expected_lon_range):
    """グリッド仕様のチェック"""
    results = []
    
    # グリッドサイズのチェック
    actual_grid = (ds.dims['latitude'], ds.dims['longitude'])
    if actual_grid == expected_grid:
        results.append(f"  ✓ グリッドサイズ: {actual_grid} (期待値と一致)")
    else:
        results.append(f"  ✗ グリッドサイズ: {actual_grid} (期待値: {expected_grid})")
    
    # 緯度範囲のチェック
    lat_values = ds.latitude.values
    actual_lat_range = (lat_values[0], lat_values[-1])
    if abs(actual_lat_range[0] - expected_lat_range[0]) < 0.01 and \
       abs(actual_lat_range[1] - expected_lat_range[1]) < 0.01:
        results.append(f"  ✓ 緯度範囲: {actual_lat_range[0]:.1f}°N ~ {actual_lat_range[1]:.1f}°N")
    else:
        results.append(f"  ✗ 緯度範囲: {actual_lat_range} (期待値: {expected_lat_range})")
    
    # 経度範囲のチェック
    lon_values = ds.longitude.values
    actual_lon_range = (lon_values[0], lon_values[-1])
    if abs(actual_lon_range[0] - expected_lon_range[0]) < 0.01 and \
       abs(actual_lon_range[1] - expected_lon_range[1]) < 0.01:
        results.append(f"  ✓ 経度範囲: {actual_lon_range[0]:.1f}°E ~ {actual_lon_range[1]:.1f}°E")
    else:
        results.append(f"  ✗ 経度範囲: {actual_lon_range} (期待値: {expected_lon_range})")
    
    return results

def check_variables(ds, required_vars, data_type):
    """必要な変数の存在チェック"""
    results = []
    missing_vars = []
    
    for var_name, required_levels in required_vars.items():
        if var_name in ds.data_vars:
            results.append(f"  ✓ 変数 '{var_name}' が存在")
            
            # L-pallの場合は気圧レベルもチェック
            if data_type == 'L-pall' and required_levels:
                if 'level' in ds[var_name].dims:
                    available_levels = ds.level.values
                    for level in required_levels:
                        if level in available_levels:
                            results.append(f"    ✓ {level}hPa面のデータあり")
                        else:
                            results.append(f"    ✗ {level}hPa面のデータなし")
                            missing_vars.append(f"{var_name}@{level}hPa")
        else:
            results.append(f"  ✗ 変数 '{var_name}' が存在しません")
            missing_vars.append(var_name)
    
    return results, missing_vars

def check_time_steps(ds):
    """時間ステップのチェック（FT=3とFT=6が必要）"""
    results = []
    
    if 'step' in ds.dims:
        steps = ds.step.values
        # numpy timedelta64をhourに変換
        if hasattr(steps[0], 'astype'):
            step_hours = steps.astype('timedelta64[h]').astype(int)
        else:
            step_hours = steps
        
        results.append(f"  利用可能なステップ: {sorted(set(step_hours))} 時間")
        
        for required_step in REQUIRED_STEPS:
            if required_step in step_hours:
                results.append(f"  ✓ FT={required_step}のデータあり")
            else:
                results.append(f"  ✗ FT={required_step}のデータなし")
    else:
        results.append("  ✗ stepディメンションが存在しません")
    
    return results

def check_precipitation_data(ds):
    """降水量データの特別チェック"""
    results = []
    
    # Prrファイルでは変数名が'unknown'になっている
    if 'unknown' in ds.data_vars:
        results.append("  ✓ 降水量データ（'unknown'変数）が存在")
        
        # ステップのチェック（FT=4,5,6が取得できるか）
        steps = ds.step.values
        if hasattr(steps[0], 'astype'):
            step_hours = steps.astype('timedelta64[h]').astype(int)
        else:
            step_hours = steps
        
        required_prec_steps = [4, 5, 6]
        for step in required_prec_steps:
            if step in step_hours:
                results.append(f"  ✓ FT={step}の降水量データあり")
            else:
                results.append(f"  ✗ FT={step}の降水量データなし")
        
        # データの単位チェック
        if 'units' in ds.unknown.attrs:
            results.append(f"  単位: {ds.unknown.attrs['units']}")
    else:
        results.append("  ✗ 降水量データが見つかりません")
    
    return results

def check_data_quality(ds, var_name, sample_size=5):
    """データ品質の簡易チェック"""
    results = []
    
    if var_name not in ds.data_vars:
        return results
    
    data = ds[var_name]
    
    # NaN/無効値のチェック
    total_size = data.size
    nan_count = np.isnan(data.values).sum()
    nan_ratio = nan_count / total_size * 100
    
    if nan_ratio < 1:
        results.append(f"  ✓ NaN値: {nan_ratio:.2f}% (正常)")
    elif nan_ratio < 10:
        results.append(f"  △ NaN値: {nan_ratio:.2f}% (やや多い)")
    else:
        results.append(f"  ✗ NaN値: {nan_ratio:.2f}% (異常に多い)")
    
    # データ範囲のチェック（サンプリング）
    valid_data = data.values[~np.isnan(data.values)]
    if len(valid_data) > 0:
        results.append(f"  データ範囲: {valid_data.min():.2f} ~ {valid_data.max():.2f}")
    
    return results

def check_single_file(filepath):
    """単一ファイルのチェック"""
    print(f"\n{'='*60}")
    print(f"チェック対象: {os.path.basename(filepath)}")
    print(f"{'='*60}")
    
    # ファイル存在チェック
    exists, msg = check_file_exists(filepath)
    if not exists:
        print(f"✗ {msg}")
        return False
    
    # データタイプの判定
    filename = os.path.basename(filepath)
    data_type = None
    for dtype in ['L-pall', 'Lsurf', 'Prr']:
        if dtype in filename:
            data_type = dtype
            break
    
    if not data_type:
        print("✗ ファイル名からデータタイプを判定できません")
        return False
    
    print(f"\nデータタイプ: {data_type}")
    
    try:
        # データセットを開く
        ds = xr.open_dataset(filepath)
        
        # 1. 基本構造のチェック
        print("\n1. 基本構造のチェック:")
        structure_results = check_dataset_structure(ds, data_type)
        for result in structure_results:
            print(result)
        
        # 2. グリッド仕様のチェック
        print("\n2. グリッド仕様のチェック:")
        spec = REQUIRED_VARIABLES[data_type]
        grid_results = check_grid_specifications(
            ds, spec['grid_size'], spec['lat_range'], spec['lon_range']
        )
        for result in grid_results:
            print(result)
        
        # 3. 変数のチェック
        print("\n3. 必要な変数のチェック:")
        var_results, missing_vars = check_variables(
            ds, spec['variables'], data_type
        )
        for result in var_results:
            print(result)
        
        # 4. 時間ステップのチェック
        print("\n4. 時間ステップのチェック:")
        step_results = check_time_steps(ds)
        for result in step_results:
            print(result)
        
        # 5. 降水量データの特別チェック（Prrの場合）
        if data_type == 'Prr':
            print("\n5. 降水量データの詳細チェック:")
            prec_results = check_precipitation_data(ds)
            for result in prec_results:
                print(result)
        
        # 6. データ品質の簡易チェック（代表的な変数）
        print("\n6. データ品質の簡易チェック:")
        if data_type == 'L-pall':
            check_var = 't'
        elif data_type == 'Lsurf':
            check_var = 't2m'
        else:
            check_var = 'unknown'
        
        quality_results = check_data_quality(ds, check_var)
        for result in quality_results:
            print(result)
        
        # 7. 時間情報のチェック
        print("\n7. 時間情報のチェック:")
        if 'time' in ds.dims:
            time_values = ds.time.values
            print(f"  時間次元のサイズ: {len(time_values)}")
            if len(time_values) > 0:
                print(f"  最初の時刻: {time_values[0]}")
        
        # 総合判定
        print("\n【総合判定】")
        if missing_vars:
            print(f"✗ 不足している変数があります: {missing_vars}")
            return False
        else:
            print("✓ 必要なデータは揃っています")
            return True
        
    except Exception as e:
        print(f"\n✗ エラーが発生しました: {str(e)}")
        return False
    finally:
        if 'ds' in locals():
            ds.close()

def check_all_files(base_dir, year_month=None):
    """複数ファイルの一括チェック"""
    pattern = f"MSM_data_{year_month}_*.nc" if year_month else "MSM_data_*.nc"
    
    import glob
    files = sorted(glob.glob(os.path.join(base_dir, pattern)))
    
    if not files:
        print(f"チェック対象のファイルが見つかりません: {pattern}")
        return
    
    print(f"\n見つかったファイル数: {len(files)}")
    
    success_count = 0
    for filepath in files:
        if check_single_file(filepath):
            success_count += 1
    
    print(f"\n\n{'='*60}")
    print(f"チェック完了: {success_count}/{len(files)} ファイルが正常")
    print(f"{'='*60}")

def main():
    """メイン処理"""
    print("MSM NetCDFデータチェックプログラム")
    print("時間内挿モデル用データの検証")
    
    # コマンドライン引数の処理
    if len(sys.argv) > 1:
        target = sys.argv[1]
        if os.path.isfile(target):
            # 単一ファイルのチェック
            check_single_file(target)
        elif os.path.isdir(target):
            # ディレクトリ内の全ファイルをチェック
            year_month = sys.argv[2] if len(sys.argv) > 2 else None
            check_all_files(target, year_month)
        else:
            print(f"エラー: {target} が見つかりません")
    else:
        # デフォルトパスでチェック
        default_dir = "output_nc"
        if os.path.exists(default_dir):
            # 201801のデータをサンプルとしてチェック
            check_all_files(default_dir, "201801")
        else:
            print("使用方法:")
            print("  python check_MSM_data_nc.py <ファイルパス>")
            print("  python check_MSM_data_nc.py <ディレクトリパス> [YYYYMM]")

if __name__ == "__main__":
    main()