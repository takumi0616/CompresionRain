#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
NetCDFファイルの詳細分析プログラム

指定されたNetCDFファイル群をxarrayを用いて開き、
次元、座標、変数の詳細情報や基本統計量をログファイルに出力します。

■ 必要なライブラリ
- xarray
- netCDF4 (xarrayが内部で使用)
- numpy

■ ライブラリのインストール方法 (ターミナルで実行)
pip install xarray netCDF4 numpy
"""

import xarray as xr
import numpy as np
import os
import sys
from datetime import datetime

def analyze_netcdf(file_path, log_file):
    """
    単一のNetCDFファイルを分析し、結果をログファイルに書き込む関数
    """
    # ログに見やすいセパレータとタイトルを書き込む
    log_file.write(f"◆◇◆ 分析開始: {os.path.basename(file_path)} ◆◇◆".center(80, '=') + "\n")
    log_file.write(f"分析時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"ファイルパス: {file_path}\n\n")

    # ファイルが存在しない場合はエラーメッセージを書き込んで終了
    if not os.path.exists(file_path):
        log_file.write("【エラー】: ファイルが見つかりませんでした。\n")
        log_file.write("=" * 80 + "\n\n")
        return

    try:
        # xarrayでデータセットを開く
        with xr.open_dataset(file_path) as ds:
            # --- 1. データセット全体の情報 ---
            log_file.write("【1. データセット概要】\n")
            # xarrayのprint出力をキャプチャしてファイルに書き込む
            original_stdout = sys.stdout
            sys.stdout = log_file
            print(ds)
            sys.stdout = original_stdout
            log_file.write("\n")

            # --- 2. グローバル属性 ---
            log_file.write("【2. グローバル属性】\n")
            if ds.attrs:
                for key, value in ds.attrs.items():
                    log_file.write(f"- {key}: {value}\n")
            else:
                log_file.write("  グローバル属性はありません。\n")
            log_file.write("\n")

            # --- 3. 次元情報 ---
            log_file.write("【3. 次元情報】\n")
            if ds.dims:
                for dim_name, size in ds.dims.items():
                    log_file.write(f"- {dim_name}: {size}\n")
            else:
                log_file.write("  次元情報はありません。\n")
            log_file.write("\n")

            # --- 4. 座標変数の詳細 ---
            log_file.write("【4. 座標変数の詳細】\n")
            if ds.coords:
                for coord_name, coord_var in ds.coords.items():
                    log_file.write(f"--- 座標: {coord_name} ---\n")
                    log_file.write(f"  次元: {coord_var.dims}\n")
                    log_file.write(f"  データ型: {coord_var.dtype}\n")
                    # 値が数値型の場合のみ統計情報を計算
                    if np.issubdtype(coord_var.dtype, np.number):
                        valid_values = coord_var.values[np.isfinite(coord_var.values)]
                        if valid_values.size > 0:
                            log_file.write(f"  最小値: {np.nanmin(valid_values):.4f}\n")
                            log_file.write(f"  最大値: {np.nanmax(valid_values):.4f}\n")
                        else:
                            log_file.write("  有効な数値データがありません。\n")
                    # 属性を出力
                    if coord_var.attrs:
                        log_file.write("  属性:\n")
                        for key, value in coord_var.attrs.items():
                            log_file.write(f"    - {key}: {value}\n")
                    log_file.write("\n")
            else:
                log_file.write("  座標変数はありません。\n")
            log_file.write("\n")

            # --- 5. データ変数の詳細と基本統計量 ---
            log_file.write("【5. データ変数の詳細と基本統計量】\n")
            if ds.data_vars:
                for var_name, data_var in ds.data_vars.items():
                    log_file.write(f"--- 変数: {var_name} ({data_var.attrs.get('long_name', 'N/A')}) ---\n")
                    log_file.write(f"  次元: {data_var.dims}\n")
                    log_file.write(f"  データ型: {data_var.dtype}\n")
                    
                    # 属性を出力
                    if data_var.attrs:
                        log_file.write("  属性:\n")
                        for key, value in data_var.attrs.items():
                             log_file.write(f"    - {key}: {value}\n")
                    
                    # 基本統計量を計算 (NaNを無視)
                    log_file.write("  基本統計量:\n")
                    mean_val = data_var.mean().item()
                    min_val = data_var.min().item()
                    max_val = data_var.max().item()
                    std_val = data_var.std().item()
                    log_file.write(f"    - 平均値: {mean_val:.4f}\n")
                    log_file.write(f"    - 最小値: {min_val:.4f}\n")
                    log_file.write(f"    - 最大値: {max_val:.4f}\n")
                    log_file.write(f"    - 標準偏差: {std_val:.4f}\n")
                    log_file.write(f"    - 欠損値(NaN)の数: {int(data_var.isnull().sum())}\n\n")
            else:
                log_file.write("  データ変数はありません。\n")

    except Exception as e:
        log_file.write(f"【エラー】: ファイルの処理中にエラーが発生しました。\n")
        log_file.write(f"  {e}\n")

    finally:
        log_file.write("=" * 80 + "\n\n")


def main():
    """
    メイン処理
    """
    # 分析対象のファイルリスト
    file_paths = [
        './MSM_data_nc/MSM_data_201801_L-pall.nc',
        './MSM_data_nc/MSM_data_201801_Lsurf.nc',
        './MSM_data_nc/MSM_data_201801_Prr.nc',
    ]
    log_file_name = 'analysis_log.txt'
    
    # ログファイルを開く (新規作成または上書きモード)
    # 追記したい場合は 'a' に変更してください
    with open(log_file_name, 'w', encoding='utf-8') as f:
        f.write("#" * 80 + "\n")
        f.write("NetCDF ファイル一括分析ログ\n")
        f.write(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("#" * 80 + "\n\n")

        # 各ファイルを分析
        for path in file_paths:
            analyze_netcdf(path, f)
    
    print(f"分析が完了しました。結果は '{log_file_name}' を確認してください。")


if __name__ == '__main__':
    main()