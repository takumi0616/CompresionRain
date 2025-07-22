#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
時間内挿モデル用のNetCDFデータセット作成プログラム (v4: IndexError修正版)
- 降水量ファイルのステップ数が想定より少ない場合にクラッシュする問題を修正
"""

import xarray as xr
import numpy as np
import pandas as pd
import os
import logging
import time
import traceback

# --- ログ設定 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_creation.log", mode='w'),
        logging.StreamHandler()
    ]
)

# --- 定数・設定 ---
TARGET_LAT = 46.95 - np.arange(480) * 0.05
TARGET_LON = 120.0 + np.arange(480) * 0.0625

def get_variable_config():
    """モデルに使用する変数の設定を返す"""
    # [新しい変数名, 元ファイル種別, 元変数名, 気圧面(なければNone), 予報時間(FT)]
    config = [
        # 地上変数
        ['Prmsl_ft3', 'Lsurf', 'prmsl', None, 3], ['Prmsl_ft6', 'Lsurf', 'prmsl', None, 6],
        ['T2m_ft3', 'Lsurf', 't2m', None, 3], ['T2m_ft6', 'Lsurf', 't2m', None, 6],
        ['U10m_ft3', 'Lsurf', 'u10', None, 3], ['U10m_ft6', 'Lsurf', 'u10', None, 6],
        ['V10m_ft3', 'Lsurf', 'v10', None, 3], ['V10m_ft6', 'Lsurf', 'v10', None, 6],
        # 上層変数
        ['U975_ft3', 'L-pall', 'u', 975, 3], ['U975_ft6', 'L-pall', 'u', 975, 6],
        ['V975_ft3', 'L-pall', 'v', 975, 3], ['V975_ft6', 'L-pall', 'v', 975, 6],
        ['T975_ft3', 'L-pall', 't', 975, 3], ['T975_ft6', 'L-pall', 't', 975, 6],
        ['U950_ft3', 'L-pall', 'u', 950, 3], ['U950_ft6', 'L-pall', 'u', 950, 6],
        ['V950_ft3', 'L-pall', 'v', 950, 3], ['V950_ft6', 'L-pall', 'v', 950, 6],
        ['T950_ft3', 'L-pall', 't', 950, 3], ['T950_ft6', 'L-pall', 't', 950, 6],
        ['U925_ft3', 'L-pall', 'u', 925, 3], ['U925_ft6', 'L-pall', 'u', 925, 6],
        ['V925_ft3', 'L-pall', 'v', 925, 3], ['V925_ft6', 'L-pall', 'v', 925, 6],
        ['T925_ft3', 'L-pall', 't', 925, 3], ['T925_ft6', 'L-pall', 't', 925, 6],
        ['R925_ft3', 'L-pall', 'r', 925, 3], ['R925_ft6', 'L-pall', 'r', 925, 6],
        ['U850_ft3', 'L-pall', 'u', 850, 3], ['U850_ft6', 'L-pall', 'u', 850, 6],
        ['V850_ft3', 'L-pall', 'v', 850, 3], ['V850_ft6', 'L-pall', 'v', 850, 6],
        ['T850_ft3', 'L-pall', 't', 850, 3], ['T850_ft6', 'L-pall', 't', 850, 6],
        ['R850_ft3', 'L-pall', 'r', 850, 3], ['R850_ft6', 'L-pall', 'r', 850, 6],
        ['GH500_ft3', 'L-pall', 'gh', 500, 3], ['GH500_ft6', 'L-pall', 'gh', 500, 6],
        ['T500_ft3', 'L-pall', 't', 500, 3], ['T500_ft6', 'L-pall', 't', 500, 6],
        ['R500_ft3', 'L-pall', 'r', 500, 3], ['R500_ft6', 'L-pall', 'r', 500, 6],
        ['GH300_ft3', 'L-pall', 'gh', 300, 3], ['GH300_ft6', 'L-pall', 'gh', 300, 6],
        ['U300_ft3', 'L-pall', 'u', 300, 3], ['U300_ft6', 'L-pall', 'u', 300, 6],
        ['V300_ft3', 'L-pall', 'v', 300, 3], ['V300_ft6', 'L-pall', 'v', 300, 6],
    ]
    return config

def format_time(seconds):
    """秒を HH:MM:SS 形式の文字列に変換する"""
    if seconds is None:
        return "N/A"
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"

def process_month(year_month_str, input_dir, output_dir):
    """指定された年月のデータを処理し、1つのNetCDFファイルにまとめる"""
    month_start_time = time.time()
    logging.info(f"処理開始: {year_month_str}")

    t0 = time.time()
    paths = {
        'L-pall': os.path.join(input_dir, f'MSM_data_{year_month_str}_L-pall.nc'),
        'Lsurf': os.path.join(input_dir, f'MSM_data_{year_month_str}_Lsurf.nc'),
        'Prr': os.path.join(input_dir, f'MSM_data_{year_month_str}_Prr.nc'),
    }
    for ftype, fpath in paths.items():
        if not os.path.exists(fpath):
            logging.warning(f"ファイルが見つかりません: {fpath}。この月はスキップします。")
            return
    logging.info(f"  [完了] ファイルパスの準備 ({time.time() - t0:.2f}秒)")

    t0 = time.time()
    try:
        ds_lpall = xr.open_dataset(paths['L-pall'])
        ds_lsurf = xr.open_dataset(paths['Lsurf'])
        ds_prr = xr.open_dataset(paths['Prr'])
    except Exception as e:
        logging.error(f"{year_month_str} のファイルを開けませんでした: {e}")
        return
    logging.info(f"  [完了] NetCDFファイルの読み込み ({time.time() - t0:.2f}秒)")

    t0 = time.time()
    ml_dataset_vars = {}
    var_config = get_variable_config()
    target_grid = xr.Dataset({'latitude': (('latitude',), TARGET_LAT), 'longitude': (('longitude',), TARGET_LON)})

    for new_name, ftype, var_name, level, ft in var_config:
        source_ds = {'L-pall': ds_lpall, 'Lsurf': ds_lsurf}[ftype]
        if not var_name in source_ds:
            logging.warning(f"  - 警告: 変数 '{var_name}' がファイル '{ftype}' に見つかりません。'{new_name}' はスキップします。")
            continue
        
        num_steps = source_ds.sizes.get('step', 1)
        # FT(予報時間)からインデックス番号に変換
        step_index_to_get = ft -1 if ft > 0 else 0
        
        if num_steps > step_index_to_get:
            data_array = source_ds[var_name].isel(step=step_index_to_get)
        else:
            logging.warning(f"  - 警告: {new_name} に必要なステップ数({num_steps})が不足しています。スキップします。")
            continue

        if level:
            data_array = data_array.sel(level=level, method='nearest')
        
        interp_da = data_array.interp_like(target_grid, method='linear')
        ml_dataset_vars[new_name] = interp_da.drop_vars(['level', 'step'], errors='ignore')
    logging.info(f"  [完了] 説明変数の抽出とリグリッド ({time.time() - t0:.2f}秒)")
    
    # --- 降水量変数の特別処理（修正箇所） ---
    t0 = time.time()
    prr_var = ds_prr['unknown']
    num_steps_prr = prr_var.sizes.get('step', 0)
    logging.info(f"  - 降水量ファイルのステップ数: {num_steps_prr}")

    # 3時間積算値
    if num_steps_prr >= 3:
        prec_ft3 = prr_var.isel(step=slice(0, 3)).sum(dim='step')
        ml_dataset_vars['Prec_ft3'] = prec_ft3.interp_like(target_grid, method='linear').drop_vars('step', errors='ignore')
    else:
        logging.warning("  - 警告: ステップ数が3未満のため、Prec_ft3 は作成できません。")

    if num_steps_prr >= 6:
        prec_4_6h = prr_var.isel(step=slice(3, 6)).sum(dim='step')
        ml_dataset_vars['Prec_4_6h_sum'] = prec_4_6h.interp_like(target_grid, method='linear').drop_vars('step', errors='ignore')
    else:
        logging.warning("  - 警告: ステップ数が6未満のため、Prec_4_6h_sum は作成できません。")
        
    if num_steps_prr >= 9:
        prec_6_9h = prr_var.isel(step=slice(6, 9)).sum(dim='step')
        ml_dataset_vars['Prec_6_9h_sum'] = prec_6_9h.interp_like(target_grid, method='linear').drop_vars('step', errors='ignore')
    else:
        logging.warning("  - 警告: ステップ数が9未満のため、Prec_6_9h_sum は作成できません。")

    # 目的変数 (1時間値)
    for ft_h in [4, 5, 6]:
        target_index = ft_h - 1
        if num_steps_prr > target_index:
            prec_target = prr_var.isel(step=target_index)
            ml_dataset_vars[f'Prec_Target_ft{ft_h}'] = prec_target.interp_like(target_grid, method='linear').drop_vars('step', errors='ignore')
        else:
            logging.warning(f"  - 警告: ステップ数({num_steps_prr})が不足しているため、目的変数 Prec_Target_ft{ft_h} は作成できません。")
    logging.info(f"  [完了] 降水量変数の処理 ({time.time() - t0:.2f}秒)")

    # --- ファイル保存 ---
    if not ml_dataset_vars:
        logging.warning(f"作成する変数がありませんでした。{year_month_str}.nc の作成をスキップします。")
        return

    t0 = time.time()
    final_ds = xr.Dataset(ml_dataset_vars)
    encoding = {}
    for var in final_ds.data_vars:
        if 'time' in final_ds[var].dims:
            dims = list(final_ds[var].dims)
            time_index = dims.index('time')
            chunks = list(final_ds[var].shape)
            chunks[time_index] = 1
            encoding[var] = {'chunksizes': tuple(chunks)}

    output_path = os.path.join(output_dir, f"{year_month_str}.nc")
    final_ds.to_netcdf(output_path, encoding=encoding, mode='w')
    logging.info(f"  [完了] ファイル保存: {output_path} ({time.time() - t0:.2f}秒)")
    
    ds_lpall.close()
    ds_lsurf.close()
    ds_prr.close()
    total_month_time = time.time() - month_start_time
    logging.info(f"処理完了: {year_month_str} (合計所要時間: {format_time(total_month_time)})")

def main():
    script_start_time = time.time()
    logging.info("==================================================")
    logging.info("データ作成プロセスを開始します。 (v4: IndexError修正版)")
    logging.info("==================================================")

    INPUT_BASE_DIR = './MSM_data_nc'
    OUTPUT_BASE_DIR = './MSM_data_convert_nc'
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    date_range = pd.date_range(start='2018-01-01', end='2023-12-31', freq='MS')
    total_months = len(date_range)
    logging.info(f"処理対象期間: 2018年1月～2023年12月 (合計 {total_months} ヶ月)")
    
    for i, dt in enumerate(date_range):
        progress_percent = (i + 1) / total_months * 100
        logging.info("--------------------------------------------------")
        logging.info(f"全体進捗: {i + 1}/{total_months} ({progress_percent:.1f}%)")
        
        try:
            year_month_str = dt.strftime('%Y%m')
            process_month(year_month_str, INPUT_BASE_DIR, OUTPUT_BASE_DIR)
        except Exception as e:
            logging.error(f"致命的なエラーが発生し {dt.strftime('%Y%m')} の処理を中断しました: {e}")
            logging.error(traceback.format_exc())

        elapsed_time = time.time() - script_start_time
        avg_time_per_month = elapsed_time / (i + 1)
        remaining_months = total_months - (i + 1)
        eta_seconds = avg_time_per_month * remaining_months
        
        logging.info(f"経過時間: {format_time(elapsed_time)}")
        if remaining_months > 0:
            logging.info(f"残り推定時間 (ETA): {format_time(eta_seconds)}")
        logging.info("--------------------------------------------------")

    total_runtime = time.time() - script_start_time
    logging.info("==================================================")
    logging.info(f"全ての処理が完了しました。")
    logging.info(f"総実行時間: {format_time(total_runtime)}")
    logging.info("==================================================")

if __name__ == '__main__':
    main()