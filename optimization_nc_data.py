#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import numpy as np
import xarray as xr
import datetime
import time
import logging
import hdf5plugin  # LZ4圧縮に必要
import dask
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
from dask.diagnostics import ProgressBar

# ==============================================================================
# --- 基本設定 (このセクションを編集して実行) ---
# ==============================================================================
# 処理対象期間
START_YEAR = 2018
END_YEAR = 2023

# 並列処理ワーカー数 (マシンのCPUコア数に合わせて調整)
# メモリ使用量とディスクI/Oを考慮し、CPUコア数より少し少なめに設定するのも良い
MAX_WORKERS = 16

# --- パス設定 ---
# このスクリプトが存在するディレクトリを基準にパスを構築
SCRIPT_DIR = Path(__file__).parent
# 変換元のNetCDFファイルが格納されているディレクトリ
INPUT_DIR = SCRIPT_DIR / "output_nc"
# 最適化されたNetCDFファイルの出力先
OUTPUT_DIR = SCRIPT_DIR / "optimization_nc"
# 正規化のための統計情報（グループごとのスケールのみ）を保存するファイル
SCALER_FILE = SCRIPT_DIR / "scaler.nc"
# 変数→グループ対応をメタとして保存（デバッグ/再現用）
SCALER_META = SCRIPT_DIR / "scaler_groups.json"

# ==============================================================================
# --- ログ設定 ---
# ==============================================================================
def setup_logging():
    """ログ設定を初期化する"""
    log_dir = SCRIPT_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    log_filename = log_dir / f"optimization_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    # tqdmは標準エラー出力にのみ表示
    tqdm.pandas(file=os.sys.stderr)

# ==============================================================================
# --- ユーティリティ/グルーピング ---
# ==============================================================================

TIME_FEATURE_VARS = ['dayofyear_sin', 'dayofyear_cos', 'hour_sin', 'hour_cos']
# data_vars に混入していても正規化対象から外したい（座標的）変数
COORD_LIKE_VARS = ['lat', 'lon', 'time']

def get_target_variables(ds):
    """
    データセットから正規化対象となる物理変数を取得する。
    - 時間特徴量（sin/cos）を除外
    - lat/lon/time のような座標的変数を除外（data_varsに存在しても除外）
    """
    coord_vars = set(ds.coords)
    exclude = set(TIME_FEATURE_VARS) | set(COORD_LIKE_VARS) | coord_vars
    return [v for v in ds.data_vars if v not in exclude]

def build_group_map(variable_names):
    """
    変数名から「同一物理量かつ関係性のあるもの」を同一スケールで正規化するためのグループを構築する。
    - 降水: Prec_Target_ft4/5/6 と Prec_4_6h_sum（および Prec_ft3）を同一グループ 'precip' にする
      → x/s + y/s + z/s = (x+y+z)/s により積算関係を保持
    - 風U/V, 温度T, 相対湿度R, ジオポテンシャルGH, 海面更正気圧Prmsl はレベル別に ft3/ft6 を同一グループ
      例: U850_ft3/U850_ft6 → グループ 'U850'
    - 上記以外は変数単位で固有グループ
    戻り値:
        var_to_group: dict[var_name] = group_name
        group_to_vars: dict[group_name] = [var_names...]
    """
    var_to_group = {}
    group_to_vars = {}

    # 1) 降水（積算関係保持のため必ず同一グループ）
    precip_aliases = set([
        'Prec_Target_ft4', 'Prec_Target_ft5', 'Prec_Target_ft6',
        'Prec_4_6h_sum', 'Prec_ft3'
    ])
    present_precip = sorted([v for v in variable_names if v in precip_aliases])
    if present_precip:
        for v in present_precip:
            var_to_group[v] = 'precip'
        group_to_vars['precip'] = present_precip

    # 2) レベル付き物理量の ft3/ft6 を結合
    #    GHxxx_ft[36], T(2m|xxx)_ft[36], U(10m|xxx)_ft[36], V(10m|xxx)_ft[36], Rxxx_ft[36], Prmsl_ft[36]
    patterns = [
        # group name, regex to capture base
        ('GH',   re.compile(r'^(GH\d{3})_ft[36]$')),
        ('T',    re.compile(r'^(T(?:2m|\d{3}))_ft[36]$')),
        ('U',    re.compile(r'^(U(?:10m|\d{3}))_ft[36]$')),
        ('V',    re.compile(r'^(V(?:10m|\d{3}))_ft[36]$')),
        ('R',    re.compile(r'^(R\d{3})_ft[36]$')),
        ('Prmsl',re.compile(r'^(Prmsl)_ft[36]$')),
        ('Prec', re.compile(r'^(Prec)_ft[36]$')),  # Prec_ft3 など（上のprecipで既に拾うが保険）
    ]

    assigned = set(var_to_group.keys())

    for v in variable_names:
        if v in assigned:
            continue
        matched = False
        for _, pat in patterns:
            m = pat.match(v)
            if m:
                group = m.group(1)
                var_to_group[v] = group
                group_to_vars.setdefault(group, []).append(v)
                matched = True
                break
        if not matched:
            # 上記に当てはまらないものは単独グループ
            var_to_group[v] = v
            group_to_vars.setdefault(v, []).append(v)

    # グループ内の変数順を安定化
    for g in group_to_vars:
        group_to_vars[g] = sorted(group_to_vars[g])

    return var_to_group, group_to_vars

# ==============================================================================
# --- スケール統計の計算（グループ単位：スケールのみ、平均は使わない） ---
# ==============================================================================

def calculate_and_save_group_scales(file_paths, scaler_path, target_variables, group_to_vars):
    """
    全データセットに渡る各グループのスケール（標準偏差相当）を計算し保存する。
    - 正規化は x / scale（オフセットなし）とすることで加法関係を保持
    - グループ内の全変数で同一の scale(lat, lon) を使用
    - スケールは「グループ内の各変数の std(time) を取り、そのvariable平均」を採用
      （他の定義でも加法関係は保持されるが、安定性のため平均を採用）
    """
    logging.info("--- Starting calculation of group scales for normalization (scale-only, no centering) ---")
    logging.info(f"Target files: {len(file_paths)}")

    logging.info("Opening all files as a single virtual dataset with Dask...")
    # time チャンクは 24 など適宜（メモリ・I/O に応じて調整）
    with xr.open_mfdataset(file_paths, parallel=True, chunks={'time': 248}) as ds:
        ds_vars = ds[target_variables]

        logging.info("Computing per-variable std over time...")
        # std over time -> dims: ('variable', 'lat', 'lon')
        std_by_var = ds_vars.std(dim='time', skipna=True).to_dataarray(name='std_by_var')

        # ゼロ分散対策（極端に小さい値は 1.0 に置換）
        std_by_var = std_by_var.where(std_by_var > 1e-6, 1.0)

        scales = []
        group_names = []
        logging.info("Aggregating std within groups to produce group scales...")
        for group, vars_in_group in group_to_vars.items():
            # 対象変数のみを選択
            sel = std_by_var.sel(variable=[v for v in vars_in_group if v in std_by_var['variable'].values])
            # グループ内平均（variable 次元で平均）: shape (lat, lon)
            group_scale = sel.mean(dim='variable', skipna=True)
            group_scale = group_scale.rename('scale')
            scales.append(group_scale)
            group_names.append(group)

        # 連結して (group, lat, lon) の DataArray に
        scale_da = xr.concat(scales, dim=xr.DataArray(group_names, dims='group', name='group'))
        # Dataset 化
        scaler_ds = xr.Dataset({'scale': scale_da})

        # 保存
        scaler_ds.to_netcdf(scaler_path)
        logging.info(f"✅ Group scales saved successfully to: {scaler_path}")
        return scaler_ds

# ==============================================================================
# --- 単一ファイルの最適化（正規化+圧縮） ---
# ==============================================================================

def optimize_monthly_netcdf(file_path, output_path, scaler_path, var_to_group, target_variables):
    """
    単一の月次NetCDFファイルを読み込み、グループ同一スケール (x/scale) の正規化、
    チャンク設定、LZ4圧縮を適用して保存する。
    - day/hour の sin/cos 特徴は完全に除去して出力しない
    """
    try:
        if output_path.exists():
            logging.info(f"Skipping: {output_path} already exists.")
            return f"Skipped: {file_path.name}"

        logging.debug(f"Processing: {file_path.name}")

        # 各ワーカー内でスケーラを開く（プロセス間共有より軽量）
        scaler_ds = xr.open_dataset(scaler_path)

        with xr.open_dataset(file_path, chunks={'time': 1}) as ds:
            # 時間特徴量は完全削除
            drop_vars = [v for v in TIME_FEATURE_VARS if v in ds]
            if drop_vars:
                ds = ds.drop_vars(drop_vars)
                logging.debug(f"Dropped time features: {drop_vars}")

            # 正規化（x / scale）: グループの scale(lat,lon) を使用
            for var_name in target_variables:
                if var_name in ds and var_name in var_to_group:
                    group = var_to_group[var_name]
                    if 'scale' not in scaler_ds or group not in scaler_ds['scale']['group'].values:
                        # スケール不在（整合性の問題）→スキップして警告
                        logging.warning(f"Scale for group '{group}' not found; skipping normalization for '{var_name}'")
                        continue
                    scale = scaler_ds['scale'].sel(group=group, drop=True)  # dims: (lat, lon)
                    # ブロードキャストで (time, lat, lon) に適用
                    ds[var_name] = ds[var_name] / scale

            # 圧縮設定
            encoding = {}
            for var in ds.data_vars:
                # time 次元がある 3次元以上の変数にチャンクを設定
                if ds[var].ndim >= 3 and 'time' in ds[var].dims:
                    # (time=1, lat, lon) チャンク
                    encoding[var] = {
                        **hdf5plugin.LZ4(),
                        'chunksizes': (1,) + ds[var].shape[1:]
                    }
                else:
                    # 2次元変数などはデフォルト圧縮のみにする
                    encoding[var] = {**hdf5plugin.LZ4()}

            write_job = ds.to_netcdf(
                output_path,
                engine='h5netcdf',
                encoding=encoding,
                mode='w',
                compute=False
            )

            with ProgressBar():
                write_job.compute()

            logging.debug(f"Successfully created: {output_path.name}")
            return f"Success: {file_path.name}"

    except Exception as e:
        logging.error(f"❌ Failed to process {file_path.name}: {e}", exc_info=True)
        return f"Failed: {file_path.name}"

# ==============================================================================
# --- メイン実行部 ---
# ==============================================================================
def main():
    """
    メイン処理。
    1) 変数のグルーピングを行い、グループ毎のスケール（lat,lon 毎）を計算し保存
       - スケールのみ（平均は使わない） → x/scale で正規化
       - 降水の積算関係を保持（Prec_4_6h_sum ≈ ft4+ft5+ft6 は正規化後も成り立つ）
       - 同一物理量の変数（レベル/予報時間違い）は同じスケール
    2) 各月ファイルを並列で最適化（時間周期特徴は完全に除去）
    """
    setup_logging()
    total_start_time = time.time()
    logging.info("===== NetCDF Optimization Process Start (Group-wise scale-only normalization) =====")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.info(f"Input directory: {INPUT_DIR}")
    logging.info(f"Output directory: {OUTPUT_DIR}")
    logging.info(f"Scaler file: {SCALER_FILE}")

    file_paths = sorted([p for p in INPUT_DIR.glob("*.nc")])
    if not file_paths:
        logging.error(f"No NetCDF files found in {INPUT_DIR}. Exiting.")
        return

    # ターゲット変数とグループ決定（先頭ファイルで構造取得）
    with xr.open_dataset(file_paths[0]) as temp_ds:
        target_variables = get_target_variables(temp_ds)
    var_to_group, group_to_vars = build_group_map(target_variables)
    logging.info(f"Target variables for normalization ({len(target_variables)}): {target_variables}")
    logging.info(f"Number of groups: {len(group_to_vars)}")

    # スケーラ準備
    if SCALER_FILE.exists():
        logging.info(f"Found existing scaler file. Loading from: {SCALER_FILE}")
        scaler_ds = xr.open_dataset(SCALER_FILE)
        # 既存スケーラが旧形式（mean/stdのみ、'scale' なし）の場合は再計算
        if 'scale' not in scaler_ds:
            logging.warning("Existing scaler is legacy (missing 'scale'). Recomputing scaler.")
            scaler_ds.close()
            scaler_ds = calculate_and_save_group_scales(file_paths, SCALER_FILE, target_variables, group_to_vars)
        existing_groups = set(scaler_ds['scale']['group'].values.tolist())

        # グループの不一致チェック（警告のみ、必要なら再計算）
        current_groups = set(group_to_vars.keys())
        if existing_groups != current_groups:
            logging.warning("Group set mismatch between existing scaler and current variables.")
            logging.warning(f"Existing groups (n={len(existing_groups)}): {sorted(existing_groups)[:10]}...")
            logging.warning(f"Current groups  (n={len(current_groups)}): {sorted(current_groups)[:10]}...")
            logging.warning("Recomputing scaler to ensure consistency.")
            scaler_ds = calculate_and_save_group_scales(file_paths, SCALER_FILE, target_variables, group_to_vars)
    else:
        logging.info("Scaler file not found. Calculating from scratch...")
        scaler_ds = calculate_and_save_group_scales(file_paths, SCALER_FILE, target_variables, group_to_vars)

    # 変数→グループのメタを保存（可読性のため JSON で別保存）
    try:
        with open(SCALER_META, 'w', encoding='utf-8') as f:
            json.dump({'var_to_group': var_to_group, 'group_to_vars': group_to_vars}, f, ensure_ascii=False, indent=2)
        logging.info(f"Saved scaler grouping meta: {SCALER_META}")
    except Exception as e:
        logging.warning(f"Failed to write scaler grouping meta: {e}")

    logging.info(f"\n--- Starting parallel optimization of {len(file_paths)} monthly files using {MAX_WORKERS} workers ---")

    process_start_time = time.time()
    with tqdm(total=len(file_paths), desc="Optimizing files", file=os.sys.stderr) as pbar:
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(
                    optimize_monthly_netcdf,
                    fp,
                    OUTPUT_DIR / fp.name,
                    str(SCALER_FILE),
                    var_to_group,
                    target_variables
                ) for fp in file_paths
            ]

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                logging.debug(result)
                pbar.update(1)

    process_elapsed = time.time() - process_start_time
    logging.info(f"--- Parallel optimization finished. Time taken: {datetime.timedelta(seconds=int(process_elapsed))} ---")

    total_elapsed = time.time() - total_start_time
    logging.info(f"\n===== All processes finished. Total execution time: {datetime.timedelta(seconds=int(total_elapsed))} =====")

if __name__ == '__main__':
    # 必要パッケージ:
    #   pip install h5netcdf hdf5plugin dask xarray
    main()
