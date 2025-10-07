#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ファイル概要:
    NetCDF 月次ファイル群に対して、物理量ごとの関係性を保つ「グループ単位の min-max 正規化」を一括適用し、
    書き出し時に time 方向のチャンクと LZ4 圧縮を設定して、学習や推論で入出力しやすく最適化するスクリプト。

背景と設計方針:
    - 対象期間のデータは期間内でスキーマが統一されていることを前提とする。
    - 変数間の物理的関係（例: 降水の積算、同一高度面の U/V/T/R/GH、海面更正気圧など）を保つため、
      ft3/ft6 等の同系列変数を同一グループにまとめ、同一スケールで正規化する。
    - 正規化はグループ単位のスカラー min/max を用いるため、格子点（lat, lon）や時刻ごとにスケールが変わらない。
      これにより時空間の勾配や積算関係が歪みにくい。
    - dayofyear_sin/cos, hour_sin/cos は時間特徴量として保持するが、正規化の対象外とする。

入出力:
    - 入力: output_nc/*.nc（月ごとの NetCDF）。例として ncdump の一例では、time=248, lat=480, lon=480 に
      多数の物理変数（Prmsl_ft3/6, U/V/T/R at 975/950/925/850/500/300, GH500/300, U10m/V10m, T2m 等）と
      降水（Prec_ft3, Prec_4_6h_sum, Prec_Target_ft4/5/6）、時間特徴量、座標(lat, lon, time) が含まれる。
    - 出力: optimization_nc/*.nc（各月ファイルを min-max 正規化＋LZ4圧縮にて再保存）。
    - 付随メタ: scaler_groups.json に var_to_group, group_to_vars, group_minmax を保存し、再実行時に再利用する。

処理の流れ:
    1) 先頭ファイルで対象変数を抽出し、正規化グループを構築。
    2) 全ファイルにまたがって各変数の全域 min/max を計算し、グループ min/max を導出。
    3) 各月ファイルについて、グループ min-max 正規化を適用し、time チャンクと LZ4 圧縮を指定して書き出す。
       書き出しは h5netcdf エンジンを使用。Dask のスレッドスケジューラでファイル内並列を行い I/O と CPU を両立させる。

注意:
    - max == min の退避や NaN の安全対策を実装。
    - 期間内でファイルスキーマが同一であることを前提（変数の過不足があると再スケーラ計算が走る）。
"""

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
from dask.diagnostics import ProgressBar
from multiprocessing.pool import ThreadPool
import gc

# ==============================================================================
# --- 基本設定 (このセクションを編集して実行) ---
# ==============================================================================
# 処理対象期間
START_YEAR = 2018
END_YEAR = 2023

# 並列処理ワーカー数 (マシンのCPUコア数に合わせて調整)
# メモリ使用量とディスクI/Oを考慮し、CPUコア数より少し少なめに設定するのも良い
# 単一ファイル内並列に使用するスレッド数（CPUコア数）
MAX_WORKERS = 40
# 出力時のtimeチャンクサイズ（I/O効率改善のため）
WRITE_TIME_CHUNK = 24

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
    """
    概要:
        ログ出力と進捗バーの表示設定を初期化する。
    引数:
        なし
    処理:
        - logs/ ディレクトリを作成し、タイムスタンプ付きファイルへ INFO レベル以上を出力。
        - コンソールにも同時出力（StreamHandler）。
        - tqdm は標準エラー出力に表示して、ログと衝突しないようにする。
    戻り値:
        なし（副作用として logging のグローバル設定と tqdm の出力先を変更）
    """
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
    概要:
        与えられた xarray.Dataset から「正規化対象となる物理変数名」のリストを抽出する。
    引数:
        ds (xarray.Dataset):
            NetCDF から読み込んだデータセット。coords と data_vars を含む。
    処理:
        - 以下を正規化の対象外として除外する。
          (1) TIME_FEATURE_VARS（dayofyear_sin/cos, hour_sin/cos）…時間特徴量のためスケール固定を避ける
          (2) COORD_LIKE_VARS（lat, lon, time）…座標的な意味を持つため正規化しない
          (3) ds.coords に含まれている変数 … たとえ data_vars に混入していても座標扱いで除外
    戻り値:
        List[str]:
            正規化対象となる物理変数名のリスト。
    """
    coord_vars = set(ds.coords)
    exclude = set(TIME_FEATURE_VARS) | set(COORD_LIKE_VARS) | coord_vars
    return [v for v in ds.data_vars if v not in exclude]

def build_group_map(variable_names):
    """
    概要:
        変数名の規則（高度面や ft3/ft6 など）に基づいて、物理的に関連する変数を同一スケールで正規化するための
        「グループ」を構築する。
    引数:
        variable_names (List[str]):
            正規化候補の変数名一覧。
    処理:
        - 降水（Prec_Target_ft4/5/6, Prec_4_6h_sum, Prec_ft3）を 'precip' グループにまとめる。
          同一スケール s で正規化すると、x/s + y/s = (x+y)/s が成り立ち、積算の関係を保ちやすくなる。
        - GHxxx_ft[36], T(2m|xxx)_ft[36], U(10m|xxx)_ft[36], V(10m|xxx)_ft[36], Rxxx_ft[36], Prmsl_ft[36] 等については
          「同一基底（高度や2m/10mなど）」で ft3/ft6 をまとめ、グループ名に基底（例: 'U850'）を用いる。
        - どのパターンにも当てはまらない変数は単独で1グループにする。
        - グループ内の変数順は安定化のためソートする。
    戻り値:
        Tuple[Dict[str, str], Dict[str, List[str]]]:
            - var_to_group: 変数名 -> グループ名
            - group_to_vars: グループ名 -> そのグループに属する変数名リスト
    例:
        U850_ft3, U850_ft6 -> グループ 'U850'
        Prmsl_ft3, Prmsl_ft6 -> グループ 'Prmsl'
        Prec_Target_ft4, Prec_Target_ft5, Prec_Target_ft6, Prec_4_6h_sum, Prec_ft3 -> グループ 'precip'
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

def calculate_and_save_group_minmax(file_paths, target_variables, group_to_vars):
    """
    概要:
        全ファイル・全時刻・全格子点にまたがって、グループ単位のグローバル min/max（スカラー）を計算する。
        得られた min/max は以降の min-max 正規化に用いられる。
    引数:
        file_paths (List[pathlib.Path]):
            入力となる月次 NetCDF ファイルのパス一覧。
        target_variables (List[str]):
            正規化対象の物理変数名のリスト（時間特徴量や座標は含まない）。
        group_to_vars (Dict[str, List[str]]):
            グループ名 -> そのグループに属する変数名のリスト。
    処理:
        - xr.open_mfdataset により複数ファイルをまとめて開き、各変数について全域 min/max を計算（NaN を除外）。
        - 変数ごとの min/max をグループ内で統合し、グループ min = min(各変数の min)、グループ max = max(各変数の max)
          を採用して、グループ単位のスカラー min/max を決定する。
        - 数値が不定/同値の場合の安全対策（微小幅の付与）を行う。
    戻り値:
        Dict[str, Dict[str, float]]:
            group_minmax。キーはグループ名、値は {'min': float, 'max': float}。
    注意:
        - lat/lon ごとや時刻ごとのスケールは計算しない（空間・時間でスケールが変わらないことが目的）。
        - TIME_FEATURE_VARS はここでは扱わない（別途保持するが正規化しない）。
    """
    logging.info("--- Starting calculation of group-wise global min/max for min-max normalization ---")
    logging.info(f"Target files: {len(file_paths)}")

    group_minmax = {}

    with xr.open_mfdataset(file_paths, parallel=True, chunks={'time': 248}) as ds:
        ds_vars = ds[target_variables]

        # まず各変数ごとの全域 min/max を計算（time, lat, lon すべての次元にわたって）
        logging.info("Computing per-variable global min/max across (time, lat, lon)...")
        var_mins = {}
        var_maxs = {}
        with tqdm(total=len(target_variables), desc="変数ごとのmin/max集計", leave=False, file=os.sys.stderr) as pbar:
            for v in target_variables:
                if v not in ds_vars:
                    pbar.update(1)
                    continue
                vmin = float(ds_vars[v].min(skipna=True).compute().item())
                vmax = float(ds_vars[v].max(skipna=True).compute().item())
                # 安全対策: max==min の場合は微小幅を与える
                if not np.isfinite(vmin):
                    vmin = 0.0
                if not np.isfinite(vmax):
                    vmax = vmin + 1.0
                if abs(vmax - vmin) < 1e-12:
                    vmax = vmin + 1e-6
                var_mins[v] = vmin
                var_maxs[v] = vmax
                pbar.set_postfix_str(v)
                pbar.update(1)

        # グループ内の全変数の min の最小値、max の最大値をグループ min/max とする
        logging.info("Aggregating variable-wise min/max into group-wise min/max (scalars)...")
        for group, vars_in_group in group_to_vars.items():
            present = [v for v in vars_in_group if v in var_mins]
            if not present:
                # グループ内に対象変数が存在しない場合
                continue
            gmin = min(var_mins[v] for v in present)
            gmax = max(var_maxs[v] for v in present)
            # max==min 対策
            if abs(gmax - gmin) < 1e-12:
                gmax = gmin + 1e-6
            group_minmax[group] = {'min': float(gmin), 'max': float(gmax)}

    logging.info("✅ Finished computing group-wise min/max (global scalars).")
    return group_minmax

# ==============================================================================
# --- 単一ファイルの最適化（正規化+圧縮） ---
# ==============================================================================

def optimize_monthly_netcdf(file_path, output_path, scaler_json_path, var_to_group, target_variables):
    """
    概要:
        単一の月次 NetCDF ファイルに対して、グループ min-max 正規化を適用し、time チャンクと LZ4 圧縮を指定して書き出す。
    引数:
        file_path (pathlib.Path):
            入力となる月次 NetCDF ファイルのパス。
        output_path (pathlib.Path):
            最適化後の NetCDF を書き出すパス。
        scaler_json_path (str):
            group_minmax を含むメタ JSON（scaler_groups.json）のパス。
        var_to_group (Dict[str, str]):
            変数名 -> グループ名 の対応。
        target_variables (List[str]):
            正規化対象の物理変数名リスト。
    処理:
        - 既に出力が存在する場合はスキップ。
        - scaler_json を読み込み、各グループの min/max（スカラー）を取得。
        - xarray でファイルを開き、time 方向のチャンクを調整。
        - target_variables のうち DS に存在する変数に対して、x_norm = (x - min_g) / (max_g - min_g) を適用。
          TIME_FEATURE_VARS はデータとして保持するが正規化対象外。
        - 書き出し時、3次元以上で time を含む変数には (time_chunk, lat, lon) のチャンクと LZ4 圧縮を設定。
          2次元変数などは圧縮のみ設定。
        - h5netcdf エンジンで compute=False で書き出し、ProgressBar で実際の計算を実行。
    戻り値:
        str:
            "Success: ファイル名" / "Skipped: ファイル名" / "Failed: ファイル名" のいずれかのステータスメッセージ。
    例外:
        例外はキャッチしてログ出力し、"Failed: ..." を返す。
    """
    try:
        if output_path.exists():
            logging.info(f"スキップ: 既に存在します -> {output_path}")
            return f"Skipped: {file_path.name}"

        logging.info(f"処理開始: {file_path.name}")

        # グループmin/maxの読み込み
        with open(scaler_json_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        group_minmax = meta.get('group_minmax', {})
        if not group_minmax:
            logging.warning(f"group_minmax が '{scaler_json_path}' に見つかりません。正規化をスキップします。")

        with xr.open_dataset(file_path, chunks='auto') as ds:
            # 時間特徴量は保持（正規化対象からは除外）
            drop_vars = []
            kept_time_vars = [v for v in TIME_FEATURE_VARS if v in ds]
            if kept_time_vars:
                logging.info(f"時間特徴は保持（非正規化）: {kept_time_vars}")

            # 読み込みチャンクをtime方向に揃えて計算・I/O効率を改善
            ds = ds.chunk({'time': min(WRITE_TIME_CHUNK, ds.sizes.get('time', 1))})

            # 正規化（min-max）：グループのスカラー min/max を使用（格子点毎ではない）
            variables_to_norm = [v for v in target_variables if v in ds and v in var_to_group]
            logging.info(f"{file_path.name}: 正規化対象変数 = {len(variables_to_norm)} 個（除外: 時間特徴 {len(kept_time_vars)} 個）")
            with tqdm(total=len(variables_to_norm), desc=f"正規化中 {file_path.name}", leave=False, file=os.sys.stderr) as pbar:
                for var_name in variables_to_norm:
                    group = var_to_group[var_name]
                    gmm = group_minmax.get(group)
                    if gmm is None:
                        logging.warning(f"Min/Max for group '{group}' not found; skipping normalization for '{var_name}'")
                        pbar.update(1)
                        continue
                    gmin = float(gmm.get('min', 0.0))
                    gmax = float(gmm.get('max', 1.0))
                    denom = gmax - gmin
                    if denom <= 1e-12:
                        denom = 1.0
                        logging.warning(f"Group '{group}' max==min; using denom=1.0 for '{var_name}'")
                    ds[var_name] = (ds[var_name] - gmin) / denom
                    pbar.update(1)

            # 圧縮設定
            encoding = {}
            for var in ds.data_vars:
                # time 次元がある 3次元以上の変数にチャンクを設定
                if ds[var].ndim >= 3 and 'time' in ds[var].dims:
                    # (time=1, lat, lon) チャンク
                    time_len = int(ds.sizes.get('time', 1))
                    time_chunk = min(WRITE_TIME_CHUNK, time_len)
                    encoding[var] = {
                        **hdf5plugin.LZ4(),
                        'chunksizes': (time_chunk,) + ds[var].shape[1:]
                    }
                else:
                    # 2次元変数などはデフォルト圧縮のみにする
                    encoding[var] = {**hdf5plugin.LZ4()}

            logging.info(f"NetCDF 書き出し設定: 変数数={len(ds.data_vars)}, 圧縮設定対象={len(encoding)}")
            write_job = ds.to_netcdf(
                output_path,
                engine='h5netcdf',
                encoding=encoding,
                mode='w',
                compute=False
            )

            with ProgressBar():
                write_job.compute()

            logging.info(f"✅ 出力完了: {output_path.name}")
            return f"Success: {file_path.name}"

    except Exception as e:
        logging.error(f"❌ Failed to process {file_path.name}: {e}", exc_info=True)
        return f"Failed: {file_path.name}"

# ==============================================================================
# --- メイン実行部 ---
# ==============================================================================
def main():
    """
    概要:
        スクリプト全体の実行エントリポイント。グループ定義/スケーラ算出/各月ファイルの最適化を順に実施する。
    引数:
        なし（モジュール定数 START_YEAR/END_YEAR, 入出力ディレクトリ, 並列設定 などを利用）
    処理:
        1) ログ設定を初期化し、入出力ディレクトリやメタファイルのパスをログに出す。
        2) 先頭ファイルを開いて対象変数を抽出し、build_group_map でグループを構築。
        3) 既存の scaler_groups.json が現在のグループ構成と一致していれば再利用。そうでなければ
           calculate_and_save_group_minmax で全域 min/max を再計算し、メタを書き出す。
        4) 各月ファイルに対して、Dask の thread スケジューラ（pool=ThreadPool(MAX_WORKERS)）を用いて
           optimize_monthly_netcdf を呼び、正規化と圧縮・チャンク設定を適用して書き出す。
        5) 各工程と経過時間をログに記録する。
    戻り値:
        なし（副作用として optimization_nc/ に最適化済み NetCDF を出力し、scaler_groups.json を作成/更新する）
    """
    setup_logging()
    total_start_time = time.time()
    logging.info("===== NetCDF Optimization Process Start (Group-wise min-max normalization) =====")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.info(f"Input directory: {INPUT_DIR}")
    logging.info(f"Output directory: {OUTPUT_DIR}")
    logging.info(f"Scaler JSON: {SCALER_META}")
    # 数値演算ライブラリのスレッド数を制御（可能なら）
    os.environ.setdefault("OMP_NUM_THREADS", str(MAX_WORKERS))
    os.environ.setdefault("MKL_NUM_THREADS", str(MAX_WORKERS))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(MAX_WORKERS))

    file_paths = sorted([p for p in INPUT_DIR.glob("*.nc")])
    if not file_paths:
        logging.error(f"No NetCDF files found in {INPUT_DIR}. Exiting.")
        return

    # ターゲット変数とグループ決定（先頭ファイルで構造取得）
    with xr.open_dataset(file_paths[0]) as temp_ds:
        target_variables = get_target_variables(temp_ds)
        dims_info = {k: int(v) for k, v in temp_ds.dims.items()}
        logging.info(f"先頭ファイルの次元: {dims_info}")
        logging.info(f"座標: {list(temp_ds.coords)}")
    var_to_group, group_to_vars = build_group_map(target_variables)
    logging.info(f"Target variables for normalization ({len(target_variables)}): {target_variables}")
    logging.info(f"Number of groups: {len(group_to_vars)}")
    preview = {g: len(vs) for g, vs in list(group_to_vars.items())[:10]}
    logging.info(f"グループの例（最大10件）: {preview}")

    # グループ単位の min/max を計算（必要に応じて）
    need_compute = True
    group_minmax = {}
    if SCALER_META.exists():
        try:
            with open(SCALER_META, 'r', encoding='utf-8') as f:
                meta_existing = json.load(f)
            existing_groups = set(meta_existing.get('group_to_vars', {}).keys())
            current_groups = set(group_to_vars.keys())
            if 'group_minmax' in meta_existing and existing_groups == current_groups:
                group_minmax = meta_existing['group_minmax']
                need_compute = False
                logging.info(f"Found existing scaler JSON with group_minmax. Using: {SCALER_META}")
            else:
                logging.info("scaler_groups.json が古い/不整合のため再計算します。")
        except Exception as e:
            logging.warning(f"Failed to read existing scaler JSON. Recomputing. ({e})")

    if need_compute:
        logging.info("Calculating group-wise global min/max from scratch...")
        group_minmax = calculate_and_save_group_minmax(file_paths, target_variables, group_to_vars)
        # メタを書き出し（var_to_group, group_to_vars, group_minmax を併せて保存）
        try:
            with open(SCALER_META, 'w', encoding='utf-8') as f:
                json.dump(
                    {'var_to_group': var_to_group, 'group_to_vars': group_to_vars, 'group_minmax': group_minmax},
                    f, ensure_ascii=False, indent=2
                )
            logging.info(f"Saved scaler grouping + minmax meta: {SCALER_META}")
        except Exception as e:
            logging.warning(f"Failed to write scaler grouping meta: {e}")

    logging.info(f"\n--- Starting per-file optimization (sequential over files, parallel within file using {MAX_WORKERS} threads, local threads scheduler) ---")

    process_start_time = time.time()
    with tqdm(total=len(file_paths), desc="Optimizing files", file=os.sys.stderr) as pbar:
        for fp in file_paths:
            # ローカルthreadsスケジューラで1ファイル内並列（送信オーバーヘッドを回避）
            logging.info(f"[{fp.name}] Using local threads scheduler (threads={MAX_WORKERS})")
            with dask.config.set(scheduler='threads', pool=ThreadPool(MAX_WORKERS)):
                result = optimize_monthly_netcdf(
                    fp,
                    OUTPUT_DIR / fp.name,
                    str(SCALER_META),
                    var_to_group,
                    target_variables
                )
            logging.info(result)
            pbar.update(1)
            gc.collect()

    process_elapsed = time.time() - process_start_time
    logging.info(f"--- Per-file optimization finished. Time taken: {datetime.timedelta(seconds=int(process_elapsed))} ---")

    total_elapsed = time.time() - total_start_time
    logging.info(f"\n===== All processes finished. Total execution time: {datetime.timedelta(seconds=int(total_elapsed))} =====")

if __name__ == '__main__':
    # 必要パッケージ:
    #   pip install h5netcdf hdf5plugin dask xarray
    main()
