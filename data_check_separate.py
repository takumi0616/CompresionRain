#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
データ全体の統計把握と equal-split(3時間積算を3分割)の妥当性チェックを行うスクリプト

対象データ: ./optimization_nc 以下の月次netCDF (例: 201801.nc ...)
出力先: ./result_data_check_separate にログと図を保存

実装内容:
- 全ファイルをオンデマンドで順次処理（メモリ節約）
- 各時刻・各画素の 1時間降水 [Prec_Target_ft4, 5, 6] と 3時間積算 [Prec_4_6h_sum] を読み取り
- S = 3時間積算 > 0 の画素で比率 r = y / S を計算し、その分散 Var(r) をSのビンごとに集計
- equal-split ベースライン y_eq = S/3 に対する SSE, RMSE を全データで集計
- 0降水(乾燥)の画素数・時刻単位・日単位の統計を集計
- 最大降水（1h, 3h積算）の値・出現日時などを記録
- 図: x軸=3時間積算S (mm), y軸=Var(r) の平均（ビン平均）をプロット

注意:
- netCDFは大きいため、open_mfdatasetは使わず、各ファイルを個別に開いて time スライス単位で処理
- NaNは 0 に置換
"""

import os
import glob
import logging
from typing import List, Tuple, Dict
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
# Register HDF5 compression filters for h5py/h5netcdf
import hdf5plugin  # noqa: F401  # needed to load external HDF5 filters (zstd/blosc/etc.)
# For robustness in some environments
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
os.environ.setdefault("HDF5_DISABLE_VERSION_CHECK", "1")

# 入力/出力ディレクトリ
DATA_DIR = os.path.join(os.path.dirname(__file__), "optimization_nc")
OUT_DIR = os.path.join(os.path.dirname(__file__), "result_data_check_separate")

# 変数名
TARGET_VARS_1H = ["Prec_Target_ft4", "Prec_Target_ft5", "Prec_Target_ft6"]
TARGET_VAR_SUM = "Prec_4_6h_sum"

# 図・集計用のビン設定（3時間積算Sに対して）
# 気象的なレンジを意識したビン（必要に応じて変更可）
S_BINS = np.array([0.0, 0.1, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 60.0, 100.0, 150.0, 200.0, np.inf], dtype=np.float64)

# 数値安定用
EPS = 1e-6


def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("data_check")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def get_monthly_files(data_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(data_dir, "*.nc")))
    return files


def date_from_timeval(time_val) -> str:
    """
    xarray の time 値 (numpy.datetime64 等) から YYYY-MM-DD 文字列を返す
    """
    try:
        # np.datetime64 -> python datetime
        ts = np.datetime_as_string(time_val, unit="s")
        # 'YYYY-MM-DDTHH:MM:SS' -> 'YYYY-MM-DD'
        return ts.split("T")[0]
    except Exception:
        # 代替: pandas に頼らず、xarrayの to_pandas を避けたいので簡易処理
        return str(time_val)[:10]


def open_dataset_robust(path: str, logger: logging.Logger) -> xr.Dataset:
    """
    Try h5netcdf (with hdf5plugin) first to support external HDF5 filters.
    Fallback to netcdf4 engine if needed.
    """
    try:
        ds = xr.open_dataset(path, engine="h5netcdf")
        return ds
    except Exception as e1:
        logger.warning(f"h5netcdfで開けませんでした。netcdf4にフォールバックします: {path} ({e1})")
        try:
            ds = xr.open_dataset(path, engine="netcdf4", lock=False)
            return ds
        except Exception as e2:
            logger.error(f"netCDFを開けませんでした: {path} ({e2})")
            raise

def process_files(files: List[str], logger: logging.Logger):
    os.makedirs(OUT_DIR, exist_ok=True)

    # 集計用カウンタ
    n_files = 0
    total_time_steps = 0
    total_pixels = 0

    # 乾燥統計
    dry_pixel_count_S = 0  # S==0の画素カウント
    dry_timesteps_all_pixels = 0  # その時刻において全画素S==0（完全乾燥）の時刻数

    # 日単位の乾燥判定（全時間・全画素でS==0）
    # 日別合計（全画素・その日の全時刻のSの合計）を積算
    day_sum_S: Dict[str, float] = {}

    # equal-splitベースラインの誤差集計
    sse_equal_1h = 0.0  # Σ(y - S/3)^2 over all (3h*pix*time)
    count_equal_1h = 0  # 有効な(3*H*W)カウント（S>0のみ or S>=0? -> S>=0で量の誤差も意味はあるので全画素対象）
    # 正規化用に真値の二乗和（参照）
    sse_true_ref = 0.0  # Σ(y^2)

    # 比率分散 Var(r) のビン集計（S>0 のみ）
    K = len(S_BINS) - 1
    sum_var_r_per_bin = np.zeros(K, dtype=np.float64)
    cnt_var_r_per_bin = np.zeros(K, dtype=np.int64)

    # 最大降水（1h, 3h積算）
    max_1h_val = -np.inf
    max_1h_info = ("", -1)  # (file, time_idx)
    max_3h_val = -np.inf
    max_3h_info = ("", -1)

    logger.info(f"入力ディレクトリ: {DATA_DIR}")
    logger.info(f"対象ファイル数（検出）: {len(files)}")

    for fp in files:
        try:
            ds = open_dataset_robust(fp, logger)
        except Exception as e:
            logger.error(f"ファイルを開けませんでした: {fp} ({e})")
            continue

        n_files += 1
        T = int(ds.sizes["time"])
        H = int(ds.sizes["lat"])
        W = int(ds.sizes["lon"])
        total_time_steps += T
        total_pixels += T * H * W

        logger.info(f"[{os.path.basename(fp)}] time={T}, HxW={H}x{W}")

        # 逐次処理
        for ti in tqdm(range(T), desc=f"Processing {os.path.basename(fp)}", leave=False):
            # 必要な変数だけ読み込み
            try:
                y4 = np.nan_to_num(ds[TARGET_VARS_1H[0]].isel(time=ti).values).astype(np.float32, copy=False)
                y5 = np.nan_to_num(ds[TARGET_VARS_1H[1]].isel(time=ti).values).astype(np.float32, copy=False)
                y6 = np.nan_to_num(ds[TARGET_VARS_1H[2]].isel(time=ti).values).astype(np.float32, copy=False)
                S = np.nan_to_num(ds[TARGET_VAR_SUM].isel(time=ti).values).astype(np.float32, copy=False)
            except Exception as e:
                logger.warning(f"変数読み込みエラー: {fp} time={ti} ({e})")
                continue

            # 等分ベースラインの SSE, 参照二乗和
            y_stack = np.stack([y4, y5, y6], axis=0)  # (3,H,W)
            y_eq = S / 3.0
            # S>=0 全画素でカウント（S<0は通常ない前提）
            diff = y_stack - y_eq
            sse_equal_1h += float(np.sum(diff * diff))
            sse_true_ref += float(np.sum(y_stack * y_stack))
            count_equal_1h += int(3 * H * W)

            # S>0 のみで比率・分散を評価
            mask_pos = S > 0.0
            if np.any(mask_pos):
                S_pos = S[mask_pos]  # (N,)
                y_pos = y_stack[:, mask_pos]  # (3, N)
                r = y_pos / (S_pos + EPS)  # (3, N)
                # Var(r) をチャネル方向で計算
                var_r = np.var(r, axis=0)  # (N,)

                # S のビンへ集計
                bin_idx = np.digitize(S_pos, S_BINS, right=False) - 1  # 0..K-1
                valid = (bin_idx >= 0) & (bin_idx < K)
                if np.any(valid):
                    # ビンごとに和/カウントを加算
                    for b in range(K):
                        sel = (bin_idx == b)
                        if np.any(sel):
                            sum_var_r_per_bin[b] += float(np.sum(var_r[sel]))
                            cnt_var_r_per_bin[b] += int(np.sum(sel))

            # 最大降水の更新（1h と 3h積算）
            local_max_1h = float(np.max(y_stack))
            if local_max_1h > max_1h_val:
                max_1h_val = local_max_1h
                max_1h_info = (fp, ti)
            local_max_3h = float(np.max(S))
            if local_max_3h > max_3h_val:
                max_3h_val = local_max_3h
                max_3h_info = (fp, ti)

            # 乾燥統計
            dry_pixel_count_S += int(np.sum(S <= 0.0))
            if np.all(S <= 0.0):
                dry_timesteps_all_pixels += 1

            # 日別合計Sの集計（全画素合計）
            try:
                time_val = ds["time"].isel(time=ti).values
                date_key = date_from_timeval(time_val)
            except Exception:
                # 万一のため
                date_key = f"{os.path.basename(fp)}#t{ti}"
            day_sum_S[date_key] = day_sum_S.get(date_key, 0.0) + float(np.sum(S))

        # ファイルクローズ
        try:
            ds.close()
        except Exception:
            pass

    # まとめ統計の計算
    dry_pixel_ratio = dry_pixel_count_S / float(total_pixels) if total_pixels > 0 else np.nan
    dry_days = sum(1 for v in day_sum_S.values() if v <= 0.0)
    total_days = len(day_sum_S)

    # equal-splitベースラインの RMSE と正規化RMSE
    rmse_equal_1h = np.sqrt(sse_equal_1h / max(count_equal_1h, 1))
    # 参照スケールとして sqrt(mean(y^2)) を分母にした正規化RMSE
    ref_rms = np.sqrt(sse_true_ref / max(count_equal_1h, 1))
    nrmse_equal_1h = rmse_equal_1h / (ref_rms + EPS)

    # Var(r) のビン平均
    mean_var_r = np.zeros(K, dtype=np.float64)
    for b in range(K):
        if cnt_var_r_per_bin[b] > 0:
            mean_var_r[b] = sum_var_r_per_bin[b] / cnt_var_r_per_bin[b]
        else:
            mean_var_r[b] = np.nan

    # ログ出力
    log_lines = []
    log_lines.append("========== Data Check (Separate) ==========")
    log_lines.append(f"Files processed: {n_files}")
    log_lines.append(f"Total time steps: {total_time_steps}")
    log_lines.append(f"Total pixels (time*H*W): {total_pixels}")
    log_lines.append("")
    log_lines.append("---- Dryness ----")
    log_lines.append(f"Dry pixels (S==0): {dry_pixel_count_S}  ({dry_pixel_ratio*100:.2f} %)")
    log_lines.append(f"Timesteps fully dry (all pixels S==0): {dry_timesteps_all_pixels}")
    log_lines.append(f"Dry days (day_sum_S==0): {dry_days} / {total_days} days")
    log_lines.append("")
    log_lines.append("---- Max Precipitation ----")
    log_lines.append(f"Max 1h precipitation: {max_1h_val:.6f} mm at {os.path.basename(max_1h_info[0])} [time_index={max_1h_info[1]}]")
    log_lines.append(f"Max 3h accumulation: {max_3h_val:.6f} mm at {os.path.basename(max_3h_info[0])} [time_index={max_3h_info[1]}]")
    log_lines.append("")
    log_lines.append("---- Equal-split Baseline ----")
    log_lines.append("Definition: y_eq = S/3 for each of (ft+4, ft+5, ft+6)")
    log_lines.append(f"RMSE (mm): {rmse_equal_1h:.6f}")
    log_lines.append(f"Reference RMS (sqrt(mean(y^2))): {ref_rms:.6f}")
    log_lines.append(f"Normalized RMSE (RMSE / refRMS): {nrmse_equal_1h:.6f}")
    log_lines.append("")
    log_lines.append("---- Var(r) by S bins (S>0 pixels only) ----")
    for b in range(K):
        left = S_BINS[b]
        right = S_BINS[b + 1]
        if np.isinf(right):
            rng = f"[{left}, inf)"
        else:
            rng = f"[{left}, {right})"
        log_lines.append(f"S bin {b:02d} {rng}: count={cnt_var_r_per_bin[b]}, mean Var(r)={mean_var_r[b]:.6f}")

    # 保存
    with open(os.path.join(OUT_DIR, "data_check_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    for ln in log_lines:
        logger.info(ln)

    # 図: x=3時間積算S, y=Var(r) の平均（ビン）
    # ビンの代表値（中心）を作る
    x_vals = []
    for b in range(K):
        left = S_BINS[b]
        right = S_BINS[b + 1]
        if np.isinf(right):
            # 無限上限は少し右に開いた見せ方。代表値は 1.2*left とする（任意）
            x_vals.append(left * 1.2 if left > 0 else np.nan)
        else:
            x_vals.append((left + right) / 2.0)
    x_vals = np.array(x_vals, dtype=np.float64)

    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, mean_var_r, "o-", label="Mean Var(r) per S-bin")
    plt.xscale("log")
    plt.xlabel("3-hour accumulation S (mm) [log scale]")
    plt.ylabel("Variance of hourly ratios Var(r4,r5,r6)")
    plt.title("Deviation from equal-split (Var(r) vs 3h accumulation S)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    fig_path = os.path.join(OUT_DIR, "variance_vs_sum.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    logger.info(f"Saved figure: {fig_path}")

    # 追加でCSV出力（ビンごとの集計）
    import csv
    csv_path = os.path.join(OUT_DIR, "variance_vs_sum_bins.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerow(["bin_index", "S_left", "S_right", "count", "mean_var_r", "x_rep"])
        for b in range(K):
            left = S_BINS[b]
            right = S_BINS[b + 1]
            writer.writerow([b, left, right, int(cnt_var_r_per_bin[b]), float(mean_var_r[b]), float(x_vals[b])])
    logger.info(f"Saved CSV: {csv_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    log_path = os.path.join(OUT_DIR, "execution.log")
    logger = setup_logger(log_path)

    logger.info("データチェックを開始します")
    files = get_monthly_files(DATA_DIR)
    if not files:
        logger.error(f"対象ファイルが見つかりません: {DATA_DIR}")
        return

    process_files(files, logger)
    logger.info("データチェックが完了しました")


if __name__ == "__main__":
    main()
