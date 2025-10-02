#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全変数の健全性チェック と 降水量の整合性チェック（厳密検証）を行うスクリプト

対象データ: ./optimization_nc 以下の月次netCDF (201801.nc ... 202312.nc)
出力先: ./result_data_check_separate に日本語ログとCSVを保存

実装概要:
1) 全変数健全性チェック（各ファイル・各変数）
   - dtype, 次元と形状
   - _FillValue の有無と値
   - 有限値数 / NaN数 / Inf数
   - 有限値の min, max, mean, std
   - 近似分位点(1%,5%,50%,95%,99%)（ランダムサンプル採用）
   - 負値の個数（負値がある変数の把握）
   - 欠損地（全時刻で NaN が続く格子点数：降水量Sで報告）

2) 降水量の整合性チェック（各ファイル）
   - 変数名:
       1時間降水: Prec_Target_ft4, Prec_Target_ft5, Prec_Target_ft6
       3時間積算: Prec_4_6h_sum
   - チェック内容（NaN/Infは除外して判定）:
       a) 非負性: y4,y5,y6,S それぞれで負の値がないか
       b) 恒等性: S ?= y4 + y5 + y6 （絶対・相対許容誤差つき）
       c) 単調性: S >= max(y4,y5,y6) （非負であれば理論的に成り立つ）
       d) 時間合計の一致性: Σ_{i,j}(S) ?= Σ_{i,j}(y4+y5+y6)（各時刻）
       e) 1h最大と3h最大の関係性（上記の整合性が成り立つ場合、Max(3h) >= Max(1h) を期待）
   - 重大不一致（最大差・最大違反）の発生地点（ファイル名, time_idx, lat/lon値, 格子インデックス）を具体的にログ
   - 欠損地（全時刻でSがNaN）・常時ゼロ地（全時刻でS≒0）の格子点数を報告

注意:
- NaN/Inf をゼロに置換はしない（問題の見逃し防止）→ isfinite で厳密にフィルタ
- メモリ節約のため time 次元を小刻み（チャンク）に読み出し
- h5netcdf優先、失敗時は netcdf4 にフォールバック
"""

import os
import glob
import csv
import math
import logging
import argparse
from typing import List, Tuple, Dict, Optional

import numpy as np
import xarray as xr
from tqdm import tqdm

# 外部HDF5フィルタ対応（h5netcdfで必要な場合がある）
import hdf5plugin  # noqa: F401

# 一部環境でのHDF5ロック問題回避
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
os.environ.setdefault("HDF5_DISABLE_VERSION_CHECK", "1")

# 入出力
DATA_DIR = os.path.join(os.path.dirname(__file__), "optimization_nc")
OUT_DIR = os.path.join(os.path.dirname(__file__), "result_data_check_separate")

# 降水関連の変数名
PREC_VARS_1H = ["Prec_Target_ft4", "Prec_Target_ft5", "Prec_Target_ft6"]
PREC_VAR_3H = "Prec_4_6h_sum"

# 期待される変数のマッピング（ユーザー提供情報に基づく）
EXPECTED_VARIABLES = {
    "surface": [
        "Prmsl_ft3", "Prmsl_ft6",
        "T2m_ft3", "T2m_ft6",
        "U10m_ft3", "U10m_ft6",
        "V10m_ft3", "V10m_ft6",
    ],
    "975hPa": [
        "U975_ft3", "U975_ft6",
        "V975_ft3", "V975_ft6",
        "T975_ft3", "T975_ft6",
    ],
    "950hPa": [
        "U950_ft3", "U950_ft6",
        "V950_ft3", "V950_ft6",
        "T950_ft3", "T950_ft6",
    ],
    "925hPa": [
        "U925_ft3", "U925_ft6",
        "V925_ft3", "V925_ft6",
        "T925_ft3", "T925_ft6",
        "R925_ft3", "R925_ft6",
    ],
    "850hPa": [
        "U850_ft3", "U850_ft6",
        "V850_ft3", "V850_ft6",
        "T850_ft3", "T850_ft6",
        "R850_ft3", "R850_ft6",
    ],
    "500hPa": [
        "GH500_ft3", "GH500_ft6",
        "T500_ft3", "T500_ft6",
        "R500_ft3", "R500_ft6",
    ],
    "300hPa": [
        "GH300_ft3", "GH300_ft6",
        "U300_ft3", "U300_ft6",
        "V300_ft3", "V300_ft6",
    ],
    "precip": [
        "Prec_ft3",          # 0-3時間 積算降水
        "Prec_4_6h_sum",     # 3-6時間 積算降水（= S）
        "Prec_6_9h_sum",     # 6-9時間 積算降水（存在しない月があれば欠如として報告）
        "Prec_Target_ft4",   # 4時間目の降水（3-6時間窓の1時間値）
        "Prec_Target_ft5",   # 5時間目の降水
        "Prec_Target_ft6",   # 6時間目の降水
    ],
    "time_aux": ["dayofyear_sin", "dayofyear_cos", "hour_sin", "hour_cos"],
}
EXPECTED_LAT_RANGE = (23.0, 46.95)       # [min, max]
EXPECTED_LON_RANGE = (120.0, 149.9375)   # [min, max]
RANGE_TOL_DEG = 0.15  # 許容誤差（度）

# 数値安定・しきい値
EPS = 1e-12
# 恒等性を判定する絶対・相対許容誤差（mm）
TOL_ABS = 1e-3
TOL_REL = 1e-6  # |diff| <= TOL_ABS + TOL_REL * |S|

# 近似分位点サンプルサイズ上限
Q_SAMPLE_MAX = 200_000

# ロガー名
LOGGER_NAME = "data_check"


def setup_logger(log_path: str, console_level: int = logging.WARNING, file_level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    # Allow all messages; handlers control what actually gets emitted.
    logger.setLevel(logging.DEBUG)
    if logger.hasHandlers():
        logger.handlers.clear()
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    fh.setLevel(file_level)
    ch.setLevel(console_level)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def get_monthly_files(data_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(data_dir, "*.nc")))


def open_dataset_robust(path: str, logger: logging.Logger) -> xr.Dataset:
    try:
        return xr.open_dataset(path, engine="h5netcdf")
    except Exception as e1:
        logger.warning(f"h5netcdfで開けませんでした。netcdf4にフォールバックします: {path} ({e1})")
        try:
            return xr.open_dataset(path, engine="netcdf4", lock=False)
        except Exception as e2:
            logger.error(f"netCDFを開けませんでした: {path} ({e2})")
            raise


def dt64_to_str(dt64) -> str:
    """numpy.datetime64 などを ISO 文字列（分精度）へ"""
    try:
        return np.datetime_as_string(dt64, unit="m")
    except Exception:
        return str(dt64)


def unique_sorted(a: np.ndarray) -> List:
    return sorted(list(set(a.tolist())))


class StreamStats:
    """ストリーミングで統計量を集計するヘルパ"""
    def __init__(self, want_quantiles: bool = True, sample_max: int = Q_SAMPLE_MAX, rng_seed: int = 1234) -> None:
        self.count = 0  # 有限値カウント
        self.sum = 0.0
        self.sumsq = 0.0
        self.min_val = math.inf
        self.max_val = -math.inf
        self.nan_count = 0
        self.inf_count = 0
        self.neg_count = 0
        self.want_quantiles = want_quantiles
        self.sample_max = sample_max
        self._samples: Optional[np.ndarray] = None
        self._rng = np.random.default_rng(rng_seed)

    def push(self, arr: np.ndarray) -> None:
        if arr.size == 0:
            return
        finite = np.isfinite(arr)
        not_finite = ~finite
        if np.any(not_finite):
            self.nan_count += int(np.isnan(arr[not_finite]).sum())
            self.inf_count += int(np.isinf(arr[not_finite]).sum())
        if not np.any(finite):
            return
        x = arr[finite].astype(np.float64, copy=False)
        self.count += x.size
        self.sum += float(x.sum())
        self.sumsq += float(np.square(x, dtype=np.float64).sum())
        xmin = float(x.min())
        xmax = float(x.max())
        if xmin < self.min_val:
            self.min_val = xmin
        if xmax > self.max_val:
            self.max_val = xmax
        self.neg_count += int((x < 0).sum())

        if self.want_quantiles:
            if self._samples is None:
                # 初期充填
                if x.size <= self.sample_max:
                    self._samples = x.copy()
                else:
                    idx = self._rng.choice(x.size, size=self.sample_max, replace=False)
                    self._samples = x[idx]
            else:
                room = self.sample_max - self._samples.size
                if room > 0:
                    take = min(room, x.size)
                    idx = self._rng.choice(x.size, size=take, replace=False)
                    self._samples = np.concatenate([self._samples, x[idx]], axis=0)
                else:
                    # reservoir sampling (置換)
                    # 低コスト化のため、ここでは間引き置換ではなく小規模置換
                    replace = min(self.sample_max // 10, x.size)
                    if replace > 0:
                        idx_new = self._rng.choice(x.size, size=replace, replace=False)
                        idx_old = self._rng.choice(self.sample_max, size=replace, replace=False)
                        self._samples[idx_old] = x[idx_new]

    def result(self) -> Dict:
        mean = (self.sum / self.count) if self.count > 0 else math.nan
        var = (self.sumsq / self.count - mean * mean) if self.count > 0 else math.nan
        std = math.sqrt(max(var, 0.0)) if self.count > 0 else math.nan
        quantiles = {}
        if self.want_quantiles and self._samples is not None and self._samples.size > 0:
            qs = [0.01, 0.05, 0.50, 0.95, 0.99]
            try:
                vals = np.quantile(self._samples, qs)
                quantiles = {f"{int(q*100)}%": float(v) for q, v in zip(qs, vals)}
            except Exception:
                quantiles = {}
        return dict(
            finite_count=self.count,
            nan_count=self.nan_count,
            inf_count=self.inf_count,
            min=self.min_val if self.count > 0 else math.nan,
            max=self.max_val if self.count > 0 else math.nan,
            mean=mean,
            std=std,
            neg_count=self.neg_count,
            quantiles=quantiles,
        )


def compute_stream_stats_for_da(da: xr.DataArray, time_chunk: int = 8) -> Dict:
    """
    DataArray のストリーミング統計（メモリ節約）
    time 次元がなければ全体を一括で処理
    """
    st = StreamStats()
    dims = list(da.dims)
    if "time" not in dims:
        arr = da.values
        st.push(arr)
        return st.result()

    T = int(da.sizes["time"])
    for t0 in range(0, T, time_chunk):
        t1 = min(T, t0 + time_chunk)
        try:
            x = da.isel(time=slice(t0, t1)).values
        except Exception:
            x = da.isel(time=slice(t0, t1)).to_numpy()
        st.push(x)
    return st.result()


def check_time_axis(ds: xr.Dataset, logger: logging.Logger, fp: str) -> None:
    logger.info(f"[{os.path.basename(fp)}] 時間軸チェックを開始")
    if "time" not in ds:
        logger.warning("time 変数が見つかりません")
        return
    t = ds["time"].values
    n = t.shape[0]
    logger.info(f"  time 長さ: {n}")
    # メタ情報
    units = ds["time"].attrs.get("units", "")
    cal = ds["time"].attrs.get("calendar", "")
    logger.info(f"  time:units={units}, calendar={cal}")
    # 先頭・末尾
    try:
        t0s = dt64_to_str(ds["time"].isel(time=0).values)
        t1s = dt64_to_str(ds["time"].isel(time=n - 1).values)
        logger.info(f"  先頭時刻: {t0s}, 末尾時刻: {t1s}")
    except Exception:
        pass
    # 差分
    try:
        if np.issubdtype(t.dtype, np.datetime64):
            dt = np.diff(t).astype("timedelta64[m]").astype(np.int64)  # 分
            uniq = unique_sorted(dt)
            logger.info(f"  時間間隔のユニーク値（分）: {uniq[:10]}{' ...' if len(uniq) > 10 else ''}")
            acceptable = (60, 180)
            bad = [v for v in uniq if v not in acceptable]  # 60分または180分以外
            if bad:
                logger.warning(f"  警告: 60分または180分以外の間隔を検出: {bad}")
            else:
                if 180 in uniq and 60 not in uniq:
                    logger.info("  時間解像度: 3時間（180分）間隔であると判断しました")
                elif 60 in uniq and 180 not in uniq:
                    logger.info("  時間解像度: 1時間（60分）間隔であると判断しました")
        else:
            # 単調性のみチェック（例えば 'hours since ...' の整数時系列）
            diffs = np.diff(t.astype(np.int64))
            uniq = unique_sorted(diffs)
            logger.info(f"  time の差分（ユニーク）: {uniq[:10]}{' ...' if len(uniq) > 10 else ''}")
            if any(v <= 0 for v in uniq):
                logger.warning("  警告: time の非単調（後退/同一）を検出")
    except Exception as e:
        logger.warning(f"  時間差分の解析に失敗: {e}")


def check_all_variables_in_file(ds: xr.Dataset, logger: logging.Logger, fp: str, time_chunk: int = 8) -> Dict[str, Dict]:
    """
    各変数の健全性統計を収集し、ログ出力。
    戻り値: var_name -> 統計辞書
    """
    logger.info(f"[{os.path.basename(fp)}] 全変数の健全性チェックを開始")
    stats_per_var: Dict[str, Dict] = {}

    # 座標 lat/lon の基本情報
    lat_vals = ds["lat"].values if "lat" in ds else None
    lon_vals = ds["lon"].values if "lon" in ds else None
    if lat_vals is not None and lon_vals is not None:
        logger.info(f"  格子: lat={len(lat_vals)}, lon={len(lon_vals)}")

    for v in ds.data_vars:
        da = ds[v]
        dtype = str(da.dtype)
        shape = tuple(int(da.sizes[d]) for d in da.dims)
        fillv = da.attrs.get("_FillValue", None)
        logger.info(f"  変数: {v}, dtype={dtype}, dims={da.dims}, shape={shape}, _FillValue={fillv}")

        st = compute_stream_stats_for_da(da, time_chunk=time_chunk)
        stats_per_var[v] = st

        # ログ: 要点
        logger.info(
            f"    有限値={st['finite_count']:,} / NaN={st['nan_count']:,} / Inf={st['inf_count']:,} "
            f"/ 負値={st['neg_count']:,}"
        )
        logger.info(
            f"    min={st['min']:.6f}, max={st['max']:.6f}, mean={st['mean']:.6f}, std={st['std']:.6f}"
        )
        if st["quantiles"]:
            qtxt = ", ".join([f"{k}={v:.6f}" for k, v in st["quantiles"].items()])
            logger.info(f"    近似分位点: {qtxt}")

    return stats_per_var


def verify_variable_mapping(ds: xr.Dataset, logger: logging.Logger, fp: str, verbose: bool = False) -> Dict:
    """
    ユーザー提供の仕様（変数→GRIB2由来/レベル/予報時刻）に対し、
    ファイル内の変数が期待通り存在するか・座標/格子/時間軸の整合性を検査
    """
    logger.info(f"[{os.path.basename(fp)}] 仕様との変数マッピング検査を開始")
    actual_vars = set(ds.data_vars.keys())
    expected_all = set(v for grp in EXPECTED_VARIABLES.values() for v in grp)

    present_expected = sorted(expected_all & actual_vars)
    missing_expected = sorted(expected_all - actual_vars)
    extra_vars = sorted(actual_vars - expected_all)

    # ft3/ft6のカバレッジ
    ft3_found = sorted([v for v in present_expected if v.endswith("_ft3")])
    ft6_found = sorted([v for v in present_expected if v.endswith("_ft6")])
    targets_found = [v for v in present_expected if v.startswith("Prec_Target_ft")]
    has_prec_6_9 = ("Prec_6_9h_sum" in actual_vars)

    # 座標範囲
    lat_min = lat_max = lon_min = lon_max = math.nan
    lat_ok = lon_ok = False
    grid_ok = False
    grid_h = grid_w = None

    if "lat" in ds and "lon" in ds:
        try:
            lat = ds["lat"].values
            lon = ds["lon"].values
            lat_min = float(np.nanmin(lat))
            lat_max = float(np.nanmax(lat))
            lon_min = float(np.nanmin(lon))
            lon_max = float(np.nanmax(lon))
            lat_ok = (lat_min >= (EXPECTED_LAT_RANGE[0] - RANGE_TOL_DEG)) and (lat_max <= (EXPECTED_LAT_RANGE[1] + RANGE_TOL_DEG))
            lon_ok = (lon_min >= (EXPECTED_LON_RANGE[0] - RANGE_TOL_DEG)) and (lon_max <= (EXPECTED_LON_RANGE[1] + RANGE_TOL_DEG))
        except Exception:
            pass

    if "lat" in ds.sizes and "lon" in ds.sizes:
        grid_h = int(ds.sizes["lat"])
        grid_w = int(ds.sizes["lon"])
        grid_ok = (grid_h == 480 and grid_w == 480)

    # 時間軸のダイジェスト
    time_minutes = []
    try:
        t = ds["time"].values
        if np.issubdtype(t.dtype, np.datetime64):
            dt = np.diff(t).astype("timedelta64[m]").astype(np.int64)
            uniq = unique_sorted(dt)
            time_minutes = uniq
        else:
            diffs = np.diff(t.astype(np.int64))
            time_minutes = unique_sorted(diffs)
    except Exception:
        pass

    # ログ出力
    logger.info("  期待される変数カテゴリごとの存在状況")
    for cat, vars_ in EXPECTED_VARIABLES.items():
        found = [v for v in vars_ if v in actual_vars]
        miss = [v for v in vars_ if v not in actual_vars]
        logger.info(f"    - {cat}: 存在 {len(found)} 個 / 欠如 {len(miss)} 個")
        if miss and verbose:
            logger.info(f"      欠如: {', '.join(miss[:20])}{' ...' if len(miss) > 20 else ''}")

    if extra_vars:
        logger.info(f"  期待外の追加変数: {len(extra_vars)} 個（例）: {', '.join(extra_vars[:20])}{' ...' if len(extra_vars) > 20 else ''}")
    logger.info(f"  ft3 変数の検出数: {len(ft3_found)}, ft6 変数の検出数: {len(ft6_found)}, 目的変数(Prec_Target_ft[4-6])の検出数: {len(targets_found)}")
    logger.info(f"  Prec_6_9h_sum の存在: {'あり' if has_prec_6_9 else 'なし'}")
    if not has_prec_6_9:
        logger.info("    注意: 仕様上は 6-9時間積算(Prec_6_9h_sum)が想定されていますが、当該ファイルでは見つかりませんでした。")

    if verbose:
        # 期待変数の属性プレビュー（units/long_name/standard_name/description/comments）
        # netCDFにGRIB2由来の詳細メタが保存されていない場合が多いため、存在すれば確認ログとして出力
        logger.info("  期待変数の属性プレビュー（units/long_name/standard_name/description/comments）")
        for v in present_expected:
            try:
                attrs = ds[v].attrs
            except Exception:
                attrs = {}
            units = attrs.get("units", "")
            long_name = attrs.get("long_name", "")
            standard_name = attrs.get("standard_name", "")
            description = attrs.get("description", "")
            comments = attrs.get("comments", "")
            logger.info(
                f"    - {v}: units='{units}', long_name='{long_name}', "
                f"standard_name='{standard_name}', description='{description}', comments='{comments}'"
            )

    logger.info("  座標と格子の検査")
    logger.info(f"    - lat範囲: [{lat_min:.5f}, {lat_max:.5f}] 期待[{EXPECTED_LAT_RANGE[0]}, {EXPECTED_LAT_RANGE[1]}] 判定: {'OK' if lat_ok else 'NG'}")
    logger.info(f"    - lon範囲: [{lon_min:.5f}, {lon_max:.5f}] 期待[{EXPECTED_LON_RANGE[0]}, {EXPECTED_LON_RANGE[1]}] 判定: {'OK' if lon_ok else 'NG'}")
    logger.info(f"    - 格子サイズ: lat={grid_h}, lon={grid_w}（期待 480×480） 判定: {'OK' if grid_ok else 'NG'}")

    if time_minutes:
        logger.info(f"  time間隔のユニーク（分）: {time_minutes[:10]}{' ...' if len(time_minutes) > 10 else ''}（3時間=180分 / 1時間=60分）")

    return dict(
        present_expected_count=len(present_expected),
        missing_count=len(missing_expected),
        extra_count=len(extra_vars),
        ft3_found_count=len(ft3_found),
        ft6_found_count=len(ft6_found),
        targets_found_count=len(targets_found),
        has_prec_6_9h_sum=bool(has_prec_6_9),
        lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max,
        lat_range_ok=bool(lat_ok), lon_range_ok=bool(lon_ok),
        grid_h=grid_h, grid_w=grid_w, grid_ok=bool(grid_ok),
        time_interval_minutes=time_minutes,
        missing_names=missing_expected[:50],  # 先頭50のみ保持
        extra_names=extra_vars[:50],
    )

def scan_time_lag_consistency(ds: xr.Dataset, logger: logging.Logger, fp: str, lags=range(-3, 4), verbose: bool = False) -> Dict:
    """
    S(=Prec_4_6h_sum) と (y4+y5+y6) の時間ラグ整合性をスキャンして評価する
    - 各ラグごとに |S - Σy| の平均と、許容誤差内割合を算出
    - ここでは配列をdaskチャンク（可能なら）にして計算負荷を抑える
    戻り値: dict(best_lag, best_mean_abs, scan=[{lag, mean_abs_diff, ok_ratio, total}...])
    """
    if (PREC_VAR_3H not in ds) or any(v not in ds for v in PREC_VARS_1H):
        return {}

    S = ds[PREC_VAR_3H]
    y4 = ds[PREC_VARS_1H[0]]
    y5 = ds[PREC_VARS_1H[1]]
    y6 = ds[PREC_VARS_1H[2]]

    # 可能ならチャンク（dask）化
    try:
        tsize = int(S.sizes.get("time", 1))
        H = int(S.sizes.get("lat", 1))
        W = int(S.sizes.get("lon", 1))
        chunk = dict(time=min(32, tsize), lat=min(128, H), lon=min(128, W))
        S = S.chunk(chunk)
        y4 = y4.chunk(chunk)
        y5 = y5.chunk(chunk)
        y6 = y6.chunk(chunk)
    except Exception:
        pass

    results = []
    best = None

    if verbose:
        logger.info(f"[{os.path.basename(fp)}] 時間ずれスキャン（S と y4+y5+y6 の整合）")
    for lag in lags:
        try:
            Ysum = y4.shift(time=lag) + y5.shift(time=lag) + y6.shift(time=lag)
            mask = np.isfinite(S) & np.isfinite(Ysum)
            diff = (S - Ysum).where(mask)
            abs_diff = np.abs(diff)

            # 平均絶対誤差
            try:
                mean_abs = float(abs_diff.mean().compute().item())
            except Exception:
                mean_abs = float(abs_diff.mean().values)

            # 許容内割合
            tol = (TOL_ABS + TOL_REL * np.abs(S.where(mask)))
            ok = (abs_diff <= tol)
            try:
                ok_count = int(ok.sum().compute().item())
                total = int(mask.sum().compute().item())
            except Exception:
                ok_count = int(ok.sum().values)
                total = int(mask.sum().values)
            ok_ratio = (ok_count / total) if total > 0 else 0.0

            results.append(dict(lag=int(lag), mean_abs_diff=mean_abs, ok_ratio=ok_ratio, total=total))
            if verbose:
                logger.info(f"  - ラグ {int(lag):+d}: |差|平均={mean_abs:.6f} mm, 許容内割合={ok_ratio*100:.2f}% (比較対象ピクセル={total:,})")
        except Exception as e:
            if verbose:
                logger.warning(f"  - ラグ {int(lag):+d}: 計算に失敗: {e}")
            else:
                logger.debug(f"  - ラグ {int(lag):+d}: 計算に失敗: {e}")

    if results:
        results_sorted = sorted(results, key=lambda d: (d["mean_abs_diff"], -d["ok_ratio"]))
        best = results_sorted[0]
        logger.info(f"  -> 最良ラグ: {best['lag']:+d} （|差|平均={best['mean_abs_diff']:.6f} mm, 許容内割合={best['ok_ratio']*100:.2f}%）")
        return dict(best_lag=best["lag"], best_mean_abs=best["mean_abs_diff"], scan=results)
    else:
        logger.info("  -> ラグ評価の結果が得られませんでした")
        return dict(best_lag=None, best_mean_abs=math.nan, scan=[])

def check_precip_strict_same_time(ds: xr.Dataset, logger: logging.Logger, fp: str, verbose: bool = False) -> Dict:
    """
    時刻座標を厳密に一致させた上で S ?= y4+y5+y6 を検証する（同一timeインデックス検証）
    """
    for name in PREC_VARS_1H + [PREC_VAR_3H]:
        if name not in ds:
            logger.error(f"  必須変数が見つかりません: {name}")
            return {}

    # 次元順をそろえ、共通の time/lat/lon でアライン
    try:
        Sa = ds[PREC_VAR_3H].transpose("time", "lat", "lon")
        y4 = ds[PREC_VARS_1H[0]].transpose("time", "lat", "lon")
        y5 = ds[PREC_VARS_1H[1]].transpose("time", "lat", "lon")
        y6 = ds[PREC_VARS_1H[2]].transpose("time", "lat", "lon")
    except Exception:
        Sa = ds[PREC_VAR_3H]
        y4 = ds[PREC_VARS_1H[0]]
        y5 = ds[PREC_VARS_1H[1]]
        y6 = ds[PREC_VARS_1H[2]]

    try:
        y4a, y5a, y6a, Sa = xr.align(y4, y5, y6, Sa, join="inner")
    except Exception as e:
        logger.error(f"  同時刻アラインメントに失敗: {e}")
        return {}

    H = int(Sa.sizes.get("lat", 1))
    W = int(Sa.sizes.get("lon", 1))

    # 計算
    Ysum = y4a + y5a + y6a
    finite_all = np.isfinite(Sa) & np.isfinite(y4a) & np.isfinite(y5a) & np.isfinite(y6a)
    diff = (Sa - Ysum).where(finite_all)
    abs_diff = np.abs(diff)

    def _compute_scalar(x):
        try:
            return float(x.compute().item())
        except Exception:
            try:
                return float(x.values)
            except Exception:
                return float(x)

    total_ok_px = int(_compute_scalar(finite_all.sum()))
    sum_abs_diff = _compute_scalar(abs_diff.sum())
    mean_abs_diff = (sum_abs_diff / total_ok_px) if total_ok_px > 0 else math.nan
    rmse = _compute_scalar(np.sqrt((np.square(diff)).mean()))

    tol = (TOL_ABS + TOL_REL * np.abs(Sa.where(finite_all)))
    ok = (abs_diff <= tol)
    count_diff_ok = int(_compute_scalar(ok.sum()))
    count_diff_bad = total_ok_px - count_diff_ok

    # 単調性
    y_max = xr.concat([y4a, y5a, y6a], dim="comp").max("comp")
    mono_margin = (Sa - y_max).where(finite_all)
    bad_mono = mono_margin < -(TOL_ABS + TOL_REL * np.abs(Sa.where(finite_all)))
    count_S_lt_maxY = int(_compute_scalar(bad_mono.sum()))

    # 時刻ごとの空間合計
    S_masked = Sa.where(finite_all)
    Y_masked = Ysum.where(finite_all)
    sumS_t = S_masked.sum(dim=("lat", "lon"), skipna=True)
    sumY_t = Y_masked.sum(dim=("lat", "lon"), skipna=True)
    d_t = sumS_t - sumY_t
    bad_t = np.abs(d_t) > (TOL_ABS * H * W)
    sumS_vs_sumY_bad = int(_compute_scalar(bad_t.sum()))

    # 欠損地・常時ゼロ地
    always_nan_S = ~(np.isfinite(Sa)).any(dim="time")
    always_zero_S = ~(np.isfinite(Sa) & (np.abs(Sa) > TOL_ABS)).any(dim="time")
    always_nan_S_count = int(_compute_scalar(always_nan_S.sum()))
    always_zero_S_count = int(_compute_scalar(always_zero_S.sum()))

    # 極値
    max_1h_val = _compute_scalar(xr.concat([y4a, y5a, y6a], dim="comp").max())
    max_3h_val = _compute_scalar(Sa.max())

    res = dict(
        neg_counts={},  # 省略（詳細はverboseパスで収集）
        alignment_best_lag=None,
        alignment_best_lag_mean_abs_diff=math.nan,
        total_ok_px=total_ok_px,
        sum_abs_diff=sum_abs_diff,
        mean_abs_diff=mean_abs_diff,
        rmse_S_vs_sumY=rmse,
        count_diff_ok=count_diff_ok,
        count_diff_bad=count_diff_bad,
        max_abs_diff=_compute_scalar(abs_diff.max()),
        max_abs_diff_info=None,
        count_S_lt_maxY=count_S_lt_maxY,
        worst_S_minus_maxY=0.0,
        worst_S_minus_maxY_info=None,
        sumS_vs_sumY_bad=sumS_vs_sumY_bad,
        sumS_vs_sumY_worst=0.0,
        sumS_vs_sumY_worst_ti=None,
        always_nan_S_count=always_nan_S_count,
        always_zero_S_count=always_zero_S_count,
        grid_H=H, grid_W=W,
        max_1h_val=max_1h_val,
        max_1h_info=None,
        max_3h_val=max_3h_val,
        max_3h_info=None,
    )

    # ログ（簡潔）
    logger.info(f"[{os.path.basename(fp)}] 降水量整合性チェック 結果（同時刻厳密検証）")
    logger.info("  b) 恒等性 S ≈ y4+y5+y6 の確認（同一time）")
    logger.info(f"    - 比較対象(有限値)ピクセル: {total_ok_px:,} 個")
    logger.info(f"    - 絶対誤差の平均 |S - Σy|: {mean_abs_diff:.6f} mm")
    logger.info(f"    - RMSE(S, Σy): {rmse:.6f} mm")
    logger.info(f"    - 許容内(OK): {count_diff_ok:,} / 許容外(NG): {count_diff_bad:,}")
    logger.info(f"  d) 時間ごとの空間合計 Σ(S) と Σ(y4+y5+y6) の一致性")
    logger.info(f"    - 不一致(許容外)の時刻数: {sumS_vs_sumY_bad:,}")
    logger.info("  e) 極値（最大降水）")
    logger.info(f"    - 1時間降水の最大: {max_1h_val:.6f} mm")
    logger.info(f"    - 3時間積算の最大: {max_3h_val:.6f} mm")

    return res

def check_precip_in_file(ds: xr.Dataset, logger: logging.Logger, fp: str, time_chunk: int = 4, verbose: bool = False) -> Dict:
    """
    降水量整合性チェック（各ファイル）
    戻り値: 主要集計の辞書
    """
    for name in PREC_VARS_1H + [PREC_VAR_3H]:
        if name not in ds:
            logger.error(f"  必須変数が見つかりません: {name}")
            return {}

    T = int(ds.sizes["time"])
    H = int(ds.sizes["lat"])
    W = int(ds.sizes["lon"])

    lat_vals = ds["lat"].values if "lat" in ds else None
    lon_vals = ds["lon"].values if "lon" in ds else None

    # 厳密な同時刻検証: 全変数の time 座標が一致していない場合はアラインして厳密検証
    try:
        tS = ds[PREC_VAR_3H]["time"].values
        t4 = ds[PREC_VARS_1H[0]]["time"].values
        t5 = ds[PREC_VARS_1H[1]]["time"].values
        t6 = ds[PREC_VARS_1H[2]]["time"].values
        times_equal = np.array_equal(tS, t4) and np.array_equal(tS, t5) and np.array_equal(tS, t6)
    except Exception:
        times_equal = False
    if not times_equal:
        logger.info("  時刻座標が一致していません。共通の時刻集合で厳密検証を実施します。")
        return check_precip_strict_same_time(ds, logger, fp, verbose=verbose)

    # 時間ラグ整合性のスキャン（小さなラグ範囲）
    align_summary = scan_time_lag_consistency(ds, logger, fp, lags=range(-3, 4), verbose=verbose)

    # 集計
    neg_counts = {k: 0 for k in (PREC_VARS_1H + [PREC_VAR_3H])}
    total_ok_px = 0
    sum_abs_diff = 0.0
    sum_sqr_diff = 0.0
    count_diff_ok = 0
    count_diff_bad = 0
    max_abs_diff = 0.0
    max_abs_diff_info = None  # (ti, i, j, S, y4,y5,y6, diff)

    count_S_lt_maxY = 0
    worst_S_minus_maxY = 0.0  # 最小（負側）→ 一番ひどい S - maxY
    worst_S_minus_maxY_info = None

    # 時間合計チェック
    sumS_vs_sumY_bad = 0
    sumS_vs_sumY_worst = 0.0
    sumS_vs_sumY_worst_ti = None

    # 欠損・常時ゼロ
    always_nan_S = np.ones((H, W), dtype=bool)
    always_zero_S = np.ones((H, W), dtype=bool)

    # 極値
    max_1h_val = -math.inf
    max_1h_info = None  # (ti, which_of_3, i, j, val)
    max_3h_val = -math.inf
    max_3h_info = None  # (ti, i, j, val)

    for t0 in range(0, T, time_chunk):
        t1 = min(T, t0 + time_chunk)

        # 読み込み
        y4 = ds[PREC_VARS_1H[0]].isel(time=slice(t0, t1)).values.astype(np.float32, copy=False)
        y5 = ds[PREC_VARS_1H[1]].isel(time=slice(t0, t1)).values.astype(np.float32, copy=False)
        y6 = ds[PREC_VARS_1H[2]].isel(time=slice(t0, t1)).values.astype(np.float32, copy=False)
        S = ds[PREC_VAR_3H].isel(time=slice(t0, t1)).values.astype(np.float32, copy=False)

        # 負値カウント（有限のみ）
        for name, arr in zip(PREC_VARS_1H + [PREC_VAR_3H], [y4, y5, y6, S]):
            finite = np.isfinite(arr)
            neg_counts[name] += int((arr[finite] < 0).sum())

        # always_nan / always_zero 更新（S）
        # tブロック → 時間方向に any で集約して、全期間条件を更新
        finite_S_blk = np.isfinite(S)  # (tb,H,W)
        any_finite_blk = np.any(finite_S_blk, axis=0)  # (H,W)
        always_nan_S &= ~any_finite_blk

        # 常時ゼロ: このブロック内の S がすべて有限かつほぼゼロであることが必要
        # → 1つでも (finite かつ |S|>TOL) があれば常時ゼロ条件は崩れる
        fin_and_nonzero_blk = np.any(np.isfinite(S) & (np.abs(S) > TOL_ABS), axis=0)
        always_zero_S &= ~fin_and_nonzero_blk

        # 恒等性・単調性・時間合計チェック
        ysum = (y4 + y5 + y6)
        finite_all = np.isfinite(S) & np.isfinite(y4) & np.isfinite(y5) & np.isfinite(y6)
        if np.any(finite_all):
            S_ok = S[finite_all]
            Y_ok = ysum[finite_all]
            diff = S_ok - Y_ok
            abs_diff = np.abs(diff)
            # RMSE用（二乗誤差の合計）
            sum_sqr_diff += float(np.square(diff, dtype=np.float64).sum())

            total_ok_px += int(S_ok.size)
            sum_abs_diff += float(abs_diff.sum())

            # 許容誤差判定
            tol_ok = abs_diff <= (TOL_ABS + TOL_REL * np.abs(S_ok))
            count_diff_ok += int(tol_ok.sum())
            count_diff_bad += int((~tol_ok).sum())

            # 最大差
            if abs_diff.size > 0:
                loc = int(abs_diff.argmax())
                max_loc_val = float(abs_diff[loc])
                if max_loc_val > max_abs_diff:
                    max_abs_diff = max_loc_val
                    # 元のインデックスへ
                    flat_idx = np.flatnonzero(finite_all.ravel())[loc]
                    tb, ii, jj = np.unravel_index(flat_idx, finite_all.shape)
                    ti = t0 + tb
                    max_abs_diff_info = dict(
                        file=os.path.basename(fp),
                        time_index=int(ti),
                        time=dt64_to_str(ds["time"].isel(time=ti).values),
                        i=int(ii),
                        j=int(jj),
                        lat=float(lat_vals[ii]) if lat_vals is not None else math.nan,
                        lon=float(lon_vals[jj]) if lon_vals is not None else math.nan,
                        S=float(S[ti - t0, ii, jj]),
                        y4=float(y4[ti - t0, ii, jj]),
                        y5=float(y5[ti - t0, ii, jj]),
                        y6=float(y6[ti - t0, ii, jj]),
                        diff=float(S[ti - t0, ii, jj] - ysum[ti - t0, ii, jj]),
                    )

            # 単調性: S >= max(y4,y5,y6) を期待
            ymax = np.maximum.reduce([y4, y5, y6])[finite_all]
            mono_margin = S_ok - ymax
            bad_mono = mono_margin < - (TOL_ABS + TOL_REL * np.abs(S_ok))
            bad_mono_count = int(bad_mono.sum())
            count_S_lt_maxY += bad_mono_count
            if bad_mono_count > 0:
                worst = float(mono_margin.min())  # 最小（負側）
                if worst < worst_S_minus_maxY:
                    worst_S_minus_maxY = worst
                    # 位置特定
                    # 最小値の場所（有限の中で）
                    worst_idx_local = int(mono_margin.argmin())
                    flat_idx_all = np.flatnonzero(finite_all.ravel())[worst_idx_local]
                    tb2, ii2, jj2 = np.unravel_index(flat_idx_all, finite_all.shape)
                    ti2 = t0 + tb2
                    worst_S_minus_maxY_info = dict(
                        file=os.path.basename(fp),
                        time_index=int(ti2),
                        time=dt64_to_str(ds["time"].isel(time=ti2).values),
                        i=int(ii2),
                        j=int(jj2),
                        lat=float(lat_vals[ii2]) if lat_vals is not None else math.nan,
                        lon=float(lon_vals[jj2]) if lon_vals is not None else math.nan,
                        S=float(S[ti2 - t0, ii2, jj2]),
                        y4=float(y4[ti2 - t0, ii2, jj2]),
                        y5=float(y5[ti2 - t0, ii2, jj2]),
                        y6=float(y6[ti2 - t0, ii2, jj2]),
                        S_minus_maxY=float(worst),
                    )

        # 時間合計（空間積算）の一致性
        for tb in range(S.shape[0]):
            ti = t0 + tb
            S_t = S[tb]
            Y_t = ysum[tb]
            finite_t = np.isfinite(S_t) & np.isfinite(Y_t)
            if not np.any(finite_t):
                continue
            sumS = float(S_t[finite_t].sum())
            sumY = float(Y_t[finite_t].sum())
            d = sumS - sumY
            if abs(d) > (TOL_ABS * H * W):  # 空間合計なので許容をH*W倍程度に
                sumS_vs_sumY_bad += 1
                if abs(d) > abs(sumS_vs_sumY_worst):
                    sumS_vs_sumY_worst = d
                    sumS_vs_sumY_worst_ti = int(ti)

        # 極値更新
        y_stack = np.stack([y4, y5, y6], axis=0)  # (3,tb,H,W)
        # 1h
        with np.errstate(invalid="ignore"):
            local_1h_max = float(np.nanmax(y_stack))
            if local_1h_max > max_1h_val:
                max_1h_val = local_1h_max
                # 位置
                idx = int(np.nanargmax(y_stack))
                c, tb, ii, jj = np.unravel_index(idx, y_stack.shape)
                ti = t0 + tb
                max_1h_info = dict(
                    file=os.path.basename(fp),
                    time_index=int(ti),
                    time=dt64_to_str(ds["time"].isel(time=ti).values),
                    which_of=("ft4","ft5","ft6")[c],
                    i=int(ii), j=int(jj),
                    lat=float(lat_vals[ii]) if lat_vals is not None else math.nan,
                    lon=float(lon_vals[jj]) if lon_vals is not None else math.nan,
                    val=float(y_stack[c, tb, ii, jj]),
                )
            # 3h
            local_3h_max = float(np.nanmax(S))
            if local_3h_max > max_3h_val:
                max_3h_val = local_3h_max
                idx = int(np.nanargmax(S))
                tb, ii, jj = np.unravel_index(idx, S.shape)
                ti = t0 + tb
                max_3h_info = dict(
                    file=os.path.basename(fp),
                    time_index=int(ti),
                    time=dt64_to_str(ds["time"].isel(time=ti).values),
                    i=int(ii), j=int(jj),
                    lat=float(lat_vals[ii]) if lat_vals is not None else math.nan,
                    lon=float(lon_vals[jj]) if lon_vals is not None else math.nan,
                    val=float(S[tb, ii, jj]),
                )

    # 結果まとめ
    res = dict(
        neg_counts=neg_counts,
        alignment_best_lag=align_summary.get("best_lag") if isinstance(align_summary, dict) else None,
        alignment_best_lag_mean_abs_diff=align_summary.get("best_mean_abs") if isinstance(align_summary, dict) else math.nan,
        total_ok_px=total_ok_px,
        sum_abs_diff=sum_abs_diff,
        mean_abs_diff=(sum_abs_diff / total_ok_px) if total_ok_px > 0 else math.nan,
        rmse_S_vs_sumY=(math.sqrt(sum_sqr_diff / total_ok_px) if total_ok_px > 0 else math.nan),
        count_diff_ok=count_diff_ok,
        count_diff_bad=count_diff_bad,
        max_abs_diff=max_abs_diff,
        max_abs_diff_info=max_abs_diff_info,
        count_S_lt_maxY=count_S_lt_maxY,
        worst_S_minus_maxY=worst_S_minus_maxY if count_S_lt_maxY > 0 else 0.0,
        worst_S_minus_maxY_info=worst_S_minus_maxY_info,
        sumS_vs_sumY_bad=sumS_vs_sumY_bad,
        sumS_vs_sumY_worst=sumS_vs_sumY_worst,
        sumS_vs_sumY_worst_ti=sumS_vs_sumY_worst_ti,
        always_nan_S_count=int(always_nan_S.sum()),
        always_zero_S_count=int(always_zero_S.sum()),
        grid_H=H, grid_W=W,
        max_1h_val=max_1h_val,
        max_1h_info=max_1h_info,
        max_3h_val=max_3h_val,
        max_3h_info=max_3h_info,
    )

    # ログ（日本語で分かりやすく）
    logger.info(f"[{os.path.basename(fp)}] 降水量整合性チェック 結果")
    logger.info("  a) 非負性の確認（負値の個数; 有限値のみカウント）")
    if verbose:
        for k, v in neg_counts.items():
            logger.info(f"    - {k}: 負値 {v:,} 個")
    total_neg = sum(neg_counts.values())
    logger.info(f"    - 負値の総数（全変数合計）: {total_neg:,} 個")

    logger.info("  b) 恒等性 S ≈ y4+y5+y6 の確認")
    logger.info(f"    - 比較対象(有限値)ピクセル: {total_ok_px:,} 個")
    logger.info(f"    - 絶対誤差の平均 |S - Σy|: {res['mean_abs_diff']:.6f} mm")
    logger.info(f"    - RMSE(S, Σy): {res['rmse_S_vs_sumY']:.6f} mm")
    logger.info(f"    - 許容内(OK): {count_diff_ok:,} / 許容外(NG): {count_diff_bad:,}")
    logger.info(f"    - 最大|S-Σy|: {max_abs_diff:.6f} mm")
    if max_abs_diff_info and verbose:
        mi = max_abs_diff_info
        logger.info(
            "      最大差の発生箇所: "
            f"file={mi['file']}, time_idx={mi['time_index']} ({mi['time']}), "
            f"(i,j)=({mi['i']},{mi['j']}), (lat,lon)=({mi['lat']:.5f},{mi['lon']:.5f}), "
            f"S={mi['S']:.6f}, y4={mi['y4']:.6f}, y5={mi['y5']:.6f}, y6={mi['y6']:.6f}, diff={mi['diff']:.6f}"
        )

    logger.info("  c) 単調性 S >= max(y4,y5,y6) の確認")
    logger.info(f"    - 破れたピクセル数: {count_S_lt_maxY:,}")
    if worst_S_minus_maxY_info and verbose:
        wi = worst_S_minus_maxY_info
        logger.info(
            "      最悪の S - max(y) (負): "
            f"{worst_S_minus_maxY:.6f} mm at file={wi['file']}, time_idx={wi['time_index']} ({wi['time']}), "
            f"(i,j)=({wi['i']},{wi['j']}), (lat,lon)=({wi['lat']:.5f},{wi['lon']:.5f}), "
            f"S={wi['S']:.6f}, y4={wi['y4']:.6f}, y5={wi['y5']:.6f}, y6={wi['y6']:.6f}"
        )

    logger.info("  d) 時間ごとの空間合計 Σ(S) と Σ(y4+y5+y6) の一致性")
    logger.info(f"    - 不一致(許容外)の時刻数: {sumS_vs_sumY_bad:,}")
    if sumS_vs_sumY_worst_ti is not None and verbose:
        tstr = dt64_to_str(ds['time'].isel(time=sumS_vs_sumY_worst_ti).values)
        logger.info(f"    - 最悪の合計差: {sumS_vs_sumY_worst:.6f} mm at time_idx={sumS_vs_sumY_worst_ti} ({tstr})")

    logger.info("  e) 極値（最大降水）")
    if max_1h_info:
        logger.info(
            f"    - 1時間降水の最大: {max_1h_val:.6f} mm at {max_1h_info['file']} "
            f"[time_index={max_1h_info['time_index']} {max_1h_info['time']}] "
            f"(which={max_1h_info['which_of']}, lat={max_1h_info['lat']:.5f}, lon={max_1h_info['lon']:.5f})"
        )
    if max_3h_info:
        logger.info(
            f"    - 3時間積算の最大: {max_3h_val:.6f} mm at {max_3h_info['file']} "
            f"[time_index={max_3h_info['time_index']} {max_3h_info['time']}] "
            f"(lat={max_3h_info['lat']:.5f}, lon={max_3h_info['lon']:.5f})"
        )
    if (max_1h_val > max_3h_val) and (count_diff_bad > 0 or count_S_lt_maxY > 0):
        logger.warning(
            "    - 注意: 1時間最大が3時間最大を上回っています。恒等性/単調性の破れがあるため、"
            "timeアラインメントの不整合や欠損値の扱いに問題がある可能性があります。"
        )

    # 欠損地・常時ゼロ地
    logger.info("  f) 欠損地・常時ゼロ地（S=3時間積算）")
    logger.info(f"    - 欠損地数（全時刻でNaN）: {res['always_nan_S_count']:,} / {H*W:,} グリッド")
    logger.info(f"    - 常時ゼロ地数（全時刻で≈0）: {res['always_zero_S_count']:,} / {H*W:,} グリッド")

    return res


def write_precip_summary_csv(path_csv: str, file_summaries: List[Dict]) -> None:
    """降水整合性チェックの要約をCSV出力（ファイル単位）"""
    fields = [
        "file",
        "total_ok_px",
        "mean_abs_diff",
        "rmse_S_vs_sumY",
        "count_diff_ok",
        "count_diff_bad",
        "max_abs_diff",
        "count_S_lt_maxY",
        "sumS_vs_sumY_bad",
        "always_nan_S_count",
        "always_zero_S_count",
        "alignment_best_lag",
        "alignment_best_lag_mean_abs_diff",
        "max_1h_val",
        "max_3h_val",
    ]
    with open(path_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in file_summaries:
            w.writerow({k: s.get(k, "") for k in fields})


def write_variable_mapping_summary_csv(path_csv: str, mapping_summaries: List[Dict]) -> None:
    """変数マッピング検査の要約をCSV出力（ファイル単位）"""
    fields = [
        "file",
        "present_expected_count",
        "missing_count",
        "extra_count",
        "ft3_found_count",
        "ft6_found_count",
        "targets_found_count",
        "has_prec_6_9h_sum",
        "lat_min", "lat_max", "lon_min", "lon_max",
        "lat_range_ok", "lon_range_ok",
        "grid_h", "grid_w", "grid_ok",
        # 欠如/追加の先頭数件（可読性のため）
        "missing_names_preview",
        "extra_names_preview",
    ]
    with open(path_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in mapping_summaries:
            row = {
                "file": s.get("file", ""),
                "present_expected_count": s.get("present_expected_count", 0),
                "missing_count": s.get("missing_count", 0),
                "extra_count": s.get("extra_count", 0),
                "ft3_found_count": s.get("ft3_found_count", 0),
                "ft6_found_count": s.get("ft6_found_count", 0),
                "targets_found_count": s.get("targets_found_count", 0),
                "has_prec_6_9h_sum": int(bool(s.get("has_prec_6_9h_sum", False))),
                "lat_min": s.get("lat_min", math.nan),
                "lat_max": s.get("lat_max", math.nan),
                "lon_min": s.get("lon_min", math.nan),
                "lon_max": s.get("lon_max", math.nan),
                "lat_range_ok": int(bool(s.get("lat_range_ok", False))),
                "lon_range_ok": int(bool(s.get("lon_range_ok", False))),
                "grid_h": s.get("grid_h", ""),
                "grid_w": s.get("grid_w", ""),
                "grid_ok": int(bool(s.get("grid_ok", False))),
                "missing_names_preview": ", ".join(s.get("missing_names", [])[:10]),
                "extra_names_preview": ", ".join(s.get("extra_names", [])[:10]),
            }
            w.writerow(row)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="詳細ログを出力（INFO/DEBUG相当）")
    parser.add_argument("--quiet", action="store_true", help="重要なメッセージのみ出力（WARNING以上）")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    log_path = os.path.join(OUT_DIR, "execution.log")

    # ログレベルの決定（デフォルト: コンソールはWARNING以上、ファイルはINFO以上）
    console_level = logging.WARNING
    file_level = logging.INFO
    if args.verbose:
        console_level = logging.INFO
        file_level = logging.INFO
    if args.quiet:
        console_level = logging.ERROR
        file_level = logging.WARNING

    logger = setup_logger(log_path, console_level=console_level, file_level=file_level)

    logger.info("データチェック（全変数健全性 → 降水量整合性）を開始します")
    logger.info(f"入力ディレクトリ: {DATA_DIR}")

    files = get_monthly_files(DATA_DIR)
    if not files:
        logger.error(f"対象ファイルが見つかりません: {DATA_DIR}")
        return
    logger.info(f"対象ファイル数（検出）: {len(files)}")

    # 期間全体の集計（降水に関する）
    global_max_1h_val = -math.inf
    global_max_1h_info = None
    global_max_3h_val = -math.inf
    global_max_3h_info = None

    file_summaries: List[Dict] = []
    mapping_summaries: List[Dict] = []

    for fp in files:
        try:
            ds = open_dataset_robust(fp, logger)
        except Exception:
            continue

        # ファイル基本情報
        dims_txt = ", ".join([f"{k}={v}" for k, v in ds.sizes.items()])
        logger.info(f"[{os.path.basename(fp)}] 基本情報: {dims_txt}")

        # 時間軸チェック
        check_time_axis(ds, logger, fp)

        # 変数マッピング検査（仕様との対応確認）
        mapping = verify_variable_mapping(ds, logger, fp, verbose=args.verbose)
        if mapping:
            mapping_summaries.append(dict(file=os.path.basename(fp), **mapping))

        # 全変数健全性チェック（詳細は冗長なため、デフォルトではスキップ。--verbose 時のみ出力）
        if args.verbose:
            _ = check_all_variables_in_file(ds, logger, fp)

        # 降水整合性チェック
        prec_res = check_precip_in_file(ds, logger, fp, verbose=args.verbose)
        if prec_res:
            # グローバル極値更新
            if prec_res["max_1h_val"] > global_max_1h_val:
                global_max_1h_val = prec_res["max_1h_val"]
                global_max_1h_info = prec_res.get("max_1h_info")
            if prec_res["max_3h_val"] > global_max_3h_val:
                global_max_3h_val = prec_res["max_3h_val"]
                global_max_3h_info = prec_res.get("max_3h_info")

            file_summaries.append(dict(
                file=os.path.basename(fp),
                total_ok_px=prec_res.get("total_ok_px", 0),
                mean_abs_diff=prec_res.get("mean_abs_diff", math.nan),
                rmse_S_vs_sumY=prec_res.get("rmse_S_vs_sumY", math.nan),
                count_diff_ok=prec_res.get("count_diff_ok", 0),
                count_diff_bad=prec_res.get("count_diff_bad", 0),
                max_abs_diff=prec_res.get("max_abs_diff", 0.0),
                count_S_lt_maxY=prec_res.get("count_S_lt_maxY", 0),
                sumS_vs_sumY_bad=prec_res.get("sumS_vs_sumY_bad", 0),
                always_nan_S_count=prec_res.get("always_nan_S_count", 0),
                always_zero_S_count=prec_res.get("always_zero_S_count", 0),
                alignment_best_lag=prec_res.get("alignment_best_lag"),
                alignment_best_lag_mean_abs_diff=prec_res.get("alignment_best_lag_mean_abs_diff"),
                max_1h_val=prec_res.get("max_1h_val", 0.0),
                max_3h_val=prec_res.get("max_3h_val", 0.0),
            ))

        # クローズ
        try:
            ds.close()
        except Exception:
            pass

    # 期間全体の極値表示
    logger.info("==== 期間全体の極値（降水） ====")
    if global_max_1h_info:
        mi = global_max_1h_info
        logger.info(
            f"期間全体 1時間降水の最大: {global_max_1h_val:.6f} mm at {mi.get('file','?')} "
            f"[time_index={mi.get('time_index','?')} {mi.get('time','?')}] "
            f"(which={mi.get('which_of','?')}, lat={mi.get('lat',math.nan):.5f}, lon={mi.get('lon',math.nan):.5f})"
        )
    if global_max_3h_info:
        gi = global_max_3h_info
        logger.info(
            f"期間全体 3時間積算の最大: {global_max_3h_val:.6f} mm at {gi.get('file','?')} "
            f"[time_index={gi.get('time_index','?')} {gi.get('time','?')}] "
            f"(lat={gi.get('lat',math.nan):.5f}, lon={gi.get('lon',math.nan):.5f})"
        )
    if (global_max_1h_val > global_max_3h_val):
        logger.warning(
            "注意: 期間全体で 1時間最大 > 3時間最大 となっています。"
            "S=y4+y5+y6 の恒等性が破れている箇所や、単調性違反（S<max(y)）がある場合、"
            "変数の時間アラインメントや欠損値処理に問題がある可能性が高いです。"
        )

    # CSV 要約
    csv_path = os.path.join(OUT_DIR, "precip_consistency_summary.csv")
    write_precip_summary_csv(csv_path, file_summaries)
    logger.info(f"降水整合性の要約CSVを保存: {csv_path}")

    map_csv_path = os.path.join(OUT_DIR, "variable_mapping_summary.csv")
    write_variable_mapping_summary_csv(map_csv_path, mapping_summaries)
    logger.info(f"変数マッピング検査の要約CSVを保存: {map_csv_path}")

    logger.info("データチェックが完了しました（詳細は本ログとCSVを参照）")


if __name__ == "__main__":
    main()
