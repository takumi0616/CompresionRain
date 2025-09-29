#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
output_nc 品質チェックスクリプト（改良版）
- 201801〜202312の yyyymm.nc を対象
- 重要事項のみを既定(INFO)でログに出力。詳細は --verbose 2 で DEBUG を確認
- 機能:
  * 変数セットの統一性、欠損/Inf、常識的レンジ外、座標・時間の整合性、cos/sin の一貫性
  * Prec_4_6h_sum ≈ (Prec_Target_ft4 + ft5 + ft6) の整合性検証（--prec-sum-tol）
  * 変数の一様値/全欠損の検出
- h5py + hdf5plugin による直読み、time 次元のブロック処理で高速化
- 出力: <dir>/output_nc_quality_report.json
"""

import argparse
from pathlib import Path
import re
import json
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import logging
import traceback
import math
import time
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

# 外部フィルタ対応（zstd/blosc等）
try:
    import hdf5plugin  # noqa: F401
    HDF5PLUGIN = True
except Exception:
    HDF5PLUGIN = False

try:
    import h5py
except Exception as e:
    raise RuntimeError("h5py が必要です。conda install -c conda-forge h5py もしくは pip install h5py") from e

EXPECTED_START = "201801"
EXPECTED_END   = "202312"
YM_PAT = re.compile(r"^(\d{6})\.nc$")
DEFAULT_TIME_CHUNK = 24  # 1日分を目安（--time-chunk で変更可能）

# ---------------------------------------------------------
# ロギングヘルパー
# ---------------------------------------------------------
def setup_logger(verbosity: int):
    """
    verbosity:
      0: WARNING以上
      1: INFO以上（推奨） -> 重要事項のみ
      2: DEBUG（最詳細）
    """
    level = logging.INFO
    if verbosity <= 0:
        level = logging.WARNING
    elif verbosity >= 2:
        level = logging.DEBUG
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)

# ---------------------------------------------------------
# 許容レンジの定義（名前・unitsから推定し、保守的なレンジで検査）
# ---------------------------------------------------------
def decide_allowed_range(var: str, units: Optional[str], sample_min: Optional[float] = None, sample_max: Optional[float] = None) -> Tuple[Optional[float], Optional[float], str]:
    v = var.lower()
    u = (units or "").lower()

    # time はレンジチェック対象外
    if v == "time":
        return None, None, "time coordinate (no range check)"

    # 正規化特徴（cos/sin）
    if v in ("dayofyear_cos", "dayofyear_sin", "hour_cos", "hour_sin"):
        return -1.001, 1.001, "unitless (cos/sin in [-1,1])"

    # 座標
    if v == "lat":
        return -90.0, 90.0, "degrees_north"
    if v == "lon":
        return -180.0, 360.0, "degrees_east"

    # 降水（mm想定）
    if v.startswith("prec_"):
        return 0.0, 1000.0, "precip (mm) expected non-negative"

    # 海面更正気圧
    if "prmsl" in v:
        if "pa" in u and "hpa" not in u:
            return 80000.0, 110000.0, "pressure (Pa)"
        if "hpa" in u or "mb" in u:
            return 800.0, 1100.0, "pressure (hPa)"
        return 80000.0, 110000.0, "pressure (Pa) [assumed]"

    # 気温
    if v.startswith("t"):
        if "k" in u:
            return 170.0, 330.0, "temperature (K)"
        if "c" in u or "degc" in u:
            return -100.0, 60.0, "temperature (degC)"
        if sample_min is not None:
            if sample_min > 100:
                return 170.0, 330.0, "temperature (K) [assumed]"
            else:
                return -100.0, 60.0, "temperature (degC) [assumed]"
        return 170.0, 330.0, "temperature (K) [fallback]"

    # 風（U/V）
    if v.startswith("u") or v.startswith("v"):
        return -200.0, 200.0, "wind component (m/s)"

    # ジオポテンシャル高度
    if v.startswith("gh"):
        return 0.0, 20000.0, "geopotential height (m)"

    # 湿度
    if v.startswith("r"):
        if "%" in u or "percent" in u:
            return 0.0, 110.0, "relative humidity (%)"
        if u in ("1", "fraction", "frac"):
            return 0.0, 1.5, "relative humidity (fraction)"
        if sample_max is not None and sample_max <= 1.5:
            return 0.0, 1.5, "relative humidity (fraction) [assumed]"
        return 0.0, 110.0, "relative humidity (%) [assumed]"

    # 不明
    return None, None, "no range check (unknown variable)"

def iter_months(start_ym: str, end_ym: str) -> List[str]:
    start = int(start_ym); end = int(end_ym)
    y, m = divmod(start, 100); y2, m2 = divmod(end, 100)
    months = []
    while (y < y2) or (y == y2 and m <= m2):
        months.append(f"{y:04d}{m:02d}")
        m += 1
        if m == 13:
            m = 1; y += 1
    return months

def as_scalar_attr(v):
    if isinstance(v, np.ndarray):
        if v.size == 1:
            return v.reshape(()).item()
        return v
    try:
        return v.item()
    except Exception:
        return v

def apply_cf_scaling(arr: np.ndarray, dset) -> np.ndarray:
    out = arr.astype(np.float64, copy=False)
    fill = dset.attrs.get('_FillValue', None)
    miss = dset.attrs.get('missing_value', None)
    if fill is not None:
        fv = as_scalar_attr(fill)
        m = (out == fv)
        if np.any(m): out[m] = np.nan
    if miss is not None:
        mv = as_scalar_attr(miss)
        m = (out == mv)
        if np.any(m): out[m] = np.nan
    sf = dset.attrs.get('scale_factor', None)
    ao = dset.attrs.get('add_offset', None)
    if sf is not None:
        out = out * float(as_scalar_attr(sf))
    if ao is not None:
        out = out + float(as_scalar_attr(ao))
    return out

def guess_time_axis(dset, h5f) -> Optional[int]:
    shape = dset.shape
    if len(shape) == 0:
        return None
    if 'time' not in h5f:
        return None
    tN = int(h5f['time'].shape[0])
    candidates = [ax for ax, n in enumerate(shape) if int(n) == tN]
    if candidates:
        return candidates[0]
    return None

def read_block(dset, time_axis: int, t0: int, t1: int) -> np.ndarray:
    ndim = dset.ndim
    sl = [slice(None)] * ndim
    sl[time_axis] = slice(t0, t1)
    data = dset[tuple(sl)]
    data = apply_cf_scaling(data, dset)
    if time_axis != 0:
        data = np.moveaxis(data, time_axis, 0)
    return data  # (tc, ...)

def read_full(dset) -> np.ndarray:
    data = dset[...]
    data = apply_cf_scaling(data, dset)
    return data

def minmax_ignore_nan(x: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    valid = np.isfinite(x)
    if not np.any(valid):
        return (None, None)
    xv = x[valid]
    return (float(np.min(xv)), float(np.max(xv)))

def update_minmax(current_min: Optional[float], current_max: Optional[float], block: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    bmin, bmax = minmax_ignore_nan(block)
    if bmin is not None:
        current_min = bmin if current_min is None else min(current_min, bmin)
    if bmax is not None:
        current_max = bmax if current_max is None else max(current_max, bmax)
    return current_min, current_max

def check_cos_sin_pair(h5f, name_cos: str, name_sin: str, time_chunk: int, tol: float = 1e-3) -> Dict[str, Any]:
    if name_cos not in h5f or name_sin not in h5f:
        logging.debug(f"[cos/sin] {name_cos} or {name_sin} not present -> skip")
        return {"present": False}
    dcos = h5f[name_cos]; dsin = h5f[name_sin]
    tax_cos = guess_time_axis(dcos, h5f); tax_sin = guess_time_axis(dsin, h5f)

    violations = 0
    total = 0
    sum_dev = 0.0
    max_dev = 0.0

    logging.debug(f"[cos/sin] Checking {name_cos} & {name_sin}, shapes: {dcos.shape}, {dsin.shape}, time_axes: {tax_cos}, {tax_sin}")

    if tax_cos is not None and tax_sin is not None and dcos.shape == dsin.shape and tax_cos == tax_sin:
        T = dcos.shape[tax_cos]
        n_chunks = math.ceil(T / time_chunk)
        for ci, t0 in enumerate(range(0, T, time_chunk), 1):
            t1 = min(T, t0 + time_chunk)
            ac = read_block(dcos, tax_cos, t0, t1)
            as_ = read_block(dsin, tax_sin, t0, t1)
            val = ac*ac + as_*as_
            dev = np.abs(val - 1.0)
            valid = np.isfinite(dev)
            total += int(valid.size)
            vdev = dev[valid]
            vio = np.count_nonzero(vdev > tol)
            violations += int(vio)
            if vdev.size > 0:
                sum_dev += float(np.sum(vdev))
                mdev = float(np.max(vdev))
                if mdev > max_dev:
                    max_dev = mdev
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug(f"[cos/sin] chunk {ci}/{n_chunks} t=[{t0}:{t1}) vio={vio}")
    else:
        ac = apply_cf_scaling(dcos[...], dcos)
        as_ = apply_cf_scaling(dsin[...], dsin)
        val = ac*ac + as_*as_
        dev = np.abs(val - 1.0)
        valid = np.isfinite(dev)
        total = int(valid.size)
        vdev = dev[valid]
        violations = int(np.count_nonzero(vdev > tol))
        sum_dev = float(np.sum(vdev)) if vdev.size > 0 else 0.0
        max_dev = float(np.max(vdev)) if vdev.size > 0 else 0.0

    mean_dev = (sum_dev / total) if total > 0 else None
    logging.debug(f"[cos/sin] result: total={total}, violations={violations}, mean_dev={mean_dev}, max_dev={max_dev}, tol={tol}")
    return {
        "present": True,
        "total": total,
        "violations": violations,
        "mean_dev": mean_dev,
        "max_dev": max_dev,
        "tol": tol
    }

def check_precip_sum(h5f, time_chunk: int, tol: float,
                     sum_name: str = "Prec_4_6h_sum",
                     comp_names: Tuple[str, str, str] = ("Prec_Target_ft4", "Prec_Target_ft5", "Prec_Target_ft6")) -> Dict[str, Any]:
    """
    Prec_4_6h_sum ≈ Prec_Target_ft4 + Prec_Target_ft5 + Prec_Target_ft6 を検証
    tol は絶対誤差 (mm) の許容値
    """
    result = {"present": False}
    if sum_name not in h5f or any(n not in h5f for n in comp_names):
        logging.debug(f"[prec-sum] Required variables missing -> skip ({sum_name}, {comp_names})")
        return result

    ds_sum = h5f[sum_name]
    ds_c1 = h5f[comp_names[0]]
    ds_c2 = h5f[comp_names[1]]
    ds_c3 = h5f[comp_names[2]]

    tax_sum = guess_time_axis(ds_sum, h5f)
    taxes = [guess_time_axis(ds_c1, h5f), guess_time_axis(ds_c2, h5f), guess_time_axis(ds_c3, h5f)]
    shapes_equal = (ds_sum.shape == ds_c1.shape == ds_c2.shape == ds_c3.shape)
    if not shapes_equal or any(t is None for t in [tax_sum] + taxes) or not all(t == tax_sum for t in taxes):
        logging.debug(f"[prec-sum] Shapes/time axis not aligned. shapes: {ds_sum.shape}, {ds_c1.shape}, {ds_c2.shape}, {ds_c3.shape}; taxes: {tax_sum}, {taxes}")
        # 形状が一致しない場合はフル読みで可能ならチェック
        try:
            a_sum = apply_cf_scaling(ds_sum[...], ds_sum)
            a_c1 = apply_cf_scaling(ds_c1[...], ds_c1)
            a_c2 = apply_cf_scaling(ds_c2[...], ds_c2)
            a_c3 = apply_cf_scaling(ds_c3[...], ds_c3)
            valid = np.isfinite(a_sum) & np.isfinite(a_c1) & np.isfinite(a_c2) & np.isfinite(a_c3)
            if not np.any(valid):
                return {"present": True, "total": 0, "violations": 0, "mean_abs_diff": None, "max_abs_diff": None, "tol": tol}
            diff = np.abs(a_sum - (a_c1 + a_c2 + a_c3))
            v = diff[valid]
            vio = int(np.count_nonzero(v > tol))
            return {"present": True, "total": int(valid.sum()), "violations": vio,
                    "mean_abs_diff": float(np.mean(v)) if v.size else None,
                    "max_abs_diff": float(np.max(v)) if v.size else None, "tol": tol}
        except Exception as e:
            return {"present": True, "error": f"shape/axis mismatch and fallback failed: {e}"}

    # ブロック処理
    T = ds_sum.shape[tax_sum]
    violations = 0
    total = 0
    sum_abs = 0.0
    max_abs = 0.0
    n_chunks = math.ceil(T / time_chunk)
    for ci, t0 in enumerate(range(0, T, time_chunk), 1):
        t1 = min(T, t0 + time_chunk)
        a_sum = read_block(ds_sum, tax_sum, t0, t1)
        a_c1 = read_block(ds_c1, tax_sum, t0, t1)
        a_c2 = read_block(ds_c2, tax_sum, t0, t1)
        a_c3 = read_block(ds_c3, tax_sum, t0, t1)
        valid = np.isfinite(a_sum) & np.isfinite(a_c1) & np.isfinite(a_c2) & np.isfinite(a_c3)
        if np.any(valid):
            diff = np.abs(a_sum - (a_c1 + a_c2 + a_c3))
            v = diff[valid]
            vio = int(np.count_nonzero(v > tol))
            violations += vio
            total += int(valid.sum())
            sabs = float(np.sum(v))
            sum_abs += sabs
            mabs = float(np.max(v))
            if mabs > max_abs:
                max_abs = mabs
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug(f"[prec-sum] chunk {ci}/{n_chunks} t=[{t0}:{t1})")
    mean_abs = (sum_abs / total) if total > 0 else None
    return {"present": True, "total": total, "violations": violations, "mean_abs_diff": mean_abs, "max_abs_diff": max_abs, "tol": tol}

def process_one_file(fp_str: str,
                     time_chunk: int,
                     chunk_log_interval: int,
                     prec_sum_tol: float,
                     global_ranges: Optional[Dict[str, Dict[str, Any]]] = None,
                     template_vars: Optional[set] = None,
                     do_circular: bool = False) -> Dict[str, Any]:
    """
    1ファイルを処理して結果を返すワーカー関数（プロセス安全）
    - fp_str: 対象ファイルパス（str）
    - global_ranges: {var: {"min": rmin, "max": rmax, "desc": desc}} が与えられればそれを用いて外れ値判定
                     None の場合、このファイルの最初のチャンクから推定し、結果に含めて返す
    - template_vars: 変数集合の差分判定に用いる（Noneならスキップ）
    - do_circular: cos/sin 一貫性チェックを行うか（最初の月のみ True）
    """
    fp = Path(fp_str)
    m = YM_PAT.match(fp.name)
    ym = m.group(1) if m else fp.name
    res: Dict[str, Any] = {
        "ym": ym,
        "var_names": [],
        "var_set_mismatch": None,
        "time_issue": None,
        "coord_issues": [],
        "circular_features": {},
        "precip_sum_check": {},
        "var_stats": {},
        "global_ranges": None,
        "file_dur": 0.0,
    }
    t0_file = time.time()
    try:
        with h5py.File(fp, "r") as f:
            var_names = sorted(list(f.keys()))
            vset = set(var_names)
            res["var_names"] = var_names

            # 変数集合の差分
            if template_vars is not None:
                miss = sorted(list(template_vars - vset))
                extra_vars = sorted(list(vset - template_vars))
                if miss or extra_vars:
                    res["var_set_mismatch"] = {"missing": miss, "extra": extra_vars}

            # time monotonic
            if "time" in f:
                t = apply_cf_scaling(f["time"][...], f["time"])
                if t.size > 1:
                    diffs = np.diff(t)
                    if not np.all(diffs > 0):
                        res["time_issue"] = "time not strictly increasing"
            else:
                res["time_issue"] = "time missing"

            # lat/lon
            coord_issues: List[str] = []
            if "lat" in f:
                lat = apply_cf_scaling(f["lat"][...], f["lat"])
                lat_min, lat_max = minmax_ignore_nan(lat)
                if lat_min is not None and (lat_min < -90.5 or lat_min > 90.5):
                    coord_issues.append(f"lat min out of range: {lat_min}")
                if lat_max is not None and (lat_max > 90.5 or lat_max < -90.5):
                    coord_issues.append(f"lat max out of range: {lat_max}")
            else:
                coord_issues.append("lat missing")
            if "lon" in f:
                lon = apply_cf_scaling(f["lon"][...], f["lon"])
                lon_min, lon_max = minmax_ignore_nan(lon)
                if lon_min is not None and lon_min < -180.5:
                    coord_issues.append(f"lon min too small: {lon_min}")
                if lon_max is not None and lon_max > 360.5:
                    coord_issues.append(f"lon max too large: {lon_max}")
            else:
                coord_issues.append("lon missing")
            if coord_issues:
                res["coord_issues"] = coord_issues

            # cos/sin 一貫性（代表月のみ）
            if do_circular:
                res["circular_features"]["dayofyear"] = check_cos_sin_pair(f, "dayofyear_cos", "dayofyear_sin", time_chunk, tol=1e-3)
                res["circular_features"]["hour"] = check_cos_sin_pair(f, "hour_cos", "hour_sin", time_chunk, tol=1e-3)

            # 降水和チェック
            res["precip_sum_check"] = check_precip_sum(f, time_chunk, tol=prec_sum_tol)

            # 変数走査
            vstats: Dict[str, Dict[str, Any]] = {}
            for name in var_names:
                dset = f[name]
                units = dset.attrs.get("units", None)
                s = {
                    "units": units,
                    "count_total": 0,
                    "count_nan": 0,
                    "count_inf": 0,
                    "count_finite": 0,
                    "min": None,
                    "max": None,
                    "out_of_range": 0,
                    "range_used": None,
                    "negative_count": 0,
                }
                tax = guess_time_axis(dset, f)
                if tax is None:
                    arr = read_full(dset)
                    total = arr.size
                    nan_mask = np.isnan(arr); inf_mask = np.isinf(arr); finite = np.isfinite(arr)
                    s["count_total"] += int(total)
                    s["count_nan"] += int(nan_mask.sum())
                    s["count_inf"] += int(inf_mask.sum())
                    s["count_finite"] += int(finite.sum())
                    s["min"], s["max"] = update_minmax(s["min"], s["max"], arr)
                    # レンジ決定
                    if s["range_used"] is None:
                        if global_ranges is not None and name in global_ranges:
                            s["range_used"] = global_ranges[name]
                        else:
                            rmin, rmax, desc = decide_allowed_range(name, units, s["min"], s["max"])
                            s["range_used"] = {"min": rmin, "max": rmax, "desc": desc}
                    r = s["range_used"]
                    if r and (r["min"] is not None and r["max"] is not None):
                        oor = ((arr < r["min"]) | (arr > r["max"])) & finite
                        s["out_of_range"] += int(np.count_nonzero(oor))
                    if name.lower().startswith("prec_"):
                        neg = (arr < 0) & finite
                        s["negative_count"] += int(np.count_nonzero(neg))
                else:
                    T = dset.shape[tax]
                    n_chunks = math.ceil(T / time_chunk)
                    for ci, t0 in enumerate(range(0, T, time_chunk), 1):
                        t1 = min(T, t0 + time_chunk)
                        arr = read_block(dset, tax, t0, t1)
                        total = arr.size
                        nan_mask = np.isnan(arr); inf_mask = np.isinf(arr); finite = np.isfinite(arr)
                        s["count_total"] += int(total)
                        s["count_nan"] += int(nan_mask.sum())
                        s["count_inf"] += int(inf_mask.sum())
                        s["count_finite"] += int(finite.sum())
                        s["min"], s["max"] = update_minmax(s["min"], s["max"], arr)
                        if s["range_used"] is None:
                            if global_ranges is not None and name in global_ranges:
                                s["range_used"] = global_ranges[name]
                            else:
                                rmin, rmax, desc = decide_allowed_range(name, units, s["min"], s["max"])
                                s["range_used"] = {"min": rmin, "max": rmax, "desc": desc}
                        r = s["range_used"]
                        if r and (r["min"] is not None and r["max"] is not None):
                            oor = ((arr < r["min"]) | (arr > r["max"])) & finite
                            s["out_of_range"] += int(np.count_nonzero(oor))
                        if name.lower().startswith("prec_"):
                            neg = (arr < 0) & finite
                            s["negative_count"] += int(np.count_nonzero(neg))
                        # DEBUGログは親プロセス側に任せる（プロセス間でログが乱れやすいため）
                vstats[name] = s

            res["var_stats"] = vstats

            # 最初のファイルの場合はグローバルレンジを返す
            if global_ranges is None:
                res["global_ranges"] = {vn: vstats[vn]["range_used"] for vn in vstats.keys()}

    except Exception as e:
        res["error"] = f"{fp.name}: {e}"
    res["file_dur"] = time.time() - t0_file
    return res

def main():
    parser = argparse.ArgumentParser(description="output_nc 品質チェック（改良版, 重要ログのみ）")
    parser.add_argument("--dir", type=str, default="", help="output_nc ディレクトリ")
    parser.add_argument("--start", type=str, default=EXPECTED_START, help="開始YYYYMM")
    parser.add_argument("--end", type=str, default=EXPECTED_END, help="終了YYYYMM")
    parser.add_argument("--time-chunk", type=int, default=DEFAULT_TIME_CHUNK, help="time ブロック長（既定: 24）")
    parser.add_argument("--verbose", type=int, default=1, help="ログ詳細度: 0=WARNING, 1=INFO(推奨), 2=DEBUG(詳細)")
    parser.add_argument("--log-chunk-every", type=int, default=5, help="チャンク進捗ログの間隔（Nチャンク毎に1回, DEBUG時のみ出力）")
    parser.add_argument("--prec-sum-tol", type=float, default=1e-6, help="Prec_4_6h_sum と和の許容絶対誤差 (mm)")
    parser.add_argument("--workers", type=int, default=0, help="並列ワーカー数（0で自動）")
    args = parser.parse_args()

    setup_logger(args.verbose)

    base_dir = Path(args.dir).resolve() if args.dir else (Path("./output_nc").resolve() if Path("./output_nc").exists() else Path("/home/devel/work_takasuka_git/docker_miniconda/src/CompresionRain/output_nc").resolve())

    logging.info("============================================================")
    logging.info(f"[INFO] チェック対象ディレクトリ: {base_dir} | 期間: {args.start}-{args.end}")
    logging.info(f"[INFO] hdf5plugin: {'ON' if HDF5PLUGIN else 'OFF'}")
    if not base_dir.exists():
        raise FileNotFoundError(f"ディレクトリが存在しません: {base_dir}")

    time_chunk = max(1, int(args.time_chunk))
    chunk_log_interval = max(1, int(args.log_chunk_every))
    logging.info(f"[CONFIG] time_chunk={time_chunk}, verbose={args.verbose}, prec_sum_tol={args.prec_sum_tol}")

    # ファイル探索
    nc_files = sorted([p for p in base_dir.glob("*.nc") if YM_PAT.match(p.name)],
                      key=lambda p: YM_PAT.match(p.name).group(1))
    months_present = [YM_PAT.match(p.name).group(1) for p in nc_files]
    months_expected = iter_months(args.start, args.end)
    missing = sorted(set(months_expected) - set(months_present))
    extra   = sorted(set(months_present) - set(months_expected))

    logging.info(f"[INFO] 検出ファイル数: {len(nc_files)}")
    if missing:
        logging.warning(f"[WARN] 欠損月: {', '.join(missing)}")
    else:
        logging.info("[OK] 欠損月はありません")
    if extra:
        logging.warning(f"[WARN] 期待範囲外の余分な月: {', '.join(extra)}")
    else:
        logging.info("[OK] 余分な月はありません")
    if not nc_files:
        raise FileNotFoundError("対象 .nc ファイルが見つかりません。")

    # 変数セットの統一性
    template_vars: Optional[set] = None
    per_file_var_mismatch: Dict[str, Dict[str, Any]] = {}
    file_time_issues: Dict[str, str] = {}
    file_coord_issues: Dict[str, List[str]] = {}
    circular_feature_results: Dict[str, Any] = {}

    # 降水合計の整合性（ファイル単位）
    precip_sum_checks: Dict[str, Dict[str, Any]] = {}

    # 変数集計（全期間合算）
    var_stats: Dict[str, Dict[str, Any]] = {}

    def init_var_stats(name: str, units: Optional[str]):
        if name not in var_stats:
            var_stats[name] = {
                "units": units,
                "count_total": 0,
                "count_nan": 0,
                "count_inf": 0,
                "count_finite": 0,
                "min": None,
                "max": None,
                "out_of_range": 0,
                "range_used": None,
                "negative_count": 0,  # 降水
            }

    total_start = time.time()

    # 先頭ファイルでテンプレート変数集合とグローバル許容レンジを決定
    first_fp = nc_files[0]
    logging.info(f"[FILE 1/{len(nc_files)}] {first_fp.name} を検査中（基準決定）...")
    first_res = process_one_file(str(first_fp), time_chunk, chunk_log_interval, args.prec_sum_tol,
                                 global_ranges=None, template_vars=None, do_circular=True)
    if "error" in first_res:
        logging.error(f"[FILE ERROR] {first_fp.name}: {first_res['error']}")
    else:
        template_vars = set(first_res["var_names"])
        circular_feature_results = first_res.get("circular_features", {})
        ym_first = first_res.get("ym")
        if ym_first:
            precip_sum_checks[ym_first] = first_res.get("precip_sum_check", {})
            if first_res.get("time_issue"):
                file_time_issues[ym_first] = first_res["time_issue"]
            if first_res.get("coord_issues"):
                file_coord_issues[ym_first] = first_res["coord_issues"]
        if first_res.get("var_set_mismatch"):
            per_file_var_mismatch[ym_first] = first_res["var_set_mismatch"]
        # グローバルレンジ
        global_ranges = first_res.get("global_ranges", {})

        # 集計に反映
        for name, s in first_res["var_stats"].items():
            init_var_stats(name, s.get("units"))
            g = var_stats[name]
            g["count_total"] += s["count_total"]
            g["count_nan"] += s["count_nan"]
            g["count_inf"] += s["count_inf"]
            g["count_finite"] += s["count_finite"]
            if s["min"] is not None:
                g["min"] = s["min"] if g["min"] is None else min(g["min"], s["min"])
            if s["max"] is not None:
                g["max"] = s["max"] if g["max"] is None else max(g["max"], s["max"])
            if g["range_used"] is None and s.get("range_used") is not None:
                g["range_used"] = s["range_used"]
            g["out_of_range"] += s["out_of_range"]
            g["negative_count"] += s["negative_count"]
        logging.info(f"[FILE DONE] {first_fp.name} (time={first_res.get('file_dur', 0.0):.2f}s)")

    # 残りファイルをプロセス並列で処理
    rest_files = nc_files[1:]
    if rest_files:
        auto_workers = os.cpu_count() or 1
        workers = int(args.workers) if int(args.workers) > 0 else auto_workers
        workers = max(1, min(workers, len(rest_files)))
        logging.info(f"[CONFIG] workers={workers} (auto={auto_workers}) for remaining {len(rest_files)} files")

        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [
                ex.submit(
                    process_one_file,
                    str(fp),
                    time_chunk,
                    chunk_log_interval,
                    args.prec_sum_tol,
                    global_ranges,
                    template_vars,
                    False
                )
                for fp in rest_files
            ]
            for fut in as_completed(futs):
                res = fut.result()
                ym = res.get("ym")
                if "error" in res:
                    logging.error(f"[FILE ERROR] {ym}: {res['error']}")
                    continue
                if res.get("var_set_mismatch"):
                    per_file_var_mismatch[ym] = res["var_set_mismatch"]
                if res.get("time_issue"):
                    file_time_issues[ym] = res["time_issue"]
                if res.get("coord_issues"):
                    file_coord_issues[ym] = res["coord_issues"]
                precip_sum_checks[ym] = res.get("precip_sum_check", {})

                # 集計に反映
                for name, s in res["var_stats"].items():
                    init_var_stats(name, s.get("units"))
                    g = var_stats[name]
                    g["count_total"] += s["count_total"]
                    g["count_nan"] += s["count_nan"]
                    g["count_inf"] += s["count_inf"]
                    g["count_finite"] += s["count_finite"]
                    if s["min"] is not None:
                        g["min"] = s["min"] if g["min"] is None else min(g["min"], s["min"])
                    if s["max"] is not None:
                        g["max"] = s["max"] if g["max"] is None else max(g["max"], s["max"])
                    if g["range_used"] is None and s.get("range_used") is not None:
                        g["range_used"] = s["range_used"]
                    g["out_of_range"] += s["out_of_range"]
                    g["negative_count"] += s["negative_count"]

                logging.info(f"[FILE DONE] {ym} (time={res.get('file_dur', 0.0):.2f}s)")

    total_dur = time.time() - total_start
    logging.info(f"[ALL FILES DONE] duration={total_dur:.2f}s")

    # 出力（人が見やすい要約）
    print("\n================== 変数セットの統一性 ==================")
    if per_file_var_mismatch:
        print("[WARN] 変数セットが基準ファイルと異なる月があります。")
        for ym, diff in sorted(per_file_var_mismatch.items()):
            print(f"  - {ym}: {diff}")
    else:
        print("[OK] 全ファイルで変数集合は一致")

    print("\n================== time/coord の整合性 ==================")
    if file_time_issues:
        print("[WARN] time の単調増加に問題がある月があります。")
        for ym, msg in sorted(file_time_issues.items()):
            print(f"  - {ym}: {msg}")
    else:
        print("[OK] 全ファイルで time は単調増加")

    if file_coord_issues:
        print("[WARN] 座標に問題がある月があります。")
        for ym, issues in sorted(file_coord_issues.items()):
            print(f"  - {ym}: {issues}")
    else:
        print("[OK] lat/lon の範囲は常識的範囲内")

    print("\n================== cos/sin 特徴の検査 ==================")
    for key, res in circular_feature_results.items():
        if not res.get("present", False):
            print(f"[INFO] {key}: 特徴が見つかりませんでした（スキップ）")
        elif "error" in res:
            print(f"[WARN] {key}: チェック中にエラー: {res['error']}")
        else:
            v = res["violations"]; tot = res["total"]
            mean_dev = res["mean_dev"]; max_dev = res["max_dev"]; tol = res["tol"]
            if v == 0:
                print(f"[OK] {key}: cos^2+sin^2≈1 (tol={tol}) 逸脱なし, mean_dev={mean_dev:.2e}, max_dev={max_dev:.2e}")
            else:
                print(f"[WARN] {key}: 逸脱 {v}/{tot} (tol={tol}), mean_dev={mean_dev:.2e}, max_dev={max_dev:.2e}")

    print("\n================== 降水 4-6h 合計の整合性 ==================")
    if not precip_sum_checks:
        print("[INFO] 該当ファイルがない/検査を実行できませんでした。")
    else:
        any_issue = False
        for ym in sorted(precip_sum_checks.keys()):
            res = precip_sum_checks[ym]
            if not res.get("present", False):
                print(f"[INFO] {ym}: 必要変数が存在せずスキップ")
                continue
            if "error" in res:
                print(f"[WARN] {ym}: チェックエラー: {res['error']}")
                any_issue = True
                continue
            v = res["violations"]; tot = res["total"]; tol = res["tol"]
            mean_abs = res["mean_abs_diff"]; max_abs = res["max_abs_diff"]
            if v == 0:
                print(f"[OK] {ym}: Prec_4_6h_sum ≈ ft4+ft5+ft6 (tol={tol}) 逸脱なし, mean_abs={0.0 if mean_abs is None else mean_abs:.3e}, max_abs={0.0 if max_abs is None else max_abs:.3e}")
            else:
                print(f"[WARN] {ym}: 逸脱 {v}/{tot} (tol={tol}), mean_abs={mean_abs:.3e}, max_abs={max_abs:.3e}")
                any_issue = True
        if not any_issue:
            print("[OK] 全月で合計の整合性に問題なし")

    print("\n================== 変数別 品質サマリ（全期間合算） ==================")
    issues_any = False
    for name in sorted(var_stats.keys()):
        s = var_stats[name]
        total = s["count_total"]
        finite = s["count_finite"]
        nan_ratio = (s["count_nan"] / total) if total > 0 else 0.0
        inf_ratio = (s["count_inf"] / total) if total > 0 else 0.0
        oor_ratio = (s["out_of_range"] / finite) if finite > 0 else 0.0
        neg_ratio = (s["negative_count"] / finite) if finite > 0 else 0.0
        rng = s["range_used"]
        rng_str = f"{rng['min']}..{rng['max']} ({rng['desc']})" if rng else "N/A"

        line = (f"{name}: total={total}, finite={finite}, "
                f"NaN={s['count_nan']} ({nan_ratio:.6%}), Inf={s['count_inf']} ({inf_ratio:.6%}), "
                f"min={s['min']}, max={s['max']}, "
                f"out_of_range={s['out_of_range']} ({oor_ratio:.6%}), allowed={rng_str}")

        flags = []
        if name.lower().startswith("prec_"):
            line += f", negative_prec={s['negative_count']} ({neg_ratio:.6%})"
            if s["negative_count"] > 0:
                flags.append("NEGATIVE_PREC")
        # 追加の健全性チェック
        if finite == 0 and total > 0:
            flags.append("ALL_MISSING")
        if s["min"] is not None and s["max"] is not None and s["min"] == s["max"]:
            flags.append("CONSTANT_VALUE")

        if s["count_nan"] > 0 or s["count_inf"] > 0 or s["out_of_range"] > 0 or flags:
            prefix = "[ERROR] " if "ALL_MISSING" in flags else "[WARN] "
            print(prefix + line + (f" | flags={flags}" if flags else "")); issues_any = True
        else:
            print("[OK]   " + line)

    if not issues_any:
        print("[OK] 全変数で NaN/Inf/外れ値の検出なし（設定レンジ内）")

    # JSONレポート
    report = {
        "dir": str(base_dir),
        "period": {"start": args.start, "end": args.end},
        "missing_months": missing,
        "extra_months": extra,
        "var_set_mismatch": per_file_var_mismatch,
        "time_issues": file_time_issues,
        "coord_issues": file_coord_issues,
        "circular_features": circular_feature_results,
        "precip_sum_checks": precip_sum_checks,
        "var_stats": var_stats,
        "notes": [
            "allowed_range は名前・units・一部の値域から保守的に推定しています。",
            "用途に応じて decide_allowed_range() のレンジを調整してください。",
            "処理は h5py + hdf5plugin により time 次元をブロック分割して高速に実行しています。",
            "ログは既定で重要事項のみを INFO に出力。詳細は --verbose 2 を使用してください。",
            "Prec_4_6h_sum の整合性は --prec-sum-tol (mm) で許容誤差を制御できます。",
        ],
    }
    out_json = base_dir / "output_nc_quality_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logging.info(f"[INFO] レポートを書き出しました: {out_json}")

if __name__ == "__main__":
    main()
