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

# =========================
# 追加: 降水ビン集計の設定
# =========================
# 1時間降水（重み用ビンと同一スケール）
BIN_EDGES_1H = np.array([1.0, 5.0, 10.0, 20.0, 30.0, 50.0], dtype=np.float64)  # 右開区間 (right=False)
BIN_LABELS_1H = [
    "<1",
    "1-5",
    "5-10",
    "10-20",
    "20-30",
    "30-50",
    "50+",
]
# 3時間積算（1hの2倍スケール）
SUM_BIN_EDGES = np.array([2.0, 10.0, 20.0, 40.0, 60.0, 100.0], dtype=np.float64)
SUM_BIN_LABELS = [
    "<2",
    "2-10",
    "10-20",
    "20-40",
    "40-60",
    "60-100",
    "100+",
]

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

# =========================
# 追加: ビン集計・相関分析ユーティリティ
# =========================
def count_bins_for_array(arr: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """
    有効値のみ対象としてビンカウントを返す。
    戻り: shape = (len(bin_edges)+1,) int64
    """
    v = np.asarray(arr).ravel()
    mask = np.isfinite(v)
    if not np.any(mask):
        return np.zeros(len(bin_edges) + 1, dtype=np.int64)
    idx = np.digitize(v[mask], bin_edges, right=False)  # 0..len(bin_edges)
    return np.bincount(idx, minlength=len(bin_edges) + 1).astype(np.int64)

def analyze_bin_distributions(nc_files: List[Path], time_chunk: int) -> Dict[str, Any]:
    """
    analyze_1h_bin_distribution.py 相当の処理をh5pyベースで統合実装。
    - 対象: Prec_Target_ft4/5/6（1h, mm/h）, Prec_4_6h_sum（3時間積算, mm/3h）
    - 出力: 全体/各変数のビン別カウントと割合
    """
    target_1h_vars = ["Prec_Target_ft4", "Prec_Target_ft5", "Prec_Target_ft6"]
    total_counts_1h = np.zeros(len(BIN_EDGES_1H) + 1, dtype=np.int64)
    per_var_counts_1h = {v: np.zeros(len(BIN_EDGES_1H) + 1, dtype=np.int64) for v in target_1h_vars}
    sum_counts = np.zeros(len(SUM_BIN_EDGES) + 1, dtype=np.int64)

    for fp in nc_files:
        try:
            with h5py.File(fp, "r") as f:
                # 1hターゲット
                present_1h = [v for v in target_1h_vars if v in f]
                if present_1h:
                    # 時間軸を合わせるため先頭のtime軸を基準に回す
                    ds0 = f[present_1h[0]]
                    tax0 = guess_time_axis(ds0, f)
                    if tax0 is None:
                        for v in present_1h:
                            arr = apply_cf_scaling(f[v][...], f[v])
                            c = count_bins_for_array(arr, BIN_EDGES_1H)
                            per_var_counts_1h[v] += c
                            total_counts_1h += c
                    else:
                        T = ds0.shape[tax0]
                        for t0 in range(0, T, time_chunk):
                            t1 = min(T, t0 + time_chunk)
                            for v in present_1h:
                                c = count_bins_for_array(read_block(f[v], tax0, t0, t1), BIN_EDGES_1H)
                                per_var_counts_1h[v] += c
                                total_counts_1h += c
                # 3時間積算
                if "Prec_4_6h_sum" in f:
                    ds = f["Prec_4_6h_sum"]; tax = guess_time_axis(ds, f)
                    if tax is None:
                        arr = apply_cf_scaling(ds[...], ds)
                        sum_counts += count_bins_for_array(arr, SUM_BIN_EDGES)
                    else:
                        T = ds.shape[tax]
                        for t0 in range(0, T, time_chunk):
                            t1 = min(T, t0 + time_chunk)
                            arr = read_block(ds, tax, t0, t1)
                            sum_counts += count_bins_for_array(arr, SUM_BIN_EDGES)
                else:
                    # 代替: 3時刻の和
                    if all(v in f for v in target_1h_vars):
                        ds = f[target_1h_vars[0]]; tax = guess_time_axis(ds, f)
                        if tax is None:
                            arr = (apply_cf_scaling(f[target_1h_vars[0]][...], f[target_1h_vars[0]]) +
                                   apply_cf_scaling(f[target_1h_vars[1]][...], f[target_1h_vars[1]]) +
                                   apply_cf_scaling(f[target_1h_vars[2]][...], f[target_1h_vars[2]]))
                            sum_counts += count_bins_for_array(arr, SUM_BIN_EDGES)
                        else:
                            T = ds.shape[tax]
                            for t0 in range(0, T, time_chunk):
                                t1 = min(T, t0 + time_chunk)
                                arr = (read_block(f[target_1h_vars[0]], tax, t0, t1) +
                                       read_block(f[target_1h_vars[1]], tax, t0, t1) +
                                       read_block(f[target_1h_vars[2]], tax, t0, t1))
                                sum_counts += count_bins_for_array(arr, SUM_BIN_EDGES)
        except Exception as e:
            logging.warning(f"[BIN] {Path(fp).name}: 集計中にエラー: {e}")

    def make_ratio(counts: np.ndarray) -> List[float]:
        tot = int(counts.sum())
        return [(float(c) / tot * 100.0) if tot > 0 else 0.0 for c in counts.tolist()]

    report = {
        "bin_edges_1h": BIN_EDGES_1H.tolist(),
        "bin_labels_1h": list(BIN_LABELS_1H),
        "overall_1h": {
            "total": int(total_counts_1h.sum()),
            "counts": total_counts_1h.tolist(),
            "ratios": make_ratio(total_counts_1h),
        },
        "per_var_1h": {
            v: {
                "total": int(per_var_counts_1h[v].sum()),
                "counts": per_var_counts_1h[v].tolist(),
                "ratios": make_ratio(per_var_counts_1h[v]),
            }
            for v in per_var_counts_1h.keys()
        },
        "bin_edges_sum": SUM_BIN_EDGES.tolist(),
        "bin_labels_sum": list(SUM_BIN_LABELS),
        "overall_sum": {
            "total": int(sum_counts.sum()),
            "counts": sum_counts.tolist(),
            "ratios": make_ratio(sum_counts),
        },
    }
    return report

def analyze_relations(nc_files: List[Path], time_chunk: int) -> Dict[str, Any]:
    """
    降水と予測変数（時間特徴/直前降水）の関係を時系列（空間平均）で解析。
    - 対象ターゲット: Prec_Target_ft4/5/6 と 3h積算（合成）
    - 特徴量: Prec_ft3, hour_cos, hour_sin, dayofyear_cos, dayofyear_sin（存在時のみ）
    - 指標: Pearson相関係数（時系列：各時刻の空間平均同士）
    """
    targets_1h = ["Prec_Target_ft4", "Prec_Target_ft5", "Prec_Target_ft6"]
    targets_all = targets_1h + ["Prec_4_6h_sum"]
    feat_candidates = ["Prec_ft3", "hour_cos", "hour_sin", "dayofyear_cos", "dayofyear_sin"]

    # 統計保持: {tgt: {feat: {n,sum_x,sum_x2,sum_y,sum_y2,sum_xy}}}
    stats: Dict[str, Dict[str, Dict[str, float]]] = {t: {} for t in targets_all}
    feats_available: set = set()

    def ensure_stat(tgt: str, feat: str):
        if feat not in stats[tgt]:
            stats[tgt][feat] = {"n": 0.0, "sum_x": 0.0, "sum_x2": 0.0, "sum_y": 0.0, "sum_y2": 0.0, "sum_xy": 0.0}

    def update_pair(tgt: str, feat: str, x: np.ndarray, y: np.ndarray):
        # x,y: shape (tc,) の時系列（空間平均済み）
        mask = np.isfinite(x) & np.isfinite(y)
        if not np.any(mask):
            return
        xv = x[mask].astype(np.float64, copy=False)
        yv = y[mask].astype(np.float64, copy=False)
        ensure_stat(tgt, feat)
        st = stats[tgt][feat]
        st["n"] += float(xv.size)
        st["sum_x"] += float(np.sum(xv))
        st["sum_x2"] += float(np.sum(xv * xv))
        st["sum_y"] += float(np.sum(yv))
        st["sum_y2"] += float(np.sum(yv * yv))
        st["sum_xy"] += float(np.sum(xv * yv))

    def reduce_to_timeseries(arr: np.ndarray) -> np.ndarray:
        # arr: (tc, ...). (tc,) -> そのまま, (tc,H,W)-> spatial mean, 他はflatten平均
        if arr.ndim == 1:
            return arr.astype(np.float64, copy=False)
        if arr.ndim >= 2:
            ax = tuple(range(1, arr.ndim))
            return np.nanmean(arr, axis=ax)
        # スカラー等は繰り返し（呼び出し側で長さ管理）
        return np.asarray(arr, dtype=np.float64)

    for fp in nc_files:
        try:
            with h5py.File(fp, "r") as f:
                # 対象が最低限存在するか
                present_tgts = [v for v in targets_1h if v in f]
                has_sum = "Prec_4_6h_sum" in f
                if not present_tgts and not has_sum:
                    continue
                # time軸
                # まず1hターゲットのいずれか、無ければsumでtime軸検出
                ref_name = present_tgts[0] if present_tgts else "Prec_4_6h_sum"
                ds_ref = f[ref_name]; tax = guess_time_axis(ds_ref, f)
                if tax is None:
                    # 全読みで一括処理
                    # ターゲット
                    tgt_series: Dict[str, np.ndarray] = {}
                    for tname in present_tgts:
                        arr = apply_cf_scaling(f[tname][...], f[tname])
                        tgt_series[tname] = reduce_to_timeseries(arr)
                    if has_sum:
                        arrs = apply_cf_scaling(f["Prec_4_6h_sum"][...], f["Prec_4_6h_sum"])
                        tgt_series["Prec_4_6h_sum"] = reduce_to_timeseries(arrs)
                    elif len(present_tgts) == 3:
                        arrs = (apply_cf_scaling(f["Prec_Target_ft4"][...], f["Prec_Target_ft4"]) +
                                apply_cf_scaling(f["Prec_Target_ft5"][...], f["Prec_Target_ft5"]) +
                                apply_cf_scaling(f["Prec_Target_ft6"][...], f["Prec_Target_ft6"]))
                        tgt_series["Prec_4_6h_sum"] = reduce_to_timeseries(arrs)
                    # 特徴量（存在時）
                    feat_series: Dict[str, np.ndarray] = {}
                    for feat in feat_candidates:
                        if feat in f:
                            arr = apply_cf_scaling(f[feat][...], f[feat])
                            feat_series[feat] = reduce_to_timeseries(arr)
                            feats_available.add(feat)
                    # 長さ整合（最小長さに合わせる）
                    lengths = [len(v) for v in list(tgt_series.values()) + list(feat_series.values()) if hasattr(v, "__len__")]
                    if not lengths:
                        continue
                    L = min(lengths)
                    for k in tgt_series.keys():
                        tgt_series[k] = tgt_series[k][:L]
                    for k in feat_series.keys():
                        feat_series[k] = feat_series[k][:L]
                    # 更新
                    for tname, ty in tgt_series.items():
                        for fname, fx in feat_series.items():
                            update_pair(tname, fname, fx, ty)
                else:
                    T = ds_ref.shape[tax]
                    for t0 in range(0, T, time_chunk):
                        t1 = min(T, t0 + time_chunk)
                        # ターゲット
                        tgt_series: Dict[str, np.ndarray] = {}
                        for tname in present_tgts:
                            arr = read_block(f[tname], tax, t0, t1)
                            tgt_series[tname] = reduce_to_timeseries(arr)
                        if has_sum:
                            arrs = read_block(f["Prec_4_6h_sum"], tax, t0, t1)
                            tgt_series["Prec_4_6h_sum"] = reduce_to_timeseries(arrs)
                        elif len(present_tgts) == 3:
                            arrs = (read_block(f["Prec_Target_ft4"], tax, t0, t1) +
                                    read_block(f["Prec_Target_ft5"], tax, t0, t1) +
                                    read_block(f["Prec_Target_ft6"], tax, t0, t1))
                            tgt_series["Prec_4_6h_sum"] = reduce_to_timeseries(arrs)
                        # 特徴量
                        feat_series: Dict[str, np.ndarray] = {}
                        for feat in feat_candidates:
                            if feat in f:
                                arr = read_block(f[feat], guess_time_axis(f[feat], f) or tax, t0, t1)
                                feat_series[feat] = reduce_to_timeseries(arr)
                                feats_available.add(feat)
                        # 長さ整合（最小長に揃える）
                        lengths = [len(v) for v in list(tgt_series.values()) + list(feat_series.values()) if hasattr(v, "__len__")]
                        if not lengths:
                            continue
                        L = min(lengths)
                        for k in tgt_series.keys():
                            tgt_series[k] = tgt_series[k][:L]
                        for k in feat_series.keys():
                            feat_series[k] = feat_series[k][:L]
                        # 更新
                        for tname, ty in tgt_series.items():
                            for fname, fx in feat_series.items():
                                update_pair(tname, fname, fx, ty)
        except Exception as e:
            logging.warning(f"[REL] {Path(fp).name}: 関係解析中にエラー: {e}")

    # 相関係数を計算
    def to_pearson(st: Dict[str, float]) -> Optional[float]:
        n = st["n"]
        if n <= 1:
            return None
        sx, sx2, sy, sy2, sxy = st["sum_x"], st["sum_x2"], st["sum_y"], st["sum_y2"], st["sum_xy"]
        num = sxy - (sx * sy / n)
        denx = sx2 - (sx * sx / n)
        deny = sy2 - (sy * sy / n)
        den = denx * deny
        if den <= 0:
            return None
        return float(num / np.sqrt(den))

    pearson: Dict[str, Dict[str, Optional[float]]] = {t: {} for t in targets_all}
    for t in stats.keys():
        for f in stats[t].keys():
            pearson[t][f] = to_pearson(stats[t][f])

    # 季節性・日周期の強さ（rのベクトル合成）
    def vec_amp(r_cos: Optional[float], r_sin: Optional[float]) -> Optional[float]:
        if r_cos is None or r_sin is None:
            return None
        return float(np.sqrt(r_cos * r_cos + r_sin * r_sin))

    diurnal_amp = {t: vec_amp(pearson[t].get("hour_cos"), pearson[t].get("hour_sin")) for t in pearson.keys()}
    seasonal_amp = {t: vec_amp(pearson[t].get("dayofyear_cos"), pearson[t].get("dayofyear_sin")) for t in pearson.keys()}

    return {
        "features_available": sorted(list(feats_available)),
        "targets": targets_all,
        "pearson_r": pearson,
        "diurnal_amplitude_r": diurnal_amp,
        "seasonal_amplitude_r": seasonal_amp,
    }

def analyze_wet_stats(nc_files: List[Path], time_chunk: int, thr: float = 0.1) -> Dict[str, Any]:
    """
    Wet-hour(>thr mm)の画素比率（全体）と日周期(0..23h)のwet率(%)を算出。
    - 対象: Prec_Target_ft4/5/6, Prec_4_6h_sum（sum欠損時はft4+ft5+ft6で代替）
    - 日周期: hour_cos/hour_sin から算出（存在時のみ）
    """
    targets_1h = ["Prec_Target_ft4", "Prec_Target_ft5", "Prec_Target_ft6"]
    target_sum_name = "Prec_4_6h_sum"
    target_all = targets_1h + [target_sum_name]

    overall = {k: {"wet": 0, "total": 0} for k in target_all}
    diurnal_wet = {k: np.zeros(24, dtype=np.int64) for k in target_all}
    diurnal_tot = {k: np.zeros(24, dtype=np.int64) for k in target_all}
    diurnal_available_any = False

    def reduce_ts(arr: np.ndarray) -> np.ndarray:
        # 任意shape -> (tc,) の時系列（空間平均）
        if arr.ndim == 1:
            return arr.astype(np.float64, copy=False)
        if arr.ndim >= 2:
            return np.nanmean(arr, axis=tuple(range(1, arr.ndim))).astype(np.float64, copy=False)
        return np.asarray(arr, dtype=np.float64)

    for fp in nc_files:
        try:
            with h5py.File(fp, "r") as f:
                # time軸（sum優先、無ければ1hのどれか）
                ref_name = target_sum_name if target_sum_name in f else (targets_1h[0] if targets_1h[0] in f else (targets_1h[1] if targets_1h[1] in f else (targets_1h[2] if targets_1h[2] in f else None)))
                if ref_name is None:
                    continue
                tax_ref = guess_time_axis(f[ref_name], f)
                T = (f[ref_name].shape[tax_ref]) if tax_ref is not None else None

                # 日周期用の hour_cos/hour_sin
                has_hour = ("hour_cos" in f and "hour_sin" in f)
                tax_hc = guess_time_axis(f["hour_cos"], f) if has_hour else None
                tax_hs = guess_time_axis(f["hour_sin"], f) if has_hour else None
                use_diurnal = has_hour and (tax_hc is not None) and (tax_hs is not None)

                if tax_ref is None:
                    # フル読み
                    # hour 時系列（オプション）
                    if use_diurnal:
                        hc = apply_cf_scaling(f["hour_cos"][...], f["hour_cos"])
                        hs = apply_cf_scaling(f["hour_sin"][...], f["hour_sin"])
                        hc_ts = reduce_ts(hc); hs_ts = reduce_ts(hs)
                        Lh = min(len(hc_ts), len(hs_ts))
                        angle = np.arctan2(hs_ts[:Lh], hc_ts[:Lh])
                        hour_idx = ((angle % (2 * np.pi)) / (2 * np.pi) * 24.0)
                        hour_idx = np.floor(hour_idx).astype(np.int64) % 24
                    else:
                        hour_idx = None

                    # 各ターゲット
                    present_1h = [v for v in targets_1h if v in f]
                    # 1h
                    for v in present_1h:
                        arr = apply_cf_scaling(f[v][...], f[v])
                        finite = np.isfinite(arr)
                        wet = (arr > thr) & finite
                        if arr.ndim == 0:
                            tc = 1
                            wet_counts = np.array([int(wet)], dtype=np.int64)
                            tot_counts = np.array([int(finite)], dtype=np.int64)
                        else:
                            # (tc,...) にそろっている保証が無いので、time次元が無い場合のケア
                            if "time" in f and arr.shape[0] == f["time"].shape[0]:
                                tc = arr.shape[0]
                                wet_counts = wet.reshape(tc, -1).sum(axis=1)
                                tot_counts = finite.reshape(tc, -1).sum(axis=1)
                            else:
                                # 時間情報なし -> 全体集計のみ
                                tc = 1
                                wet_counts = np.array([int(wet.sum())], dtype=np.int64)
                                tot_counts = np.array([int(finite.sum())], dtype=np.int64)

                        overall[v]["wet"] += int(wet_counts.sum())
                        overall[v]["total"] += int(tot_counts.sum())
                        if hour_idx is not None and len(hour_idx) >= len(wet_counts):
                            diurnal_available_any = True
                            L = min(len(hour_idx), len(wet_counts))
                            for i in range(L):
                                h = int(hour_idx[i])
                                diurnal_wet[v][h] += int(wet_counts[i])
                                diurnal_tot[v][h] += int(tot_counts[i])

                    # sum
                    if target_sum_name in f:
                        arrs = apply_cf_scaling(f[target_sum_name][...], f[target_sum_name])
                    elif len(present_1h) == 3:
                        arrs = (apply_cf_scaling(f[targets_1h[0]][...], f[targets_1h[0]]) +
                                apply_cf_scaling(f[targets_1h[1]][...], f[targets_1h[1]]) +
                                apply_cf_scaling(f[targets_1h[2]][...], f[targets_1h[2]]))
                    else:
                        arrs = None
                    if arrs is not None:
                        finite = np.isfinite(arrs)
                        wet = (arrs > thr) & finite
                        if arrs.ndim == 0:
                            wet_counts = np.array([int(wet)], dtype=np.int64)
                            tot_counts = np.array([int(finite)], dtype=np.int64)
                        else:
                            if "time" in f and arrs.shape[0] == f["time"].shape[0]:
                                tc = arrs.shape[0]
                                wet_counts = wet.reshape(tc, -1).sum(axis=1)
                                tot_counts = finite.reshape(tc, -1).sum(axis=1)
                            else:
                                tc = 1
                                wet_counts = np.array([int(wet.sum())], dtype=np.int64)
                                tot_counts = np.array([int(finite.sum())], dtype=np.int64)
                        overall[target_sum_name]["wet"] += int(wet_counts.sum())
                        overall[target_sum_name]["total"] += int(tot_counts.sum())
                        if hour_idx is not None and len(hour_idx) >= len(wet_counts):
                            diurnal_available_any = True
                            L = min(len(hour_idx), len(wet_counts))
                            for i in range(L):
                                h = int(hour_idx[i])
                                diurnal_wet[target_sum_name][h] += int(wet_counts[i])
                                diurnal_tot[target_sum_name][h] += int(tot_counts[i])

                else:
                    # ブロック処理
                    T = f[ref_name].shape[tax_ref]
                    for t0 in range(0, T, time_chunk):
                        t1 = min(T, t0 + time_chunk)
                        # hour
                        if use_diurnal:
                            hc = read_block(f["hour_cos"], tax_hc, t0, t1)
                            hs = read_block(f["hour_sin"], tax_hs, t0, t1)
                            hc_ts = reduce_ts(hc); hs_ts = reduce_ts(hs)
                            Lh = min(len(hc_ts), len(hs_ts))
                            angle = np.arctan2(hs_ts[:Lh], hc_ts[:Lh])
                            hour_idx = ((angle % (2 * np.pi)) / (2 * np.pi) * 24.0)
                            hour_idx = np.floor(hour_idx).astype(np.int64) % 24
                        else:
                            hour_idx = None

                        # 1hターゲット
                        present_1h = [v for v in targets_1h if v in f]
                        for v in present_1h:
                            arr = read_block(f[v], tax_ref, t0, t1)
                            finite = np.isfinite(arr)
                            wet = (arr > thr) & finite
                            if arr.ndim >= 2:
                                tc = arr.shape[0]
                                wet_counts = wet.reshape(tc, -1).sum(axis=1)
                                tot_counts = finite.reshape(tc, -1).sum(axis=1)
                            else:
                                wet_counts = np.array([int(wet.sum())], dtype=np.int64)
                                tot_counts = np.array([int(finite.sum())], dtype=np.int64)
                            overall[v]["wet"] += int(wet_counts.sum())
                            overall[v]["total"] += int(tot_counts.sum())
                            if hour_idx is not None and len(hour_idx) >= len(wet_counts):
                                diurnal_available_any = True
                                L = min(len(hour_idx), len(wet_counts))
                                for i in range(L):
                                    h = int(hour_idx[i])
                                    diurnal_wet[v][h] += int(wet_counts[i])
                                    diurnal_tot[v][h] += int(tot_counts[i])

                        # sum
                        if target_sum_name in f:
                            arrs = read_block(f[target_sum_name], tax_ref, t0, t1)
                        elif len(present_1h) == 3:
                            arrs = (read_block(f[targets_1h[0]], tax_ref, t0, t1) +
                                    read_block(f[targets_1h[1]], tax_ref, t0, t1) +
                                    read_block(f[targets_1h[2]], tax_ref, t0, t1))
                        else:
                            arrs = None
                        if arrs is not None:
                            finite = np.isfinite(arrs)
                            wet = (arrs > thr) & finite
                            if arrs.ndim >= 2:
                                tc = arrs.shape[0]
                                wet_counts = wet.reshape(tc, -1).sum(axis=1)
                                tot_counts = finite.reshape(tc, -1).sum(axis=1)
                            else:
                                wet_counts = np.array([int(wet.sum())], dtype=np.int64)
                                tot_counts = np.array([int(finite.sum())], dtype=np.int64)
                            overall[target_sum_name]["wet"] += int(wet_counts.sum())
                            overall[target_sum_name]["total"] += int(tot_counts.sum())
                            if hour_idx is not None and len(hour_idx) >= len(wet_counts):
                                diurnal_available_any = True
                                L = min(len(hour_idx), len(wet_counts))
                                for i in range(L):
                                    h = int(hour_idx[i])
                                    diurnal_wet[target_sum_name][h] += int(wet_counts[i])
                                    diurnal_tot[target_sum_name][h] += int(tot_counts[i])
        except Exception as e:
            logging.warning(f"[WET] {Path(fp).name}: Wet-hour解析中にエラー: {e}")

    # 率に変換
    overall_out = {}
    for k, d in overall.items():
        wet = int(d["wet"]); tot = int(d["total"])
        ratio = (wet / tot * 100.0) if tot > 0 else 0.0
        overall_out[k] = {"wet": wet, "total": tot, "ratio_percent": ratio}

    diurnal_ratio = None
    if diurnal_available_any:
        diurnal_ratio = {}
        for k in target_all:
            w = diurnal_wet[k].astype(np.float64)
            t = diurnal_tot[k].astype(np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.where(t > 0, (w / t) * 100.0, 0.0)
            diurnal_ratio[k] = r.tolist()

    return {"threshold_mm": float(thr), "overall": overall_out, "diurnal_ratio": diurnal_ratio}

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
    # 追加オプション: 詳細分析
    parser.add_argument("--analyze-bins", action="store_true",
                        help="1時間降水/3時間積算の強度ビン分布を併せて出力（元 analyze_1h_bin_distribution.py の統合）")
    parser.add_argument("--analyze-relations", action="store_true",
                        help="降水（時系列の空間平均）と予測変数（Prec_ft3, day/hourのcos/sin）の相関関係を出力")
    parser.add_argument("--analyze-wet", action="store_true",
                        help="Wet-hour(>閾値)の画素比率や日周期(0..23h)分布を出力")
    parser.add_argument("--wet-thr", type=float, default=0.1,
                        help="Wet-hour判定のしきい値(mm)。既定=0.1mm")
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

    # 追加: 降水ビン分布と関係解析
    bin_dist_report = None
    relations_report = None
    wet_stats_report = None

    if args.analyze_bins:
        print("\n================== 降水 強度ビン分布（統合版） ==================")
        t0 = time.time()
        bin_dist_report = analyze_bin_distributions(nc_files, time_chunk)
        # 表示（全体 + 各1hターゲット + 積算）
        overall = bin_dist_report["overall_1h"]
        print(f"[1h 合計] 総画素数: {overall['total']:,}")
        for i, label in enumerate(BIN_LABELS_1H):
            cnt = int(overall["counts"][i]); ratio = float(overall["ratios"][i])
            print(f"  Bin {label:>6}: count={cnt:,}  ratio={ratio:6.3f}%")
        print("------------------------------------------------------------------------------------")
        for v, d in bin_dist_report["per_var_1h"].items():
            print(f"{v}: total={int(d['total']):,}")
            for i, label in enumerate(BIN_LABELS_1H):
                cnt = int(d["counts"][i]); ratio = float(d["ratios"][i])
                print(f"  Bin {label:>6}: count={cnt:,}  ratio={ratio:6.3f}%")
            print("------------------------------------------------------------------------------------")
        overall_sum = bin_dist_report["overall_sum"]
        print(f"[3h 積算] 総画素数: {overall_sum['total']:,}")
        for i, label in enumerate(SUM_BIN_LABELS):
            cnt = int(overall_sum["counts"][i]); ratio = float(overall_sum["ratios"][i])
            print(f"  Bin {label:>6}: count={cnt:,}  ratio={ratio:6.3f}%")
        print(f"[INFO] ビン分布解析 所要時間: {time.time() - t0:.2f}s")

    if args.analyze_relations:
        print("\n================== 降水と予測変数の関係（時系列・空間平均の相関） ==================")
        t0 = time.time()
        relations_report = analyze_relations(nc_files, time_chunk)
        feats = relations_report["features_available"]
        print(f"[INFO] 解析対象特徴量: {', '.join(feats) if feats else '(なし)'}")
        if feats:
            pear = relations_report["pearson_r"]
            for tgt in relations_report["targets"]:
                if tgt not in pear:
                    continue
                print(f"--- Target: {tgt} ---")
                for feat in feats:
                    r = pear[tgt].get(feat, None)
                    if r is None:
                        print(f"  r({feat}) = N/A")
                    else:
                        print(f"  r({feat}) = {r:+.3f}")
                # ベクトル合成の強さ
                dh = relations_report["diurnal_amplitude_r"].get(tgt)
                ds = relations_report["seasonal_amplitude_r"].get(tgt)
                if dh is not None:
                    print(f"  |diurnal (hour_cos/sin)| = {dh:.3f}")
                if ds is not None:
                    print(f"  |seasonal (doy_cos/sin)| = {ds:.3f}")
        print(f"[INFO] 関係解析 所要時間: {time.time() - t0:.2f}s")

    if args.analyze_wet:
        print("\n================== Wet-hour 比率と日周期分布 ==================")
        t0 = time.time()
        wet_stats_report = analyze_wet_stats(nc_files, time_chunk, thr=float(args.wet_thr))
        thr = wet_stats_report.get("threshold_mm", 0.1)
        overall = wet_stats_report.get("overall", {})
        for var, d in overall.items():
            wet = int(d.get("wet", 0)); tot = int(d.get("total", 0))
            ratio = (wet / tot * 100.0) if tot > 0 else 0.0
            print(f"{var:16s}: wet={wet:,} / total={tot:,}  ratio={ratio:6.3f}%  (thr={thr}mm)")
        diurnal = wet_stats_report.get("diurnal_ratio", None)
        if diurnal:
            print("\n[日周期（hour=0..23 のwet率%）]")
            for var, arr in diurnal.items():
                arrf = [f"{x:5.2f}" for x in arr]
                print(f"{var:16s}: " + " ".join(arrf))
        print(f"[INFO] Wet-hour解析 所要時間: {time.time() - t0:.2f}s")

    # JSONレポート
    # 追加キーのためのローカル変数を保持（None可）
    _bin_dist_for_json = bin_dist_report
    _relations_for_json = relations_report
    _wet_stats_for_json = wet_stats_report

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
        "bin_distribution": _bin_dist_for_json,
        "relations": _relations_for_json,
        "wet_stats": _wet_stats_for_json,
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
