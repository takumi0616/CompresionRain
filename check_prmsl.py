#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最適化済み NetCDF (optimization_nc/*.nc) に対して、降水量（mm）を逆変換して以下を解析・可視化するスクリプト。

目的:
  - optimization_nc/ の各 .nc をすべて読み込み、降水グループ(precip)の min-max 正規化を scaler_groups.json から逆変換（mm）に戻す
  - "Prec_Target_ft4", "Prec_Target_ft5", "Prec_Target_ft6" の3つの値が同一時刻でどれぐらい離れているか（分散/最大-最小）
  - "Prec_4_6h_sum"/3 と各 ft4/5/6 の乖離（誤差）を統計化
  - 参考として 3つのターゲット間の相関（冗長性の手掛かり）と、(ft4+ft5+ft6) と Prec_4_6h_sum の整合性も提示
  - 図（ヒストグラム・Hexbin）とログファイル、Markdownレポートを作成して結果を視覚的に把握しやすくする

特徴:
  - HDF5(LZ4) 直読み (h5py + hdf5plugin) による time 次元のストリーミング処理（メモリ効率）
  - NaN/Inf の安全対策、形状・time軸の不一致に対するフォールバック
  - 出力: 要約を JSON + PNG 図 + Markdown レポートへ保存、ログファイルも出力

必要条件:
  - optimization_nc_data_v2.py で出力された optimization_nc/*.nc（h5netcdf, LZ4, time-chunk=1）
  - 同ディレクトリに scaler_groups.json（group_minmax に precip[min,max]を含む）

使い方(例):
  # 実行ディレクトリ: /app/src/CompresionRain
  # 最低限: これだけでOK（入出力は ./ 配下に自動出力）
  python check_prmsl.py

  # 任意の詳細オプション:
  python check_prmsl.py \
    --dir ./optimization_nc \
    --scaler ./scaler_groups.json \
    --time-chunk 1 \
    --out ./analysis_result/precip_analysis.json \
    --fig-dir ./analysis_result/precip_analysis_figs \
    --verbose 1
"""

from __future__ import annotations
import argparse
from pathlib import Path
import json
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import logging
import time
import math
import os
import re
# 可視化（任意）
try:
    import matplotlib
    matplotlib.use("Agg")  # 非GUI環境でも保存可能
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

try:
    import seaborn as sns  # 任意
    HAS_SNS = True
except Exception:
    HAS_SNS = False

# 日本語フォント（japanize-matplotlib 利用、無い場合はフォールバック）
JP_FONT = False
if 'HAS_MPL' in globals() and HAS_MPL:
    try:
        import japanize_matplotlib  # noqa: F401
        JP_FONT = True
    except Exception:
        JP_FONT = False
    try:
        import matplotlib as mpl
        mpl.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

# HDF5 フィルタ(LZ4等)の登録
try:
    import hdf5plugin  # noqa: F401
    HDF5PLUGIN = True
except Exception:
    HDF5PLUGIN = False

try:
    import h5py
except Exception as e:
    raise RuntimeError("h5py が必要です。conda install -c conda-forge h5py もしくは pip install h5py") from e


# =========================
# ロギング
# =========================
def setup_logger(verbosity: int):
    """
    verbosity:
      0: WARNING以上
      1: INFO以上（推奨）
      2: DEBUG（最詳細）
    """
    level = logging.INFO
    if verbosity <= 0:
        level = logging.WARNING
    elif verbosity >= 2:
        level = logging.DEBUG
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    # 既存ハンドラをクリアして二重出力回避
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)


# =========================
# HDF5ユーティリティ
# =========================
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
    """
    CF の _FillValue/missing_value/scale_factor/add_offset に従って numpy 配列を実数に変換
    （optimization_nc は通常スケーリング無し想定だが安全のため実装）
    """
    out = np.asarray(arr).astype(np.float64, copy=False)
    fill = dset.attrs.get('_FillValue', None)
    miss = dset.attrs.get('missing_value', None)
    if fill is not None:
        fv = as_scalar_attr(fill)
        m = (out == fv)
        if np.any(m):
            out = out.copy()
            out[m] = np.nan
    if miss is not None:
        mv = as_scalar_attr(miss)
        m = (out == mv)
        if np.any(m):
            out = out.copy()
            out[m] = np.nan
    sf = dset.attrs.get('scale_factor', None)
    ao = dset.attrs.get('add_offset', None)
    if sf is not None:
        out = out * float(as_scalar_attr(sf))
    if ao is not None:
        out = out + float(as_scalar_attr(ao))
    return out

def guess_time_axis(dset, h5f) -> Optional[int]:
    shape = dset.shape
    if len(shape) == 0 or 'time' not in h5f:
        return None
    tN = int(h5f['time'].shape[0])
    candidates = [ax for ax, n in enumerate(shape) if int(n) == tN]
    if candidates:
        return candidates[0]
    return None

def read_block(dset, time_axis: int, t0: int, t1: int) -> np.ndarray:
    """
    指定時間区間 [t0:t1) を抽出し、time 次元を先頭に移動して返す: (tc, ...)
    """
    ndim = dset.ndim
    sl = [slice(None)] * ndim
    sl[time_axis] = slice(t0, t1)
    data = dset[tuple(sl)]
    data = apply_cf_scaling(data, dset)
    if time_axis != 0:
        data = np.moveaxis(data, time_axis, 0)
    return data


# =========================
# スケーラの読み込みと逆変換
# =========================
def load_precip_minmax(scaler_json_path: Path) -> Tuple[float, float]:
    """
    scaler_groups.json から precip グループの min/max を取得
    """
    with open(scaler_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    gmm = meta.get("group_minmax", {}).get("precip", None)
    if not gmm:
        raise RuntimeError(f"'precip' の group_minmax が見つかりません: {scaler_json_path}")
    gmin = float(gmm.get("min", 0.0))
    gmax = float(gmm.get("max", 1.0))
    if not np.isfinite(gmin) or not np.isfinite(gmax) or gmax <= gmin:
        raise RuntimeError(f"不正な precip min/max: min={gmin}, max={gmax}")
    return gmin, gmax

def inv_minmax_precip(x_norm: np.ndarray, gmin: float, gmax: float) -> np.ndarray:
    """
    正規化 [0,1] を mm に逆変換（グループ一律のスカラー min/max を使用）
    """
    return x_norm * (gmax - gmin) + gmin


# =========================
# 統計アグリゲータ
# =========================
class ErrorStats:
    """
    誤差 |e| の平均, 二乗平均平方根(RMSE), 符号付き平均, 最大絶対値 を逐次集計
    """
    def __init__(self, name: str):
        self.name = name
        self.n = 0
        self.sum_abs = 0.0
        self.sum_sq = 0.0
        self.sum_signed = 0.0
        self.max_abs = 0.0

    def update(self, e: np.ndarray):
        v = np.asarray(e, dtype=np.float64).ravel()
        mask = np.isfinite(v)
        if not np.any(mask):
            return
        vv = v[mask]
        self.n += int(vv.size)
        self.sum_abs += float(np.sum(np.abs(vv)))
        self.sum_sq += float(np.sum(vv * vv))
        self.sum_signed += float(np.sum(vv))
        mabs = float(np.max(np.abs(vv)))
        if mabs > self.max_abs:
            self.max_abs = mabs

    def to_dict(self) -> Dict[str, Any]:
        mae = (self.sum_abs / self.n) if self.n > 0 else None
        rmse = (np.sqrt(self.sum_sq / self.n) if self.n > 0 else None)
        mean_signed = (self.sum_signed / self.n) if self.n > 0 else None
        return {
            "count": int(self.n),
            "mae": None if mae is None else float(mae),
            "rmse": None if rmse is None else float(rmse),
            "mean_signed": None if mean_signed is None else float(mean_signed),
            "max_abs": float(self.max_abs),
        }


class DistStats:
    """
    量の平均/分散/最大 を逐次集計（分散は一括変換しやすいよう2乗和を保持）
    """
    def __init__(self, name: str):
        self.name = name
        self.n = 0
        self.sum = 0.0
        self.sum2 = 0.0
        self.max = 0.0

    def update(self, x: np.ndarray):
        v = np.asarray(x, dtype=np.float64).ravel()
        mask = np.isfinite(v)
        if not np.any(mask):
            return
        vv = v[mask]
        self.n += int(vv.size)
        self.sum += float(np.sum(vv))
        self.sum2 += float(np.sum(vv * vv))
        m = float(np.max(vv))
        if m > self.max:
            self.max = m

    def to_dict(self) -> Dict[str, Any]:
        mean = (self.sum / self.n) if self.n > 0 else None
        var = (self.sum2 / self.n - (mean * mean)) if (self.n > 0 and mean is not None) else None
        std = (np.sqrt(var) if (var is not None and var >= 0) else None)
        return {
            "count": int(self.n),
            "mean": None if mean is None else float(mean),
            "stddev": None if std is None else float(std),
            "max": float(self.max),
        }


class PairCorr:
    """
    2変数の相関係数 Pearson r を逐次集計
    """
    def __init__(self, name: str):
        self.name = name
        self.n = 0.0
        self.sum_x = 0.0
        self.sum_y = 0.0
        self.sum_x2 = 0.0
        self.sum_y2 = 0.0
        self.sum_xy = 0.0

    def update(self, x: np.ndarray, y: np.ndarray):
        x = np.asarray(x, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64).ravel()
        mask = np.isfinite(x) & np.isfinite(y)
        if not np.any(mask):
            return
        xv = x[mask]; yv = y[mask]
        self.n += float(xv.size)
        self.sum_x += float(np.sum(xv))
        self.sum_y += float(np.sum(yv))
        self.sum_x2 += float(np.sum(xv * xv))
        self.sum_y2 += float(np.sum(yv * yv))
        self.sum_xy += float(np.sum(xv * yv))

    def value(self) -> Optional[float]:
        n = self.n
        if n <= 1:
            return None
        sx, sy = self.sum_x, self.sum_y
        sx2, sy2, sxy = self.sum_x2, self.sum_y2, self.sum_xy
        num = sxy - (sx * sy / n)
        denx = sx2 - (sx * sx / n)
        deny = sy2 - (sy * sy / n)
        den = denx * deny
        if den <= 0:
            return None
        return float(num / np.sqrt(den))


# =========================
# サンプリング（図用、メモリ抑制）
# =========================
class ReservoirSampler:
    def __init__(self, capacity: int, batch_cap: int = 2000, seed: int = 42):
        self.capacity = int(max(0, capacity))
        self.batch_cap = int(max(1, batch_cap))
        self.rng = np.random.RandomState(seed)
        self.buf: Optional[np.ndarray] = None

    def add(self, arr: np.ndarray):
        if self.capacity <= 0:
            return
        v = np.asarray(arr, dtype=np.float64).ravel()
        m = np.isfinite(v)
        if not np.any(m):
            return
        v = v[m]
        take = min(self.batch_cap, v.size)
        if take <= 0:
            return
        idx = self.rng.choice(v.size, size=take, replace=False)
        new = v[idx]
        if self.buf is None:
            if new.size > self.capacity:
                sel = self.rng.choice(new.size, size=self.capacity, replace=False)
                self.buf = new[sel]
            else:
                self.buf = new.copy()
            return
        merged = np.concatenate([self.buf, new], axis=0)
        if merged.size > self.capacity:
            sel = self.rng.choice(merged.size, size=self.capacity, replace=False)
            merged = merged[sel]
        self.buf = merged

    def get(self) -> np.ndarray:
        return np.array([]) if self.buf is None else np.asarray(self.buf)

class PairReservoirSampler:
    def __init__(self, capacity: int, batch_cap: int = 2000, seed: int = 123):
        self.capacity = int(max(0, capacity))
        self.batch_cap = int(max(1, batch_cap))
        self.rng = np.random.RandomState(seed)
        self.buf: Optional[np.ndarray] = None  # shape (n,2)

    def add(self, x: np.ndarray, y: np.ndarray):
        if self.capacity <= 0:
            return
        xv = np.asarray(x, dtype=np.float64).ravel()
        yv = np.asarray(y, dtype=np.float64).ravel()
        m = np.isfinite(xv) & np.isfinite(yv)
        if not np.any(m):
            return
        xv = xv[m]
        yv = yv[m]
        if xv.size == 0:
            return
        take = min(self.batch_cap, xv.size)
        idx = self.rng.choice(xv.size, size=take, replace=False)
        new = np.stack([xv[idx], yv[idx]], axis=1)
        if self.buf is None:
            if new.shape[0] > self.capacity:
                sel = self.rng.choice(new.shape[0], size=self.capacity, replace=False)
                self.buf = new[sel]
            else:
                self.buf = new.copy()
            return
        merged = np.concatenate([self.buf, new], axis=0)
        if merged.shape[0] > self.capacity:
            sel = self.rng.choice(merged.shape[0], size=self.capacity, replace=False)
            merged = merged[sel]
        self.buf = merged

    def get(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.buf is None or self.buf.size == 0:
            return np.array([]), np.array([])
        arr = np.asarray(self.buf)
        return arr[:, 0], arr[:, 1]

# =========================
# 主処理
# =========================
VARS_1H = ("Prec_Target_ft4", "Prec_Target_ft5", "Prec_Target_ft6")
VAR_SUM = "Prec_4_6h_sum"
YM_PAT = re.compile(r"^(\d{6})\.nc$")

def process_file(fp: Path, gmin: float, gmax: float, time_chunk: int,
                 combined_aggregators: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    1ファイル分を逐次集計（全データと有降水のみの両方）
    combined_aggregators: {"all": aggregators, "precip_only": aggregators_precip_only}
    """
    aggregators = combined_aggregators["all"]
    aggregators_precip_only = combined_aggregators["precip_only"]
    res: Dict[str, Any] = {"file": fp.name, "present": False, "dur": 0.0, "counts": {}}
    t0 = time.perf_counter()
    try:
        with h5py.File(fp, "r") as f:
            present = [v for v in VARS_1H if v in f]
            has_sum = (VAR_SUM in f)
            if not present:
                logging.info(f"[SKIP] {fp.name}: ターゲットが見つかりません")
                return res

            res["present"] = True

            # time軸は先頭のターゲット基準
            ref_name = present[0]
            ds_ref = f[ref_name]
            tax = guess_time_axis(ds_ref, f)

            # ローカルカウンタ（有効画素数）
            local_counts = {
                "triplet_points": 0,
                "sum_points": 0,
            }

            def handle_block(a4n, a5n, a6n, asumn):
                # 正規化 -> mm へ逆変換
                a4 = inv_minmax_precip(a4n, gmin, gmax)
                a5 = inv_minmax_precip(a5n, gmin, gmax)
                a6 = inv_minmax_precip(a6n, gmin, gmax)
                if asumn is None:
                    # 合計が無い場合は 1h の3つから合成
                    asum = a4 + a5 + a6
                else:
                    asum = inv_minmax_precip(asumn, gmin, gmax)

                # 形状を揃えて 1次元へ
                a4v = a4.ravel(); a5v = a5.ravel(); a6v = a6.ravel(); asumv = asum.ravel()

                # 有効マスク
                m3 = np.isfinite(a4v) & np.isfinite(a5v) & np.isfinite(a6v)
                ms = np.isfinite(asumv)
                m3s = m3 & ms

                # 3つの分散・max-min（3つすべて有限）
                if np.any(m3):
                    x4 = a4v[m3]; x5 = a5v[m3]; x6 = a6v[m3]
                    mean3 = (x4 + x5 + x6) / 3.0
                    var3 = ((x4 - mean3)**2 + (x5 - mean3)**2 + (x6 - mean3)**2) / 3.0
                    sd3 = np.sqrt(var3)

                    # max-min
                    max3 = np.maximum(np.maximum(x4, x5), x6)
                    min3 = np.minimum(np.minimum(x4, x5), x6)
                    rng = max3 - min3

                    local_counts["triplet_points"] += int(x4.size)

                    # 集約（全データ）
                    aggregators["sd_triplet"].update(sd3)
                    aggregators["maxmin_triplet"].update(rng)
                    aggregators["sample_sd"].add(sd3)
                    aggregators["sample_rng"].add(rng)

                    ths = aggregators["maxmin_thresholds"]["thresholds"]
                    for i, thr in enumerate(ths):
                        aggregators["maxmin_thresholds"]["counts"][i] += int(np.count_nonzero(rng <= thr))
                    aggregators["maxmin_thresholds"]["total"] += int(rng.size)

                    # 有降水のみ（3つのターゲットと積算がすべて>0）
                    # m3sは既に3つとsumが有限なマスク。さらに全て>0をチェック
                    if np.any(m3s):
                        xs = asumv[m3s]
                        precip_mask = (x4 > 0) & (x5 > 0) & (x6 > 0) & (xs[m3] > 0) if np.any(m3s) else np.zeros_like(x4, dtype=bool)
                        # m3内のインデックスで有降水判定
                        # m3sはm3のサブセットなので、m3内で再度チェック
                        x4_m3s = a4v[m3s]; x5_m3s = a5v[m3s]; x6_m3s = a6v[m3s]; s_m3s = asumv[m3s]
                        precip_mask_m3s = (x4_m3s > 0) & (x5_m3s > 0) & (x6_m3s > 0) & (s_m3s > 0)
                        
                        if np.any(precip_mask_m3s):
                            x4p = x4_m3s[precip_mask_m3s]
                            x5p = x5_m3s[precip_mask_m3s]
                            x6p = x6_m3s[precip_mask_m3s]
                            mean3p = (x4p + x5p + x6p) / 3.0
                            var3p = ((x4p - mean3p)**2 + (x5p - mean3p)**2 + (x6p - mean3p)**2) / 3.0
                            sd3p = np.sqrt(var3p)
                            max3p = np.maximum(np.maximum(x4p, x5p), x6p)
                            min3p = np.minimum(np.minimum(x4p, x5p), x6p)
                            rngp = max3p - min3p

                            aggregators_precip_only["sd_triplet"].update(sd3p)
                            aggregators_precip_only["maxmin_triplet"].update(rngp)
                            aggregators_precip_only["sample_sd"].add(sd3p)
                            aggregators_precip_only["sample_rng"].add(rngp)

                            ths_p = aggregators_precip_only["maxmin_thresholds"]["thresholds"]
                            for i, thr in enumerate(ths_p):
                                aggregators_precip_only["maxmin_thresholds"]["counts"][i] += int(np.count_nonzero(rngp <= thr))
                            aggregators_precip_only["maxmin_thresholds"]["total"] += int(rngp.size)

                # 3h合計/3 との乖離（sum が有限かつ3変数有限）
                if np.any(m3s):
                    x4 = a4v[m3s]; x5 = a5v[m3s]; x6 = a6v[m3s]; s = asumv[m3s]
                    s3 = s / 3.0
                    e4 = x4 - s3
                    e5 = x5 - s3
                    e6 = x6 - s3

                    local_counts["sum_points"] += int(x4.size)
                    
                    # 全データ集計
                    aggregators["err_ft4_vs_sum3"].update(e4)
                    aggregators["err_ft5_vs_sum3"].update(e5)
                    aggregators["err_ft6_vs_sum3"].update(e6)
                    aggregators["sample_e4"].add(e4)
                    aggregators["sample_e5"].add(e5)
                    aggregators["sample_e6"].add(e6)

                    e_sum = (x4 + x5 + x6) - s
                    aggregators["err_sum_consistency"].update(e_sum)
                    aggregators["sample_esum"].add(e_sum)

                    # 有降水のみ集計
                    precip_mask = (x4 > 0) & (x5 > 0) & (x6 > 0) & (s > 0)
                    if np.any(precip_mask):
                        x4p = x4[precip_mask]; x5p = x5[precip_mask]; x6p = x6[precip_mask]; sp = s[precip_mask]
                        s3p = sp / 3.0
                        e4p = x4p - s3p
                        e5p = x5p - s3p
                        e6p = x6p - s3p
                        e_sump = (x4p + x5p + x6p) - sp

                        aggregators_precip_only["err_ft4_vs_sum3"].update(e4p)
                        aggregators_precip_only["err_ft5_vs_sum3"].update(e5p)
                        aggregators_precip_only["err_ft6_vs_sum3"].update(e6p)
                        aggregators_precip_only["sample_e4"].add(e4p)
                        aggregators_precip_only["sample_e5"].add(e5p)
                        aggregators_precip_only["sample_e6"].add(e6p)
                        aggregators_precip_only["err_sum_consistency"].update(e_sump)
                        aggregators_precip_only["sample_esum"].add(e_sump)

                # 相関（冗長性の指標） + 図用ペアサンプル
                aggregators["corr_ft4_ft5"].update(a4v, a5v)
                aggregators["corr_ft4_ft6"].update(a4v, a6v)
                aggregators["corr_ft5_ft6"].update(a5v, a6v)
                aggregators["pair_ft4_ft5"].add(a4v, a5v)
                aggregators["pair_ft4_ft6"].add(a4v, a6v)
                aggregators["pair_ft5_ft6"].add(a5v, a6v)

                # 有降水のみの相関
                if np.any(m3s):
                    x4 = a4v[m3s]; x5 = a5v[m3s]; x6 = a6v[m3s]; s = asumv[m3s]
                    precip_mask = (x4 > 0) & (x5 > 0) & (x6 > 0) & (s > 0)
                    if np.any(precip_mask):
                        x4p = x4[precip_mask]; x5p = x5[precip_mask]; x6p = x6[precip_mask]
                        aggregators_precip_only["corr_ft4_ft5"].update(x4p, x5p)
                        aggregators_precip_only["corr_ft4_ft6"].update(x4p, x6p)
                        aggregators_precip_only["corr_ft5_ft6"].update(x5p, x6p)
                        aggregators_precip_only["pair_ft4_ft5"].add(x4p, x5p)
                        aggregators_precip_only["pair_ft4_ft6"].add(x4p, x6p)
                        aggregators_precip_only["pair_ft5_ft6"].add(x5p, x6p)

            if tax is None:
                # time 軸判定不可 -> フル読み
                a4n = apply_cf_scaling(f["Prec_Target_ft4"][...], f["Prec_Target_ft4"]) if "Prec_Target_ft4" in f else None
                a5n = apply_cf_scaling(f["Prec_Target_ft5"][...], f["Prec_Target_ft5"]) if "Prec_Target_ft5" in f else None
                a6n = apply_cf_scaling(f["Prec_Target_ft6"][...], f["Prec_Target_ft6"]) if "Prec_Target_ft6" in f else None
                asumn = apply_cf_scaling(f[VAR_SUM][...], f[VAR_SUM]) if has_sum else None
                if a4n is None or a5n is None or a6n is None:
                    logging.warning(f"[WARN] {fp.name}: 必要変数が不足（ft4/ft5/ft6）")
                else:
                    handle_block(a4n, a5n, a6n, asumn)
            else:
                # ブロック処理
                T = ds_ref.shape[tax]
                for t0 in range(0, T, time_chunk):
                    t1 = min(T, t0 + time_chunk)
                    # 1h ターゲット
                    a4n = read_block(f["Prec_Target_ft4"], tax, t0, t1) if "Prec_Target_ft4" in f else None
                    a5n = read_block(f["Prec_Target_ft5"], tax, t0, t1) if "Prec_Target_ft5" in f else None
                    a6n = read_block(f["Prec_Target_ft6"], tax, t0, t1) if "Prec_Target_ft6" in f else None
                    asumn = read_block(f[VAR_SUM], tax, t0, t1) if has_sum else None
                    if a4n is None or a5n is None or a6n is None:
                        logging.warning(f"[WARN] {fp.name}: 必要変数が不足（ft4/ft5/ft6）")
                        break
                    handle_block(a4n, a5n, a6n, asumn)

            res["counts"] = local_counts
    except Exception as e:
        logging.error(f"[ERROR] {fp.name}: 解析中にエラー: {e}", exc_info=True)
    res["dur"] = time.perf_counter() - t0
    return res


def main():
    parser = argparse.ArgumentParser(description="optimization_nc の降水(precip)分析: 逆変換(mm)して3ターゲット間の乖離と sum/3 との差を評価 + 図/ログ/HTML出力")
    parser.add_argument("--dir", type=str, default="", help="optimization_nc ディレクトリ（省略時: スクリプトと同階層の optimization_nc）")
    parser.add_argument("--scaler", type=str, default="", help="scaler_groups.json のパス（省略時: スクリプトと同階層）")
    parser.add_argument("--time-chunk", type=int, default=1, help="time ブロック長（最適化ファイルは通常1）")
    parser.add_argument("--verbose", type=int, default=1, help="ログ詳細度: 0=WARNING, 1=INFO(推奨), 2:DEBUG")
    parser.add_argument("--out", type=str, default="", help="結果 JSON の保存先（省略時: optimization_nc/precip_analysis.json）")
    parser.add_argument("--fig-dir", type=str, default="", help="図の出力ディレクトリ（省略時: optimization_nc/precip_analysis_figs）")
    parser.add_argument("--no-figs", action="store_true", help="図の保存を無効化")
    parser.add_argument("--sample-points", type=int, default=200000, help="図作成用のサンプル上限（全体からランダム抽出）")
    parser.add_argument("--hexbin-gridsize", type=int, default=60, help="Hexbin のグリッドサイズ")
    args = parser.parse_args()

    setup_logger(args.verbose)

    script_dir = Path(__file__).parent.resolve()
    base_dir = Path(args.dir).resolve() if args.dir else (script_dir / "optimization_nc").resolve()
    scaler_path = Path(args.scaler).resolve() if args.scaler else (script_dir / "scaler_groups.json").resolve()
    output_dir = (script_dir / "analysis_result").resolve()
    out_path = Path(args.out).resolve() if args.out else (output_dir / "precip_analysis.json").resolve()
    fig_dir = Path(args.fig_dir).resolve() if args.fig_dir else (output_dir / "precip_analysis_figs").resolve()
    log_path = (output_dir / "precip_analysis.log").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ログファイルハンドラ追加
    try:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.getLogger().level)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logging.getLogger().addHandler(fh)
    except Exception:
        pass

    logging.info("============================================================")
    logging.info(f"[INFO] 対象ディレクトリ: {base_dir}")
    logging.info(f"[INFO] scaler JSON   : {scaler_path}")
    logging.info(f"[INFO] hdf5plugin    : {'ON' if HDF5PLUGIN else 'OFF'}")
    logging.info(f"[INFO] matplotlib    : {'ON' if HAS_MPL else 'OFF'} | seaborn: {'ON' if HAS_SNS else 'OFF'}")
    logging.info(f"[INFO] 図ディレクトリ  : {fig_dir}")
    logging.info(f"[INFO] ログファイル   : {log_path}")

    if not base_dir.exists():
        raise FileNotFoundError(f"ディレクトリが存在しません: {base_dir}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"scaler_groups.json が見つかりません: {scaler_path}")

    # スケーラ読み込み（precip）
    gmin, gmax = load_precip_minmax(scaler_path)
    logging.info(f"[INFO] precip group min/max: min={gmin}, max={gmax}")

    # ファイル列挙（yyyymm.nc 優先、見つからなければ全 *.nc）
    nc_files = sorted([p for p in base_dir.glob("*.nc") if re.match(YM_PAT, p.name)], key=lambda p: re.match(YM_PAT, p.name).group(1) if re.match(YM_PAT, p.name) else p.name)
    if not nc_files:
        nc_files = sorted([p for p in base_dir.glob("*.nc")])

    logging.info(f"[INFO] 検出ファイル数: {len(nc_files)}")
    if not nc_files:
        raise FileNotFoundError("解析対象 .nc ファイルが見つかりません。")

    # 集計器の準備（全データ用）
    aggregators: Dict[str, Any] = {
        # 分散・レンジ
        "sd_triplet": DistStats("sd_triplet_mm"),
        "maxmin_triplet": DistStats("max_minus_min_mm"),
        # max-min が閾値以下の割合
        "maxmin_thresholds": {
            "thresholds": np.array([0.0, 0.1, 0.5, 1.0, 2.0], dtype=np.float64),
            "counts": np.zeros(5, dtype=np.int64),
            "total": 0
        },
        # sum/3 との誤差
        "err_ft4_vs_sum3": ErrorStats("Prec_Target_ft4 - Prec_4_6h_sum/3 [mm]"),
        "err_ft5_vs_sum3": ErrorStats("Prec_Target_ft5 - Prec_4_6h_sum/3 [mm]"),
        "err_ft6_vs_sum3": ErrorStats("Prec_Target_ft6 - Prec_4_6h_sum/3 [mm]"),
        # 合計の整合性 (ft4+ft5+ft6) - sum
        "err_sum_consistency": ErrorStats("(ft4+ft5+ft6) - Prec_4_6h_sum [mm]"),
        # 相関
        "corr_ft4_ft5": PairCorr("corr(ft4, ft5)"),
        "corr_ft4_ft6": PairCorr("corr(ft4, ft6)"),
        "corr_ft5_ft6": PairCorr("corr(ft5, ft6)"),

        # 図用サンプラ
        "sample_sd": ReservoirSampler(args.sample_points),
        "sample_rng": ReservoirSampler(args.sample_points),
        "sample_e4": ReservoirSampler(args.sample_points),
        "sample_e5": ReservoirSampler(args.sample_points),
        "sample_e6": ReservoirSampler(args.sample_points),
        "sample_esum": ReservoirSampler(args.sample_points),
        "pair_ft4_ft5": PairReservoirSampler(min(args.sample_points, 300000)),
        "pair_ft4_ft6": PairReservoirSampler(min(args.sample_points, 300000)),
        "pair_ft5_ft6": PairReservoirSampler(min(args.sample_points, 300000)),
    }

    # 集計器の準備（有降水のみ用: 3つのターゲットと積算がすべて0でないケース）
    aggregators_precip_only: Dict[str, Any] = {
        "sd_triplet": DistStats("sd_triplet_mm_precip_only"),
        "maxmin_triplet": DistStats("max_minus_min_mm_precip_only"),
        "maxmin_thresholds": {
            "thresholds": np.array([0.0, 0.1, 0.5, 1.0, 2.0], dtype=np.float64),
            "counts": np.zeros(5, dtype=np.int64),
            "total": 0
        },
        "err_ft4_vs_sum3": ErrorStats("Prec_Target_ft4 - Prec_4_6h_sum/3 [mm] (precip_only)"),
        "err_ft5_vs_sum3": ErrorStats("Prec_Target_ft5 - Prec_4_6h_sum/3 [mm] (precip_only)"),
        "err_ft6_vs_sum3": ErrorStats("Prec_Target_ft6 - Prec_4_6h_sum/3 [mm] (precip_only)"),
        "err_sum_consistency": ErrorStats("(ft4+ft5+ft6) - Prec_4_6h_sum [mm] (precip_only)"),
        "corr_ft4_ft5": PairCorr("corr(ft4, ft5) precip_only"),
        "corr_ft4_ft6": PairCorr("corr(ft4, ft6) precip_only"),
        "corr_ft5_ft6": PairCorr("corr(ft5, ft6) precip_only"),
        "sample_sd": ReservoirSampler(args.sample_points),
        "sample_rng": ReservoirSampler(args.sample_points),
        "sample_e4": ReservoirSampler(args.sample_points),
        "sample_e5": ReservoirSampler(args.sample_points),
        "sample_e6": ReservoirSampler(args.sample_points),
        "sample_esum": ReservoirSampler(args.sample_points),
        "pair_ft4_ft5": PairReservoirSampler(min(args.sample_points, 300000)),
        "pair_ft4_ft6": PairReservoirSampler(min(args.sample_points, 300000)),
        "pair_ft5_ft6": PairReservoirSampler(min(args.sample_points, 300000)),
    }

    # 実行
    t0_all = time.perf_counter()
    per_file_results: List[Dict[str, Any]] = []
    processed = 0
    for i, fp in enumerate(nc_files, 1):
        logging.info(f"[{i:03d}/{len(nc_files)}] {fp.name} を解析中...")
        # 全データと有降水のみの両方を集計（同じファイルを2回処理しないように aggregators を渡す辞書を統合）
        combined_aggs = {"all": aggregators, "precip_only": aggregators_precip_only}
        res = process_file(fp, gmin, gmax, max(1, int(args.time_chunk)), combined_aggs)
        per_file_results.append(res)
        if res.get("present"):
            processed += 1
        logging.info(f"[DONE] {fp.name}: time={res.get('dur', 0.0):.2f}s | present={res.get('present')}")

    dur_all = time.perf_counter() - t0_all
    logging.info(f"[ALL DONE] files={processed}/{len(nc_files)} | duration={dur_all:.2f}s")

    # 結果整形
    ths = aggregators["maxmin_thresholds"]["thresholds"].tolist()
    cnts = aggregators["maxmin_thresholds"]["counts"].astype(np.int64).tolist()
    total_thr = int(aggregators["maxmin_thresholds"]["total"])
    ratios = [(c / total_thr * 100.0) if total_thr > 0 else 0.0 for c in cnts]

    # 有降水のみの結果整形
    ths_p = aggregators_precip_only["maxmin_thresholds"]["thresholds"].tolist()
    cnts_p = aggregators_precip_only["maxmin_thresholds"]["counts"].astype(np.int64).tolist()
    total_thr_p = int(aggregators_precip_only["maxmin_thresholds"]["total"])
    ratios_p = [(c / total_thr_p * 100.0) if total_thr_p > 0 else 0.0 for c in cnts_p]

    out: Dict[str, Any] = {
        "dir": str(base_dir),
        "files_processed": processed,
        "duration_sec": dur_all,
        "precip_group_minmax": {"min": gmin, "max": gmax},
        "metrics": {
            "sd_triplet_mm": aggregators["sd_triplet"].to_dict(),
            "max_minus_min_mm": aggregators["maxmin_triplet"].to_dict(),
            "max_minus_min_thresholds_mm": {
                "thresholds_mm": ths,
                "counts": cnts,
                "ratios_percent": ratios,
                "total_points": total_thr,
                "desc": "max(ft4,ft5,ft6) - min(ft4,ft5,ft6) が各閾値以下となる割合"
            },
            "errors_vs_sum3": {
                "ft4_minus_sum3": aggregators["err_ft4_vs_sum3"].to_dict(),
                "ft5_minus_sum3": aggregators["err_ft5_vs_sum3"].to_dict(),
                "ft6_minus_sum3": aggregators["err_ft6_vs_sum3"].to_dict(),
                "desc": "Prec_Target_ftk - Prec_4_6h_sum/3 (mm)"
            },
            "sum_consistency_error": aggregators["err_sum_consistency"].to_dict(),
        },
        "metrics_precip_only": {
            "sd_triplet_mm": aggregators_precip_only["sd_triplet"].to_dict(),
            "max_minus_min_mm": aggregators_precip_only["maxmin_triplet"].to_dict(),
            "max_minus_min_thresholds_mm": {
                "thresholds_mm": ths_p,
                "counts": cnts_p,
                "ratios_percent": ratios_p,
                "total_points": total_thr_p,
                "desc": "有降水のみ: max(ft4,ft5,ft6) - min(ft4,ft5,ft6) が各閾値以下となる割合"
            },
            "errors_vs_sum3": {
                "ft4_minus_sum3": aggregators_precip_only["err_ft4_vs_sum3"].to_dict(),
                "ft5_minus_sum3": aggregators_precip_only["err_ft5_vs_sum3"].to_dict(),
                "ft6_minus_sum3": aggregators_precip_only["err_ft6_vs_sum3"].to_dict(),
                "desc": "有降水のみ: Prec_Target_ftk - Prec_4_6h_sum/3 (mm)"
            },
            "sum_consistency_error": aggregators_precip_only["err_sum_consistency"].to_dict(),
        },
        "pairwise_correlation": {
            "corr(ft4,ft5)": aggregators["corr_ft4_ft5"].value(),
            "corr(ft4,ft6)": aggregators["corr_ft4_ft6"].value(),
            "corr(ft5,ft6)": aggregators["corr_ft5_ft6"].value(),
        },
        "pairwise_correlation_precip_only": {
            "corr(ft4,ft5)": aggregators_precip_only["corr_ft4_ft5"].value(),
            "corr(ft4,ft6)": aggregators_precip_only["corr_ft4_ft6"].value(),
            "corr(ft5,ft6)": aggregators_precip_only["corr_ft5_ft6"].value(),
        },
        "per_file": per_file_results,
        "notes": [
            "値はすべて mm に逆変換した後に算出。",
            "metrics: 全データ（降水0を含む）の統計。",
            "metrics_precip_only: 3つのターゲットと積算がすべて>0のデータのみの統計。",
            "sd_triplet は各時刻・各画素で3つのターゲットの標準偏差を取り、その平均/分散/最大を全域で集計。",
            "max_minus_min は各時刻・各画素で max-min を取り、その平均/分散/最大を集計。閾値別割合も併記。",
            "errors_vs_sum3 は Prec_Target_ftk - Prec_4_6h_sum/3 の MAE/RMSE/平均符号/最大絶対値。",
            "sum_consistency_error は (ft4+ft5+ft6) - Prec_4_6h_sum の誤差。precip グループは同一スケールで正規化しているため、"
            "理論上は線形関係が保存されるが、入出力やNaN処理に起因する微小差があればここに現れる。",
        ],
    }

    # コンソール概要（全データ）
    print("\n================== 降水ターゲットの乖離（mm） ==================")
    sd = out["metrics"]["sd_triplet_mm"]; rng = out["metrics"]["max_minus_min_mm"]
    print(f"sd_triplet_mm: mean={sd['mean']:.6f}  std={0.0 if sd['stddev'] is None else sd['stddev']:.6f}  max={sd['max']:.6f}  (count={sd['count']})")
    print(f"max_minus_min_mm: mean={rng['mean']:.6f}  std={0.0 if rng['stddev'] is None else rng['stddev']:.6f}  max={rng['max']:.6f}  (count={rng['count']})")
    thsec = out["metrics"]["max_minus_min_thresholds_mm"]
    print("max-min 閾値割合:")
    for thr, c, r in zip(thsec["thresholds_mm"], thsec["counts"], thsec["ratios_percent"]):
        print(f"  <= {thr:4.1f} mm : {c:,} ({r:6.3f}%)")
    print("\n================== sum/3 との誤差（mm） ==================")
    for key in ("ft4_minus_sum3", "ft5_minus_sum3", "ft6_minus_sum3"):
        d = out["metrics"]["errors_vs_sum3"][key]
        print(f"{key}: MAE={d['mae']:.6f}  RMSE={d['rmse']:.6f}  mean_signed={d['mean_signed']:.6f}  max_abs={d['max_abs']:.6f}  (N={d['count']})")
    dsum = out["metrics"]["sum_consistency_error"]
    print(f"\n(ft4+ft5+ft6) - Prec_4_6h_sum: MAE={dsum['mae']:.6f}  RMSE={dsum['rmse']:.6f}  mean_signed={dsum['mean_signed']:.6f}  max_abs={dsum['max_abs']:.6f}  (N={dsum['count']})")
    print("\n================== ターゲット間の相関（全点, mm） ==================")
    corr = out["pairwise_correlation"]
    for k in ("corr(ft4,ft5)", "corr(ft4,ft6)", "corr(ft5,ft6)"):
        r = corr.get(k, None)
        print(f"{k}: r={('N/A' if r is None else f'{r:+.6f}')}")

    # コンソール概要（有降水のみ）
    print("\n================== 【有降水のみ】降水ターゲットの乖離（mm） ==================")
    sd_p = out["metrics_precip_only"]["sd_triplet_mm"]; rng_p = out["metrics_precip_only"]["max_minus_min_mm"]
    print(f"sd_triplet_mm: mean={sd_p['mean']:.6f}  std={0.0 if sd_p['stddev'] is None else sd_p['stddev']:.6f}  max={sd_p['max']:.6f}  (count={sd_p['count']})")
    print(f"max_minus_min_mm: mean={rng_p['mean']:.6f}  std={0.0 if rng_p['stddev'] is None else rng_p['stddev']:.6f}  max={rng_p['max']:.6f}  (count={rng_p['count']})")
    thsec_p = out["metrics_precip_only"]["max_minus_min_thresholds_mm"]
    print("max-min 閾値割合:")
    for thr, c, r in zip(thsec_p["thresholds_mm"], thsec_p["counts"], thsec_p["ratios_percent"]):
        print(f"  <= {thr:4.1f} mm : {c:,} ({r:6.3f}%)")
    print("\n================== 【有降水のみ】sum/3 との誤差（mm） ==================")
    for key in ("ft4_minus_sum3", "ft5_minus_sum3", "ft6_minus_sum3"):
        d = out["metrics_precip_only"]["errors_vs_sum3"][key]
        print(f"{key}: MAE={d['mae']:.6f}  RMSE={d['rmse']:.6f}  mean_signed={d['mean_signed']:.6f}  max_abs={d['max_abs']:.6f}  (N={d['count']})")
    dsum_p = out["metrics_precip_only"]["sum_consistency_error"]
    print(f"\n(ft4+ft5+ft6) - Prec_4_6h_sum: MAE={dsum_p['mae']:.6f}  RMSE={dsum_p['rmse']:.6f}  mean_signed={dsum_p['mean_signed']:.6f}  max_abs={dsum_p['max_abs']:.6f}  (N={dsum_p['count']})")
    print("\n================== 【有降水のみ】ターゲット間の相関（mm） ==================")
    corr_p = out["pairwise_correlation_precip_only"]
    for k in ("corr(ft4,ft5)", "corr(ft4,ft6)", "corr(ft5,ft6)"):
        r = corr_p.get(k, None)
        print(f"{k}: r={('N/A' if r is None else f'{r:+.6f}')}")

    # JSON 書き出し
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    logging.info(f"[INFO] 解析結果を書き出しました: {out_path}")

    # 可視化
    if not args.no_figs and HAS_MPL:
        fig_dir.mkdir(parents=True, exist_ok=True)

        def robust_range(x: np.ndarray, pad: float = 0.05) -> Tuple[float, float]:
            if x.size == 0:
                return (0.0, 1.0)
            q1, q99 = np.nanpercentile(x, [1.0, 99.0])
            span = max(1e-6, q99 - q1)
            return (q1 - pad * span, q99 + pad * span)

        def wrap_caption_text(text: Optional[str], width: int = 40) -> str:
            """
            長い説明文を改行して折り返す（画像内で見切れないようにする）。
            日本語全角でもおおむね width 文字で改行。textwrap が無い/失敗時は固定幅で分割。
            """
            if not text:
                return ""
            try:
                import textwrap
                return "\n".join(textwrap.wrap(text, width=width, break_long_words=True, replace_whitespace=False))
            except Exception:
                lines: List[str] = []
                w = max(10, int(width))
                for i in range(0, len(text), w):
                    lines.append(text[i:i+w])
                return "\n".join(lines)

        def save_hist(data: np.ndarray, title: str, xlabel: str, path: Path, bins: int = 100, vline_at_zero: bool = False, caption: Optional[str] = None):
            if data.size == 0:
                logging.warning(f"[FIG] No data for {title}")
                return
            lo, hi = robust_range(data)
            plt.figure(figsize=(7.5, 5.0))
            if HAS_SNS:
                sns.histplot(data, bins=bins, stat="density", edgecolor=None)
            else:
                plt.hist(data, bins=bins, density=True, alpha=0.8)
            if vline_at_zero:
                plt.axvline(0.0, color="k", linestyle="--", linewidth=1.0)
            plt.xlim(lo, hi)
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel("密度")
            if caption:
                cap = wrap_caption_text(caption, width=40)
                nlines = cap.count("\n") + 1
                bottom_margin = min(0.35, 0.12 + 0.022 * nlines)
                plt.tight_layout(rect=(0.02, bottom_margin, 0.98, 0.98))
                plt.gcf().text(0.5, bottom_margin * 0.45, cap, ha="center", va="bottom", fontsize=10)
            else:
                plt.tight_layout()
            plt.savefig(path.as_posix(), dpi=150)
            plt.close()
            logging.info(f"[FIG] Saved: {path}")

        def save_bar_thresholds(ths: List[float], ratios: List[float], path: Path, caption: Optional[str] = None):
            x = [f"≤{t:g}" for t in ths]
            y = ratios
            plt.figure(figsize=(7.5, 5.0))
            bars = plt.bar(x, y, color="#4C72B0")
            plt.ylabel("割合(%)")
            plt.title("max(ft4,ft5,ft6) - min(ft4,ft5,ft6) 閾値割合")
            for b, v in zip(bars, y):
                plt.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
            if caption:
                cap = wrap_caption_text(caption, width=46)
                nlines = cap.count("\n") + 1
                bottom_margin = min(0.35, 0.12 + 0.022 * nlines)
                plt.tight_layout(rect=(0.02, bottom_margin, 0.98, 0.98))
                plt.gcf().text(0.5, bottom_margin * 0.45, cap, ha="center", va="bottom", fontsize=10)
            else:
                plt.tight_layout()
            plt.savefig(path.as_posix(), dpi=150)
            plt.close()
            logging.info(f"[FIG] Saved: {path}")

        def save_hexbin(x: np.ndarray, y: np.ndarray, title: str, xlabel: str, ylabel: str, path: Path, gridsize: int = 60, caption: Optional[str] = None):
            if x.size == 0 or y.size == 0:
                logging.warning(f"[FIG] No data for {title}")
                return
            plt.figure(figsize=(7.0, 6.0))
            hb = plt.hexbin(x, y, gridsize=gridsize, cmap="viridis", mincnt=1)
            cb = plt.colorbar(hb)
            cb.set_label("カウント")
            limx = robust_range(x)
            limy = robust_range(y)
            lo = min(limx[0], limy[0]); hi = max(limx[1], limy[1])
            plt.plot([lo, hi], [lo, hi], "k--", linewidth=1.0)  # y=x
            plt.xlim(limx); plt.ylim(limy)
            plt.title(title)
            plt.xlabel(xlabel); plt.ylabel(ylabel)
            if caption:
                cap = wrap_caption_text(caption, width=48)
                nlines = cap.count("\n") + 1
                bottom_margin = min(0.35, 0.10 + 0.02 * nlines)
                plt.tight_layout(rect=(0.02, bottom_margin, 0.98, 0.98))
                plt.gcf().text(0.5, bottom_margin * 0.45, cap, ha="center", va="bottom", fontsize=10)
            else:
                plt.tight_layout()
            plt.savefig(path.as_posix(), dpi=150)
            plt.close()
            logging.info(f"[FIG] Saved: {path}")

        # サンプルの取得
        s_sd = aggregators["sample_sd"].get()
        s_rng = aggregators["sample_rng"].get()
        s_e4 = aggregators["sample_e4"].get()
        s_e5 = aggregators["sample_e5"].get()
        s_e6 = aggregators["sample_e6"].get()
        s_esum = aggregators["sample_esum"].get()
        p45x, p45y = aggregators["pair_ft4_ft5"].get()
        p46x, p46y = aggregators["pair_ft4_ft6"].get()
        p56x, p56y = aggregators["pair_ft5_ft6"].get()

        # 図のキャプション（日本語）
        cap_sd = "各時刻・各格子点で3つのターゲット(1時間降水)の標準偏差(sd)の分布。値が小さいほど3者の予測が一致しており、尾の太さは不一致事例の頻度を示します。"
        cap_rng = "同一時刻・格子における3者の最大−最小レンジの分布。小さいほど一致度が高く、右側の尾が長いほど乖離の大きいケースが多いことを示します。"
        cap_e4 = "ft4 − (3時間積算/3) の誤差分布。0付近に集中していれば一貫性が高い。正側/負側への偏りは系統的なズレを示唆します。"
        cap_e5 = "ft5 − (3時間積算/3) の誤差分布。0近傍集中なら良好、一方で広がりはばらつきの大きさを表します。"
        cap_e6 = "ft6 − (3時間積算/3) の誤差分布。0からの偏りは種別ごとの系統誤差やイベント依存性を示します。"
        cap_esum = "(ft4+ft5+ft6) − (3時間積算) の差。理想は0で、非ゼロは入出力/NaN処理などの差分が疑われます。"
        cap_bar = "レンジ(max−min)が各閾値以下となる割合。例: ≤0.1mmの割合が高いほど3者がほぼ一致している領域が多いことを意味します。"
        cap_p45 = "ft4 と ft5 の散布図(Hexbin)。y=xに沿って分布するほど二者の整合性が高い。斜めからの系統的な外れは偏りを示します。"
        cap_p46 = "ft4 と ft6 の散布図(Hexbin)。y=xからの広がりは両者の差異の大きさを示します。"
        cap_p56 = "ft5 と ft6 の散布図(Hexbin)。帯が太いほど多くの点が存在し、y=xから離れるほど不一致が大きいことを示します。"

        # ヒストグラム
        save_hist(s_sd, "3ターゲット標準偏差の分布", "sd_triplet (mm)", fig_dir / "precip_sd_triplet_hist.png", caption=cap_sd)
        save_hist(s_rng, "3ターゲットのレンジ(max−min)分布", "range (mm)", fig_dir / "precip_range_hist.png", caption=cap_rng)
        save_hist(s_e4, "ft4 − sum/3 の誤差分布", "error (mm)", fig_dir / "precip_err_ft4_minus_sum3_hist.png", vline_at_zero=True, caption=cap_e4)
        save_hist(s_e5, "ft5 − sum/3 の誤差分布", "error (mm)", fig_dir / "precip_err_ft5_minus_sum3_hist.png", vline_at_zero=True, caption=cap_e5)
        save_hist(s_e6, "ft6 − sum/3 の誤差分布", "error (mm)", fig_dir / "precip_err_ft6_minus_sum3_hist.png", vline_at_zero=True, caption=cap_e6)
        save_hist(s_esum, "(ft4+ft5+ft6) − sum の誤差分布", "error (mm)", fig_dir / "precip_err_sum_consistency_hist.png", vline_at_zero=True, caption=cap_esum)

        # 閾値割合（棒グラフ）
        ths = out["metrics"]["max_minus_min_thresholds_mm"]["thresholds_mm"]
        ratios = out["metrics"]["max_minus_min_thresholds_mm"]["ratios_percent"]
        save_bar_thresholds(ths, ratios, fig_dir / "precip_range_threshold_bars.png", caption=cap_bar)

        # ペア散布（Hexbin）
        save_hexbin(p45x, p45y, "ft4 vs ft5", "ft4 (mm)", "ft5 (mm)", fig_dir / "precip_pair_ft4_ft5_hexbin.png", gridsize=int(args.hexbin_gridsize), caption=cap_p45)
        save_hexbin(p46x, p46y, "ft4 vs ft6", "ft4 (mm)", "ft6 (mm)", fig_dir / "precip_pair_ft4_ft6_hexbin.png", gridsize=int(args.hexbin_gridsize), caption=cap_p46)
        save_hexbin(p56x, p56y, "ft5 vs ft6", "ft5 (mm)", "ft6 (mm)", fig_dir / "precip_pair_ft5_ft6_hexbin.png", gridsize=int(args.hexbin_gridsize), caption=cap_p56)

        # Markdown レポート
        md_path = (output_dir / "precip_analysis_README.md").resolve()
        try:
            lines: List[str] = []
            lines.append("# 降水(precip) 分析レポート")
            lines.append("")
            lines.append(f"- 対象ディレクトリ: {out['dir']}")
            lines.append(f"- 処理ファイル数  : {out['files_processed']}")
            lines.append(f"- 所要時間        : {out['duration_sec']:.2f} 秒")
            lines.append("")
            mm = out["precip_group_minmax"]
            lines.append(f"- 逆変換に使用した precip min/max: min={mm['min']}, max={mm['max']}")
            lines.append("")
            lines.append("## 指標の要約")
            sd = out["metrics"]["sd_triplet_mm"]
            rng = out["metrics"]["max_minus_min_mm"]
            lines.append(f"- sd_triplet: 平均={sd['mean']:.6f}, 標準偏差={0.0 if sd['stddev'] is None else sd['stddev']:.6f}, 最大={sd['max']:.6f} (N={sd['count']})")
            lines.append(f"- max−min  : 平均={rng['mean']:.6f}, 標準偏差={0.0 if rng['stddev'] is None else rng['stddev']:.6f}, 最大={rng['max']:.6f} (N={rng['count']})")
            ev = out["metrics"]["errors_vs_sum3"]
            for key, jp in [("ft4_minus_sum3","ft4 − sum/3"), ("ft5_minus_sum3","ft5 − sum/3"), ("ft6_minus_sum3","ft6 − sum/3")]:
                d = ev[key]
                lines.append(f"- {jp}: MAE={d['mae']:.6f}, RMSE={d['rmse']:.6f}, 符号付平均={d['mean_signed']:.6f}, 最大絶対値={d['max_abs']:.6f} (N={d['count']})")
            dsum = out["metrics"]["sum_consistency_error"]
            lines.append(f"- (ft4+ft5+ft6) − 3h積算: MAE={dsum['mae']:.6f}, RMSE={dsum['rmse']:.6f}, 符号付平均={dsum['mean_signed']:.6f}, 最大絶対値={dsum['max_abs']:.6f} (N={dsum['count']})")
            lines.append("")
            lines.append("## 図と読み方")
            lines.append("各図は ./precip_analysis_figs 以下に出力されています。")
            lines.append("")
            lines.append("### 3ターゲット標準偏差の分布")
            lines.append("![sd_triplet](precip_analysis_figs/precip_sd_triplet_hist.png)")
            lines.append(cap_sd)
            lines.append("")
            lines.append("### 3ターゲットのレンジ(max−min)分布")
            lines.append("![range](precip_analysis_figs/precip_range_hist.png)")
            lines.append(cap_rng)
            lines.append("")
            lines.append("### 誤差分布: ft4 − sum/3")
            lines.append("![e4](precip_analysis_figs/precip_err_ft4_minus_sum3_hist.png)")
            lines.append(cap_e4)
            lines.append("")
            lines.append("### 誤差分布: ft5 − sum/3")
            lines.append("![e5](precip_analysis_figs/precip_err_ft5_minus_sum3_hist.png)")
            lines.append(cap_e5)
            lines.append("")
            lines.append("### 誤差分布: ft6 − sum/3")
            lines.append("![e6](precip_analysis_figs/precip_err_ft6_minus_sum3_hist.png)")
            lines.append(cap_e6)
            lines.append("")
            lines.append("### 合計整合性の誤差: (ft4+ft5+ft6) − 3h積算")
            lines.append("![esum](precip_analysis_figs/precip_err_sum_consistency_hist.png)")
            lines.append(cap_esum)
            lines.append("")
            lines.append("### レンジ閾値割合（≤0, 0.1, 0.5, 1.0, 2.0 mm）")
            lines.append("![bars](precip_analysis_figs/precip_range_threshold_bars.png)")
            lines.append(cap_bar)
            lines.append("")
            lines.append("### ペア散布(Hexbin): ft4 vs ft5")
            lines.append("![p45](precip_analysis_figs/precip_pair_ft4_ft5_hexbin.png)")
            lines.append(cap_p45)
            lines.append("")
            lines.append("### ペア散布(Hexbin): ft4 vs ft6")
            lines.append("![p46](precip_analysis_figs/precip_pair_ft4_ft6_hexbin.png)")
            lines.append(cap_p46)
            lines.append("")
            lines.append("### ペア散布(Hexbin): ft5 vs ft6")
            lines.append("![p56](precip_analysis_figs/precip_pair_ft5_ft6_hexbin.png)")
            lines.append(cap_p56)
            lines.append("")
            corr = out["pairwise_correlation"]
            lines.append("## ターゲット間の相関係数")
            for k in ("corr(ft4,ft5)", "corr(ft4,ft6)", "corr(ft5,ft6)"):
                r = corr.get(k, None)
                lines.append(f"- {k}: r={('N/A' if r is None else f'{r:+.6f}')}")
            md_path.parent.mkdir(parents=True, exist_ok=True)
            with open(md_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            logging.info(f"[INFO] Markdownレポートを書き出しました: {md_path}")
        except Exception as e:
            logging.warning(f"[MD] Markdown生成に失敗: {e}")

if __name__ == "__main__":
    main()
