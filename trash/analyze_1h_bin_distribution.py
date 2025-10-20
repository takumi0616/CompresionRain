
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1時間降水(Prec_Target_ft4/5/6)の画素数を強度ビンごとに集計し、比率(%)を出力するスクリプト。
- 対象データ: ./optimization_nc 下の TRAIN_YEARS 月別ファイル
- 対象変数: Prec_Target_ft4, Prec_Target_ft5, Prec_Target_ft6 (mm/h)
- ビン境界: 重み用のビン [1, 5, 10, 20, 30, 50] を使用
  ※ torch.bucketize(right=False) 相当の右開区間に合わせ、区間は:
     (-inf, 1), [1, 5), [5, 10), [10, 20), [20, 30), [30, 50), [50, +inf)
"""
import os
import glob
import numpy as np
import xarray as xr
import hdf5plugin  # HDF5圧縮フィルタ（zstd, blosc等）対応を有効化

# 設定
DATA_DIR = "./optimization_nc"
TRAIN_YEARS = [2018, 2019, 2020, 2021]

TARGET_VARS_1H = ["Prec_Target_ft4", "Prec_Target_ft5", "Prec_Target_ft6"]  # mm/h

# 重み用ビン（1h）に合わせる
BIN_EDGES = np.array([1.0, 5.0, 10.0, 20.0, 30.0, 50.0], dtype=np.float64)  # 右開区間の境界（right=False）
# ラベル（表示用）
BIN_LABELS = [
    "<1",
    "1-5",
    "5-10",
    "10-20",
    "20-30",
    "30-50",
    "50+",
]

# 3時間積算用のビン（v5の重み用スケールに合わせる: [2, 10, 20, 40, 60, 100]）
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

def get_monthly_files(data_dir, years):
    files = []
    for y in years:
        files.extend(sorted(glob.glob(os.path.join(data_dir, f"{y}*.nc"))))
    return files

def count_bins_for_array(arr, bin_edges):
    """
    arr: 任意shape (mm/h), NaN可
    bin_edges: np.array([...]) for numpy.digitize with right=False
    戻り: ビンごとのカウント (len(bin_edges)+1,)
    """
    # 有効値のみ対象
    v = arr.ravel()
    mask = np.isfinite(v)
    v = v[mask]
    if v.size == 0:
        return np.zeros(len(bin_edges) + 1, dtype=np.int64)
    # numpy.digitize right=False → (-inf, e0), [e0, e1), ..., [e_last, +inf)
    idx = np.digitize(v, bin_edges, right=False)  # 0..len(bin_edges)
    # カウント
    counts = np.bincount(idx, minlength=len(bin_edges) + 1).astype(np.int64)
    return counts

def main():
    files = get_monthly_files(DATA_DIR, TRAIN_YEARS)
    if not files:
        print(f"[ERROR] No files found under {DATA_DIR} for years={TRAIN_YEARS}")
        return

    total_counts = np.zeros(len(BIN_EDGES) + 1, dtype=np.int64)
    per_var_counts = {var: np.zeros(len(BIN_EDGES) + 1, dtype=np.int64) for var in TARGET_VARS_1H}
    # 3時間積算（Prec_4_6h_sum）用
    sum_counts = np.zeros(len(SUM_BIN_EDGES) + 1, dtype=np.int64)

    print(f"[INFO] Found {len(files)} files. Starting aggregation over TRAIN_YEARS={TRAIN_YEARS}")
    for i, fp in enumerate(files, 1):
        try:
            # まずは netcdf4 エンジンで開く（本番コードと同様）
            ds = xr.open_dataset(fp, engine="netcdf4")
        except Exception as e1:
            # フィルタ未定義等で失敗した場合は h5netcdf へフォールバック
            print(f"[INFO] Falling back to h5netcdf for {os.path.basename(fp)} due to: {e1}")
            try:
                ds = xr.open_dataset(fp, engine="h5netcdf")
            except Exception as e2:
                print(f"[WARN] Failed to open: {fp} (h5netcdf also failed: {e2})")
                continue

        try:
            # 変数ごとに一括ロード（ファイル単位）
            for var in TARGET_VARS_1H:
                if var not in ds.variables:
                    print(f"[WARN] Variable '{var}' not found in {os.path.basename(fp)}. Skipped.")
                    continue
                # 全time,lat,lon 読み込み（ファイル単位でまとめて）
                arr = ds[var].values  # shape: (time, lat, lon)
                counts = count_bins_for_array(arr, BIN_EDGES)
                per_var_counts[var] += counts
                total_counts += counts

            # 3時間積算の分布（可能なら直接変数から、なければ3時刻の和で計算）
            sum_var_name = "Prec_4_6h_sum"
            if sum_var_name in ds.variables:
                arr_sum = ds[sum_var_name].values  # shape: (time, lat, lon)
            else:
                # 代替: 3時刻の和（全time,lat,lon）
                missing = [v for v in TARGET_VARS_1H if v not in ds.variables]
                if len(missing) == 0:
                    arr_sum = ds[TARGET_VARS_1H[0]].values + ds[TARGET_VARS_1H[1]].values + ds[TARGET_VARS_1H[2]].values
                else:
                    arr_sum = None
                    print(f"[WARN] Cannot compute 3h sum in {os.path.basename(fp)} (missing: {missing})")
            if arr_sum is not None:
                counts_sum = count_bins_for_array(arr_sum, SUM_BIN_EDGES)
                sum_counts += counts_sum
        finally:
            ds.close()

        if i % 10 == 0 or i == len(files):
            print(f"[PROG] Processed {i}/{len(files)} files.")

    # 出力
    grand_total = int(total_counts.sum())
    print("\n================ Bin Distribution (TRAIN, 1-hour, using weight bins) ================")
    print(f"Total pixels (all files, all 3 targets): {grand_total}")
    print("Bins (right=False): (-inf,1), [1,5), [5,10), [10,20), [20,30), [30,50), [50, +inf)")
    print("------------------------------------------------------------------------------------")
    print("Overall (3 targets combined):")
    for i, label in enumerate(BIN_LABELS):
        cnt = int(total_counts[i])
        ratio = (cnt / grand_total * 100.0) if grand_total > 0 else 0.0
        print(f"  Bin {label:>5}: count={cnt:,}  ratio={ratio:6.3f}%")
    print("------------------------------------------------------------------------------------")
    for var in TARGET_VARS_1H:
        cnts = per_var_counts[var]
        subtotal = int(cnts.sum())
        print(f"{var}: total={subtotal:,}")
        for i, label in enumerate(BIN_LABELS):
            cnt = int(cnts[i])
            ratio = (cnt / subtotal * 100.0) if subtotal > 0 else 0.0
            print(f"  Bin {label:>5}: count={cnt:,}  ratio={ratio:6.3f}%")
        print("------------------------------------------------------------------------------------")

    # 3時間積算の出力
    grand_total_sum = int(sum_counts.sum())
    print("\n================ Bin Distribution (TRAIN, 3-hour accumulation) =====================")
    print(f"Total pixels (all files): {grand_total_sum}")
    print("Bins (right=False): (-inf,2), [2,10), [10,20), [20,40), [40,60), [60,100), [100, +inf)")
    print("------------------------------------------------------------------------------------")
    for i, label in enumerate(SUM_BIN_LABELS):
        cnt = int(sum_counts[i])
        ratio = (cnt / grand_total_sum * 100.0) if grand_total_sum > 0 else 0.0
        print(f"  Bin {label:>6}: count={cnt:,}  ratio={ratio:6.3f}%")
    print("------------------------------------------------------------------------------------")

if __name__ == "__main__":
    main()
