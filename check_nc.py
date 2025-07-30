import netCDF4 as nc
import numpy as np
import sys
import logging
import os
import datetime
import argparse # 💡コマンドライン引数を扱うライブラリを追加

# --- ここに課題の要件を定義 ---
# (REQUIREMENTS辞書は変更なし)
REQUIREMENTS = {
    "dimensions": {
        "lat": 480,
        "lon": 480,
    },
    "coordinates": {
        "lat": 46.95 - np.arange(480) * 0.05,
        "lon": 120.0 + np.arange(480) * 0.0625,
    },
    "input_variables": [
        'Prmsl_ft3', 'U10m_ft3', 'V10m_ft3', 'T2m_ft3', 'U975_ft3', 'V975_ft3',
        'T975_ft3', 'U950_ft3', 'V950_ft3', 'T950_ft3', 'U925_ft3', 'V925_ft3',
        'T925_ft3', 'R925_ft3', 'U850_ft3', 'V850_ft3', 'T850_ft3', 'R850_ft3',
        'GH500_ft3', 'T500_ft3', 'R500_ft3', 'GH300_ft3', 'U300_ft3', 'V300_ft3',
        'Prmsl_ft6', 'U10m_ft6', 'V10m_ft6', 'T2m_ft6', 'U975_ft6', 'V975_ft6',
        'T975_ft6', 'U950_ft6', 'V950_ft6', 'T950_ft6', 'U925_ft6', 'V925_ft6',
        'T925_ft6', 'R925_ft6', 'U850_ft6', 'V850_ft6', 'T850_ft6', 'R850_ft6',
        'GH500_ft6', 'T500_ft6', 'R500_ft6', 'GH300_ft6', 'U300_ft6', 'V300_ft6',
        'Prec_ft3', 'Prec_4_6h_sum'
    ],
    "target_variables": [
        'Prec_Target_ft4', 'Prec_Target_ft5', 'Prec_Target_ft6'
    ],
    # 2018年はうるう年ではないので365日。1日8回更新。
    # 1ヶ月分(31日)の場合: 31 * 8 = 248
    "time_steps_expected_year": 365 * 8,
    "time_steps_expected_month": 31 * 8
}
# --- 要件定義ここまで ---

def setup_logger(log_file_path):
    """ロガーを設定し、ファイルとコンソールに出力するようにする"""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def validate_netcdf(file_path, reqs):
    """NetCDFファイルが課題の要件を満たしているか詳細に検証する"""
    log_file = "check_nc.log"
    setup_logger(log_file)
    logging.info(f"NetCDFファイル検証レポート: {file_path}")
    logging.info(f"レポート生成日時: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"ログファイル: {os.path.abspath(log_file)}\n")
    if not os.path.exists(file_path):
        logging.error(f"❌ [エラー] 指定されたファイルが見つかりません: {file_path}")
        return
    try:
        dataset = nc.Dataset(file_path, 'r')
    except OSError as e:
        logging.error(f"❌ [エラー] ファイルを開けません。ファイルが破損しているか、NetCDF形式ではない可能性があります: {e}")
        return

    # (ここから下の検証ロジックは変更ありません)
    # --- 1. 必須変数の存在チェック ---
    logging.info("## 1. 必須変数の存在チェック ##")
    all_vars = list(dataset.variables.keys())
    missing_inputs = [v for v in reqs["input_variables"] if v not in all_vars]
    missing_targets = [v for v in reqs["target_variables"] if v not in all_vars]

    if not missing_inputs:
        logging.info("✅ [OK] 全ての入力変数が見つかりました。")
    else:
        logging.warning(f"⚠️ [WARN] 不足している入力変数があります: {missing_inputs}")

    if not missing_targets:
        logging.info("✅ [OK] 全ての目的変数が見つかりました。")
    else:
        logging.warning(f"⚠️ [WARN] 不足している目的変数があります: {missing_targets}")
    logging.info("-" * 50)

    # --- 2. 次元と座標系のチェック ---
    logging.info("## 2. 次元と座標系のチェック ##")
    # 次元サイズのチェック
    for dim, size in reqs["dimensions"].items():
        if dim in dataset.dimensions and dataset.dimensions[dim].size == size:
            logging.info(f"✅ [OK] 次元 '{dim}' のサイズは期待通り ({size}) です。")
        else:
            actual_size = dataset.dimensions.get(dim)
            actual_size = actual_size.size if actual_size else "存在しない"
            logging.error(f"❌ [NG] 次元 '{dim}' のサイズが期待値 ({size}) と異なります。実際のサイズ: {actual_size}")

    # 座標値のチェック
    for coord, expected_values in reqs["coordinates"].items():
        if coord in dataset.variables:
            actual_values = dataset.variables[coord][:]
            if np.allclose(actual_values, expected_values):
                logging.info(f"✅ [OK] 座標変数 '{coord}' の値は仕様と一致します。")
            else:
                logging.warning(f"⚠️ [WARN] 座標変数 '{coord}' の値が仕様とわずかに異なります。")
        else:
            logging.error(f"❌ [NG] 座標変数 '{coord}' が見つかりません。")
    logging.info("-" * 50)
    
    # --- 3. 時間軸のチェック ---
    logging.info("## 3. 時間軸のチェック ##")
    if 'time' in dataset.dimensions:
        actual_steps = dataset.dimensions['time'].size
        logging.info(f"- 検出された時間ステップ数: {actual_steps}")
        
        # 月次ファイルか年次ファイルか自動で判断
        if actual_steps == reqs["time_steps_expected_year"]:
            logging.info(f"✅ [OK] 時間ステップ数は期待通りです (1年分: {reqs['time_steps_expected_year']})。")
        elif 240 <= actual_steps <= reqs["time_steps_expected_month"]: # 1ヶ月(28-31日)の範囲
            logging.info(f"✅ [OK] 時間ステップ数は月次ファイルとして妥当です。")
        else:
            logging.warning(f"⚠️ [WARN] 時間ステップ数が年次({reqs['time_steps_expected_year']})または月次({28*8}~{31*8})の期待値と異なります。")
        
        if 'time' in dataset.variables:
            time_var = dataset.variables['time']
            dates = nc.num2date(time_var[:], units=time_var.units, calendar=time_var.calendar)
            
            # 時間間隔チェック
            diffs = np.diff(dates[:5])
            if all(d.total_seconds() == 3 * 3600 for d in diffs): # 3時間間隔かチェック
                logging.info("✅ [OK] 時間間隔は3時間で一定のようです（冒頭部分で確認）。")
            else:
                logging.warning(f"⚠️ [WARN] 時間間隔が3時間で一定ではありません。間隔: {[d.total_seconds()/3600 for d in diffs]} 時間")
            
            # === 追加部分：初期時刻の分布チェック ===
            logging.info("\n--- 3a. 初期時刻（UTC）の分布チェック ---")
            
            # 各初期時刻ごとのデータ数をカウント
            hour_counts = {}
            for date in dates:
                hour = date.hour
                if hour not in hour_counts:
                    hour_counts[hour] = 0
                hour_counts[hour] += 1
            
            # 期待される初期時刻
            expected_hours = [0, 3, 6, 9, 12, 15, 18, 21]
            
            # 結果を表示
            logging.info("初期時刻ごとのデータ数:")
            for hour in expected_hours:
                count = hour_counts.get(hour, 0)
                if count > 0:
                    logging.info(f"  - {hour:02d} UTC: {count} データ")
                else:
                    logging.warning(f"  - {hour:02d} UTC: {count} データ ⚠️ データがありません！")
            
            # 予期しない時刻のデータがないかチェック
            unexpected_hours = set(hour_counts.keys()) - set(expected_hours)
            if unexpected_hours:
                logging.warning(f"⚠️ [WARN] 予期しない初期時刻のデータが含まれています: {sorted(unexpected_hours)}")
            else:
                logging.info("✅ [OK] 全てのデータが期待される初期時刻（00, 03, 06, 09, 12, 15, 18, 21 UTC）から取得されています。")
            
            # 各日の初期時刻の完全性をチェック
            logging.info("\n--- 3b. 日別の初期時刻完全性チェック ---")
            
            # 日付ごとにグループ化
            daily_hours = {}
            for date in dates:
                day_key = date.strftime('%Y-%m-%d')
                if day_key not in daily_hours:
                    daily_hours[day_key] = set()
                daily_hours[day_key].add(date.hour)
            
            # 不完全な日をチェック
            incomplete_days = []
            for day, hours in sorted(daily_hours.items()):
                missing_hours = set(expected_hours) - hours
                if missing_hours:
                    incomplete_days.append((day, sorted(missing_hours)))
            
            if incomplete_days:
                logging.warning(f"⚠️ [WARN] 以下の日で一部の初期時刻のデータが欠けています:")
                for day, missing in incomplete_days[:5]:  # 最初の5日分のみ表示
                    logging.warning(f"  - {day}: 欠落時刻 {missing}")
                if len(incomplete_days) > 5:
                    logging.warning(f"  ... 他 {len(incomplete_days) - 5} 日")
            else:
                logging.info("✅ [OK] 全ての日で8つの初期時刻（3時間ごと）のデータが揃っています。")
            
            # === 追加部分終了 ===
            
        else:
            logging.error("❌ [NG] 'time' 変数が見つかりません。")
    else:
        logging.error("❌ [NG] 'time' 次元が見つかりません。")
    logging.info("-" * 50)

    # --- 4. データ品質チェック (欠損値 & 整合性) ---
    logging.info("## 4. データ品質チェック ##")
    logging.info("--- 4a. 各変数の欠損値（NaN）の割合 ---")
    for var_name, var_obj in dataset.variables.items():
        if len(var_obj.shape) > 1:
            data = var_obj[:]
            nan_count = np.isnan(data).sum()
            if hasattr(var_obj, '_FillValue'):
                fill_value = var_obj._FillValue
                if not np.isnan(fill_value):
                    nan_count += (data == fill_value).sum()
            total_count = data.size
            if total_count > 0:
                nan_percentage = (nan_count / total_count) * 100
                log_msg = f"- {var_name}: {nan_percentage:.2f}%"
                if nan_percentage > 50:
                    logging.warning(f"⚠️ {log_msg}  <-- 欠損値が50%を超えています！")
                else:
                    logging.info(log_msg)
    logging.info("\n--- 4b. 降水量データの整合性チェック ---")
    prec_sum_var = 'Prec_4_6h_sum'
    target_prec_vars = ['Prec_Target_ft4', 'Prec_Target_ft5', 'Prec_Target_ft6']
    if all(v in dataset.variables for v in [prec_sum_var] + target_prec_vars):
        sum_data = dataset.variables[prec_sum_var][:]
        target_sum = np.zeros_like(sum_data, dtype=np.float32)
        for var in target_prec_vars:
            target_sum += dataset.variables[var][:]
        mask = ~np.isnan(sum_data) & ~np.isnan(target_sum)
        diff = np.abs(sum_data[mask] - target_sum[mask])
        if np.allclose(sum_data[mask], target_sum[mask]):
            logging.info("✅ [OK] Prec_4_6h_sum は、ft4, ft5, ft6 の合計と一致しました。")
        else:
            logging.warning(f"⚠️ [WARN] Prec_4_6h_sum と ft4,5,6の合計が一致しません。平均絶対誤差: {diff.mean():.6f}")
    else:
        logging.warning("⚠️ [WARN] 降水量の整合性チェックに必要な変数が不足しています。")

    dataset.close()
    logging.info("\n検証が完了しました。")


if __name__ == "__main__":
    # 💡---ここから修正---
    # コマンドラインからファイルパスを受け取るように変更
    parser = argparse.ArgumentParser(description="NetCDFファイルの課題要件適合性を検証します。")
    parser.add_argument("netcdf_file", type=str, help="検証するNetCDFファイルのパス")
    args = parser.parse_args()

    # 引数で受け取ったファイルパスを使って検証を実行
    validate_netcdf(args.netcdf_file, REQUIREMENTS)
    # 💡---ここまで修正---
    # python check_nc.py output_nc/201801.nc