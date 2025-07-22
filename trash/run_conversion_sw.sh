#!/bin/bash

# ==============================================================================
# MSM GRIB2データから機械学習用NetCDFデータセットを作成するスクリプト
# ==============================================================================
#
# 機能:
# 1. 統計量計算 (calc_stats): 指定期間のデータの平均・標準偏差を計算し保存する。
# 2. データ変換 (convert): 統計量を用いてデータを正規化し、年ごとのNetCDFファイルに変換する。
#
# 実行方法:
#   ./run_conversion.sh             # 全ての処理 (統計量計算 -> データ変換) を実行
#   ./run_conversion.sh calc_stats  # 統計量計算のみ実行
#   ./run_conversion.sh convert     # データ変換のみ実行 (事前に統計量ファイルが必要)
#
# バックグラウンド実行:
#   nohup ./run_conversion.sh > conversion.log 2>&1 &
#

# --- スクリプト設定 ---
# set -e: コマンドがエラーになった時点でスクリプトを終了する
# set -o pipefail: パイプラインのいずれかのコマンドが失敗した場合にエラーとする
set -e
set -o pipefail

# --- 基本設定 ---
START_YEAR=2018
END_YEAR=2023
# 学習データ期間 (統計量計算用)
TRAIN_START_YEAR=2018
TRAIN_END_YEAR=2023

# 並列処理ワーカー数 (マシンのCPUコア数に合わせて調整)
MAX_WORKERS=22

# --- パス設定 ---
# スクリプトの場所を基準にプロジェクトディレクトリを決定
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
BASE_DIR=$(dirname "${SCRIPT_DIR}") # src/CompresionRain -> src

# 修正点: データディレクトリのパスを SCRIPT_DIR 基準に変更
MSM_DIR="${SCRIPT_DIR}/MSM_data/"
OUTPUT_DIR="${SCRIPT_DIR}/output_nc"
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
TEMP_ROOT_DIR="${OUTPUT_DIR}/temp"
STATS_FILE="${OUTPUT_DIR}/stats_${TRAIN_START_YEAR}-${TRAIN_END_YEAR}.nc"

# --- 実行モード ---
# "all", "calc_stats", "convert" から選択
MODE=${1:-"all"}

# --- 準備 (必要なディレクトリを作成) ---
echo "INFO: Creating necessary directories..."
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${TEMP_ROOT_DIR}"
echo "INFO: Directories are ready."

# --- 1. 統計量の計算 ---
if [[ "$MODE" == "all" || "$MODE" == "calc_stats" ]]; then
    echo "=================================================="
    echo " MODE: Running Statistics Calculation"
    echo "=================================================="
    train_start_date="${TRAIN_START_YEAR}-01-01"
    train_end_date="${TRAIN_END_YEAR}-12-31"
    
    echo "Calculating statistics from ${train_start_date} to ${train_end_date}"
    echo "Statistics file will be saved to: ${STATS_FILE}"
    echo "Temporary files will be stored in: ${TEMP_ROOT_DIR}"
    
    # 既存の統計量ファイルがあれば削除
    if [ -f "${STATS_FILE}" ]; then
        echo "WARN: Existing statistics file found. Removing it: ${STATS_FILE}"
        rm -f "${STATS_FILE}"
    fi

    python3 "${SCRIPT_DIR}/convert_data.py" \
        --mode calc_stats \
        "${train_start_date}" "${train_end_date}" \
        --stats_file "${STATS_FILE}" \
        --msm_dir "${MSM_DIR}" \
        --max_workers "${MAX_WORKERS}" \
        --temp_dir "${TEMP_ROOT_DIR}"
    
    echo "SUCCESS: Successfully calculated and saved statistics."
fi

# --- 2. データ変換 ---
if [[ "$MODE" == "all" || "$MODE" == "convert" ]]; then
    echo "=================================================="
    echo " MODE: Running Data Conversion"
    echo "=================================================="
    
    if [ ! -f "${STATS_FILE}" ]; then
        echo "ERROR: Statistics file not found: ${STATS_FILE}" >&2
        echo "Please run in 'calc_stats' mode first or run with 'all' mode." >&2
        exit 1
    fi
    
    for year in $(seq ${START_YEAR} ${END_YEAR}); do
        echo "--------------------------------------------------"
        echo "Processing Year: ${year}"
        start_date="${year}-01-01"
        end_date="${year}-12-31"
        output_file="${OUTPUT_DIR}/${year}.nc"
        
        echo "Date Range: ${start_date} to ${end_date}"
        echo "Output File: ${output_file}"

        python3 "${SCRIPT_DIR}/convert_data_sw.py" \
            --mode convert \
            "${start_date}" "${end_date}" \
            --stats_file "${STATS_FILE}" \
            --output_file "${output_file}" \
            --msm_dir "${MSM_DIR}" \
            --max_workers "${MAX_WORKERS}" \
            --temp_dir "${TEMP_ROOT_DIR}"
        
        echo "SUCCESS: Successfully processed year ${year}."
    done
fi

echo "=================================================="
echo "All processing finished successfully."
echo "Final NetCDF files are in: ${OUTPUT_DIR}"
echo "=================================================="