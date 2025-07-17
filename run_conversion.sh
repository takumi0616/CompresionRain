#!/bin/bash

# --- 設定 ---
START_YEAR=2018
END_YEAR=2023
# 学習データ期間 (統計量計算用)
TRAIN_START_YEAR=2018
TRAIN_END_YEAR=2018 # 統計量を計算する期間を定義

MAX_WORKERS=22 # マシンのコア数に合わせて調整

# --- パス設定 (改善点①: /home 配下に明示的に指定) ---
# ユーザーのホームディレクトリを基準にパスを構築
USER_HOME=$(eval echo ~${SUDO_USER:-$USER})
# プロジェクトのベースディレクトリを適切に設定してください
BASE_DIR="${USER_HOME}/work_takasuka_git/docker_miniconda/src/CompresionRain"
MSM_DIR="${BASE_DIR}/MSM_data/"
OUTPUT_DIR="${BASE_DIR}/output_nc"
TEMP_ROOT_DIR="${OUTPUT_DIR}/temp"
STATS_FILE="${OUTPUT_DIR}/stats_${TRAIN_START_YEAR}-${TRAIN_END_YEAR}.nc"

# --- 実行モード ---
# "all", "calc_stats", "convert" から選択
# 例: ./run_conversion.sh calc_stats -> 統計量計算のみ実行
# 例: ./run_conversion.sh convert   -> 変換のみ実行
# 例: ./run_conversion.sh           -> 両方実行
MODE=${1:-"all"}

# --- 準備 ---
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${TEMP_ROOT_DIR}"

# --- 1. 統計量の計算 ---
if [[ "$MODE" == "all" || "$MODE" == "calc_stats" ]]; then
    echo "=================================================="
    echo " MODE: Running Statistics Calculation"
    echo "=================================================="
    train_start_date="${TRAIN_START_YEAR}-01-01"
    train_end_date="${TRAIN_END_YEAR}-12-31"
    temp_dir_for_stats="${TEMP_ROOT_DIR}/stats_calc"

    echo "Calculating statistics from ${train_start_date} to ${train_end_date}"
    echo "Statistics file will be saved to: ${STATS_FILE}"

    python3 convert_data.py \
        --mode calc_stats \
        "${train_start_date}" "${train_end_date}" \
        --stats_file "${STATS_FILE}" \
        --msm_dir "${MSM_DIR}" \
        --max_workers "${MAX_WORKERS}" \
        --temp_dir "${temp_dir_for_stats}"

    if [ $? -ne 0 ]; then
        echo "Error during statistics calculation. Exiting."
        exit 1
    fi
    echo "Successfully calculated and saved statistics."
fi

# --- 2. データ変換 ---
if [[ "$MODE" == "all" || "$MODE" == "convert" ]]; then
    echo "=================================================="
    echo " MODE: Running Data Conversion"
    echo "=================================================="
    
    if [ ! -f "${STATS_FILE}" ]; then
        echo "Statistics file not found: ${STATS_FILE}" >&2
        echo "Please run in 'calc_stats' mode first." >&2
        exit 1
    fi

    for year in $(seq ${START_YEAR} ${END_YEAR}); do
        echo "--------------------------------------------------"
        echo "Processing Year: ${year}"
        start_date="${year}-01-01"
        end_date="${year}-12-31"
        output_file="${OUTPUT_DIR}/${year}.nc"
        temp_dir_for_year="${TEMP_ROOT_DIR}/${year}"

        echo "Date Range: ${start_date} to ${end_date}"
        echo "Output File: ${output_file}"
        
        python3 convert_data.py \
            --mode convert \
            "${start_date}" "${end_date}" \
            --stats_file "${STATS_FILE}" \
            --output_file "${output_file}" \
            --msm_dir "${MSM_DIR}" \
            --max_workers "${MAX_WORKERS}" \
            --temp_dir "${temp_dir_for_year}"
        
        if [ $? -eq 0 ]; then
            echo "Successfully processed ${year}."
        else
            echo "Error processing ${year}. Exiting." >&2
            exit 1
        fi
    done
fi

echo "=================================================="
echo "All processing finished successfully."
echo "Final NetCDF files are in: ${OUTPUT_DIR}"
echo "=================================================="