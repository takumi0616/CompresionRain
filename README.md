# コマンド

## 現在

```bash
notify-run gpu01 -- nohup python swinunet_main_v5.py > swinunet_main_v5.log 2>&1 &

notify-run gpu02 -- nohup python swinunet_main_v5.py > swinunet_main_v5.log 2>&1 &

notify-run gpu01 -- nohup python swinunet_main_v6.py > swinunet_main_v6.log 2>&1 &

notify-run gpu02 -- nohup python swinunet_main_v6.py > swinunet_main_v6.log 2>&1 &

pkill -f "swinunet_main_v5.py"

pkill -f "swinunet_main_v6.py"
```

```bash
for year in MSM_data/*/; do
    echo "Processing: $year"
    rm -f "$year"*.5b7b6.idx
done
```

DDP による GPU エラー時のコマンド

```bash
sudo kill -9 1493785
```

## gpu01 → mac

```bash
rsync -avz --progress gpu01:/home/devel/work_takasuka_git/docker_miniconda/src/CompresionRain/swin-unet_main_result_v6_singleGPU /Users/takumi0616/Develop/docker_miniconda/src/CompresionRain/result_gpu01/
```

## 過去

```bash
notify-run gpu01 -- nohup python swinunet_main_v1.py > swinunet_main_v1.log 2>&1 &

notify-run gpu02 -- nohup python optimization_nc_data.py > optimization_nc_data.log 2>&1 &

notify-run gpu02 -- nohup python optimization_nc_data_v2.py > optimization_nc_data_v2.log 2>&1 &

notify-run gpu01 -- nohup python optimization_nc_data.py > optimization_nc_data.log 2>&1 &

notify-run gpu01 -- nohup python optimization_nc_data_v2.py > optimization_nc_data_v2.log 2>&1 &

notify-run gpu01 -- nohup python3 check_data.py > check_data.log 2>&1 &

notify-run gpu01 -- nohup python3 convert_msm_data.py > convert_msm_data.log 2>&1 &

notify-run gpu01 -- nohup python3 convert_nc.py > convert_nc.log 2>&1 &

notify-run gpu01 -- nohup python3 check_output_nc.py > check_output_nc.log 2>&1 &

notify-run gpu01 -- nohup python swinunet_main_v2.py > swinunet_main_v2.log 2>&1 &

notify-run gpu02 -- nohup python analyze_1h_bin_distribution.py > analyze_1h_bin_distribution.log 2>&1 &

notify-run gpu01 -- nohup python separate_main_v1.py > separate_main_v1.log 2>&1 &

notify-run gpu01 -- nohup python separate_main_v2.py > separate_main_v2.log 2>&1 &

nohup python data_check_separate.py > data_check_separate.log 2>&1 &

nohup python check_prmsl.py > check_prmsl.log 2>&1 &

pkill -f "swinunet_main_v1.py"

pkill -f "optimization_nc_data.py"

pkill -f "optimization_nc_data_v2.py"

pkill -f "convert_data_v1.py"

pkill -f "convert_msm_data.py"

pkill -f "convert_nc.py"

pkill -f "swinunet_main_v3.py"

pkill -f "swinunet_main_v5.py"

pkill -f "separate_main_v1.py"

pkill -f "check_output_nc.py"
```

# 実装の詳細

本プロジェクトは「既知の 3 時間積算降水量 S=Prec_4_6h_sum を拘束条件とし、各画素ごとに t+4/t+5/t+6 の配分比率 w=(w4,w5,w6) を予測する」タスクに最適化した実装です。  
モデルは配分ロジット（3 チャネル）を出力し、softmax によって比率へ変換、既知の積算 S を画素ごとに按分して 1 時間値（mm）を生成します。これにより各画素で Σt pred_1h = S が常に成立します。

- 対象コード:
  - モデルとパイプライン: `src/CompresionRain/swinunet_main_v5.py`
  - モデル本体（Swin U-Net）: `src/CompresionRain/swin_unet.py`
  - 設定: `src/CompresionRain/swinunet_main_v5_config.py`
  - スケーリング係数: `src/CompresionRain/scaler_groups.json`

## 全体フロー（処理の順序）

1. 設定読み込み・ログ初期化（`swinunet_main_v5_config.py` / `setup_loggers()`）
2. NetCDF 月次ファイル列挙（`get_monthly_files`）
3. データセット構築（`NetCDFDataset`）
   - 入力テンソル `input` と、ターゲット `target_1h`（mm, 3ch）、`target_sum`（mm, 1ch）を返す
4. モデル初期化（`SwinTransformerSys(**MODEL_ARGS)`）
5. 学習ループ（各エポック）
   - AMP による forward（bf16/fp16）
   - 損失計算（配分 →mm 変換後の加重 MSE）
   - 逆伝播・最適化・LR スケジューラ
   - 検証（拡張メトリクス算出）
   - 最良モデル保存
6. ランク 0 のみで最終評価（テキスト出力）・等分(S/3)との詳細比較評価・可視化・曲線プロット・動画化

## データ I/O とスケーリング

- データ読み込み: `NetCDFDataset`
  - 変数定義は `swinunet_main_v5_config.CFG["DATA"]` による
    - 共通入力（ft+3/ft+6 の 2 面）: `INPUT_VARS_COMMON`
    - 降水入力: `INPUT_VARS_PREC=["Prec_ft3"]`
    - 時間特徴: `TIME_VARS=["dayofyear_sin","dayofyear_cos","hour_sin","hour_cos"]`
  - 返却するテンソル（PyTorch, float32）
    - `input` 形状: `(C_in, H, W)`  
      C_in = `len(INPUT_VARS_COMMON)*2 + len(INPUT_VARS_PREC) + len(TIME_VARS)`
    - `target_1h` 形状: `(3, H, W)` … `["Prec_Target_ft4","Prec_Target_ft5","Prec_Target_ft6"]`
    - `target_sum` 形状: `(1, H, W)` … `"Prec_4_6h_sum"`
  - 降水のスケール: optimization_nc は 0-1 正規化で保存
    - `inverse_precip_tensor(t, group_minmax)` により mm へ逆変換（`scaler_groups.json` の `group_minmax["precip"]` を使用）

## モデル（Swin U-Net）と出力解釈

- モデル: `SwinTransformerSys`（`swin_unet.py`）
  - 入力: `(B, C_in, H, W)`
  - 出力: `(B, 3, H, W)` の「配分ロジット」
  - 重要パラメータ（設定ファイル `swinunet_main_v5_config.py`）
    - `MODEL_ARGS` … embed_dim, depths, num_heads, window_size など
    - `ALLOCATION_TAU`（`CFG["MODEL"]["allocation_softmax_tau"]` をエクスポート）… softmax 温度 τ

### 比率 →mm 変換（核心ロジック）

実体は `swinunet_main_v5.py` の `allocation_logits_to_mm(logits, sum_mm, tau)`:

```python
weights = torch.softmax(logits / tau, dim=1)  # (B,3,H,W) 各画素でΣ=1, 非負
pred_1h_mm = weights * sum_mm                 # (B,3,H,W), sum_mm は (B,1,H,W) の既知S
```

- `sum_mm` はデータセットが返す `target_sum`（mm, 既知の 3h 積算）
- 特徴:
  - 物理制約 Σt pred_1h = S を常に満たす
  - S=0 の画素は自動的に pred_1h=0 になる
  - τ（`ALLOCATION_TAU`）で分配の尖り/平坦化を制御（τ<1 で尖る、>1 で均す）

## 損失関数（mm 直比較）

`custom_loss_function(output_logits, targets)`（`swinunet_main_v5.py`）

- 事前整形:
  - `target_1h` / `target_sum` は mm へ逆変換済み（Dataset 内）
  - `pred_1h_mm = allocation_logits_to_mm(output_logits, target_sum, ALLOCATION_TAU)`
- 誤差:
  - 1 時間（3ch）: 強度重み付き MSE（`CFG["LOSS"]` のビン・重みを使用）
  - 積算（1ch）: 強度重み付き MSE（理論上一致だが丸め誤差対策で残置）
- ログ用 RMSE:
  - `unweighted_mse_1h = MSE(pred_1h_mm, target_1h)`
  - `unweighted_mse_sum = MSE(Σpred_1h_mm, target_sum)`

### 強度重み（クラス不均衡対策）

- 1h 用ビン/重み: `INTENSITY_WEIGHT_BINS_1H`, `INTENSITY_WEIGHT_VALUES_1H`
- 3h 積算用ビン/重み: `INTENSITY_WEIGHT_BINS_SUM`, `INTENSITY_WEIGHT_VALUES_SUM`
- 実装: `_get_weight_map_from_bins` + `_weighted_mse`

## 検証指標（validate_one_epoch）

- すべて mm スケールで計算（出力は `allocation_logits_to_mm` で mm 化）
- バイナリ指標（閾値ごと; 時刻別/積算）
  - Accuracy, Precision, Recall, F1, CSI（`BINARY_THRESHOLDS_MM_1H / _SUM`）
- カテゴリ指標（`CATEGORY_BINS_MM_1H / _SUM`）
  - overall accuracy、クラス別 precision / recall
- RMSE
  - 1h（モデル vs 正解）
  - 等分ベースライン（equal-split: S/3）との 1h RMSE 比較（時刻をまたぐ全画素）
  - 積算 RMSE
- Wet-hour パターン（0/1/2/3 個の湿潤画素）の一致度

## 等分(S/3)ベースラインとの詳細比較（新実装）

「各格子点において、単純な S/3 よりモデルの比率予測の方が良いか」を詳細に確認するため、次を追加:

- `evaluate_equal_split_baseline_detailed(model_path, valid_dataset, device, out_txt, out_npz, logger)`
  - 位置: `swinunet_main_v5.py`
  - 出力:
    - `RESULT_DIR/equal_split_detailed_v5.txt`
      - Overall: 1h RMSE（モデル/等分 S/3）, 改善画素比率（SSE_model < SSE_base）
      - Per-hour RMSE（FT+4/5/6）
      - 強度ビン（S のカテゴリ）別の改善率
    - `RESULT_DIR/equal_split_maps_v5.npz`
      - `improved_count_map`（勝った回数 / 画素）
      - `total_count_map`（評価回数 / 画素）
      - `mean_sse_diff_map` = 平均(SSE_model - SSE_base)（負ならモデル有利）
  - 算出の要点:
    - モデル予測: `pred_1h = w × S`
    - 等分: `base_1h = S/3` を 3 チャネルに複製
    - SSE（3 時刻合計）を画素単位で比較し、改善の有無・差を集計

## 可視化（PNG）と動画化（MP4）

- `visualize_final_results`（mm スケール）
  - 2x4 パネル: 予測（積算, FT+4/5/6）、正解（積算, FT+4/5/6）
  - カラースケール: LogNorm（自動 vmax, `LOGNORM_VMIN_MM` 下限）
  - 追加で線形 0-600mm 版（RN ディレクトリ）も保存
- `create_video_from_images`
  - `validation_*.png` を mp4 へ連結（`VIDEO_FPS`）

## 分散学習・AMP・最適化

- DDP（GPU 数 > 1 で自動使用, `nccl/gloo`）
- AMP: `AMP_DTYPE` に応じて `torch.float16` or `torch.bfloat16`
- Optimizer: AdamW（`LEARNING_RATE * world_size`）
- LR Scheduler: CosineAnnealingLR（`T_max=NUM_EPOCHS`）
- 勾配クリップ: `clip_grad_norm_(..., max_norm=1.0)`
- GradScaler: fp16 時のみ有効化

## 設定項目（主要）

- パス系: `RESULT_DIR`, `MODEL_SAVE_PATH`, `RESULT_IMG_DIR(_RN)`, `EVALUATION_LOG_PATH`, `VIDEO_OUTPUT_PATH` …
- データ: `TRAIN_YEARS`, `VALID_YEARS`, `IMG_SIZE`, `PATCH_SIZE`, `NUM_CLASSES=3`
- 入力チャネル数: `IN_CHANS = len(INPUT_VARS_COMMON)*2 + len(INPUT_VARS_PREC) + len(TIME_VARS)`
- 学習: `BATCH_SIZE`, `NUM_EPOCHS`, `LEARNING_RATE`
- 損失: `ENABLE_INTENSITY_WEIGHTED_LOSS` と各ビン/重み
- モデル: `MODEL_ARGS`（Swin U-Net 構成）, `ALLOCATION_TAU`（softmax 温度 τ）

## テンソル形状まとめ

- `input`: `(B, C_in, H, W)`
- モデル出力（配分ロジット）: `(B, 3, H, W)`
- `target_1h`（mm）: `(B, 3, H, W)`（Dataset 内で mm へ逆変換済）
- `target_sum`（mm）: `(B, 1, H, W)`（Dataset 内で mm へ逆変換済）
- 予測 mm: `pred_1h_mm = softmax(logits/τ) * target_sum` → `(B, 3, H, W)`

## 実行コマンド例

```bash
# GPU ノードで学習・評価・可視化・動画化まで一括
notify-run gpu01 -- nohup python swinunet_main_v5.py > swinunet_main_v5.log 2>&1 &
```

## 実装上のポイント（設計意図）

- 物理量拘束（Σt pred = S）を出力層の解釈で保証（post-process 的に S を掛けるだけ）
- 比率の非負・総和 1 は softmax により自動保証
- 等分(S/3)ベースラインとの比較を「全体 RMSE」だけでなく「格子点・強度帯・時刻別」まで掘り下げて可視化・数値化
- 既存の評価・出力の枠組み（図・動画）は mm スケールで一貫して維持
