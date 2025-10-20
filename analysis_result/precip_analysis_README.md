# 降水(precip) 分析レポート

- 対象ディレクトリ: /app/src/CompresionRain/optimization_nc
- 処理ファイル数  : 72
- 所要時間        : 1651.04 秒

- 逆変換に使用した precip min/max: min=0.0, max=599.4375

## 指標の要約
- sd_triplet: 平均=0.093272, 標準偏差=0.503160, 最大=99.536912 (N=4038451200)
- max−min  : 平均=0.214806, 標準偏差=1.156436, 最大=223.249996 (N=4038451200)
- ft4 − sum/3: MAE=0.095134, RMSE=0.558773, 符号付平均=0.000740, 最大絶対値=139.916661 (N=4038451200)
- ft5 − sum/3: MAE=0.064449, RMSE=0.406736, 符号付平均=-0.000110, 最大絶対値=129.499992 (N=4038451200)
- ft6 − sum/3: MAE=0.095036, RMSE=0.554929, 符号付平均=-0.000630, 最大絶対値=129.416665 (N=4038451200)
- (ft4+ft5+ft6) − 3h積算: MAE=0.000000, RMSE=0.000000, 符号付平均=0.000000, 最大絶対値=0.000031 (N=4038451200)

## 図と読み方
各図は ./precip_analysis_figs 以下に出力されています。

### 3ターゲット標準偏差の分布
![sd_triplet](precip_analysis_figs/precip_sd_triplet_hist.png)
各時刻・各格子点で3つのターゲット(1時間降水)の標準偏差(sd)の分布。値が小さいほど3者の予測が一致しており、尾の太さは不一致事例の頻度を示します。

### 3ターゲットのレンジ(max−min)分布
![range](precip_analysis_figs/precip_range_hist.png)
同一時刻・格子における3者の最大−最小レンジの分布。小さいほど一致度が高く、右側の尾が長いほど乖離の大きいケースが多いことを示します。

### 誤差分布: ft4 − sum/3
![e4](precip_analysis_figs/precip_err_ft4_minus_sum3_hist.png)
ft4 − (3時間積算/3) の誤差分布。0付近に集中していれば一貫性が高い。正側/負側への偏りは系統的なズレを示唆します。

### 誤差分布: ft5 − sum/3
![e5](precip_analysis_figs/precip_err_ft5_minus_sum3_hist.png)
ft5 − (3時間積算/3) の誤差分布。0近傍集中なら良好、一方で広がりはばらつきの大きさを表します。

### 誤差分布: ft6 − sum/3
![e6](precip_analysis_figs/precip_err_ft6_minus_sum3_hist.png)
ft6 − (3時間積算/3) の誤差分布。0からの偏りは種別ごとの系統誤差やイベント依存性を示します。

### 合計整合性の誤差: (ft4+ft5+ft6) − 3h積算
![esum](precip_analysis_figs/precip_err_sum_consistency_hist.png)
(ft4+ft5+ft6) − (3時間積算) の差。理想は0で、非ゼロは入出力/NaN処理などの差分が疑われます。

### レンジ閾値割合（≤0, 0.1, 0.5, 1.0, 2.0 mm）
![bars](precip_analysis_figs/precip_range_threshold_bars.png)
レンジ(max−min)が各閾値以下となる割合。例: ≤0.1mmの割合が高いほど3者がほぼ一致している領域が多いことを意味します。

### ペア散布(Hexbin): ft4 vs ft5
![p45](precip_analysis_figs/precip_pair_ft4_ft5_hexbin.png)
ft4 と ft5 の散布図(Hexbin)。y=xに沿って分布するほど二者の整合性が高い。斜めからの系統的な外れは偏りを示します。

### ペア散布(Hexbin): ft4 vs ft6
![p46](precip_analysis_figs/precip_pair_ft4_ft6_hexbin.png)
ft4 と ft6 の散布図(Hexbin)。y=xからの広がりは両者の差異の大きさを示します。

### ペア散布(Hexbin): ft5 vs ft6
![p56](precip_analysis_figs/precip_pair_ft5_ft6_hexbin.png)
ft5 と ft6 の散布図(Hexbin)。帯が太いほど多くの点が存在し、y=xから離れるほど不一致が大きいことを示します。

## ターゲット間の相関係数
- corr(ft4,ft5): r=+0.685944
- corr(ft4,ft6): r=+0.476895
- corr(ft5,ft6): r=+0.689786

## 降水強度ビン分布と提案重み

- 1h bins (mm/h): [1.0, 5.0, 10.0, 20.0, 30.0, 50.0]
- 1h 提案重み: [1.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
- sum bins (mm/3h): [2.0, 10.0, 20.0, 40.0, 60.0, 100.0]
- sum 提案重み: [1.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]

### 分布棒グラフ（全点）
![1h_all](precip_analysis_figs/precip_1h_bins_all_bars.png)
![sum_all](precip_analysis_figs/precip_sum_bins_all_bars.png)

### 分布棒グラフ（有降水のみ）
![1h_precip](precip_analysis_figs/precip_1h_bins_precip_only_bars.png)
![sum_precip](precip_analysis_figs/precip_sum_bins_precip_only_bars.png)

### 提案重みの棒グラフ
![w1h](precip_analysis_figs/suggested_weights_1h_bars.png)
![wsum](precip_analysis_figs/suggested_weights_sum_bars.png)

### 設定ファイルへの反映例（swinunet_main_v5_config.py）
```python
CFG['LOSS']['intensity_weight_values_1h'] = [1.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
CFG['LOSS']['intensity_weight_values_sum'] = [1.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
```