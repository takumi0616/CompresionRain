# 降水(precip) 分析レポート

- 対象ディレクトリ: /app/src/CompresionRain/optimization_nc
- 処理ファイル数 : 72
- 所要時間 : 1596.10 秒

- 逆変換に使用した precip min/max: min=0.0, max=599.4375

## 指標の要約

- sd_triplet: 平均=0.093272, 標準偏差=0.503160, 最大=99.536912 (N=4038451200)
- max−min : 平均=0.214806, 標準偏差=1.156436, 最大=223.249996 (N=4038451200)
- ft4 − sum/3: MAE=0.095134, RMSE=0.558773, 符号付平均=0.000740, 最大絶対値=139.916661 (N=4038451200)
- ft5 − sum/3: MAE=0.064449, RMSE=0.406736, 符号付平均=-0.000110, 最大絶対値=129.499992 (N=4038451200)
- ft6 − sum/3: MAE=0.095036, RMSE=0.554929, 符号付平均=-0.000630, 最大絶対値=129.416665 (N=4038451200)
- (ft4+ft5+ft6) − 3h 積算: MAE=0.000000, RMSE=0.000000, 符号付平均=0.000000, 最大絶対値=0.000031 (N=4038451200)

## 図と読み方

各図は ./precip_analysis_figs 以下に出力されています。

### 3 ターゲット標準偏差の分布

![sd_triplet](precip_analysis_figs/precip_sd_triplet_hist.png)
各時刻・各格子点で 3 つのターゲット(1 時間降水)の標準偏差(sd)の分布。値が小さいほど 3 者の予測が一致しており、尾の太さは不一致事例の頻度を示します。

### 3 ターゲットのレンジ(max−min)分布

![range](precip_analysis_figs/precip_range_hist.png)
同一時刻・格子における 3 者の最大 − 最小レンジの分布。小さいほど一致度が高く、右側の尾が長いほど乖離の大きいケースが多いことを示します。

### 誤差分布: ft4 − sum/3

![e4](precip_analysis_figs/precip_err_ft4_minus_sum3_hist.png)
ft4 − (3 時間積算/3) の誤差分布。0 付近に集中していれば一貫性が高い。正側/負側への偏りは系統的なズレを示唆します。

### 誤差分布: ft5 − sum/3

![e5](precip_analysis_figs/precip_err_ft5_minus_sum3_hist.png)
ft5 − (3 時間積算/3) の誤差分布。0 近傍集中なら良好、一方で広がりはばらつきの大きさを表します。

### 誤差分布: ft6 − sum/3

![e6](precip_analysis_figs/precip_err_ft6_minus_sum3_hist.png)
ft6 − (3 時間積算/3) の誤差分布。0 からの偏りは種別ごとの系統誤差やイベント依存性を示します。

### 合計整合性の誤差: (ft4+ft5+ft6) − 3h 積算

![esum](precip_analysis_figs/precip_err_sum_consistency_hist.png)
(ft4+ft5+ft6) − (3 時間積算) の差。理想は 0 で、非ゼロは入出力/NaN 処理などの差分が疑われます。

### レンジ閾値割合（≤0, 0.1, 0.5, 1.0, 2.0 mm）

![bars](precip_analysis_figs/precip_range_threshold_bars.png)
レンジ(max−min)が各閾値以下となる割合。例: ≤0.1mm の割合が高いほど 3 者がほぼ一致している領域が多いことを意味します。

### ペア散布(Hexbin): ft4 vs ft5

![p45](precip_analysis_figs/precip_pair_ft4_ft5_hexbin.png)
ft4 と ft5 の散布図(Hexbin)。y=x に沿って分布するほど二者の整合性が高い。斜めからの系統的な外れは偏りを示します。

### ペア散布(Hexbin): ft4 vs ft6

![p46](precip_analysis_figs/precip_pair_ft4_ft6_hexbin.png)
ft4 と ft6 の散布図(Hexbin)。y=x からの広がりは両者の差異の大きさを示します。

### ペア散布(Hexbin): ft5 vs ft6

![p56](precip_analysis_figs/precip_pair_ft5_ft6_hexbin.png)
ft5 と ft6 の散布図(Hexbin)。帯が太いほど多くの点が存在し、y=x から離れるほど不一致が大きいことを示します。

## ターゲット間の相関係数

- corr(ft4,ft5): r=+0.685944
- corr(ft4,ft6): r=+0.476895
- corr(ft5,ft6): r=+0.689786
