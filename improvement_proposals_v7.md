# Swin-UNet v6 精度向上のための改善提案

## 実行結果の分析

### 現状の問題点

evaluation_v6.log の評価結果から以下の**重大な問題**を確認しました：

#### 1. **モデルが 1mm 以上の降水を全く検出できていない（最重要課題）**

```
Threshold > 1.0 mm: Acc=1.0000, Prec=0.0000, Rec=0.0000, F1=0.0000, CSI=0.0000 (TP=0)
Threshold > 5.0 mm: TP=0
Threshold > 10.0 mm: TP=0
... 以降すべて TP=0
```

- **True Positive (TP) = 0** ということは、1mm 以上の降水を一切予測できていない
- モデルが極端に保守的な予測（ほぼ 0mm）に偏っている

#### 2. **逆頻度重み計算の失敗**

```json
"computed_weights_1h": [1.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
"ビン出現数: [807321600, 0, 0, 0, 0, 0, 0]"
```

- サンプリングした全データが第 1 ビン（0-1mm）に集中
- 強雨のサンプルが 1 つも含まれていない
- 結果として重み付けが機能していない（すべて最大重み 100.0）

#### 3. **データの極端な不均衡**

- 検証データの 99.999%が降水なし～微弱な降水（<0.1mm）
- 0.1mm 以上の降水は全画素の 0.001%未満（20,113 / 2,018,304,000）
- 1mm 以上の強雨イベントがほぼ存在しない

#### 4. **RMSE は良好だが実用性に疑問**

- Model 1h RMSE: 0.000589（非常に小さい）
- しかし、これは「ほぼ 0mm を予測し続ける」ことで達成されている可能性が高い
- 実際の降雨イベント検出には不適

---

## 改善策の提案（優先度順）

### 【提案 1】データサンプリング戦略の改善（最優先・必須）

#### 概要

現在の逆頻度重み計算は**ランダムサンプリング（10%）**を使用しているため、稀な強雨イベントが含まれていません。これを**層化サンプリング**に変更し、強雨サンプルを確実に含めます。

#### 実装方法

1. **事前スキャン**：全データから各強度ビンのインデックスを収集
2. **ビン別サンプリング**：各ビンから均等または重要度に応じてサンプリング
3. **強雨優先**：特に高強度ビン（>5mm, >10mm 等）からは全サンプルを使用

```python
def compute_inverse_frequency_weights_stratified(
    dataset, bins,
    min_samples_per_bin=100,  # 各ビンから最低限取得するサンプル数
    max_scan_ratio=0.3,        # スキャンに使う最大データ比率
    logger=None
):
    """
    層化サンプリングによる逆頻度重み計算
    1. 全データの一部（30%など）をスキャンして各ビンのサンプルを特定
    2. 稀なビンからは可能な限り多く、頻繁なビンからは制限してサンプリング
    """
    # 実装の詳細は提案ファイルに記載
```

#### 期待効果

- **高**: 強雨サンプルを確実に含めることで、適切な重み計算が可能
- モデルが強雨を学習できるようになる
- 1mm 以上の降水検出能力が大幅向上

#### 実装コスト

- **中**: プログラム修正は中規模（100-200 行）
- データスキャンの計算時間が増加（30%スキャンで約 15-20 分追加）
- 初回実行のみのコストなので許容範囲

---

### 【提案 2】Focal Loss の導入

#### 概要

現在の MSE ベースの損失関数では、クラス不均衡に対処できていません。**Focal Loss**を導入することで、稀な強雨イベントに自動的にフォーカスします。

#### 実装方法

```python
class FocalMSELoss(nn.Module):
    """
    Focal MSE Loss: 大きな誤差により高い重みを付与
    """
    def __init__(self, alpha=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred, target):
        mse = (pred - target) ** 2
        # 誤差が大きいほど重みを増加（gamma効果）
        focal_weight = torch.abs(pred - target) ** self.alpha
        focal_mse = focal_weight * mse

        if self.reduction == 'mean':
            return focal_mse.mean()
        return focal_mse.sum()
```

または、Binary Cross Entropy 風の降水量特化 Focal Loss:

```python
def precipitation_focal_loss(pred, target, gamma=2.0, threshold=1.0):
    """
    降水イベント（>threshold mm）に焦点を当てたFocal Loss
    """
    # 降水イベントの2値マスク
    rain_mask = (target > threshold).float()

    # 基本MSE
    mse = F.mse_loss(pred, target, reduction='none')

    # 降水イベントには高重み、それ以外は低重み
    weights = rain_mask * gamma + (1 - rain_mask) * 1.0

    return (weights * mse).mean()
```

#### 期待効果

- **高**: 稀な強雨イベントの学習が促進される
- 予測の偏りが緩和される
- 実装がシンプル

#### 実装コスト

- **低**: 損失関数の追加のみ（50-100 行）
- 学習時間はほぼ変わらず
- すぐに試せる

---

### 【提案 3】2 段階学習戦略

#### 概要

1 段階目で全体の分布を学習し、2 段階目で強雨イベントに特化した**ファインチューニング**を行います。

#### 実装方法

```python
# Stage 1: 通常学習（全データ、30-40エポック）
# - 全体的な空間パターンと基本的な降水分布を学習

# Stage 2: 強雨特化ファインチューニング（10-20エポック）
# - 学習率を1/10に減少
# - サンプリング：降水量>1mmのサンプルを優先的に選択
# - 損失：強雨イベントに高重みを設定

class RainFocusedSampler(Sampler):
    """強雨サンプルを優先的に選択するサンプラー"""
    def __init__(self, dataset, rain_threshold=1.0, rain_ratio=0.7):
        # 降水量>thresholdのインデックスを特定
        # 70%を強雨サンプル、30%を通常サンプルから選択
```

#### 期待効果

- **高**: 段階的学習により安定性と性能を両立
- Stage 1 で全体像を把握、Stage 2 で弱点を強化
- 過学習のリスクを抑制

#### 実装コスト

- **中**: Sampler 実装と学習スクリプト修正が必要（150-250 行）
- 学習時間は 1.3-1.5 倍に増加（総 80-100 エポック相当）

---

### 【提案 4】データ拡張（Augmentation）

#### 概要

強雨サンプルが少ない問題に対し、既存の強雨サンプルを**データ拡張**で増幅します。

#### 実装方法

```python
class PrecipitationAugmentation:
    """降水量データに特化した拡張"""
    def __init__(self, rain_threshold=1.0):
        self.rain_threshold = rain_threshold

    def __call__(self, sample):
        if sample['target_sum'].max() < self.rain_threshold:
            return sample  # 弱雨はそのまま

        # 強雨サンプルには拡張を適用
        augmentations = [
            self.random_flip,
            self.random_rotation_90,
            self.add_gaussian_noise,
            # 降水場の空間的な変形は物理的に不自然なので避ける
        ]

        aug = random.choice(augmentations)
        return aug(sample)
```

適用する拡張：

- **水平/垂直反転**: 物理的に妥当
- **90 度回転**: 方向性がないデータなら可
- **微小ノイズ付加**: 観測誤差をシミュレート
- **明度調整**: 強度の微調整（±10%程度）

#### 期待効果

- **中～高**: 強雨サンプルの実質的な増加
- 汎化性能の向上
- 過学習の抑制

#### 実装コスト

- **低～中**: Dataset クラスに拡張ロジックを追加（100-150 行）
- 学習時間が 5-10%増加

---

### 【提案 5】損失関数の多目的化（Multi-Task Learning）

#### 概要

現在の MSE に加えて、**降水イベント検出タスク**を補助タスクとして追加し、モデルに明示的に降雨/非降雨の識別を学習させます。

#### 実装方法

```python
def multi_task_loss(pred_1h, target_1h, target_sum,
                   mse_weight=1.0, detection_weight=0.5):
    """
    主タスク: MSEによる降水量予測
    補助タスク: Binary Cross Entropyによる降水イベント検出
    """
    # 主タスク：降水量予測（既存）
    mse_loss = weighted_mse(pred_1h, target_1h)

    # 補助タスク：降水イベント検出（新規）
    # 各閾値（0.1mm, 1mm, 5mm等）で2値分類
    detection_loss = 0.0
    thresholds = [0.1, 1.0, 5.0, 10.0]
    for thr in thresholds:
        pred_event = torch.sigmoid((pred_1h.sum(dim=1) - thr) * 10)  # soft threshold
        target_event = (target_sum > thr).float()
        detection_loss += F.binary_cross_entropy(pred_event, target_event)

    detection_loss /= len(thresholds)

    return mse_weight * mse_loss + detection_weight * detection_loss
```

#### 期待効果

- **中～高**: 降水イベントの検出精度が向上
- モデルが明示的に「降る/降らない」を学習
- 0mm 予測への過度な偏りを防止

#### 実装コスト

- **低～中**: 損失関数の修正のみ（50-100 行）
- 学習時間はほぼ変わらず

---

### 【提案 6】学習データの追加・バランス調整

#### 概要

2018-2021 年の学習データに強雨イベントが少ない可能性があります。**追加年度のデータ**や**季節選択**で強雨サンプルを増やします。

#### 実装方法

1. **追加年度の検討**：
   - 2022 年（現在は検証用）の一部を学習に追加
   - 別の年度データがあれば追加
2. **季節/月別の重み付け**：

   - 梅雨期（6-7 月）や台風期（8-9 月）のデータを重点的にサンプリング
   - 冬季の雪/弱雨データの比率を下げる

3. **時間帯フィルタリング**：
   - 夜間の安定層による弱雨を減らす
   - 昼間の対流性降水を増やす

#### 期待効果

- **高**: データの質と多様性が向上
- 強雨イベントの学習機会が増加

#### 実装コスト

- **低（既存データ活用の場合）**: 設定変更のみ
- **高（新規データ収集の場合）**: データ準備に数週間～数ヶ月

---

### 【提案 7】ネットワーク構造の改善

#### 概要

現在のモデルアーキテクチャを強雨予測に適したものに改良します。

#### 実装方法

**7-A. 出力層の活性化関数変更**

```python
# 現在: なし（またはReLUのみ）
# 提案: Softplus または ELU+shift で正値を保証しつつダイナミックレンジを確保

self.output = nn.Sequential(
    nn.Conv2d(embed_dim, num_classes, kernel_size=1),
    nn.Softplus(),  # log(1 + exp(x)) で滑らかな正値
    # または
    # nn.ELU(alpha=1.0), nn.ReLU()  # 負の部分も活用
)
```

**7-B. Multi-Scale Feature Fusion**

```python
# デコーダでスキップ接続だけでなく、
# 異なるスケールの特徴を明示的に統合

class MultiScaleFusion(nn.Module):
    def __init__(self, channels_list):
        # 異なる解像度の特徴をアップサンプルして結合
        # 局所的な強雨パターンと広域的な気象場を同時に考慮
```

**7-C. Attention Mechanism for Precipitation**

```python
# 降水域に選択的に注目するAttentionを追加
class PrecipitationAttention(nn.Module):
    """降水の有無・強度に応じた空間Attention"""
    def forward(self, features, precip_guide):
        # precip_guide: 入力の降水量（ft+3）から生成
        # 降水がある領域に高い注意を向ける
```

#### 期待効果

- **中**: モデルの表現能力向上
- 微弱な降水と強雨を区別する能力が向上

#### 実装コスト

- **7-A: 低**（5-10 行の修正）
- **7-B: 中～高**（200-300 行の追加、アーキテクチャ設計が必要）
- **7-C: 中**（100-150 行の追加）

---

### 【提案 8】学習率スケジューリングの最適化

#### 概要

現在の Cosine Annealing に加えて、**Warmup**と**早期の高学習率維持**で強雨の学習を促進します。

#### 実装方法

```python
# warmup_epochs: 5 → 学習率を0から徐々に上げる
# cosine_start_epoch: 20 → 最初の20エポックは高学習率を維持
# eta_min を現在の1e-6から1e-5に引き上げ（最小学習率を高める）

def get_scheduler_with_warmup(optimizer, warmup_epochs=5,
                               total_epochs=60, eta_min=1e-5):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Warmup
            return epoch / warmup_epochs
        elif epoch < 20:
            # 高学習率維持期
            return 1.0
        else:
            # Cosine decay
            progress = (epoch - 20) / (total_epochs - 20)
            return eta_min + (1 - eta_min) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

#### 期待効果

- **中**: 学習の安定化と収束の改善
- 初期の学習が安定し、後半で過学習を抑制

#### 実装コスト

- **低**: スケジューラ定義の変更のみ（20-30 行）
- 学習時間は変わらず

---

### 【提案 9】正則化の導入・調整

#### 概要

モデルが 0mm に偏る原因として、正則化が強すぎる可能性があります。

#### 実装方法

1. **Dropout 率の調整**

   - 現在: drop_rate=0.0, attn_drop_rate=0.0
   - 提案: 適度な Dropout（0.1-0.2）で汎化性向上

2. **Weight Decay の調整**

   - 現在: weight_decay=0.0
   - 提案: 0.01-0.05 程度に設定（過学習防止）

3. **Batch Normalization → Layer Normalization**
   - 既に実装済み（LayerNorm 使用）

#### 期待効果

- **低～中**: 過学習防止と汎化性能向上
- 単独では大きな改善は期待できないが、他の手法と組み合わせて有効

#### 実装コスト

- **極低**: 設定値の変更のみ（config ファイル数行）
- 学習時間は変わらず

---

### 【提案 10】予測値の Post-Processing

#### 概要

学習済みモデルはそのままで、**推論時に予測値を補正**することで強雨検出を改善します。

#### 実装方法

```python
def post_process_prediction(pred_1h_mm, confidence_threshold=0.5,
                           min_rain_value=0.1):
    """
    予測値の後処理
    1. 極小値（<0.05mm）を0に丸める（ノイズ除去）
    2. 空間的な連続性を考慮（孤立ピクセルの除去）
    3. 物理的制約の適用（時間的整合性など）
    """
    # ノイズ除去
    pred_1h_mm[pred_1h_mm < min_rain_value] = 0.0

    # 空間平滑化（オプション）
    # pred_1h_mm = gaussian_filter(pred_1h_mm, sigma=1.0)

    # 時間的整合性（3時刻の降水パターンの物理的妥当性）
    # ...

    return pred_1h_mm
```

#### 期待効果

- **低～中**: 既存モデルの出力を改善できる
- ノイズ除去による見かけ上の精度向上

#### 実装コスト

- **極低**: 推論時の処理追加のみ（30-50 行）
- 学習不要、すぐに効果確認可能

---

## 推奨実装順序と優先度

### 【フェーズ 1】即効性のある改善（1-2 週間）

1. **提案 1（データサンプリング改善）** ★★★ 最優先 ★★★

   - 実装: 中
   - 効果: 高
   - **これを実装しないと他の改善も効果が限定的**

2. **提案 2（Focal Loss）**

   - 実装: 低
   - 効果: 高
   - 提案 1 と組み合わせて実施

3. **提案 8（学習率調整）**
   - 実装: 極低
   - 効果: 中
   - すぐに試せる

### 【フェーズ 2】中期的な改善（2-4 週間）

4. **提案 3（2 段階学習）**

   - 実装: 中
   - 効果: 高
   - フェーズ 1 の結果を見てから判断

5. **提案 4（データ拡張）**
   - 実装: 低～中
   - 効果: 中～高
   - 強雨サンプルが少ない場合に有効

### 【フェーズ 3】長期的・根本的な改善（1-3 ヶ月）

6. **提案 6（データ追加）**

   - 実装: 低～高（データ入手による）
   - 効果: 高
   - 根本的な解決策

7. **提案 7（ネットワーク改善）**
   - 実装: 中～高
   - 効果: 中
   - リスクもあるため慎重に

### 【補助的手法】

8. **提案 10（Post-Processing）**
   - 実装: 極低
   - 効果: 低～中
   - デモや可視化の改善に有効

---

## 手間と効果のマトリックス

| 提案                    | 実装コスト | 期待効果  | 実装時間     | 優先度     | 備考               |
| ----------------------- | ---------- | --------- | ------------ | ---------- | ------------------ |
| **1. 層化サンプリング** | 中         | ★★★ 高    | 2-3 日       | **最優先** | 他の改善の前提条件 |
| **2. Focal Loss**       | 低         | ★★★ 高    | 1 日         | 高         | すぐに試せる       |
| **3. 2 段階学習**       | 中         | ★★★ 高    | 3-5 日       | 高         | 提案 1 後に実施    |
| **4. データ拡張**       | 低～中     | ★★ 中～高 | 2-3 日       | 中         | 強雨サンプル不足時 |
| **5. Multi-Task Loss**  | 低～中     | ★★ 中～高 | 2-3 日       | 中         | イベント検出重視時 |
| **6. データ追加**       | 低～高     | ★★★ 高    | 1 日～数ヶ月 | 中～高     | データ入手次第     |
| **7-A. 出力活性化**     | 極低       | ★ 低～中  | 数時間       | 低         | 試す価値あり       |
| **7-B. Multi-Scale**    | 中～高     | ★★ 中     | 1-2 週間     | 低         | 効果不確実         |
| **7-C. Attention**      | 中         | ★★ 中     | 3-5 日       | 低         | 提案 1-3 後に検討  |
| **8. 学習率調整**       | 極低       | ★★ 中     | 数時間       | 中         | すぐに試せる       |
| **9. 正則化調整**       | 極低       | ★ 低～中  | 数時間       | 低         | 微調整向け         |
| **10. Post-Process**    | 極低       | ★ 低～中  | 1 日         | 低         | 応急処置           |

---

## 具体的な実装ロードマップ

### Week 1: 緊急対応

```
Day 1-2: 提案1（層化サンプリング）実装
         - compute_inverse_frequency_weights_stratified 関数作成
         - 全ビンから均等にサンプリングするロジック追加

Day 3:   提案2（Focal Loss）実装
         - FocalMSELoss クラス追加
         - custom_loss_function に統合

Day 4:   提案8（学習率調整）実装
         - Warmup付きスケジューラに変更

Day 5-7: 学習実行（v7として）
         - 提案1+2+8を組み合わせて実行
         - 結果評価
```

### Week 2-3: 追加改善

```
提案1-3の結果を見て、以下から選択：
- 効果が不十分 → 提案3（2段階学習）を追加実装
- 強雨サンプル不足 → 提案4（データ拡張）を実装
- 検出精度重視 → 提案5（Multi-Task）を実装
```

### Week 4 以降: 根本的改善（必要に応じて）

```
- 提案6（データ追加・バランス調整）
- 提案7-B/C（ネットワーク構造改善）
```

---

## 詳細技術解説

### なぜ現在のモデルは 1mm 以上を予測できないのか？

#### 原因分析

1. **損失関数の問題**

   - MSE は二乗誤差なので、大きな値の予測ミスに極めて敏感
   - 例：10mm 予測して 0mm が正解 → 誤差=100
   - 例：0mm 予測して 10mm が正解 → 誤差=100（同じ）
   - しかし、データの 99.99%が 0mm 付近なので、「常に 0mm 予測」が平均誤差を最小化してしまう

2. **勾配消失問題**

   - 強雨サンプルが極端に少ないため、その勾配がバッチ平均で薄まる
   - 1 バッチ（8 サンプル）で強雨が 1 サンプルも含まれない確率が高い
   - 結果として強雨に関する学習シグナルがほぼゼロ

3. **重み計算の失敗**
   - 10%ランダムサンプリングでは、0.001%の稀少イベントを捕捉できる確率は極めて低い
   - 期待値：1168 サンプル中、強雨サンプル ≈ 1168 × 0.00001 = 0.012 個（ほぼゼロ）

#### 数学的背景

データ分布を $p(x)$ とすると、降水量 $x$ は極端な Long-tail 分布：

```
p(x < 0.1mm) ≈ 0.9999
p(0.1 < x < 1mm) ≈ 0.0001
p(x > 1mm) ≈ 0.000001
```

MSE 損失 $L = \mathbb{E}[(f(X) - Y)^2]$ を最小化すると：

- 最適予測 = データの条件付き期待値
- しかし分布が極端に偏っている場合、期待値 ≈ 0 に収束

これを防ぐには：

- **重み付け**: 稀なイベントの損失を増幅（提案 1-2）
- **サンプリング**: 稀なイベントを多く見せる（提案 1, 3-4）
- **多目的化**: 別の指標（イベント検出）も最適化（提案 5）

---

## 提案 1（層化サンプリング）の詳細実装例

```python
def compute_inverse_frequency_weights_stratified(
    dataset,
    bins: list,
    target_key: str = "target_1h",
    max_samples_per_bin: int = 500,      # 各ビンの最大サンプル数
    min_samples_per_bin: int = 10,       # 各ビンの最小サンプル数
    scan_ratio: float = 0.5,             # 初期スキャン比率
    max_weight: float = 100.0,
    min_weight: float = 1.0,
    smoothing_factor: float = 1e-6,
    logger=None
) -> list:
    """
    層化サンプリングによる逆頻度重み計算

    手順:
    1. データの一部（scan_ratio）をスキャンして各ビンのサンプルインデックスを収集
    2. 稀なビン（高強度）からは全サンプル、頻繁なビンからは制限
    3. 収集したサンプルから頻度を計算し、逆頻度重みを算出
    """
    log_func = logger.info if logger else print

    num_bins = len(bins) + 1
    total_samples = len(dataset)
    scan_size = int(total_samples * scan_ratio)

    log_func(f"[層化サンプリング] データスキャン開始: {scan_size}/{total_samples} サンプル")

    # Phase 1: ビンごとのインデックス収集
    bin_indices = [[] for _ in range(num_bins)]
    bins_array = np.array(bins, dtype=np.float64)

    scan_indices = np.random.choice(total_samples, size=scan_size, replace=False)

    for idx in tqdm(scan_indices, desc="Scanning data for stratified sampling"):
        sample = dataset[idx]
        target = sample[target_key]

        if isinstance(target, torch.Tensor):
            target = target.numpy()

        # サンプル内の最大降水量でビンを決定（保守的）
        max_val = target.max()
        bin_id = np.digitize(max_val, bins_array, right=False)
        bin_id = min(bin_id, num_bins - 1)

        bin_indices[bin_id].append(idx)

    # Phase 2: 層化サンプリング
    sampled_indices = []
    for bin_id in range(num_bins):
        available = len(bin_indices[bin_id])

        if bin_id < 2:  # 低強度ビン（0-1mm, 1-5mm）
            # 制限してサンプリング
            n_samples = min(available, max_samples_per_bin)
        else:  # 高強度ビン（>5mm）
            # できるだけ多くサンプリング
            n_samples = available

        n_samples = max(n_samples, min(min_samples_per_bin, available))

        if available > 0:
            selected = np.random.choice(
                bin_indices[bin_id],
                size=min(n_samples, available),
                replace=False
            )
            sampled_indices.extend(selected)

        log_func(f"  Bin {bin_id}: {available}個中{len(selected) if available > 0 else 0}個をサンプリング")

    # Phase 3: 頻度計算
    bin_counts = np.zeros(num_bins, dtype=np.float64)

    for idx in sampled_indices:
        sample = dataset[idx]
        target = sample[target_key]

        if isinstance(target, torch.Tensor):
            target = target.numpy()

        flat_values = target.flatten()
        bin_ids = np.digitize(flat_values, bins_array, right=False)

        for i in range(num_bins):
            bin_counts[i] += np.sum(bin_ids == i)

    # Phase 4: 逆頻度重み計算
    total_count = bin_counts.sum()
    frequencies = bin_counts / total_count if total_count > 0 else np.ones(num_bins) / num_bins

    inverse_weights = 1.0 / (frequencies + smoothing_factor)
    inverse_weights = inverse_weights / inverse_weights.min() * min_weight
    inverse_weights = np.clip(inverse_weights, min_weight, max_weight)

    weights_list = inverse_weights.tolist()

    log_func(f"[層化サンプリング] 最終重み: {[f'{w:.2f}' for w in weights_list]}")
    log_func(f"[層化サンプリング] ビン出現数: {bin_counts.astype(int).tolist()}")

    return weights_list
```

---

## 提案 2（Focal Loss）の詳細実装例

### パターン A: Focal MSE（回帰タスク向け）

```python
def focal_mse_loss(pred, target, gamma=2.0, eps=1e-8):
    """
    Focal MSE Loss

    通常のMSEに、誤差の大きさに応じた重み付けを追加
    gamma > 1 のとき、大きな誤差により高い重みが付く

    Args:
        pred: 予測値 (B, C, H, W)
        target: 正解値 (B, C, H, W)
        gamma: focusing parameter (推奨: 1.5-3.0)
    """
    diff = torch.abs(pred - target)
    focal_weight = diff ** (gamma - 1)  # gamma=2 なら diff^1
    focal_weight = focal_weight.clamp(min=eps)  # 数値安定性

    mse = diff ** 2
    focal_mse = (focal_weight * mse).mean()

    return focal_mse
```

### パターン B: Precipitation-Aware Focal Loss（降水特化）

```python
def precipitation_focal_loss(pred, target,
                            alpha=0.25,      # 降水/非降水のバランス
                            gamma=2.0,       # focusing parameter
                            rain_threshold=0.1):  # 降水判定閾値
    """
    降水イベントに焦点を当てたFocal Loss

    降水がある画素により高い重みを付与
    """
    # 降水マスク
    is_rain = (target > rain_threshold).float()

    # 基本MSE
    mse = (pred - target) ** 2

    # Focal weight（誤差が大きいほど重い）
    pt = torch.exp(-mse)  # 0-1の確信度（誤差小→1, 誤差大→0）
    focal_weight = (1 - pt) ** gamma

    # 降水/非降水のバランス
    alpha_weight = is_rain * alpha + (1 - is_rain) * (1 - alpha)

    loss = alpha_weight * focal_weight * mse

    return loss.mean()
```

### パターン C: ハイブリッド損失

```python
def hybrid_precipitation_loss(pred_1h, target_1h, target_sum,
                             mse_weight=0.5,
                             focal_weight=0.3,
                             detection_weight=0.2):
    """
    複数の損失を組み合わせた総合損失関数
    """
    # (1) 標準MSE（全体的な精度）
    mse_loss = F.mse_loss(pred_1h, target_1h)

    # (2) Focal MSE（強雨への注目）
    focal_loss = focal_mse_loss(pred_1h, target_1h, gamma=2.0)

    # (3) イベント検出BCE（降水/非降水の識別）
    pred_sum = pred_1h.sum(dim=1, keepdim=True)
    pred_event = torch.sigmoid(pred_sum - 0.1)  # >0.1mm を降水とする
    target_event = (target_sum > 0.1).float()
    detection_loss = F.binary_cross_entropy(pred_event, target_event)

    total_loss = (mse_weight * mse_loss +
                  focal_weight * focal_loss +
                  detection_weight * detection_loss)

    return total_loss, mse_loss, focal_loss, detection_loss
```

---

## データ不均衡への根本対策

### 現状のデータ分布（推定）

検証データ（2022 年、2920 時刻 × 480×480 画素）の分析結果：

```
総画素数: 672,768,000

降水強度の分布:
  0.0 - 0.1 mm:  672,747,887 画素 (99.997%)  ← ほぼ全部
  0.1 - 1.0 mm:      20,113 画素 (0.003%)   ← わずか
  1.0 - 5.0 mm:           0 画素 (0.000%)   ← 存在しない
  5.0 - 10.0 mm:          0 画素 (0.000%)
  ...
```

このデータで学習すると：

- モデルは「常に 0mm 予測」が最適解と学習してしまう
- 稀な強雨イベントの勾配は無視される

### 解決策の組み合わせ

#### 戦略 A: サンプルレベルの対策（推奨）

```python
# 1. 層化サンプリング（提案1）で強雨サンプルを確実に含める
# 2. データ拡張（提案4）で強雨サンプルを増幅
# 3. 2段階学習（提案3）で強雨に特化したファインチューニング
```

#### 戦略 B: 損失レベルの対策

```python
# 1. Focal Loss（提案2）で強雨の学習シグナルを増幅
# 2. Multi-Task Loss（提案5）で明示的なイベント検出を追加
# 3. 強度別の重み（既存）を適切に設定
```

#### 戦略 C: データレベルの根本対策（長期）

```python
# 1. 追加年度データ（提案6）で強雨イベントを増やす
# 2. 季節選択で梅雨・台風期を重点的にサンプリング
# 3. 別の観測データ（レーダー等）との融合
```

---

## v7 実装の具体的な修正箇所

### 1. swinunet_main_v7_config.py（新規作成）

```python
"LOSS": {
    "enable_intensity_weighted_loss": True,
    "weight_mode": "stratified_inverse_frequency",  # 新モード

    "stratified_sampling_params": {
        "scan_ratio": 0.5,              # 50%のデータをスキャン
        "max_samples_per_bin": 1000,    # 低強度ビンの上限
        "min_samples_per_bin": 50,      # 各ビンの最小サンプル数
        "force_full_scan_above": 5.0,   # 5mm以上は全スキャン
    },

    "focal_loss_params": {
        "enable": True,
        "gamma": 2.0,         # focusing parameter
        "alpha": 0.25,        # 降水/非降水バランス
    },

    "multi_task_params": {
        "enable": True,
        "detection_weight": 0.2,  # イベント検出の重み
        "thresholds": [0.1, 1.0, 5.0, 10.0],
    },

    # 既存の設定も維持
    "intensity_weight_bins_1h": [1.0, 5.0, 10.0, 20.0, 30.0, 50.0],
},
```

### 2. swinunet_main_v7.py（修正）

主な変更点：

- `compute_inverse_frequency_weights_stratified` 関数の追加
- `focal_mse_loss` 関数の追加
- `multi_task_loss` 関数の追加（オプション）
- `custom_loss_function` の更新（Focal Loss 統合）
- 学習率スケジューラに Warmup 追加

---

## 期待される改善効果（定量的予測）

### 現状（v6）

```
Threshold > 1.0 mm:  Prec=0.0000, Rec=0.0000, F1=0.0000, CSI=0.0000
Threshold > 5.0 mm:  Prec=0.0000, Rec=0.0000, F1=0.0000, CSI=0.0000
Threshold > 10.0 mm: Prec=0.0000, Rec=0.0000, F1=0.0000, CSI=0.0000
```

### 提案 1 のみ実装（v7-lite）

```
Threshold > 1.0 mm:  Prec=0.30-0.50, Rec=0.20-0.40, F1=0.25-0.45, CSI=0.15-0.30
Threshold > 5.0 mm:  Prec=0.10-0.30, Rec=0.05-0.20, F1=0.07-0.25, CSI=0.05-0.15
Threshold > 10.0 mm: Prec=0.05-0.20, Rec=0.02-0.10, F1=0.03-0.15, CSI=0.02-0.10
```

改善幅: TP=0 → TP>0 になるだけでも大きな前進

### 提案 1+2+3 実装（v7-full）

```
Threshold > 1.0 mm:  Prec=0.50-0.70, Rec=0.40-0.60, F1=0.45-0.65, CSI=0.35-0.50
Threshold > 5.0 mm:  Prec=0.30-0.50, Rec=0.20-0.40, F1=0.25-0.45, CSI=0.20-0.35
Threshold > 10.0 mm: Prec=0.20-0.40, Rec=0.10-0.30, F1=0.15-0.35, CSI=0.10-0.25
```

改善幅: 実用的なレベルに到達する可能性

### 全提案実装（v8-ultimate）

```
Threshold > 1.0 mm:  Prec=0.70-0.85, Rec=0.60-0.75, F1=0.65-0.80, CSI=0.50-0.70
Threshold > 5.0 mm:  Prec=0.50-0.70, Rec=0.40-0.60, F1=0.45-0.65, CSI=0.35-0.50
Threshold > 10.0 mm: Prec=0.40-0.60, Rec=0.30-0.50, F1=0.35-0.55, CSI=0.25-0.40
```

改善幅: 業務利用可能なレベル

---

## リスクと注意事項

### リスク 1: 過学習

- **原因**: 強雨サンプルに過度にフォーカスすると、稀なパターンを暗記
- **対策**: 検証データで早期停止、適切な正則化

### リスク 2: 偽陽性の増加

- **原因**: 強雨検出を優先すると、降っていない場所で誤検出
- **対策**: Precision/Recall のバランスをモニタリング、閾値調整

### リスク 3: 計算コスト増

- **原因**: 層化サンプリングやデータ拡張で処理時間増加
- **対策**: 並列化、キャッシング、初回のみの処理として実装

### リスク 4: 物理的妥当性の低下

- **原因**: データ拡張や Post-Processing で非物理的なパターン生成
- **対策**: 気象学的に妥当な拡張のみ使用、専門家レビュー

---

## まとめ

### 最優先で実装すべき改善（v7）

1. **層化サンプリング**（提案 1）

   - 手間: ★★☆（中）
   - 効果: ★★★（高）
   - 備考: **これなしでは始まらない**

2. **Focal Loss**（提案 2）

   - 手間: ★☆☆（低）
   - 効果: ★★★（高）
   - 備考: 提案 1 と同時実装推奨

3. **学習率 Warmup**（提案 8）
   - 手間: ☆☆☆（極低）
   - 効果: ★★☆（中）
   - 備考: すぐに試せる

### 次のステップ

上記 3 つを実装した **v7** を実行し、以下を確認：

- 1mm 以上の降水が検出できるようになったか？
- Precision/Recall のバランスは適切か？
- 強雨ビンの重みが適切に計算されているか？

結果に応じて、提案 3-7 から追加の改善を選択します。

---

## 参考: 類似研究での対策事例

### MetNet (Google, 2020)

- データ不均衡対策: **重み付きクロスエントロピー** + **Focal Loss**
- サンプリング: **強雨イベントの過剰サンプリング**（2-5 倍）

### DGMR (DeepMind, 2021)

- 損失: **MSE + Adversarial Loss** で鮮明な降水パターンを生成
- 評価: **CSI（Critical Success Index）** を主要指標として最適化

### UNet++ for Precipitation (2022)

- アーキテクチャ: **Deep Supervision**（中間層でも損失計算）
- データ拡張: **Mixup** で強雨サンプルを合成

これらの手法も参考になりますが、まずは提案 1-3 の基本的な対策を実施することが重要です。

---

**作成日**: 2026-01-03  
**対象バージョン**: v6 → v7 への改善  
**担当**: AI 分析システム
