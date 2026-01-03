# Swin-UNet GPU 選択オプション使用方法

## 🎯 概要

v5 および v6 のプログラムに`-use-gpu`オプションを追加し、以下の 3 つのモードで実行できるようになりました：

| オプション     | 動作                     | 使用 GPU        |
| -------------- | ------------------------ | --------------- |
| `-use-gpu 0`   | シングル GPU モード      | GPU 0 番のみ    |
| `-use-gpu 1`   | シングル GPU モード      | GPU 1 番のみ    |
| `-use-gpu ddp` | DDP モード（デフォルト） | GPU 0 番と 1 番 |

---

## 📝 実行コマンド例

### 1. GPU 0 番のみ使用（シングル GPU）

```bash
# v5
python swinunet_main_v5.py -use-gpu 0

# v6
python swinunet_main_v6.py -use-gpu 0

# バックグラウンド実行（nohup + notify-run）
notify-run gpu01 -- nohup python swinunet_main_v5.py -use-gpu 0 > swinunet_main_v5.log 2>&1 &
notify-run gpu01 -- nohup python swinunet_main_v6.py -use-gpu 0 > swinunet_main_v6.log 2>&1 &
```

### 2. GPU 1 番のみ使用（シングル GPU）

```bash
# v5
python swinunet_main_v5.py -use-gpu 1

# v6
python swinunet_main_v6.py -use-gpu 1

# バックグラウンド実行
notify-run gpu01 -- nohup python swinunet_main_v5.py -use-gpu 1 > swinunet_main_v5.log 2>&1 &
notify-run gpu01 -- nohup python swinunet_main_v6.py -use-gpu 1 > swinunet_main_v6.log 2>&1 &
```

### 3. GPU 0 番と 1 番を使用（DDP モード）

```bash
# v5（デフォルト）
python swinunet_main_v5.py
# または明示的に
python swinunet_main_v5.py -use-gpu ddp

# v6（デフォルト）
python swinunet_main_v6.py
# または明示的に
python swinunet_main_v6.py -use-gpu ddp

# バックグラウンド実行
notify-run gpu01 -- nohup python swinunet_main_v5.py -use-gpu ddp > swinunet_main_v5.log 2>&1 &
notify-run gpu01 -- nohup python swinunet_main_v6.py -use-gpu ddp > swinunet_main_v6.log 2>&1 &
```

---

## 🔧 仕組み

### CUDA_VISIBLE_DEVICES の設定

プログラム内部で環境変数を自動設定します：

```python
# -use-gpu 0 の場合
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# → PyTorchからは GPU 0 のみが見える

# -use-gpu 1 の場合
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# → PyTorchからは GPU 1 のみが見える（論理的にはcuda:0として認識）

# -use-gpu ddp の場合
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# → PyTorchから GPU 0, 1 の両方が見える
```

### DDP モードの動作

```
GPU 0 (cuda:0)                GPU 1 (cuda:1)
┌──────────────┐             ┌──────────────┐
│ プロセス0    │             │ プロセス1    │
│ rank=0       │◄─ NCCL ─────►│ rank=1       │
│              │   通信      │              │
│ バッチ0,2,4..│             │ バッチ1,3,5..│
│              │             │              │
│ 勾配計算     │             │ 勾配計算     │
└──────────────┘             └──────────────┘
        │                            │
        └──────── All-Reduce ─────────┘
                    ↓
            平均勾配で両GPUを更新
```

---

## ⚡ パフォーマンス比較

| モード                | 学習速度        | メモリ効率 | 推奨用途             |
| --------------------- | --------------- | ---------- | -------------------- |
| シングル GPU (0 or 1) | 基準（1.0×）    | 高い       | デバッグ、小規模実験 |
| DDP (0+1)             | **約 1.8-1.9×** | 中程度     | **本番学習（推奨）** |

**注意点:**

- DDP は 2 倍にはならない理由：通信オーバーヘッド（5-10%）があるため
- バッチサイズ 8 の場合、DDP では実効バッチサイズ 16（8×2）になります

---

## 📊 各モードの使い分け

### シングル GPU モード（-use-gpu 0 or 1）

**こんな時に使う:**

- ✅ プログラムのデバッグ時
- ✅ 小規模な実験（数エポックだけ試す）
- ✅ 他の人が別の GPU を使っていて 1 つしか使えない時
- ✅ メモリが足りるか確認したい時

**メリット:**

- シンプル（プロセス 1 つだけ）
- デバッグしやすい
- GPU 1 個分のメモリで済む

**デメリット:**

- 学習に時間がかかる（約 2 倍）

### DDP モード（-use-gpu ddp）

**こんな時に使う:**

- ✅ **本番の長時間学習（60 エポック等）**
- ✅ できるだけ早く結果が欲しい時
- ✅ 2 つの GPU が空いている時

**メリット:**

- **学習速度が約 2 倍**
- 実効バッチサイズが 2 倍（精度向上の可能性）

**デメリット:**

- 2 つの GPU が必要
- やや複雑（DDP 通信が入る）
- メモリは各 GPU に必要

---

## 🛠️ トラブルシューティング

### Q1: DDP モードでエラーが出る

```
RuntimeError: NCCL error: unhandled cuda error
```

**対処法:**

```bash
# シングルGPUモードで実行
python swinunet_main_v6.py -use-gpu 0
```

### Q2: GPU 1 番が使われているか確認したい

```bash
# 実行中に別のターミナルで
nvidia-smi

# GPU 1番に python プロセスが表示されればOK
```

### Q3: どの GPU が空いているか確認したい

```bash
nvidia-smi

# 使用率を確認
# GPU 0: 95% → 使用中
# GPU 1: 0%  → 空いている → -use-gpu 1 を使用
```

### Q4: バックグラウンド実行の進捗確認

```bash
# ログファイルを確認
tail -f swinunet_main_v6.log

# または
less swinunet_main_v6.log  # Shift+G で最後尾へ
```

---

## 💡 推奨設定

### 開発・デバッグ時

```bash
# 短時間で動作確認
python swinunet_main_v6.py -use-gpu 0
```

### 本番学習時（推奨）

```bash
# DDPで高速化
notify-run gpu01 -- nohup python swinunet_main_v6.py -use-gpu ddp > swinunet_v6.log 2>&1 &

# またはシンプルに
python swinunet_main_v6.py -use-gpu ddp
```

### 他の人が GPU 0 を使っている時

```bash
# GPU 1番のみ使用
python swinunet_main_v6.py -use-gpu 1
```

---

## 📌 ログの確認

実行するとログファイルが生成されます：

```
swin-unet_main_result_v6/
├── main_v6.log          # メインログ（エポックごとの損失など）
├── execution_v6.log     # 実行ログ（tqdmの進捗バー）
└── evaluation_v6.log    # 評価結果（最終メトリクス）
```

**確認方法:**

```bash
# リアルタイム監視
tail -f swin-unet_main_result_v6/main_v6.log

# GPU モード確認（先頭付近に出力される）
head -n 30 swin-unet_main_result_v6/main_v6.log | grep "GPU Mode"
# 出力例: [GPU Mode] -use-gpu=1, CUDA_VISIBLE_DEVICES=1
```

---

## ⚙️ 内部実装の詳細

### argparse の実装

```python
import argparse

parser = argparse.ArgumentParser(description='Swin-UNet v6 Training Script')
parser.add_argument('-use-gpu', '--use-gpu', type=str, default='ddp',
                    choices=['0', '1', 'ddp'],
                    help='GPU selection: 0 (GPU0 only), 1 (GPU1 only), ddp (GPU0+1 with DDP)')
args = parser.parse_args()

# GPU設定
if args.use_gpu == '0':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    force_single_gpu = True
elif args.use_gpu == '1':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    force_single_gpu = True
else:  # ddp
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    force_single_gpu = False
```

### 実行モード分岐

```python
if force_single_gpu:
    # シングルGPUモード
    main_worker(0, 1, train_files, valid_files)
else:
    # DDPモード
    world_size = torch.cuda.device_count()  # 2
    mp.spawn(main_worker,
             args=(world_size, train_files, valid_files),
             nprocs=world_size,  # 2プロセス起動
             join=True)
```

---

## 🎓 まとめ

### 基本的な使い方

```bash
# GPU 0番で実行
python swinunet_main_v6.py -use-gpu 0

# GPU 1番で実行
python swinunet_main_v6.py -use-gpu 1

# 両方のGPUで実行（推奨）
python swinunet_main_v6.py -use-gpu ddp
```

### notify-run と組み合わせた実行

```bash
# ご要望の形式
notify-run gpu01 -- nohup python swinunet_main_v6.py -use-gpu 1 > swinunet_main_v6.log 2>&1 &
```

**解説:**

- `notify-run gpu01`: 終了時に通知
- `nohup`: ターミナルを閉じても継続
- `-use-gpu 1`: GPU 1 番を使用
- `> swinunet_main_v6.log 2>&1`: 標準出力とエラーをログファイルへ
- `&`: バックグラウンド実行

これで、他の GPU を使っている人と競合せず、スムーズに学習できます！
