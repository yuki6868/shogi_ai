
# 教育用「せめぎ合い」将棋AI

## 概要

このプロジェクトは、
「最強AI」ではなく、

- 子供には子供レベル
- 中級者には中級者レベル
- 上級者には上級者レベル

で対戦し、

「あと少しで勝てそう」
「次は勝ちたい」

と思わせる “せめぎ合い” を演出する教育用将棋AIです。

---

# コンセプト

通常の将棋AI：

- 常に最善手を目指す
- 勝率最大化が目的

本プロジェクト：

- 相手の棋力に合わせる
- 接戦を作る
- 学習意欲を引き出す

---

# 現在のAI構成

```text
合法手生成
↓
Policy AI（自然な候補手）
↓
Value AI（局面評価）
↓
強さ制御AI（相手レベル調整）
↓
最終着手
```

---

# 実装済み機能

## フロントエンド

- 9×9将棋盤UI
- 駒移動
- 成り
- 持ち駒
- AI評価値表示
- 勝率表示
- 候補手表示
- playerLevelによる強さ制御

---

## バックエンド

### 将棋エンジン

- 合法手生成
- 王手判定
- 詰み判定
- 持ち駒処理
- 成り処理

### Policy AI

役割：

```text
自然な手を選ぶ
```

学習：

```text
局面 → 実際に指された手
```

教師データ：

```text
Floodgate CSA棋譜
```

---

### Value AI

役割：

```text
局面評価
```

学習：

```text
局面 → 最終勝敗
```

出力：

```text
enemy側勝率
```

---

### 強さ制御AI

役割：

```text
相手の棋力に合わせる
```

特徴：

- 初心者には少し弱く
- 上級者には強く
- ただし自然な手を維持

---

# ディレクトリ構成

```text
shogi_ai/
├── frontend/
│   └── index.html
│
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   │
│   ├── ai/
│   │   ├── board.py
│   │   ├── evaluator.py
│   │   ├── move_selector.py
│   │   ├── move_encoder.py
│   │   ├── board_tensor.py
│   │   ├── kifu_parser.py
│   │   │
│   │   ├── policy_model.py
│   │   ├── policy_inference.py
│   │   ├── train_policy.py
│   │   │
│   │   ├── value_model.py
│   │   ├── value_inference.py
│   │   ├── value_dataset.py
│   │   ├── train_value.py
│   │
│   ├── models/
│   │   ├── policy_model.pt
│   │   └── value_model.pt
│
├── dataset/
│   └── floodgate/
```

---

# 必要環境

- Python 3.11 推奨
- PyTorch
- FastAPI
- Uvicorn

---

# インストール

## 仮想環境作成

```bash
python -m venv .venv
```

---

## 仮想環境有効化

### Mac/Linux

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

---

## ライブラリインストール

```bash
pip install -r backend/requirements.txt
```

---

# サーバ起動

```bash
cd backend

python main.py
```

または

```bash
uvicorn main:app --reload
```

---

# フロント起動

`frontend/index.html` をブラウザで開く。

---

# Policy AI 学習

## 学習実行

```bash
cd backend

python -m ai.train_policy --max-files 500 --epochs 5
```

---

## 学習済みモデル出力

```text
backend/models/policy_model.pt
```

---

# Value AI 学習

## 学習実行

```bash
cd backend

python -m ai.train_value --max-files 500 --epochs 5
```

---

## 学習済みモデル出力

```text
backend/models/value_model.pt
```

---

# 学習データ

使用データ：

```text
Floodgate CSA棋譜
```

配置場所：

```text
dataset/floodgate/
```

---

# API

## AI着手

```http
POST /api/ai-move
```

---

## 合法手取得

```http
POST /api/legal-moves
```

---

## 局面評価

```http
POST /api/evaluate
```

---

## 王手判定

```http
POST /api/check-state
```

---

## Policy AI状態確認

```http
GET /api/policy-status
```

---

## Value AI状態確認

```http
GET /api/value-status
```

---

# 現在の課題

## 学習量不足

現在：

```text
数十〜数百棋譜
```

理想：

```text
数千〜数万棋譜
```

---

## 探索不足

現在：

```text
1手評価
```

今後：

```text
ミニマックス探索
αβ探索
```

---

# 今後の予定

## 短期

- Policy AI強化
- Value AI強化
- playerLevel改善
- 接戦制御改善

---

## 中期

- 探索導入
- 詰み回避強化
- 戦術理解強化

---

## 長期

- 本格的な棋力推定
- 教育カリキュラム連携
- 成長型AI
- 対局履歴分析

---

# このAIの特徴

普通の将棋AI：

```text
強さだけ
```

このAI：

```text
強さ
+
自然さ
+
接戦
+
教育効果
```

を重視。

---

# 開発目的

「勝てないからやめる」ではなく、

```text
あと少しで勝てそう
↓
次は勝ちたい
↓
勉強したい
```

を作るAIを目指しています。
