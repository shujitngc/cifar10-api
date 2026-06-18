CIFAR-10 Image Classification API
概要

PyTorchで学習したResNet18モデルを用いて、画像をCIFAR-10の10クラスに分類する画像分類APIです。

FastAPIを用いて推論APIを構築し、Dockerによるコンテナ化を行ったうえでRenderへデプロイしました。

アップロードされた画像に対して前処理を実施し、推論結果をJSON形式で返します。

使用技術
Python
PyTorch
torchvision
FastAPI
Uvicorn
Docker
Render
GitHub
分類クラス
airplane
automobile
bird
cat
deer
dog
frog
horse
ship
truck
API URL
https://xxxxx.onrender.com/docs
エンドポイント
ヘルスチェック
GET /health

レスポンス

{
  "status": "ok"
}
クラス一覧取得
GET /labels

レスポンス

{
  "classes": [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
  ]
}
画像分類
POST /predict

レスポンス例

{
  "class_id": 3,
  "class_name": "cat",
  "prob": 0.9821
}
システム構成
Client
  ↓
FastAPI
  ↓
ResNet18
  ↓
Prediction Result(JSON)
工夫した点
FastAPIを利用して機械学習モデルをAPI化
Dockerによるコンテナ化を実施
Renderへデプロイし外部公開
MIMEタイプチェックによる不正ファイル対策
ファイルサイズ制限による安全性向上
学習済みモデル未配置時はDummyModelで起動可能な設計
