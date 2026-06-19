# CIFAR-10 Image Classification API

## 概要
PyTorchで学習したResNet18モデルを用いて、画像をCIFAR-10の10クラスに分類する画像分類APIです。
FastAPIを用いて推論APIを構築し、Dockerによるコンテナ化を行ったうえでRenderへデプロイしました。
アップロードされた画像に対して前処理を実施し、推論結果をJSON形式で返します。

## 使用技術
Python / PyTorch / FastAPI / Docker / Render / GitHub

## 分類クラス
airplane / automobile / bird / cat / deer / dog / frog / horse / ship / truck

## API URL
https://cifar10-api-7440.onrender.com/docs

## Demo App
Streamlitで作成したデモ画面です。画像をアップロードすると、Render上のFastAPIにリクエストを送り、分類結果を表示します。
https://cifar10-api-fnnjgyxvusvnetqxslbyjd.streamlit.app/

## エンドポイント
GET /health
GET /labels
POST /predict

## 実行方法
docker build
docker run

## 工夫した点
- FastAPIを利用して機械学習モデルをAPI化
- Dockerによるコンテナ化を実施
- Renderへデプロイし外部公開
- MIMEタイプチェックによる不正ファイル対策
- ファイルサイズ制限による安全性向上
- 学習済みモデル未配置時はDummyModelで起動可能な設計
