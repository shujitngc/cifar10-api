# CIFAR-10 Image Classification API

## 概要
画像をアップロードすると、CIFAR-10の10クラスに分類するAPIです。

## 使用技術
Python / PyTorch / FastAPI / Docker / Render

## API URL
https://cifar10-api-7440.onrender.com/docs

## エンドポイント
GET /health
GET /labels
POST /predict

## 実行方法
docker build
docker run

## 工夫した点
- FastAPIで推論API化
- Docker対応
- Renderにデプロイ
- 画像サイズ制限とMIMEチェックを実装
