import io
import os
from typing import Dict

import torch
import torch.nn as nn
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms
from torchvision.models import resnet18

# 環境変数（本番/CIで上書き可）
MODEL_PATH = os.getenv("MODEL_PATH", "model.pt")
SKIP_MODEL_LOAD = os.getenv("SKIP_MODEL_LOAD", "0") == "1"  # CIでモデルなしでも起動するため
MAX_BYTES = int(os.getenv("MAX_BYTES", 5 * 1024 * 1024))    # 受け入れる最大ファイルサイズ（5MB）

# 前処理（学習時と合わせる）
# Resize(256) -> CenterCrop(224) -> ImageNet正規化
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# CIFAR-10 ラベル
CLASSES = ['airplane','automobile','bird','cat','deer',
           'dog','frog','horse','ship','truck']

# 学習時と同じ構成：
# ResNet18の1000出力 → Linear(1000→10)
class NetAPI(nn.Module):
    def __init__(self):
        super().__init__()
        # API側は state_dict をロードするので weights=None（アーキテクチャだけ使う）
        self.feature = resnet18(weights=None)
        self.fc = nn.Linear(1000, 10)

    def forward(self, x):
        h = self.feature(x)  # [B, 1000]
        y = self.fc(h)       # [B, 10]
        return y

# CI/開発用のダミーモデル
# （MODEL_PATHが無い時やSKIP_MODEL_LOAD=1の時に利用）
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
    def eval(self):  # 互換API
        return self
    def forward(self, x):
        # 何も学習していないダミー出力（ゼロロジット）
        return torch.zeros((x.shape[0], 10))

def load_model(weight_path: str = MODEL_PATH) -> nn.Module:
    """state_dict をロードしてモデルを返す。無ければ DummyModel を返す。"""
    if SKIP_MODEL_LOAD or not os.path.exists(weight_path):
        print(f"[WARN] Model not found or skipped. Using DummyModel. path={weight_path}")
        return DummyModel().eval()

    m = NetAPI()
    sd = torch.load(weight_path, map_location="cpu")
    state_dict = sd.get("state_dict", sd)  # {"state_dict": ...}／純state_dict の両対応
    m.load_state_dict(state_dict, strict=True)
    m.eval()
    return m

# FastAPI アプリ本体
app = FastAPI(title="CIFAR-10 Classifier", version="1.0.0")

# CORS（必要に応じて Origin を絞ってOK）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model(MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/labels")
def labels():
    return {"classes": CLASSES}

@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict:
    # MIME チェック
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid content type")

    # サイズ制限
    img_bytes = await file.read()
    if len(img_bytes) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large")

    # 画像として開けるか
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")

    # 前処理 & 推論
    x = preprocess(img).unsqueeze(0)  # [1, 3, 224, 224]
    with torch.no_grad():
        logits = model(x)
        probs = logits.softmax(1)[0]
        idx = int(probs.argmax().item())
        prob = float(probs[idx].item())

    return {
        "class_id": idx,
        "class_name": CLASSES[idx],
        "prob": round(prob, 4)
    }
