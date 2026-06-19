from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_predict_invalid_mime():
    files = {"file": ("bad.txt", b"not image", "text/plain")}
    r = client.post("/predict", files=files)
    assert r.status_code == 400

def test_predict_too_large():
    big = b"\x00" * (5 * 1024 * 1024 + 1)  # 5MB+1
    files = {"file": ("big.jpg", big, "image/jpeg")}
    r = client.post("/predict", files=files)
    assert r.status_code == 413
