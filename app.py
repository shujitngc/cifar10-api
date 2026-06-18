import requests
import streamlit as st
from PIL import Image

API_URL = "https://cifar10-api-7440.onrender.com/predict"

st.title("CIFAR-10 画像分類アプリ")
st.write("画像をアップロードすると、AIモデルがCIFAR-10のクラスに分類します。")

uploaded_file = st.file_uploader(
    "画像をアップロードしてください",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="アップロード画像", use_container_width=True)

    if st.button("分類する"):
        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type
            )
        }

        with st.spinner("推論中..."):
            response = requests.post(API_URL, files=files)

        if response.status_code == 200:
            result = response.json()

            st.success("分類が完了しました！")
            st.write(f"予測クラス: **{result['class_name']}**")
            st.write(f"クラスID: `{result['class_id']}`")
            st.write(f"確率: `{result['prob']}`")
        else:
            st.error("推論に失敗しました。")
            st.write(response.text)