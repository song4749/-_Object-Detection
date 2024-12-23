import streamlit as st
from PIL import Image
from io import BytesIO
from ultralytics import YOLO
import cv2

st.set_page_config(layout="wide", page_title="점자블록 객체인식")

st.write("## 점자블록 객체인식")
st.write(
    "점자블록이 포함된 이미지를 업로드하면 점자블록을 탐지해줍니다"
)
st.sidebar.write("## Upload and download :gear:")

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB


# Download the fixed image
def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im


def fix_image(upload):
    image = Image.open(upload)
    col1.write("Original Image :camera:")
    col1.image(image)

    model = YOLO("BackgroundRemoval/best.onnx")
    result = model.predict(image, save=True, imgsz=640, conf=0.5)
    fixed_img_bgr = result[0].plot()
    fixed_img = cv2.cvtColor(fixed_img_bgr, cv2.COLOR_BGR2RGB)  # BGR -> RGB 변환
    col2.write("Fixed Image :wrench:")
    col2.image(fixed_img)

    fixed_image = Image.fromarray(fixed_img)  # NumPy 배열을 PIL 이미지로 변환
    st.sidebar.markdown("\n")
    st.sidebar.download_button("Download fixed image", convert_image(fixed_image), "fixed.png", "image/png")


col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    if my_upload.size > MAX_FILE_SIZE:
        st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
    else:
        fix_image(upload=my_upload)
else:
    fix_image("BackgroundRemoval/20230301_144828.jpg")
