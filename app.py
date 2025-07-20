import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import streamlit as st
import gdown

# กำหนดชื่อคลาสแค่ 2 ตัว
CLASS_NAMES = ["Galaxy_A32", "iPhone_11"]

# โหลดโมเดลจากไฟล์ .pth (ไฟล์ต้องอยู่ในโฟลเดอร์เดียวกับ app.py)
@st.cache_resource
def load_model(model_path):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))  # 2 คลาส
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# ฟังก์ชันดาวน์โหลดโมเดลจาก Google Drive ถ้าไฟล์ยังไม่มี
def download_model():
    model_path = "best_model_fixed.pth"
    if not os.path.exists(model_path):
        file_id = "1vud0Qk1PHy7_jLgSUwwurUN7KKw1WeIw"  # File ID จาก Google Drive
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_path, quiet=False)
    return model_path

# ฟังก์ชันสำหรับ preprocessing ภาพก่อนทำนาย
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_image(model, img):
    image = Image.open(img).convert('RGB')
    image = val_transform(image).unsqueeze(0)  # เพิ่ม batch dim

    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
    return CLASS_NAMES[preds.item()]

# Streamlit UI
st.title("Mobile Phone Battery Classifier")

model_path = download_model()
model = load_model(model_path)

uploaded_file = st.file_uploader("Upload a phone image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    predicted_class = predict_image(model, uploaded_file)
    st.write(f"Predicted Model: **{predicted_class}**")
