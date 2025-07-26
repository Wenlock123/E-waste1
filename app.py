import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import streamlit as st
import gdown
import pandas as pd
import re

# ==== CONFIG ====
CLASS_NAMES = ["Galaxy_A32", "iPhone_11"]
CSV_DRIVE_URL = "https://drive.google.com/uc?id=1xEccDMzWIHPEop58SlQdJwITr5y50mNj"
MODEL_DRIVE_URL = "https://drive.google.com/uc?id=1vud0Qk1PHy7_jLgSUwwurUN7KKw1WeIw"
MODEL_FILENAME = "best_model_fixed.pth"
CSV_FILENAME = "phone_battery_info.csv"

# ==== โหลดโมเดล ====
@st.cache_resource
def load_model(model_path):
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def download_model():
    if not os.path.exists(MODEL_FILENAME):
        gdown.download(MODEL_DRIVE_URL, MODEL_FILENAME, quiet=False)
    return MODEL_FILENAME

# ==== โหลด CSV ====
@st.cache_data
def load_battery_data():
    if not os.path.exists(CSV_FILENAME):
        gdown.download(CSV_DRIVE_URL, CSV_FILENAME, quiet=False)
    return pd.read_csv(CSV_FILENAME)

# ==== เตรียมภาพ ====
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_image(model, img):
    image = Image.open(img).convert('RGB')
    image = val_transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
    return CLASS_NAMES[preds.item()]

# ==== ค้นหาชื่อรุ่นใกล้เคียงใน CSV ====
def find_closest_model(df, predicted_class):
    predicted_normalized = predicted_class.lower().replace("_", "").replace("-", "").replace(" ", "")
    for _, row in df.iterrows():
        model_name = str(row['model']).lower().replace("_", "").replace("-", "").replace(" ", "")
        if predicted_normalized in model_name or model_name in predicted_normalized:
            return row
    return None

# ==== คำนวณ Danger Score ====
def grade_battery_danger(row):
    score = 0

    info = str(row['battery_info'])
    if 'Li-Po' in info:
        score += 2
    elif 'Li-Ion' in info:
        score += 1

    capacity_text = str(row['mAh'])
    match = re.search(r'(\d{3,5})', capacity_text)
    if match:
        capacity = int(match.group(1))
        if capacity <= 3000:
            score += 1
        elif 3000 < capacity <= 4000:
            score += 2
        else:
            score += 3

    removable = str(row['remove']).lower()
    if 'non-removable' in removable:
        score += 3
    else:
        score += 1

    wh_text = str(row['wh'])
    if re.search(r'\d+\.\d+', wh_text):
        score += 2

    return score * 100  # ✅ ปรับให้คะแนนอยู่ในพันต้น ๆ

# ==== Streamlit UI ====
st.title("🔋 Mobile Battery Danger Classifier")

model_path = download_model()
model = load_model(model_path)
df = load_battery_data()

uploaded_file = st.file_uploader("📤 Upload a phone image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="📷 Uploaded Image", use_column_width=True)
    predicted_class = predict_image(model, uploaded_file)
    st.success(f"📱 Predicted Model: **{predicted_class}**")

    row = find_closest_model(df, predicted_class)
    if row is not None:
        score = grade_battery_danger(row)
        st.markdown(f"💥 **Danger Score:** `{score}`")
        st.markdown("🔎 **Battery Info:**")
        st.write({
            "Battery Type": row["battery_info"],
            "Capacity (mAh)": row["mAh"],
            "Removable": row["remove"],
            "Energy (Wh)": row["wh"]
        })
    else:
        st.warning("⚠️ ไม่พบข้อมูลแบตเตอรี่สำหรับรุ่นนี้ในไฟล์ CSV.")
