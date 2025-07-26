import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import streamlit as st
import gdown
import pandas as pd
import re

# ===== CONFIG =====
CLASS_NAMES = ["Galaxy_A32", "iPhone_11"]
CSV_FILE_ID = "1xEccDMzWIHPEop58SlQdJwITr5y50mNj"
MODEL_FILE_ID = "1vud0Qk1PHy7_jLgSUwwurUN7KKw1WeIw"
CSV_PATH = "phone_battery_info.csv"
MODEL_PATH = "best_model_fixed.pth"

# ===== LOAD MODEL FROM GOOGLE DRIVE =====
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

# ===== LOAD CSV FROM GOOGLE DRIVE =====
@st.cache_data
def load_battery_data():
    if not os.path.exists(CSV_PATH):
        url = f"https://drive.google.com/uc?id={CSV_FILE_ID}"
        gdown.download(url, CSV_PATH, quiet=False)
    return pd.read_csv(CSV_PATH)

# ===== IMAGE TRANSFORM =====
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ===== PREDICT IMAGE =====
def predict_image(model, img):
    image = Image.open(img).convert('RGB')
    image = val_transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
    return CLASS_NAMES[preds.item()]

# ===== DANGER SCORE =====
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

    return score * 1000

# ===== STREAMLIT UI =====
st.set_page_config(page_title="🔋 Battery Classifier", layout="centered")
st.title("🔋 Mobile Battery Classifier with Danger Score")

model = load_model()
df = load_battery_data()

uploaded_file = st.file_uploader("📤 Upload a phone image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="📷 Uploaded Image", use_column_width=True)
    predicted_class = predict_image(model, uploaded_file)
    st.success(f"📱 Predicted Model: **{predicted_class}**")

    row = df[df['model'].str.contains(predicted_class, case=False, na=False)]
    if not row.empty:
        row = row.iloc[0]
        score = grade_battery_danger(row)
        st.markdown(f"💥 **Danger Score:** `{score}`")
        st.markdown("🔎 **Battery Info:**")
        st.write({
            "Battery": row["battery_info"],
            "Capacity (mAh)": row["mAh"],
            "Removable": row["remove"],
            "Wh": row["wh"]
        })
    else:
        st.warning("⚠️ ไม่พบข้อมูลแบตเตอรี่สำหรับรุ่นนี้ในไฟล์ CSV.")
