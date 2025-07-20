import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import re

# ====== CONFIG ======
CLASS_NAMES = ["Galaxy_A32", "iPhone_11", "iPhone_12"]
MAPPING_DICT = {
    "Galaxy_A32": "Samsung Galaxy A32",
    "iPhone_11": "iPhone 11",
    "iPhone_12": "iPhone 12"
}

# ====== LOAD MODEL ======
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load("best_model_fixed.pth", map_location=device))
    model.to(device)
    model.eval()
    return model, device

model, device = load_model()

# ====== TRANSFORM ======
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ====== LOAD BATTERY INFO ======
@st.cache_data
def load_battery_data():
    df = pd.read_csv("phone_battery_info.csv")

    def grade_battery_danger(row):
        score = 0
        info = str(row['battery_info'])
        if 'Li-Po' in info:
            score += 2
        elif 'Li-Ion' in info:
            score += 1

        match = re.search(r'(\d{3,5})', str(row['mAh']))
        if match:
            capacity = int(match.group(1))
            if capacity <= 3000:
                score += 1
            elif capacity <= 4000:
                score += 2
            else:
                score += 3

        removable = str(row['remove']).lower()
        score += 3 if 'non-removable' in removable else 1

        if re.search(r'\d+\.\d+', str(row['wh'])):
            score += 2

        return score

    df['danger_score'] = df.apply(grade_battery_danger, axis=1)
    return df

df_battery = load_battery_data()

# ====== IMAGE PREDICTION ======
def predict_image(img: Image.Image):
    image = val_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, pred = torch.max(outputs, 1)
    return CLASS_NAMES[pred.item()]

# ====== STREAMLIT UI ======
st.title("🔋 ตรวจสอบมือถือและความอันตรายของแบตเตอรี่")
st.write("อัปโหลดภาพมือถือ แล้วระบบจะทำนายรุ่น พร้อมคำนวณคะแนนความอันตรายของแบตเตอรี่")

uploaded_file = st.file_uploader("📷 อัปโหลดภาพมือถือ", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="รูปที่อัปโหลด", use_column_width=True)

    pred_class = predict_image(img)
    st.success(f"✅ ระบบทำนายว่าเป็น: **{pred_class}**")

    predicted_model = MAPPING_DICT.get(pred_class)
    if predicted_model:
        battery_row = df_battery[df_battery['model'] == predicted_model]

        if len(battery_row) == 1:
            st.markdown(f"### 📱 รุ่น: **{predicted_model}**")
            st.markdown("### 🔋 ข้อมูลแบตเตอรี่:")
            st.text(battery_row['battery_info'].values[0])
            st.markdown(f"### ⚠️ คะแนนความอันตรายของแบตเตอรี่: **{battery_row['danger_score'].values[0]}/10**")
        else:
            st.warning("ไม่พบข้อมูลในฐานข้อมูลแบตเตอรี่สำหรับรุ่นนี้")
    else:
        st.warning("ไม่สามารถจับคู่ชื่อรุ่นได้ กรุณาตรวจสอบ mapping_dict")
