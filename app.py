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
CLASS_NAMES = [
    "Galaxy_A06",
    "Galaxy_A05S",
    "Galaxy_A32",
    "iPhone_11",
    "iPhone_12",
    "iPhone_13",
    "iPhone_15"
]

CSV_DRIVE_URL = "https://drive.google.com/uc?id=1xEccDMzWIHPEop58SlQdJwITr5y50mNj"
MODEL_DRIVE_URL = "https://drive.google.com/uc?id=1PfH1JF82_OOo_W24SITmB-aiCUYMxlPN"
MODEL_FILENAME = "best_model_7classes.pth"
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

    return score * 100

# ==== Streamlit UI ====
st.set_page_config(page_title="E-WASTE", page_icon="♻️", layout="centered")

# Header
st.markdown(
    """
    <h1 style='text-align:center; color:green;'>♻️ E-WASTE</h1>
    <p style='text-align:center; font-size:18px;'>“ยินดีด้วย! คุณได้เป็นหนึ่งในคนที่ช่วยโลกเอาไว้”</p>
    """,
    unsafe_allow_html=True
)

model_path = download_model()
model = load_model(model_path)
df = load_battery_data()

# Upload or Take Photo
st.subheader("📤 อัปโหลดภาพถ่าย หรือ ถ่ายภาพ")
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
# st.camera_input("📸 ถ่ายภาพ")  # ถ้าต้องการเปิดใช้กล้อง

if uploaded_file is not None:
    st.image(uploaded_file, caption="📷 Uploaded Image", use_column_width=True)
    predicted_class = predict_image(model, uploaded_file)

    # แสดงชื่อรุ่นมือถือที่ตรวจพบ
    st.markdown(f"### 📱 รุ่นมือถือที่ตรวจพบ: **{predicted_class}**")

    row = find_closest_model(df, predicted_class)

    if row is not None:
        score = grade_battery_danger(row)
        st.markdown(f"**Score:** {score}")
        
        # ผลเสีย
        if "Li-Po" in str(row['battery_info']):
            st.write("📱 แบตเตอรี่ Li-Po ประกอบด้วยสารเคมีอันตราย เช่น ลิเทียม ที่ติดไฟง่าย เจลโพลิเมอร์ที่ไวไฟ และโลหะหนักอย่างโคบอลต์ นิกเกิล และแมงกานีส ซึ่งอาจก่อให้เกิดพิษต่อร่างกาย มะเร็ง หรือปนเปื้อนสิ่งแวดล้อม หากแบตรั่ว บวม หรือถูกเผา")

        # ปุ่มเลือกศูนย์จัดส่ง
        st.markdown(
            """
            <div style='background-color:#90EE90; padding:10px; border-radius:8px; text-align:center; font-weight:bold;'>
                เลือกจัดส่งศูนย์ที่ใกล้ที่สุด
            </div>
            """,
            unsafe_allow_html=True
        )
        st.button("ศูนย์ AIS เซ็นทรัลแอร์พอร์ต")
        st.button("Siam TV สาขาหางดง")
        st.button("ศูนย์ True เซ็นทรัลเฟสติวัลเชียงใหม่")
    else:
        st.warning("⚠️ ไม่พบข้อมูลแบตเตอรี่สำหรับรุ่นนี้ในไฟล์ CSV.")
