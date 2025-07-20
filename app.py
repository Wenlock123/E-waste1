import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import pandas as pd
import re

# === โหลดโมเดลและข้อมูล ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 3)  # แก้เป็นจำนวน class ที่คุณมี
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# class label ต้องตรงกับ folder ชื่อ class ใน dataset
class_names = ["Galaxy_A32", "iPhone_11", "iPhone_12"]  # เพิ่มตามที่มี

mapping_dict = {
    "Galaxy_A32": "Samsung Galaxy A32",
    "iPhone_11": "iPhone 11",
    "iPhone_12": "iPhone 12",
}

# แปลงภาพ
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# โหลดข้อมูล CSV
df = pd.read_csv("phone_battery_info.csv")  # ต้องมีไฟล์นี้บน Streamlit Cloud ด้วย

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
        score += 1 if capacity <= 3000 else 2 if capacity <= 4000 else 3

    removable = str(row['remove']).lower()
    score += 3 if 'non-removable' in removable else 1

    if re.search(r'\d+\.\d+', str(row['wh'])):
        score += 2

    return score

df['danger_score'] = df.apply(grade_battery_danger, axis=1)

def predict_image(img: Image.Image):
    image = val_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, pred = torch.max(outputs, 1)
    return class_names[pred.item()]

# === UI ด้วย Streamlit ===
st.title("📱 ตรวจสอบมือถือและอันตรายจากแบตเตอรี่")
st.write("อัปโหลดภาพมือถือเพื่อดูว่าคือรุ่นอะไร และคำนวณคะแนนความอันตรายของแบตเตอรี่")

uploaded_file = st.file_uploader("อัปโหลดรูปภาพ", type=["jpg", "jpeg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="รูปที่อัปโหลด", use_column_width=True)

    pred_class = predict_image(img)
    st.success(f"📷 ทำนายว่าเป็น: **{pred_class}**")

    predicted_model = mapping_dict.get(pred_class)
    battery_row = df[df['model'] == predicted_model]

    if predicted_model and len(battery_row) == 1:
        score = battery_row['danger_score'].values[0]
        battery_info = battery_row['battery_info'].values[0]
        st.markdown(f"### 🔋 ข้อมูลแบตเตอรี่สำหรับ {predicted_model}")
        st.text(battery_info)
        st.markdown(f"### ⚠️ คะแนนความอันตราย: **{score}/10**")
    else:
        st.warning("ไม่พบข้อมูลแบตเตอรี่สำหรับรุ่นนี้ในฐานข้อมูล")
