import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
import json
import os

st.title("🔍 Mobile Battery Classifier")

# ----- ตรวจสอบไฟล์ labels.json และ model.pth -----
if not os.path.exists("labels.json"):
    st.error("❌ ไม่พบไฟล์ labels.json กรุณาอัปโหลดไฟล์นี้พร้อมกับแอปของคุณ")
    st.stop()

if not os.path.exists("model.pth"):
    st.error("❌ ไม่พบไฟล์ model.pth กรุณาอัปโหลดไฟล์นี้พร้อมกับแอปของคุณ")
    st.stop()

# ----- โหลด labels -----
with open("labels.json") as f:
    idx_to_label = json.load(f)

num_classes = len(idx_to_label)

# ----- โหลดโมเดล -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

# ----- Transform -----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ----- ฟังก์ชันให้คะแนนความอันตราย -----
def get_battery_risk_score(phone_model):
    risk_scores = {
        "iPhone 11": 3.5,
        "Galaxy A32": 4.2,
        # เพิ่มได้ตามต้องการ
    }
    return risk_scores.get(phone_model, "ไม่ทราบข้อมูล")

# ----- Streamlit UI -----
uploaded_file = st.file_uploader("อัปโหลดรูปมือถือ", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 รูปที่อัปโหลด", use_column_width=True)

    # ทำนาย
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        class_id = str(predicted.item())
        phone_model = idx_to_label.get(class_id, "Unknown")

    # แสดงผล
    st.markdown(f"### 📱 รุ่นมือถือ: `{phone_model}`")
    score = get_battery_risk_score(phone_model)
    st.markdown(f"### ⚠️ คะแนนความอันตรายของแบตเตอรี่: `{score}`")
