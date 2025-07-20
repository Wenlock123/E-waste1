import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import json

# ----- โหลดโมเดล -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# โมเดลควรเหมือนตอน train
model = ...  # ใส่โค้ดสร้างโมเดลของคุณ เช่น resnet18(pretrained=False)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

# ----- โหลด label -----
with open("labels.json") as f:
    idx_to_label = json.load(f)

# ----- Transform -----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ----- ฟังก์ชันให้คะแนนความอันตราย -----
def get_battery_risk_score(phone_model):
    risk_scores = {
        "iPhone 11": 3.5,
        "Galaxy A32": 4.2,
        # เพิ่มได้อีก
    }
    return risk_scores.get(phone_model, "ไม่ทราบข้อมูล")

# ----- Streamlit UI -----
st.title("🔍 Mobile Battery Classifier")

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
