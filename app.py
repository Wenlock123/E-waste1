import os
import gdown
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import streamlit as st

# กำหนดชื่อคลาสของโมเดล
CLASS_NAMES = ["Galaxy_A32", "iPhone_11", "iPhone_12"]

# ฟังก์ชันดาวน์โหลดไฟล์โมเดลจาก Google Drive
@st.cache_resource(show_spinner=False)
def download_model():
    model_path = "best_model.pth"
    if not os.path.exists(model_path):
        file_id = "1vud0Qk1PHy7_jLgSUwwurUN7KKw1WeIw"  # เปลี่ยนเป็นไฟล์ไอดีของคุณ
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_path, quiet=False)
    return model_path

# ฟังก์ชันโหลดโมเดล PyTorch
@st.cache_resource(show_spinner=False)
def load_model(model_path, num_classes=len(CLASS_NAMES)):
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# ฟังก์ชัน preprocess รูปภาพก่อนเข้าโมเดล
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return preprocess(image).unsqueeze(0)  # เพิ่มมิติ batch size

# Streamlit UI
st.title("Mobile Phone Model Classifier with PyTorch")

# โหลดโมเดล
model_path = download_model()
model = load_model(model_path)
st.success(f"Model loaded from {model_path}")

uploaded_file = st.file_uploader("Upload a phone image to classify", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = preprocess_image(image)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, preds = torch.max(outputs, 1)
        predicted_class = CLASS_NAMES[preds.item()]

    st.write(f"**Predicted class:** {predicted_class}")
