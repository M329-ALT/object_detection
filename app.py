import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

@st.cache_resource
def load_optimized_model():
    model_path = "best_coco128_model.pt"
    if os.path.exists(model_path):
        return YOLO(model_path)
    else:
        st.warning("⚠️ النموذج غير موجود. سيتم استخدام نموذج افتراضي.")
        return YOLO("yolov8m.pt")

model = load_optimized_model()

st.title("🔍 كشف الأهداف - YOLOv8")
uploaded_file = st.file_uploader("ارفع صورة", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="الصورة الأصلية", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        image.save(tmp.name)
        tmp_path = tmp.name

    results = model(tmp_path, conf=0.25)
    annotated_img = results[0].plot()
    st.image(annotated_img, caption="النتائج", use_column_width=True)
    st.success(f"تم اكتشاف {len(results[0].boxes)} هدف/أهداف")

    os.remove(tmp_path)
