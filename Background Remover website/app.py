import streamlit as st
import torch
import torchvision
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101
from PIL import Image, ImageEnhance
import numpy as np
import io
import zipfile
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Vision Extract", layout="wide", page_icon="🎯")

# --- CUSTOM DARK MODE & UI STYLING ---
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #E0E0E0; }
    .main-header {
        font-size: 3rem; font-weight: 800;
        background: -webkit-linear-gradient(#FF4B4B, #FF8383);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        padding-bottom: 10px;
    }
    div[data-testid="stExpander"] { background-color: #161B22; border: 1px solid #30363D; border-radius: 12px; }
    .stButton>button { border-radius: 8px; background-color: #FF4B4B; color: white; border: none; font-weight: bold; width: 100%; }
    .stMetric { background-color: #1c2128; padding: 10px; border-radius: 10px; border: 1px solid #30363D; }
    </style>
    """, unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_model_local():
    model_path = 'model/deeplabv3_resnet101_coco.pth'
    model = deeplabv3_resnet101(weights=None, weights_backbone=None, aux_classifier=True)
    
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model
    else:
        st.error(f"❌ Model weights not found.")
        st.stop()

model = load_model_local()

# --- IMAGE DISPLAY UTILITY (SILENCES WARNINGS) ---
def safe_image(img, caption):
    """Displays image using the correct parameter for your Streamlit version."""
    try:
        st.image(img, caption=caption, use_container_width=True)
    except TypeError:
        st.image(img, caption=caption, use_column_width=True)

# --- PROCESSING HELPERS ---
def crop_to_subject(image, mask):
    coords = np.argwhere(mask)
    if coords.size == 0: return image
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    return image.crop((x0, y0, x1, y1))

def process_image(input_image, bg_choice, brightness, contrast, auto_crop):
    preprocess = transforms.Compose([
        transforms.Resize(520),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    
    output_predictions = output.argmax(0).byte().cpu().numpy()
    mask = output_predictions > 0
    mask_image = Image.fromarray(mask).resize(input_image.size, resample=Image.NEAREST)
    mask_np = np.array(mask_image)
    img_np = np.array(input_image.convert("RGB"))
    
    if bg_choice == "Black": bg_color = [0, 0, 0]
    elif bg_choice == "White": bg_color = [255, 255, 255]
    else: bg_color = [45, 45, 45]

    result_np = np.full_like(img_np, bg_color)
    result_np[mask_np] = img_np[mask_np]
    result_img = Image.fromarray(result_np)
    
    if brightness != 1.0: result_img = ImageEnhance.Brightness(result_img).enhance(brightness)
    if contrast != 1.0: result_img = ImageEnhance.Contrast(result_img).enhance(contrast)

    coverage = (np.sum(mask_np) / mask_np.size) * 100
    if auto_crop: result_img = crop_to_subject(result_img, mask_np)

    return result_img, coverage

# --- MAIN UI ---
st.markdown('<p class="main-header">AI Vision Extract</p>', unsafe_allow_html=True)

with st.sidebar:
    st.title("🎚️ Settings")
    bg_style = st.selectbox("Background Fill", ["Black", "White", "Studio Gray"])
    do_crop = st.checkbox("Auto-Crop Subject", value=False)
    st.divider()
    st.subheader("Enhancements")
    bright_val = st.slider("Brightness", 0.5, 2.0, 1.0, 0.05)
    cont_val = st.slider("Contrast", 0.5, 2.0, 1.0, 0.05)

uploaded_files = st.file_uploader("Upload Image(s)", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    processed_images = []
    for uploaded_file in uploaded_files:
        with st.expander(f"📷 {uploaded_file.name}", expanded=True):
            img = Image.open(uploaded_file)
            col1, col2 = st.columns(2)
            with col1:
                safe_image(img, "Original Input")
            with col2:
                with st.spinner('Analyzing Pixels...'):
                    result, coverage = process_image(img, bg_style, bright_val, cont_val, do_crop)
                    safe_image(result, "Isolated Output")
                    processed_images.append((uploaded_file.name, result))
            
            # Action Row
            m1, m2, m3 = st.columns([1,1,2])
            m1.metric("Coverage", f"{coverage:.1f}%")
            m2.info(f"**Res:** {img.size[0]}x{img.size[1]}")
            buf = io.BytesIO()
            result.save(buf, format="PNG")
            m3.download_button(f"📥 Download Result", buf.getvalue(), f"extract_{uploaded_file.name}.png", "image/png")

    if len(processed_images) > 1:
        st.markdown("### 📦 Batch Operations")
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            for name, img in processed_images:
                img_buf = io.BytesIO()
                img.save(img_buf, format="PNG")
                zip_file.writestr(f"vision_extract_{name}.png", img_buf.getvalue())
        st.download_button("🚀 Download All as ZIP Archive", zip_buffer.getvalue(), "vision_export.zip", "application/zip")
else:
    st.info("System Ready. Please upload images to begin extraction.")