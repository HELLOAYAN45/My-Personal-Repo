import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import matplotlib.cm as cm
import ollama  # <-- NEW: The LLM connector

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="NeuroScan AI | Multimodal", page_icon="🧠", layout="wide")

# --- 2. CUSTOM CSS INJECTION ---
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        font-size: 3rem;
        font-weight: 700;
        background: -webkit-linear-gradient(#4A90E2, #50E3C2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding-bottom: 0px;
    }
    .sub-title {
        text-align: center;
        color: #A0AEC0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    div[data-testid="metric-container"] {
        background-color: #1E2530;
        border-radius: 12px;
        padding: 15px 20px;
        border: 1px solid #2D3748;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .stImage > img {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.4);
    }
    .llm-box {
        background-color: #1A202C;
        border-left: 5px solid #50E3C2;
        padding: 20px;
        border-radius: 8px;
        margin-top: 20px;
        font-family: 'Courier New', Courier, monospace;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. LOAD CV MODEL ---
@st.cache_resource 
def load_unet():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'unet_brats_best.keras')
    return tf.keras.models.load_model(model_path, compile=False)

model = load_unet()

# --- 4. GRAD-CAM MATH ENGINE ---
def make_gradcam_heatmap(img_array, model):
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer_name = layer.name
            break

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[0, :, :, 0]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
        
    return heatmap.numpy()

def overlay_gradcam(img_array, heatmap, alpha=0.5):
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = tf.image.resize(jet_heatmap, (img_array.shape[0], img_array.shape[1])).numpy()
    original_rgb = np.stack((img_array,) * 3, axis=-1).astype(np.float32)
    superimposed_img = jet_heatmap * alpha + original_rgb * (1 - alpha)
    return np.clip(superimposed_img, 0, 1)

# --- 5. SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Engine Settings")
    st.write("Tune the CV sensitivity.")
    min_tumor_area = st.slider("Noise Filter (Pixels)", min_value=10, max_value=1000, value=500, step=10)
    st.markdown("---")
    st.caption("CV: ResNet34 | XAI: Grad-CAM | NLP: Llama 3")

# --- 6. MAIN UI ---
st.markdown('<div class="main-title">NeuroScan AI Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Multimodal Anomaly Detection: Computer Vision + Large Language Models</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Drop a 2D MRI Brain Slice here (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Preprocess Image
    image = Image.open(uploaded_file).convert('L')
    image = image.resize((240, 240))
    img_array = np.array(image)

    if np.max(img_array) > 0:
        normalized_img = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array))
    else:
        normalized_img = img_array
    
    padded_img = np.pad(normalized_img, ((8, 8), (8, 8)), mode='constant', constant_values=0)
    img_array_3c = np.stack((padded_img, padded_img, padded_img), axis=-1)
    input_tensor = np.expand_dims(img_array_3c, axis=0)

    # Run AI Inference & Grad-CAM
    with st.spinner('🔬 Running ResNet34 vision inference...'):
        prediction_raw = model.predict(input_tensor)[0].squeeze()
        prediction_cropped = prediction_raw[8:-8, 8:-8]
        predicted_mask = (prediction_cropped > 0.5).astype(np.uint8) 
        
        heatmap = make_gradcam_heatmap(input_tensor, model)
        heatmap_cropped = heatmap[8:-8, 8:-8]
        gradcam_visual = overlay_gradcam(normalized_img, heatmap_cropped)

    # Metrics
    tumor_pixel_count = np.sum(predicted_mask)
    has_tumor = tumor_pixel_count > min_tumor_area 

    if has_tumor:
        confidence_score = np.mean(prediction_cropped[predicted_mask == 1]) * 100
    else:
        confidence_score = (1.0 - np.max(prediction_cropped)) * 100

    brain_pixel_count = np.sum(img_array > 0)
    tumor_percentage = (tumor_pixel_count / brain_pixel_count) * 100 if brain_pixel_count > 0 else 0.0

    original_rgb = np.stack((normalized_img,) * 3, axis=-1).astype(np.float32)
    red_mask = np.zeros_like(original_rgb)
    
    if has_tumor:
        red_mask[:, :, 0] = predicted_mask 

    alpha_overlay = 0.4 
    overlay_img = np.where(
        np.expand_dims(predicted_mask, axis=-1) > 0 if has_tumor else np.zeros_like(np.expand_dims(predicted_mask, axis=-1), dtype=bool),
        (original_rgb * (1 - alpha_overlay)) + (red_mask * alpha_overlay),
        original_rgb
    )

    st.markdown("---")
    
    if has_tumor:
        st.error("⚠️ **CRITICAL FINDING: Anomaly Detected**")
    else:
        st.success("✅ **DIAGNOSIS: Normal Scan (Clear)**")
        
    m1, m2, m3 = st.columns(3)
    m1.metric("CV Confidence", f"{confidence_score:.1f}%")
    m2.metric("Affected Area", f"{tumor_pixel_count} px" if has_tumor else "0 px (Filtered)")
    m3.metric("Brain Vol. Occupied", f"{tumor_percentage:.1f}%" if has_tumor else "0.0%")
        
    st.markdown("<br>", unsafe_allow_html=True)
        
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<h4 style='text-align: center; color: #E2E8F0;'>Original Scan</h4>", unsafe_allow_html=True)
        st.image(image, use_column_width=True)
        
    with col2:
        st.markdown("<h4 style='text-align: center; color: #E2E8F0;'>XAI Heatmap (Grad-CAM)</h4>", unsafe_allow_html=True)
        st.image(gradcam_visual, use_column_width=True, clamp=True)
        
    with col3:
        st.markdown("<h4 style='text-align: center; color: #E2E8F0;'>Segmented Overlay</h4>", unsafe_allow_html=True)
        if has_tumor:
            st.image(overlay_img, use_column_width=True, clamp=True)
        else:
            st.image(normalized_img, use_column_width=True, clamp=True)

    # --- 7. LLAMA 3 CLINICAL ASSISTANT ---
    if has_tumor:
        st.markdown("---")
        st.subheader("🤖 Llama 3 Clinical Assistant")
        st.write("Generating multimodal analysis based on ResNet34 telemetry...")
        
        # We craft a dynamic prompt injecting the exact math from the CV model
        prompt = f"""
        You are an AI neuro-assistant. A diagnostic computer vision model just analyzed a 2D axial MRI brain slice. 
        It detected a space-occupying lesion taking up {tumor_percentage:.1f}% of the visible brain area (Size: {tumor_pixel_count} pixels) with {confidence_score:.1f}% confidence.
        
        Provide a concise, 3-paragraph clinical summary:
        1. Explain what a lesion of this relative size generally means for intracranial pressure or patient symptoms.
        2. Discuss what general regions of the brain are visible in standard axial MRI slices and how masses affect them.
        3. Provide a strict medical disclaimer that this is an AI screening tool and a radiologist must review the scan.
        Do not use bold text, just provide clean paragraphs. Keep the tone highly professional and objective.
        """
        
        with st.spinner("Llama 3 is analyzing the data..."):
            # Stream the response live into a stylized HTML box
            response_box = st.empty()
            full_response = ""
            
            try:
                for chunk in ollama.chat(model='llama3', messages=[{'role': 'user', 'content': prompt}], stream=True):
                    full_response += chunk['message']['content']
                    response_box.markdown(f'<div class="llm-box">{full_response}▌</div>', unsafe_allow_html=True)
                
                # Final output without the blinking cursor
                response_box.markdown(f'<div class="llm-box">{full_response}</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Failed to connect to Ollama. Ensure the Llama 3 model is downloaded and running. Error: {e}")
