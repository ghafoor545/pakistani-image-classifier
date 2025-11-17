
import streamlit as st
import torch
from PIL import Image
import json
from pathlib import Path
import requests
from utils.transform import get_transform
import timm

# --- Page Config ---
st.set_page_config(
    page_title="Pakistani AI Vision Pro",
    page_icon="🇵🇰",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Beautiful CSS ---
st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh;}
    .stApp {background: transparent;}
    h1 {font-family: 'Poppins', sans-serif; color: #fff; text-align: center; font-size: 3.5rem; 
        text-shadow: 0 0 20px rgba(255,255,255,0.5);}
    .upload-box {background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); 
                 border-radius: 20px; border: 2px dashed #fff; padding: 2rem; text-align: center;}
    .result-card {background: rgba(255,255,255,0.2); backdrop-filter: blur(15px);
                  border-radius: 25px; padding: 2rem; margin: 2rem 0; box-shadow: 0 20px 40px rgba(0,0,0,0.3);}
    .pred-text {font-size: 3.5rem; font-weight: bold; background: linear-gradient(90deg, #ff6b6b, #feca57);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center;}
    .conf-bar {height: 28px; background: rgba(255,255,255,0.25); border-radius: 15px; overflow: hidden; margin: 15px 0;}
    .conf-fill {height: 100%; background: linear-gradient(90deg, #a8edea, #fed6e3); border-radius: 15px;}
    .top5 {background: rgba(0,0,0,0.25); border-radius: 15px; padding: 1rem; margin: 12px 0;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🇵🇰 Pakistani AI Vision Pro</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#fff; font-size:1.4rem;'>Upload karo photo – AI batayega kya hai!</p>", unsafe_allow_html=True)

# --- Model Loading ---
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure model folder exists
    Path("model").mkdir(exist_ok=True)
    model_path = Path("model/refined_vit_model.pth")
    class_info_path = Path("model/refined_vit_class_info.json")

    # Download model if missing
    if not model_path.exists():
        url = "https://huggingface.co/Ghafoor545/refined-vit-model/resolve/main/refined_vit_model.pth"
        with open(model_path, "wb") as f:
            r = requests.get(url)
            f.write(r.content)

    # Download class info if missing
    if not class_info_path.exists():
        url = "https://huggingface.co/Ghafoor545/refined-vit-model/resolve/main/refined_vit_class_info.json"
        with open(class_info_path, "wb") as f:
            r = requests.get(url)
            f.write(r.content)

    # Load classes
    with open(class_info_path, "r") as f:
        classes = json.load(f)["classes"]

    # Initialize and load model
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=len(classes))
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    model.eval()
    model.to(device)
    return model, classes, device

model, class_names, device = load_model()

# --- File Uploader ---
uploaded_file = st.file_uploader(
    "Upload your image here",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    label_visibility="collapsed"
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.image(image, caption="Tumhari tasveer", use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        with st.spinner("AI soch raha hai..."):
            transform = get_transform()
            input_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.nn.functional.softmax(output[0], dim=0)
                conf, idx = torch.max(probs, 0)
                pred_class = class_names[idx.item()]
                confidence = conf.item() * 100

        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.markdown(f"<p class='pred-text'>{pred_class.upper()}</p>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='color:#fff; text-align:center; margin:10px 0;'>Confidence: {confidence:.2f}%</h2>", unsafe_allow_html=True)
        st.markdown(f"<div class='conf-bar'><div class='conf-fill' style='width:{confidence}%'></div></div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # --- Top-5 Predictions ---
        st.markdown("<h3 style='color:#fff; text-align:center;'>Top 5 Guesses</h3>", unsafe_allow_html=True)
        top5_prob, top5_idx = torch.topk(probs, 5)
        for i in range(5):
            label = class_names[top5_idx[i]].title()
            prob = top5_prob[i].item() * 100
            st.markdown(f"""
            <div class='top5'>
                <h3 style='color:#fff; margin:0; display:inline'>{i+1}. {label}</h3>
                <span style='float:right; color:#feca57; font-weight:bold'>{prob:.1f}%</span>
                <div style='height:10px; background:rgba(255,255,255,0.2); border-radius:5px; margin-top:8px;'>
                    <div style='width:{prob}%; height:100%; background:linear-gradient(90deg,#a8edea,#fed6e3); border-radius:5px;'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.balloons()

st.markdown("<br><hr><p style='text-align:center; color:#ddd; font-size:1rem;'>Made with ❤️ in Pakistan</p>", unsafe_allow_html=True)
