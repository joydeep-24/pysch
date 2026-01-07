# ===================== SYSTEM SAFETY (VERY IMPORTANT) =====================
import os
# Force TensorFlow (DeepFace) to CPU, PyTorch can use GPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ===================== STANDARD IMPORTS =====================
import streamlit as st
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import time

from streamlit_webrtc import webrtc_streamer

# ===================== PROJECT MODULES =====================
from text_analyzer import TextAnalyzer
from vision_analyzer import analyze_facial_expression_vector
from conversational_model import ConversationalModel
from fusion_model import MultimodalFusionModel

# ===================== EMOTION → VALENCE–AROUSAL MAP =====================
TEXT_EMOTION_VA = {
    0: (0.8, 0.6),   # joy
    1: (-0.7, 0.3),  # sadness
    2: (-0.8, 0.8),  # anger
    3: (-0.9, 0.9),  # fear
    4: (0.4, 0.6),   # surprise
    5: (-0.6, 0.4),  # disgust
    6: (0.0, 0.1),   # neutral
}

VISION_EMOTION_VA = {
    0: (-0.8, 0.8),  # angry
    1: (-0.6, 0.4),  # disgust
    2: (-0.9, 0.9),  # fear
    3: (0.9, 0.6),   # happy
    4: (-0.7, 0.3),  # sad
    5: (0.4, 0.6),   # surprise
    6: (0.0, 0.1),   # neutral
}

CLASS_LABELS = ["Calm", "Positive", "Neutral", "Stressed", "Distressed"]

# ===================== UTILITY FUNCTIONS =====================
def emotion_probs_to_va(vec, mapping):
    v, a = 0.0, 0.0
    for i, p in enumerate(vec):
        mv, ma = mapping[i]
        v += p * mv
        a += p * ma
    return np.array([v, a], dtype=np.float32)

def extract_context_features(chat_history):
    user_msgs = [m["content"] for m in chat_history if m["role"] == "user"]
    if not user_msgs:
        return np.zeros(4, dtype=np.float32)

    total_len = sum(len(m) for m in user_msgs)
    negativity = sum(1 for m in user_msgs if any(w in m.lower() for w in ["sad", "angry", "upset", "not"]))
    repetition = len(user_msgs) / max(len(set(user_msgs)), 1)

    return np.array([
        min(total_len / 500, 1.0),         # emotional intensity
        -0.2 * negativity,                 # sentiment trend
        negativity / len(user_msgs),       # negativity ratio
        min(repetition, 1.0)               # repetition score
    ], dtype=np.float32)

# ===================== MODEL LOADERS =====================
@st.cache_resource
def load_core_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text_model = TextAnalyzer()

    fusion_model = MultimodalFusionModel().to(device)
    fusion_model.load_state_dict(
        torch.load("outputs/fusion_model_v3.pth", map_location=device)
    )
    fusion_model.eval()

    return text_model, fusion_model, device

@st.cache_resource
def load_llm():
    return ConversationalModel()

# ===================== STREAMLIT PAGE CONFIG =====================
st.set_page_config(
    page_title="Multimodal Emotional Intelligence System",
    layout="wide"
)

st.title("🧠 Multimodal Emotional Intelligence System")

# ===================== LOAD CORE MODELS =====================
with st.spinner("Loading emotion analysis models..."):
    text_analyzer, fusion_model, device = load_core_models()

st.success("✅ Emotion analysis models loaded")

# ===================== SESSION STATE =====================
if "history" not in st.session_state:
    st.session_state.history = [{"role": "system", "content": "You are a caring assistant."}]
    st.session_state.latest_frame = None
    st.session_state.latest_state = None
    st.session_state.latest_conf = None
    st.session_state.llm_loaded = False

# ===================== UI LAYOUT =====================
col1, col2 = st.columns(2)

# ===================== LEFT: CHAT =====================
with col1:
    st.header("💬 Conversation")

    for msg in st.session_state.history:
        if msg["role"] != "system":
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if user_text := st.chat_input("How are you feeling today?"):
        st.session_state.history.append({"role": "user", "content": user_text})

        # -------- TEXT EMOTION --------
        text_vec = text_analyzer.predict_vector(user_text)
        text_va = emotion_probs_to_va(text_vec, TEXT_EMOTION_VA)

        # -------- VISION EMOTION --------
        if st.session_state.latest_frame is not None:
            vision_vec = analyze_facial_expression_vector(st.session_state.latest_frame)
        else:
            vision_vec = np.zeros(7, dtype=np.float32)

        vision_va = emotion_probs_to_va(vision_vec, VISION_EMOTION_VA)

        # -------- CONTEXT --------
        context_vec = extract_context_features(st.session_state.history)

        # -------- FUSION --------
        fusion_input = torch.tensor(
            np.concatenate([text_va, vision_va, context_vec]),
            dtype=torch.float32
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = fusion_model(fusion_input)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        idx = int(np.argmax(probs))
        st.session_state.latest_state = CLASS_LABELS[idx]
        st.session_state.latest_conf = probs[idx]

        # -------- LLM LOADING (LAZY) --------
        if not st.session_state.llm_loaded:
            progress = st.progress(0, text="Loading conversational model (one-time)...")
            for i in range(5):
                time.sleep(0.3)
                progress.progress((i + 1) * 20)
            st.session_state.llm = load_llm()
            st.session_state.llm_loaded = True
            progress.empty()

        # -------- RESPONSE --------
        ai_reply = st.session_state.llm.generate_response(st.session_state.history)
        st.session_state.history.append({"role": "assistant", "content": ai_reply})

        st.rerun()

# ===================== RIGHT: VISION + RESULTS =====================
with col2:
    st.header("📷 Live Analysis")

    webrtc_ctx = webrtc_streamer(key="camera")

    if webrtc_ctx.video_receiver:
        try:
            frame = webrtc_ctx.video_receiver.get_frame(timeout=5)
            img = frame.to_ndarray(format="bgr24")
            st.session_state.latest_frame = img
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Live Camera")
        except:
            st.info("Camera initializing...")

    st.subheader("🧠 Fusion Output")

    if st.session_state.latest_state:
        st.metric(
            "Detected Psychological State",
            st.session_state.latest_state,
            f"Confidence: {st.session_state.latest_conf:.2f}"
        )
    else:
        st.info("Awaiting user input...")
