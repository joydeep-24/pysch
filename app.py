# app.py
import streamlit as st
from streamlit_webrtc import webrtc_streamer
from av import VideoFrame
import numpy as np

# Import modules
from text_analyzer import TextAnalyzer
from vision_analyzer import analyze_facial_expression
from conversational_model import ConversationalModel

st.set_page_config(layout="wide", page_title="Multimodal AI Psychological Analyzer")

@st.cache_resource
def load_models():
    text_model = TextAnalyzer(model_path="/content/drive/MyDrive/fine-tuned-analyzer-7labels")
    conv_model = ConversationalModel(model_path="/content/drive/MyDrive/models/llama-3-8b-instruct")
    return text_model, conv_model

st.title("🧠 Multimodal Psychological Analyzer")
text_analyzer, conversational_model = load_models()

if "history" not in st.session_state:
    st.session_state.history = [{"role": "system", "content": "You are a caring assistant."}]
    st.session_state.latest_analysis = {}

col1, col2 = st.columns(2)

with col1:
    st.header("Conversation")
    for message in st.session_state.history:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if user_prompt := st.chat_input("How are you feeling?"):
        text_analysis = text_analyzer.predict(user_prompt)
        facial_analysis = st.session_state.get("latest_facial_analysis", {"dominant_emotion": "unknown"})

        fused_analysis = {"text_analysis": text_analysis, "vision_analysis": facial_analysis}
        st.session_state.latest_analysis = fused_analysis

        st.session_state.history.append({"role": "user", "content": user_prompt})
        with st.spinner("Thinking..."):
            ai_response = conversational_model.generate_response(st.session_state.history)
        st.session_state.history.append({"role": "assistant", "content": ai_response})

        st.rerun()

with col2:
    st.header("Real-Time Analysis")
    webrtc_ctx = webrtc_streamer(key="webcam")
    if webrtc_ctx.video_receiver:
        try:
            video_frame = webrtc_ctx.video_receiver.get_frame(timeout=10)
            image = video_frame.to_ndarray(format="bgr24")
            facial_analysis = analyze_facial_expression(image)
            st.session_state.latest_facial_analysis = facial_analysis
        except Exception:
            st.warning("⚠️ Webcam feed stopped.")

    st.subheader("Latest Fused Analysis")
    st.json(st.session_state.latest_analysis)
