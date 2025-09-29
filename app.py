# app.py
import streamlit as st
from streamlit_webrtc import webrtc_streamer
from av import VideoFrame
import numpy as np

# Import our AI modules
from text_analyzer import TextAnalyzer
from vision_analyzer import analyze_facial_expression
from conversational_model import ConversationalModel

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Multimodal AI Assessment")

# --- Model Loading ---
# Use caching to load models only once
@st.cache_resource
def load_models():
    text_model = TextAnalyzer()
    conversational_model = ConversationalModel()
    return text_model, conversational_model

st.title("Multimodal AI Psychological Assessment 🤖")
text_analyzer, conversational_model = load_models()

# --- State Management ---
if "history" not in st.session_state:
    st.session_state.history = [{"role": "system", "content": "You are a caring assistant."}]
    st.session_state.latest_analysis = {}

# --- UI Layout ---
col1, col2 = st.columns(2)

with col1:
    st.header("Conversation")
    
    # Display chat history
    for message in st.session_state.history:
        if message['role'] != 'system':
            with st.chat_message(message['role']):
                st.markdown(message['content'])

    # Chat input
    if user_prompt := st.chat_input("How are you feeling?"):
        # 1. Get Vision Analysis
        facial_analysis = st.session_state.get('latest_facial_analysis', {"dominant_emotion": "unknown"})
        
        # 2. Get Text Analysis
        full_transcript = " ".join([m['content'] for m in st.session_state.history if m['role'] == 'user'])
        text_analysis = text_analyzer.predict(full_transcript + " " + user_prompt)
        
        # 3. FUSION (Mock for now)
        # In a real system, you'd train a model to combine these.
        # For now, we'll just merge them for display.
        fused_analysis = {**text_analysis, **facial_analysis}
        st.session_state.latest_analysis = fused_analysis
        
        # 4. Generate Conversational Reply
        st.session_state.history.append({"role": "user", "content": user_prompt})
        with st.spinner("Thinking..."):
            ai_response = conversational_model.generate_response(st.session_state.history)
        st.session_state.history.append({"role": "assistant", "content": ai_response})
        
        # Rerun to display the new messages
        st.rerun()

with col2:
    st.header("Real-Time Analysis")
    
    # Webcam Feed
    webrtc_ctx = webrtc_streamer(key="webcam")
    if webrtc_ctx.video_receiver:
        try:
            video_frame = webrtc_ctx.video_receiver.get_frame(timeout=10)
            image = video_frame.to_ndarray(format="bgr24")
            facial_analysis = analyze_facial_expression(image)
            st.session_state.latest_facial_analysis = facial_analysis
        except Exception:
            st.warning("Webcam feed stopped.")
    
    # Display latest analysis
    st.subheader("Latest Fused Analysis:")

    st.json(st.session_state.latest_analysis)
