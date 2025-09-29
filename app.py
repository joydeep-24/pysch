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
        if message
