# streamlit_app.py

import streamlit as st
# streamlit_app.py

import os
import logging
import torch
import nest_asyncio
from chatbot import load_preprocessed_data, generate_reponse_bot

# Fix event loop issues
nest_asyncio.apply()

# Disable static file watcher
os.environ["STREAMLIT_SERVER_ENABLE_STATIC_FILE_WATCHER"] = "false"

# Limit PyTorch threads
torch.set_num_threads(1)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit UI
st.title("Financial Chatbot")

# Load preprocessed data
logger.info("Loading preprocessed data...")
try:
    load_preprocessed_data()
    logger.info("Preprocessed data loaded successfully!")
except Exception as e:
    logger.error(f"Error loading preprocessed data: {e}")
    st.error("Failed to load preprocessed data. Please check the logs.")
    st.stop()

# User input
user_query = st.text_input("Enter your financial query:")

if user_query:
    logger.info(f"Processing query: {user_query}")
    response, confidence = generate_reponse_bot(user_query)
    st.write(f"Response: {response}")
    st.write(f"Confidence: {confidence:.2f}")