# streamlit_app.py

import streamlit as st
import logging
from chatbot import load_preprocessed_data, generate_reponse_bot

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache the preprocessed data to avoid reloading on every interaction
@st.cache_resource(allow_output_mutation=True)
def load_data():
    logger.info("Loading preprocessed data...")
    try:
        load_preprocessed_data()
        logger.info("Preprocessed data loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"Error loading preprocessed data: {e}")
        return False

# Streamlit UI
st.title("Financial Chatbot")

# Load preprocessed data
if not load_data():
    st.error("Failed to load preprocessed data. Please check the logs.")
    st.stop()

# User input
user_query = st.text_input("Enter your financial query:")

if user_query:
    logger.info(f"Processing query: {user_query}")
    response, confidence = generate_reponse_bot(user_query)
    st.write(f"Response: {response}")
    st.write(f"Confidence: {confidence:.2f}")