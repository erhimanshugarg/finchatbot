# streamlit_app.py

import streamlit as st
# streamlit_app.py

import os
import logging
import torch
import nest_asyncio
import asyncio
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

# Function to handle asyncio in Streamlit
def run_async_function(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # No event loop in the current context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    else:
        return asyncio.run(coro)  # Ensures proper execution of async functions

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
    st.write("🔍 Thinking...")

    try:
        # If `generate_reponse_bot` is an async function, handle it properly
        if asyncio.iscoroutinefunction(generate_reponse_bot):
            response, confidence = run_async_function(generate_reponse_bot(user_query))
        else:
            response, confidence = generate_reponse_bot(user_query)

        st.write(f"🧠 Chatbot Response: {response}")
        st.write(f"📊 Confidence Score: {confidence:.2f}")
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        st.error("⚠️ Failed to generate a response. Please check logs.")