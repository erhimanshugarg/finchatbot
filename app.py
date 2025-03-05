# app.py

import streamlit as st
import os
from chatbot import generate_reponse_bot, load_preprocessed_data

# Set page config
st.set_page_config(page_title="Retrieval-Augmented Generation (RAG)", layout="wide")

# Check if data files exist, if not, generate them
if not os.path.exists("financial_index.faiss") or not os.path.exists("text_chunks.pkl"):
    import chatbot
    chatbot.main()

# Cache the data so that it is not loaded multiple times
@st.cache_resource
def load_data():
    # Load preprocessed data
    load_preprocessed_data()

# Load preprocessed data
load_data()

# Initialize session state
if "response_data" not in st.session_state:
    st.session_state.response_data = None
if "user_query" not in st.session_state:
    st.session_state.user_query = ""

# Title and divider
st.subheader("Financial Chatbot", divider=True)

# Form for user input
with st.form("query_form"):
    # Input field for user query
    user_query = st.text_input(
        "Enter your query:",
        placeholder="E.g., what is revenue of Broadcom?",
        value=st.session_state.user_query  # Preserve the user query in the form
    )
    send_button = st.form_submit_button("Send")

    st.write("Developed 🧑‍💻 by Group 77")

    # Handle form submission
    if send_button:
        if user_query:
            # Store the user query in session state
            st.session_state.user_query = user_query

            # Show a loader while fetching the result
            with st.spinner("Fetching results..."):
                try:
                    # Generate the response
                    response, confidence = generate_reponse_bot(user_query)

                    # Store the response data in session state
                    st.session_state.response_data = {
                        "response": response,
                        "confidence": confidence
                    }
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a query.")

# Display the response outside the form
if st.session_state.response_data:
    st.subheader("Response:", divider=True)
    st.write(st.session_state.response_data["response"])

    st.write(f"Confidence: {st.session_state.response_data['confidence']}")

    # Clear the session state after displaying the response
    st.session_state.response_data = None
    st.session_state.user_query = ""  # Clear the user query