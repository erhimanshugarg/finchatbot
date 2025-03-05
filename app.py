# streamlit_app.py

import streamlit as st
from chatbot import load_preprocessed_data, generate_reponse_bot

# Load preprocessed data
load_preprocessed_data()

# Streamlit UI
st.title("Financial Chatbot")

user_query = st.text_input("Enter your financial query:")

if user_query:
    response, confidence = generate_reponse_bot(user_query)
    st.write(f"Response: {response}")
    st.write(f"Confidence: {confidence:.2f}")