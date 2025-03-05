# chatbot.py

import os
import torch
from data_processing import download_report, parse_pdf, load_text_chunks, save_text_chunks
from retrieval_and_generation import build_faiss_index, load_faiss_index, load_text_chunks_from_pickle, hybrid_retrieval, generate_response
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from pathlib import Path
import logging
import re

# Global variables for FAISS index, text chunks, and models
faiss_index = None
text_chunks = None
model = None
slm_model = None
tokenizer = None

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_preprocessed_data():
    """Load preprocessed data (FAISS index, text chunks, models, etc.)."""
    global faiss_index, text_chunks, model, slm_model, tokenizer
    logger.info(f"inside load preprocessed data...{faiss_index} {text_chunks} {model} {slm_model} {tokenizer}")
    # Load FAISS index
    index_path = "financial_index.faiss"
    if os.path.exists(index_path):
        faiss_index = load_faiss_index(index_path)
        logger.info(f"FAISS index loaded successfully from {index_path} and assigned to faiss_index.")
    else:
        logger.error(f"FAISS index not found at {index_path}")
        raise FileNotFoundError(f"FAISS index not found at {index_path}")

    # Load text chunks
    chunk_path = "text_chunks.pkl"
    if os.path.exists(chunk_path):
        text_chunks = load_text_chunks_from_pickle(chunk_path)
        logger.info(f"Text chunks loaded successfully from {chunk_path} and assigned to text_chunks.")
    else:
        logger.error(f"Text chunks not found at {chunk_path}")
        raise FileNotFoundError(f"Text chunks not found at {chunk_path}")

    # Load models
    model = SentenceTransformer("all-MiniLM-L6-v2")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    slm_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-1_5", torch_dtype=torch.float16
    ).to(device)

def retrieve_and_generate_response(query):
    """Retrieve relevant chunks and generate a response."""
    global faiss_index, text_chunks, model, slm_model, tokenizer  # Access global variables
    logger.info(f"inside retrieve and generate response...{faiss_index} {text_chunks} {model} {slm_model} {tokenizer}")
    if faiss_index is None or text_chunks is None:
        logger.error("Missing FAISS index or chunk data! Ensure data is available.")
        return "Missing FAISS index or chunk data! Ensure data is available.", 0
    retrieved_chunks = hybrid_retrieval(query, model, faiss_index, text_chunks)
    if retrieved_chunks:
        answer, confidence = generate_response(query, retrieved_chunks, slm_model, tokenizer)
        logger.info(f"Confidence: {confidence}")
        return answer, confidence
    return "No relevant results found.", 0

def filter_query(query):
    """Filter inappropriate or off-topic queries."""
    if not query or not isinstance(query, str):  # ✅ Ensure query is a string
        return "Invalid input. Please enter a valid financial query.", True

    prohibited_keywords = ['violence', 'harm', 'racism', 'fraud', 'scam']
    for word in prohibited_keywords:
        if re.search(rf"\b{word}\b", query, re.IGNORECASE):
            return "Your query contains inappropriate content and has been rejected.", True
    
    if not any(keyword in query.lower() for keyword in ['financial', 'stock', 'revenue', 'earnings', 'profit']):
        return "Please provide a query related to finance.", True
    
    return query, False

def filter_answer(answer):
    """Filter inappropriate or off-topic answers."""
    if not answer or not isinstance(answer, str):  # ✅ Ensure answer is a string
        return "The response is invalid or missing. Please try again.", True

    forbidden_terms = ['capital', 'science', 'sports', 'france']
    for term in forbidden_terms:
        if term in answer.lower():
            return "The answer seems off-topic. Please refine your query.", True
    
    if len(answer.split()) < 5:  # If the answer is too vague
        return "The answer seems too vague. Please refine your query for a more detailed response.", True
    
    return answer, False

def generate_reponse_bot(query):
    """Generate a response to the user's query."""
    logger.info(f"inside generate response bot...{query}")
    res, is_filtered = filter_query(query)
    if is_filtered:  # Query is filtered
        return res, 1
    answer, confidence = retrieve_and_generate_response(query)
    logger.info(f"got the result Confidence: {confidence}")
    res, is_filtered = filter_answer(answer)
    if is_filtered:  # Response is filtered
        return res, 1
    else:
        return answer, confidence

def main():
    """Run preprocessing steps."""
    # Create the folder if it doesn't exist
    logger.info("inside main of chatbot...")
    folder_name = "financial_reports"
    os.makedirs(folder_name, exist_ok=True)

    # Download reports
    reports = {
        "2023": "https://investors.broadcom.com/static-files/2b98b262-4fbb-4731-b3dd-88f6ca187002",
        "2024": "https://investors.broadcom.com/static-files/2e0788d2-4c75-4ed9-bde2-a96a7abb8996"
    }
    for year, url in reports.items():
        download_report(url, f"broadcom_{year}.pdf")
    logger.info("Downloaded reports successfully!")
    # Extract and clean text
    for file in os.listdir("financial_reports"):
        logger.info(f"inside for loop...{file}")
        if file.endswith(".pdf"):
            pdf_path = os.path.join("financial_reports", file)
            output_txt_path = os.path.join("financial_reports", f"{Path(file).stem}_cleaned.txt")
            parse_pdf(pdf_path, output_txt_path)

    # Build FAISS index
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index_path = "financial_index.faiss"
    chunk_path = "text_chunks.pkl"
    text_chunks = load_text_chunks("financial_reports")
    build_faiss_index(text_chunks, model, index_path, chunk_path)

# Run preprocessing if this script is executed directly
if __name__ == "__main__":
    main()