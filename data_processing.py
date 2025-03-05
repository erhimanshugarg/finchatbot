# data_processing.py

import os
import requests
import pdfplumber
import re
import pickle
import logging
from pdfminer.pdfdocument import PDFSyntaxError

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_report(url, filename):
    """Download a PDF report from a URL and save it locally."""
    response = requests.get(url)
    if response.status_code == 200:
        file_path = os.path.join("financial_reports", filename)
        with open(file_path, "wb") as file:
            file.write(response.content)
        print(f"PDF downloaded successfully and saved as {filename}")
    else:
        print(f"Failed to download PDF. Status code: {response.status_code}")

def extract_text_from_pdf(pdf_path):
    """Extract text from a financial PDF statement."""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
    except PDFSyntaxError:
        print(f"Error: {pdf_path} is not a valid PDF file.")
        return None
    return text

def clean_text(text):
    """Remove extra spaces, special characters, and unnecessary line breaks."""
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^a-zA-Z0-9.,$% ]', '', text)  # Keep financial characters
    return text.strip()

def save_cleaned_text(text, output_path):
    """Save cleaned text to a file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)

def parse_pdf(pdf_path, output_txt_path):
    """Extract text from a financial PDF statement and save it to a text file."""
    if os.path.exists(pdf_path):
        extracted_text = extract_text_from_pdf(pdf_path)
        cleaned_text = clean_text(extracted_text)
        save_cleaned_text(cleaned_text, output_txt_path)
        logger.info(f"{pdf_path} cleaned and saved at {output_txt_path}!")
        print(f"{pdf_path} cleaned and saved at {output_txt_path}!")
    else:
        print("PDF not found! Please check the path.")
        logger.error(f"PDF not found at {pdf_path}!")

def load_text_chunks(directory, chunk_size=200):
    """Load text chunks from processed .txt files with smaller chunk sizes."""
    all_chunks = []
    for filename in os.listdir(directory):
        logger.info(f"filename inside load_Text_chunks: {filename}")
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            logger.info(f"file_path inside load_Text_chunks: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                while True:
                    text = f.read(chunk_size)
                    if not text:
                        break
                    all_chunks.extend(text.split("\n"))  # Assuming chunks are line-separated
                    logger.info(f"all_chunks inside load_Text_chunks created")
    return all_chunks

def save_text_chunks(chunks, chunk_path):
    """Save text chunks to a pickle file."""
    with open(chunk_path, "wb") as f:
        pickle.dump(chunks, f)
        logger.info(f"Text chunks saved to {chunk_path}! from save_text_chunks")     