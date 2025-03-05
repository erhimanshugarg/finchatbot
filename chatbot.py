# chatbot.py

import os
import requests
import pdfplumber
import re
from pathlib import Path
import numpy as np
import faiss
import pickle
import torch
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from transformers import AutoModelForCausalLM, AutoTokenizer
import accelerate
from pdfminer.pdfdocument import PDFSyntaxError

# Global variables for FAISS index, text chunks, and models
faiss_index = None
text_chunks = None
model = None
slm_model = None
tokenizer = None

# Step 1: Download Financial Reports
def download_report(url, filename):
    """Download a PDF report from a URL and save it locally."""
    response = requests.get(url)
    if response.status_code == 200:
        file_path = os.path.join("financial_reports", filename)
        with open(file_path, "wb") as file:
            file.write(response.content)
        print(f"PDF downloaded successfully and saved as {filename}")
        # with open(os.path.join("financial_reports", filename), "wb") as file:
        #     file.write(response.content)
        # print(f"PDF downloaded successfully and saved as {filename}")
        # Log the first 100 bytes of the file to verify it's a PDF
        with open(file_path, "rb") as f:
            print(f"File header: {f.read(100)}")
    else:
        print(f"Failed to download PDF. Status code: {response.status_code}")

# Step 2: Extract and Clean Text from PDF
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
        print(f"{pdf_path} cleaned and saved at {output_txt_path}!")
    else:
        print("PDF not found! Please check the path.")

# Step 3: Load Text Chunks for RAG
def load_text_chunks(directory, chunk_size=200):
    """Load text chunks from processed .txt files with smaller chunk sizes."""
    all_chunks = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                while True:
                    text = f.read(chunk_size)
                    if not text:
                        break
                    all_chunks.extend(text.split("\n"))  # Assuming chunks are line-separated
    return all_chunks

# Step 4: Build FAISS Index
def build_faiss_index(chunks, model, index_path, chunk_path, batch_size=32):
    """Build and save a FAISS index from text chunks, ensuring retrieval consistency."""
    if not chunks:
        raise ValueError("No text chunks provided for FAISS indexing.")

    index = None
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        embeddings = model.encode(batch, convert_to_numpy=True)

        if index is None:
            embedding_dim = embeddings.shape[1]
            index = faiss.IndexFlatL2(embedding_dim)

        index.add(embeddings)

    faiss.write_index(index, index_path)

    # Save chunks to maintain consistency with FAISS indices
    with open(chunk_path, "wb") as f:
        pickle.dump(chunks, f)

    print(f"FAISS index built with {index.ntotal} embeddings and saved to {index_path}.")

# Step 5: Load FAISS Index
def load_faiss_index(index_path):
    """Load FAISS index from file."""
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found at {index_path}")
    return faiss.read_index(index_path)

# Step 6: Load Text Chunks from Pickle File
def load_text_chunks_from_pickle(chunk_path):
    """Load text chunks from a pickle file."""
    if not os.path.exists(chunk_path):
        raise FileNotFoundError(f"Chunk file not found at {chunk_path}")
    with open(chunk_path, "rb") as f:
        return pickle.load(f)

# Step 7: Hybrid Retrieval (BM25 + FAISS)
def bm25_retrieval(query, chunks):
    tokenized_chunks = [chunk.split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    scores = bm25.get_scores(query.split())
    ranked_indices = np.argsort(scores)[::-1][:5]
    return [chunks[i] for i in ranked_indices]

def faiss_retrieval(query, model, faiss_index, chunks, top_k=5):
    if faiss_index.ntotal == 0:
        return []
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = faiss_index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0] if 0 <= i < len(chunks)]

def hybrid_retrieval(query, model, faiss_index, chunks):
    return list(set(bm25_retrieval(query, chunks) + faiss_retrieval(query, model, faiss_index, chunks)))

# Step 8: Generate Response
def extract_answer(text):
    """Extract the answer from the generated text."""
    match = re.search(r'Provide a concise and informative answer: (.*?)\n', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def generate_response(query, context, model, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    formatted_context = "\n\n".join(context)
    prompt = (f"You are a financial assistant. Given the following context, answer the user's question accurately.\n\n"
              f"Context:\n{formatted_context}\n\n"
              f"Question: {query}\n\n"
              f"Provide a concise and informative answer:")
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048*3).to(device)
    
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=0.1, output_scores=True, return_dict_in_generate=True)
    
    # Decode the response
    res = tokenizer.decode(output.sequences[0], skip_special_tokens=True).strip()
    
    # Extract answer
    answer = extract_answer(res)
    
    # Calculate confidence score based on logits
    logits = output.scores[0]  # The logits for the generated sequence
    softmax_scores = torch.nn.functional.softmax(logits, dim=-1)  # Convert logits to probabilities
    confidence = softmax_scores.max().item()  # Maximum probability as the confidence score
    
    return answer, confidence

# Step 9: Retrieve and Generate Response
def retrieve_and_generate_response(query):
    """Retrieve relevant chunks and generate a response."""
    global faiss_index, text_chunks, model, slm_model, tokenizer  # Access global variables
    if faiss_index is None or text_chunks is None:
        return "Missing FAISS index or chunk data! Ensure data is available.", 0
    retrieved_chunks = hybrid_retrieval(query, model, faiss_index, text_chunks)
    if retrieved_chunks:
        answer, confidence = generate_response(query, retrieved_chunks, slm_model, tokenizer)
        return answer, confidence
    return "No relevant results found.", 0

# Step 10: Guardrails
def filter_query(query):
    """Filter inappropriate or off-topic queries."""
    prohibited_keywords = ['violence', 'harm', 'racism', 'fraud', 'scam']
    for word in prohibited_keywords:
        if re.search(rf"\b{word}\b", query, re.IGNORECASE):
            return "Your query contains inappropriate content and has been rejected.", True
    
    if not any(keyword in query.lower() for keyword in ['financial', 'stock', 'revenue', 'earnings', 'profit']):
        return "Please provide a query related to finance.", True
    
    return query, False

def filter_answer(answer):
    """Filter inappropriate or off-topic answers."""
    forbidden_terms = ['capital', 'science', 'sports', 'france']
    for term in forbidden_terms:
        if term in answer.lower():
            return "The answer seems off-topic. Please refine your query.", True
    
    if len(answer.split()) < 5:  # If the answer is too vague
        return "The answer seems too vague. Please refine your query for a more detailed response.", True
    
    return answer, False

# Step 11: Main Chatbot Function
def generate_reponse_bot(query):
    """Generate a response to the user's query."""
    res, is_filtered = filter_query(query)
    if is_filtered:  # Query is filtered
        return res, 1
    answer, confidence = retrieve_and_generate_response(query)
    res, is_filtered = filter_answer(answer)
    if is_filtered:  # Response is filtered
        return res, 1
    else:
        return answer, confidence

# Step 12: Load Preprocessed Data
def load_preprocessed_data():
    """Load preprocessed data (FAISS index, text chunks, models, etc.)."""
    global faiss_index, text_chunks, model, slm_model, tokenizer

    # Load FAISS index
    index_path = "financial_index.faiss"
    if os.path.exists(index_path):
        faiss_index = load_faiss_index(index_path)
    else:
        raise FileNotFoundError(f"FAISS index not found at {index_path}")

    # Load text chunks
    chunk_path = "text_chunks.pkl"
    if os.path.exists(chunk_path):
        text_chunks = load_text_chunks_from_pickle(chunk_path)
    else:
        raise FileNotFoundError(f"Text chunks not found at {chunk_path}")

    # Load models
    model = SentenceTransformer("all-MiniLM-L6-v2")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    slm_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-1_5", torch_dtype=torch.float16
    ).to(device)

# Step 13: Main Function for Preprocessing
def main():
    """Run preprocessing steps."""
    # Create the folder if it doesn't exist
    folder_name = "financial_reports"
    os.makedirs(folder_name, exist_ok=True)

    # Download reports
    reports = {
        "2023": "https://investors.broadcom.com/static-files/2b98b262-4fbb-4731-b3dd-88f6ca187002",
        "2024": "https://investors.broadcom.com/static-files/2e0788d2-4c75-4ed9-bde2-a96a7abb8996"
    }
    for year, url in reports.items():
        download_report(url, f"broadcom_{year}.pdf")

    # Extract and clean text
    for file in os.listdir("financial_reports"):
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

    # Load models and data
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    slm_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-1_5", torch_dtype=torch.float16
    ).to(device)

# Run preprocessing if this script is executed directly
if __name__ == "__main__":
    main()