# retrieval_and_generation.py

import numpy as np
import faiss
import torch
from rank_bm25 import BM25Okapi
import re
import pickle
import os

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

def load_faiss_index(index_path):
    """Load FAISS index from file."""
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found at {index_path}")
    return faiss.read_index(index_path)

def load_text_chunks_from_pickle(chunk_path):
    """Load text chunks from a pickle file."""
    if not os.path.exists(chunk_path):
        raise FileNotFoundError(f"Chunk file not found at {chunk_path}")
    with open(chunk_path, "rb") as f:
        return pickle.load(f)

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