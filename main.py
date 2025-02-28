from fastapi import FastAPI, HTTPException
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import os

# Ensure the FAISS index file exists
faiss_file = "faiss_indexes.pkl"
if not os.path.exists(faiss_file):
    raise RuntimeError(f"File {faiss_file} not found. Ensure it is uploaded to Render.")

# Load FAISS indexes and metadata
try:
    with open(faiss_file, "rb") as f:
        faiss_indexes, metadata = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Error loading FAISS index: {e}")

# Load Sentence Transformer Model
try:
    model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

# Initialize FastAPI app
app = FastAPI()

@app.post("/search/")
def search(query: dict):
    print("Received search request:", query)  
    
    results = []
    
    for namespace, index in faiss_indexes.items():
        query_text = query.get(namespace, "")
        if not query_text:
            print(f"Skipping namespace {namespace} (No query provided)")
            continue  

        query_vector = np.array(model.encode(query_text)).reshape(1, -1).astype("float32")
        distances, indices = index.search(query_vector, k=1)  

        match_index = indices[0][0]
        if match_index >= 0:
            match_score = float(1 / (1 + distances[0][0]))  
            match_data = metadata[namespace][match_index]
            
            
            result = {
                "namespace": namespace,
                "score": round(match_score, 2),
                "data": match_data
            }
            print("Match found:", result)  
            results.append(result)
    
    print("Returning response:", results)  
    return results
