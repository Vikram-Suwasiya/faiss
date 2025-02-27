from fastapi import FastAPI, HTTPException
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import os

# Ensure file exists (Render may not persist uploads)
faiss_file = "faiss_indexes.pkl"
if not os.path.exists(faiss_file):
    raise RuntimeError(f"File {faiss_file} not found. Ensure it is uploaded to Render.")

# Load FAISS Indexes and Metadata
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

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    results = []

    for namespace, index in faiss_indexes.items():
        query_text = query.get(namespace, "").strip()
        if not query_text:
            print(f"Skipping namespace '{namespace}' (No query provided)")
            continue  

        query_vector = np.array(model.encode(query_text)).reshape(1, -1).astype("float32")
        
        # Correct usage of faiss.IndexFlat.search
        k = 1  # Number of nearest neighbors
        distances = np.zeros((1, k), dtype=np.float32)
        indices = np.zeros((1, k), dtype=np.int64)
        index.search(query_vector, k, distances, indices)

        match_index = indices[0][0]
        if match_index >= 0:
            match_score = float(1 / (1 + distances[0][0]))  
            match_data = metadata.get(namespace, {}).get(match_index, {})

            if not match_data:
                print(f"Warning: Metadata missing for index {match_index} in '{namespace}'")
                continue

            result = {
                "data": match_data
            }
            print("Match found:", result)
            results.append(result)

    print("Returning response:", results)
    return results
