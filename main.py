from fastapi import FastAPI, HTTPException
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import os

# Load FAISS indexes and metadata
faiss_file = "faiss_indexes.pkl"

if not os.path.exists(faiss_file):
    raise RuntimeError(f"File {faiss_file} not found. Ensure it's uploaded to the correct path.")

try:
    with open(faiss_file, "rb") as f:
        faiss_indexes, metadata = pickle.load(f)
    if not isinstance(faiss_indexes, dict) or not isinstance(metadata, dict):
        raise RuntimeError("Invalid FAISS index format. Expected dictionary.")
except Exception as e:
    raise RuntimeError(f"Error loading {faiss_file}: {e}")

# Load the Sentence Transformer model
try:
    model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    raise RuntimeError(f"Error loading SentenceTransformer model: {e}")

app = FastAPI()

@app.post("/search/")
def search(query: dict):
    """Searches FAISS index using the provided query."""
    if not query:
        raise HTTPException(status_code=400, detail="Query dictionary cannot be empty.")

    print("Received search request:", query)
    results = []

    print("Available FAISS namespaces:", list(faiss_indexes.keys()))

    for namespace, index in faiss_indexes.items():
        query_text = query.get(namespace, "").strip()
        
        if not query_text:
            print(f"Skipping namespace '{namespace}' (No query provided)")
            continue  

        if not isinstance(index, faiss.Index):
            print(f"Skipping namespace '{namespace}' (Invalid FAISS index)")
            continue

        # Check if index has any vectors stored
        if index.ntotal == 0:
            print(f"Skipping namespace '{namespace}' (Index is empty)")
            continue

        # Convert text to vector
        try:
            query_vector = np.array(model.encode(query_text), dtype=np.float32).reshape(1, -1)
            print(f"Query vector shape for '{namespace}':", query_vector.shape)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error encoding query: {e}")

        # Perform FAISS search
        k = min(5, index.ntotal)  # Retrieve up to 5 matches
        distances, indices = index.search(query_vector, k=k)

        print(f"Search results for '{namespace}': indices={indices}, distances={distances}")

        for i in range(len(indices[0])):
            match_index = indices[0][i]
            if match_index < 0:
                continue  # Ignore invalid matches

            match_score = float(1 / (1 + distances[0][i]))  # Convert distance to similarity
            match_data = metadata.get(namespace, {}).get(match_index, {})

            if not match_data:
                print(f"Warning: No metadata found for index {match_index} in namespace '{namespace}'")
                continue

            result = {
                "namespace": namespace,
                "score": round(match_score, 2),
                "data": match_data
            }
            print("Match found:", result)
            results.append(result)

    print("Final response:", results)
    return results
