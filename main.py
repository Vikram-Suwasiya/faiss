from fastapi import FastAPI, HTTPException
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import os

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

print("Loaded FAISS Indexes:")
for namespace, index in faiss_indexes.items():
    print(f"Namespace: {namespace}, Total Vectors: {index.ntotal}")

print("Loaded Metadata:")
for namespace, entries in metadata.items():
    print(f"Namespace: {namespace}, Total Metadata Entries: {len(entries)}")

try:
    model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    raise RuntimeError(f"Error loading SentenceTransformer model: {e}")

app = FastAPI()

@app.post("/search/")
def search(query: dict):
    if not query:
        raise HTTPException(status_code=400, detail="Query dictionary cannot be empty.")

    print("Received search request:", query)
    print("Expected namespace keys:", list(faiss_indexes.keys()))

    results = []

    for namespace, index in faiss_indexes.items():
        query_text = query.get(namespace, "").strip()

        if not query_text:
            print(f"Skipping namespace '{namespace}' (No query provided)")
            continue

        if not isinstance(index, faiss.Index):
            print(f"Skipping namespace '{namespace}' (Invalid FAISS index)")
            continue

        if index.ntotal == 0:
            print(f"Skipping namespace '{namespace}' (Index is empty)")
            continue

        try:
            query_vector = model.encode(query_text)

            if query_vector is None:
                raise RuntimeError(f"Encoding failed for query: {query_text}")

            query_vector = np.array(query_vector, dtype=np.float32).reshape(1, -1)
            print(f"Query vector shape for '{namespace}':", query_vector.shape)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error encoding query: {e}")

        k = min(5, index.ntotal)
        distances, indices = index.search(query_vector, k=k)

        print(f"Search results for '{namespace}': indices={indices}, distances={distances}")

        for i in range(len(indices[0])):
            match_index = indices[0][i]
            if match_index < 0:
                continue

            match_score = float(1 / (1 + distances[0][i])) 
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
