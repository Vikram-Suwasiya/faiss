from fastapi import FastAPI
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

with open("faiss_indexes.pkl", "rb") as f:
    faiss_indexes, metadata = pickle.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")

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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Default to 10000 if PORT is not set
    uvicorn.run(app, host="0.0.0.0", port=port)
