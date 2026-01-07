import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_DIR = "data/index"

def search(query, top_k=5):
    with open(f"{INDEX_DIR}/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    # Load TF-IDF
    with open(f"{INDEX_DIR}/tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)

    with open(f"{INDEX_DIR}/tfidf_matrix.pkl", "rb") as f:
        tfidf_matrix = pickle.load(f)

    query_vec = tfidf.transform([query])
    scores = (tfidf_matrix @ query_vec.T).toarray().ravel()
    top_tfidf = scores.argsort()[-top_k:][::-1]

    # Semantic Search
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_emb = model.encode([query])

    index = faiss.read_index(f"{INDEX_DIR}/faiss.index")
    _, top_semantic = index.search(np.array(query_emb), top_k)

    results = set(top_tfidf.tolist() + top_semantic[0].tolist())

    return [chunks[i] for i in results]
