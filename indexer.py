import os
import pickle
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from db import fetch_all_chunks

INDEX_DIR = "data/index"


def build_indexes(chunks=None):
    os.makedirs(INDEX_DIR, exist_ok=True)

    if chunks is None:
        chunks = fetch_all_chunks()

    texts = [c["text"] for c in chunks]

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(texts)

    with open(f"{INDEX_DIR}/tfidf.pkl", "wb") as f:
        pickle.dump(tfidf, f)

    with open(f"{INDEX_DIR}/tfidf_matrix.pkl", "wb") as f:
        pickle.dump(tfidf_matrix, f)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, convert_to_numpy=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, f"{INDEX_DIR}/faiss.index")

    with open(f"{INDEX_DIR}/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
