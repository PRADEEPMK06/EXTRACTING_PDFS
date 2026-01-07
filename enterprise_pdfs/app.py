import streamlit as st
import os

from pdf_loader import extract_text_from_pdf
from text_processor import clean_text, chunk_text
from indexer import build_indexes
from db import init_db, insert_document, insert_chunks
from rag_engine import generate_answer


# -------------------------------------------------
# Streamlit Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Enterprise PDF Knowledge System",
    layout="wide"
)

st.title("Enterprise PDF Knowledge Conversion & RAG System")

# -------------------------------------------------
# Ensure Required Directories
# -------------------------------------------------
os.makedirs("data", exist_ok=True)
os.makedirs("data/pdfs", exist_ok=True)
os.makedirs("data/index", exist_ok=True)

# -------------------------------------------------
# Initialize Database (SQLite – auto setup)
# -------------------------------------------------
init_db()

# -------------------------------------------------
# PDF Upload Section
# -------------------------------------------------
st.header("Upload Enterprise PDFs")

uploaded_files = st.file_uploader(
    "Upload one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

# -------------------------------------------------
# Process PDFs Button
# -------------------------------------------------
if st.button("Process & Index PDFs"):
    if not uploaded_files:
        st.warning("Please upload at least one PDF.")
    else:
        all_chunks = []

        with st.spinner("Processing PDFs and building indexes..."):
            for file in uploaded_files:
                pdf_path = os.path.join("data/pdfs", file.name)

                # Save PDF
                with open(pdf_path, "wb") as f:
                    f.write(file.read())

                # Insert document record into SQL
                document_id = insert_document(file.name)

                # Extract text (OCR fallback)
                pages = extract_text_from_pdf(pdf_path)

                document_chunks = []

                for p in pages:
                    cleaned_text = clean_text(p["text"])
                    chunks = chunk_text(cleaned_text)

                    for c in chunks:
                        document_chunks.append({
                            "text": c,
                            "page": p["page"]
                        })

                # Store chunks in SQL
                insert_chunks(document_id, document_chunks)

                # Prepare chunks for indexing
                for c in document_chunks:
                    all_chunks.append({
                        "text": c["text"],
                        "document": file.name,
                        "page": c["page"]
                    })

            # Build FAISS + TF-IDF indexes
            build_indexes(all_chunks)

        st.success("PDFs processed, stored in SQL, and indexed successfully.")

# -------------------------------------------------
# RAG Question Answering Section
# -------------------------------------------------
st.header("🔍 Ask Questions (RAG Enabled)")

query = st.text_input(
    "Ask a question based on the uploaded PDFs"
)

if query:
    with st.spinner("Retrieving knowledge and generating answer..."):
        answer, citations = generate_answer(query)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources")
    if citations:
        for i, c in enumerate(citations, start=1):
            st.markdown(
                f"[{i}] **Document:** {c['document']} | **Page:** {c['page']}"
            )
    else:
        st.write("No sources available.")
