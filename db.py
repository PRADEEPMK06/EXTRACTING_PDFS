import sqlite3
from datetime import datetime

DB_PATH = "data/knowledge.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        uploaded_at TEXT NOT NULL
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id INTEGER,
        page INTEGER,
        text TEXT,
        FOREIGN KEY(document_id) REFERENCES documents(id)
    )
    """)

    conn.commit()
    conn.close()


def insert_document(filename: str) -> int:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO documents (filename, uploaded_at) VALUES (?, ?)",
        (filename, datetime.utcnow().isoformat())
    )

    doc_id = cur.lastrowid
    conn.commit()
    conn.close()

    return doc_id


def insert_chunks(document_id: int, chunks: list):
    conn = get_connection()
    cur = conn.cursor()

    cur.executemany(
        "INSERT INTO chunks (document_id, page, text) VALUES (?, ?, ?)",
        [(document_id, c["page"], c["text"]) for c in chunks]
    )

    conn.commit()
    conn.close()


def fetch_all_chunks():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    SELECT 
        chunks.id,
        chunks.text,
        documents.filename,
        chunks.page
    FROM chunks
    JOIN documents ON chunks.document_id = documents.id
    """)

    rows = cur.fetchall()
    conn.close()

    return [
        {
            "id": r[0],
            "text": r[1],
            "document": r[2],
            "page": r[3]
        }
        for r in rows
    ]
