import re

def clean_text(text: str) -> str:
    """
    Cleans raw extracted text from PDFs by normalizing whitespace.
    """
    if not text:
        return ""

    # Replace multiple whitespace/newlines with single space
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', ' ', text)

    return text.strip()


def chunk_text(text: str, chunk_size: int = 5):
    """
    Splits text into chunks of sentences.
    No external NLP dependencies (NLTK-free).
    """
    if not text:
        return []

    # Sentence splitting using regex (., !, ?)
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    for i in range(0, len(sentences), chunk_size):
        chunk = " ".join(sentences[i:i + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)

    return chunks
