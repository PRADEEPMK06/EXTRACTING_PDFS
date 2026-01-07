import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from search_engine import search

MODEL_NAME = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.eval()


def answer_from_chunk(question: str, chunk_text: str) -> str:
    prompt = (
        "Answer the question using ONLY the information below.\n"
        "If the answer is not present, say: Not mentioned.\n\n"
        f"Information:\n{chunk_text}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            temperature=0.0
        )

    return tokenizer.decode(output[0], skip_special_tokens=True).strip()


def synthesize_answers(answers: list) -> str:
    combined_text = " ".join(answers)

    prompt = (
        "Rewrite the following text into ONE clear, well-structured paragraph.\n"
        "Do not add new information.\n"
        "Remove repetition.\n\n"
        f"Text:\n{combined_text}\n\n"
        "Paragraph answer:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=180,
            do_sample=False,
            temperature=0.0
        )

    return tokenizer.decode(output[0], skip_special_tokens=True).strip()


def generate_answer(query: str, top_k: int = 4):
    chunks = search(query, top_k=top_k)

    if not chunks:
        return "No relevant information found in documents.", []

    partial_answers = []
    citations = []

    for chunk in chunks:
        ans = answer_from_chunk(query, chunk["text"])
        if ans and ans.lower() != "not mentioned.":
            partial_answers.append(ans)
            citations.append({
                "document": chunk["document"],
                "page": chunk["page"]
            })

    if not partial_answers:
        return "Information not found in documents.", citations

    final_answer = synthesize_answers(partial_answers)
    return final_answer, citations
