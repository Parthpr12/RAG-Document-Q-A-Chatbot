"""
rag_chatbot.py
================
A Retrieval-Augmented Generation (RAG) chatbot for document Q&A.

Built from scratch (no LangChain/LlamaIndex) to demonstrate a clear
understanding of the full RAG pipeline: chunking -> embedding -> retrieval
-> augmentation -> generation.

Author: <your name>
"""

import os
import re
import sys
import argparse
import numpy as np


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
CHUNK_SIZE = 500          # characters per chunk
CHUNK_OVERLAP = 100       # overlap between consecutive chunks
TOP_K = 3                 # number of chunks retrieved per query
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GENERATION_MODEL = "gemini-3.6-flash"

SAMPLE_TEXT = """
Retrieval-Augmented Generation (RAG) is a technique that combines information
retrieval with text generation. Instead of relying solely on knowledge baked
into a language model's parameters during training, RAG retrieves relevant
documents from an external knowledge base at query time and uses them as
context for generation. This reduces hallucination and allows the model to
answer questions about information it was never trained on, such as private
company documents or content published after the model's training cutoff.

A typical RAG pipeline has five stages. First, documents are split into
smaller chunks, because embedding models have limited input length and
smaller chunks improve retrieval precision. Second, each chunk is converted
into a dense vector using an embedding model. Third, these vectors are stored
in a vector index that supports fast similarity search. Fourth, when a user
asks a question, the question itself is embedded and compared against the
stored vectors using cosine similarity, and the top-k most similar chunks are
retrieved. Fifth, the retrieved chunks are inserted into a prompt along with
the original question, and this augmented prompt is sent to a large language
model to generate the final answer.
"""


# ---------------------------------------------------------------------------
# STEP 1: DOCUMENT LOADING
# ---------------------------------------------------------------------------
def load_document(pdf_path: str | None) -> str:
    """Load text from a PDF file, or fall back to a built-in sample."""
    if pdf_path and os.path.exists(pdf_path):
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        text = "".join(page.extract_text() or "" for page in reader.pages)
        print(f"Loaded PDF: {pdf_path} ({len(reader.pages)} pages)")
        return text
    if pdf_path:
        print(f"File not found: '{pdf_path}'. Using built-in sample text instead.")
    else:
        print("No PDF provided. Using built-in sample text about RAG.")
    return SAMPLE_TEXT


# ---------------------------------------------------------------------------
# STEP 2: CHUNKING (with overlap so boundary information isn't lost)
# ---------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return [c for c in chunks if len(c.strip()) > 20]


# ---------------------------------------------------------------------------
# STEP 3: EMBEDDINGS (local, free, runs offline)
# ---------------------------------------------------------------------------
class Embedder:
    """Wraps a local sentence-transformers model for generating embeddings."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.array(self.model.encode(texts))


# ---------------------------------------------------------------------------
# STEP 4: VECTOR INDEX (cosine similarity search)
# ---------------------------------------------------------------------------
class VectorIndex:
    """Simple in-memory vector index using cosine similarity.

    For large-scale use, swap this out for FAISS or Chroma.
    """

    def __init__(self, vectors: np.ndarray):
        self.vectors = vectors

    def search(self, query_vector: np.ndarray, k: int = TOP_K):
        norms = np.linalg.norm(self.vectors, axis=1) * (np.linalg.norm(query_vector) + 1e-10)
        similarities = (self.vectors @ query_vector) / (norms + 1e-10)
        top_idx = np.argsort(similarities)[::-1][:k]
        return top_idx, similarities[top_idx]


# ---------------------------------------------------------------------------
# STEP 5: PROMPT AUGMENTATION + GENERATION
# ---------------------------------------------------------------------------
def build_prompt(query: str, retrieved_chunks: list[str]) -> str:
    context = "\n\n".join(f"[Chunk {i+1}]: {c}" for i, c in enumerate(retrieved_chunks))
    return f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not contained in the context, say "I don't have enough information in the document to answer that."

Context:
{context}

Question: {query}

Answer:"""


def generate_answer(prompt: str, api_key: str) -> str:
    """Calls the Gemini API for the final generation step."""
    if not api_key:
        return "[No GEMINI_API_KEY set. Set it as an environment variable to get real answers.]"
    from google import genai
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
    )
    return response.text


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def run_pipeline(pdf_path: str | None, query: str, api_key: str):
    raw_text = load_document(pdf_path)
    chunks = chunk_text(raw_text)
    print(f"Document split into {len(chunks)} chunks "
          f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    embedder = Embedder()
    chunk_vectors = embedder.encode(chunks)
    index = VectorIndex(chunk_vectors)

    query_vector = embedder.encode([query])[0]
    top_idx, scores = index.search(query_vector)
    retrieved_chunks = [chunks[i] for i in top_idx]
    print(f"Retrieved top-{TOP_K} chunks, similarity scores: "
          f"{[round(float(s), 3) for s in scores]}")

    prompt = build_prompt(query, retrieved_chunks)
    answer = generate_answer(prompt, api_key)
    return answer


def main():
    parser = argparse.ArgumentParser(description="RAG-based document Q&A chatbot")
    parser.add_argument("--pdf", type=str, default=None, help="Path to a PDF file")
    parser.add_argument("--query", type=str, default=None, help="A single question to ask (non-interactive mode)")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY", "")

    if args.query:
        # Non-interactive single-question mode
        answer = run_pipeline(args.pdf, args.query, api_key)
        print(f"\nQ: {args.query}\nA: {answer}")
        return

    # Interactive mode: build the index once, then answer multiple questions
    raw_text = load_document(args.pdf)
    chunks = chunk_text(raw_text)
    embedder = Embedder()
    chunk_vectors = embedder.encode(chunks)
    index = VectorIndex(chunk_vectors)

    print("\nRAG chatbot ready. Ask questions about the document. Type 'exit' to quit.\n")
    while True:
        query = input("Your question: ").strip()
        if query.lower() in ("exit", "quit"):
            break
        query_vector = embedder.encode([query])[0]
        top_idx, scores = index.search(query_vector)
        retrieved_chunks = [chunks[i] for i in top_idx]
        prompt = build_prompt(query, retrieved_chunks)
        answer = generate_answer(prompt, api_key)
        print(f"A: {answer}\n")


if __name__ == "__main__":
    main()
