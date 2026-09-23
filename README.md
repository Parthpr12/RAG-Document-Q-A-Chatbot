# RAG Document Q&A Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about
any PDF document, grounded strictly in that document's content. Built from
scratch — no LangChain/LlamaIndex — to demonstrate a clear, first-principles
understanding of the full RAG pipeline.

## How it works

```
PDF ──▶ Chunking (with overlap) ──▶ Local Embeddings ──▶ Vector Index
                                                              │
User Question ──▶ Embed Query ──▶ Cosine Similarity Search ◀─┘
                                          │
                                          ▼
                              Top-k Relevant Chunks
                                          │
                                          ▼
                        Prompt = Question + Retrieved Context
                                          │
                                          ▼
                              Gemini API (generation)
                                          │
                                          ▼
                                  Grounded Answer
```

1. **Chunking** — the document is split into overlapping chunks so no
   information is lost at chunk boundaries.
2. **Embedding** — each chunk is converted to a vector using a local
   `sentence-transformers` model (`all-MiniLM-L6-v2`). Embeddings run
   entirely offline — free, fast, no API rate limits.
3. **Retrieval** — at query time, the question is embedded and compared
   against all chunk vectors using cosine similarity; the top-k most
   relevant chunks are retrieved.
4. **Augmentation** — the retrieved chunks are inserted into a prompt as
   context, along with an instruction to answer only from that context.
5. **Generation** — the augmented prompt is sent to the Gemini API
   (`gemini-3.6-flash`) to produce the final answer.

Keeping retrieval fully local and only calling the LLM API for the final
generation step keeps the pipeline cheap and fast, even on large documents.

## Setup

```bash
git clone https://github.com/<your-username>/rag-chatbot.git
cd rag-chatbot
pip install -r requirements.txt
```

Get a free Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)
and set it as an environment variable:

```bash
export GEMINI_API_KEY="your_key_here"      # Mac/Linux
setx GEMINI_API_KEY "your_key_here"        # Windows
```

## Usage

**Interactive mode:**
```bash
python rag_chatbot.py --pdf your_document.pdf
```

**Single question (non-interactive):**
```bash
python rag_chatbot.py --pdf your_document.pdf --query "What are the key skills listed?"
```

**No PDF?** The script falls back to a built-in sample text about RAG itself,
so you can test the pipeline immediately:
```bash
python rag_chatbot.py --query "What is RAG?"
```

## Tech stack

| Component   | Tool                                  |
|-------------|----------------------------------------|
| Embeddings  | `sentence-transformers` (local, free) |
| Similarity  | Cosine similarity (numpy)             |
| Generation  | Google Gemini API (`gemini-3.6-flash`)|
| PDF parsing | `pypdf`                               |

## Design notes

- **No LangChain** — the pipeline is implemented manually so every step
  (chunking, embedding, retrieval, prompt construction) is transparent and
  easy to reason about or extend.
- **Local embeddings over API embeddings** — avoids per-chunk API calls,
  rate limits, and cost when indexing large documents. Only the final
  generation call hits the API.
- **Chunk overlap** — prevents important information from being lost when
  it falls near a chunk boundary.
- **Grounded answers only** — the prompt explicitly instructs the model to
  say "I don't have enough information" rather than hallucinate an answer
  not present in the retrieved context.

## Possible extensions

- Swap the in-memory cosine similarity index for FAISS or Chroma for
  large-scale document collections.
- Add re-ranking (cross-encoder) for higher retrieval precision.
- Support multiple documents with source citations.
- Add automated evaluation (e.g. with RAGAS) for answer faithfulness.

## License

MIT
