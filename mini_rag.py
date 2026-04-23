"""
mini_rag.py

A tiny, beginner-friendly Retrieval-Augmented Generation (RAG) example.

This script shows four clear stages:
1) document setup
2) embeddings
3) retrieval
4) generation

It uses:
- OpenAI Embeddings API for vector creation
- FAISS for nearest-neighbor retrieval
- OpenAI Chat Completions API for final answer generation
"""

import os
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from openai import OpenAI
from pypdf import PdfReader

# Your requested Windows folder location.
# You can also override this with environment variable RAG_DOCS_DIR.
DEFAULT_DOCS_DIR = r"C:\Users\mad177\OneDrive - CSIRO\Documents\Important Papers"
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}
MAX_CHARS_PER_DOC = 3000
MAX_DOCS = 30


def print_section(title: str) -> None:
    """Prints a readable section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def build_sample_documents() -> List[str]:
    """
    Returns a small hardcoded set of sample documents.
    We use these only as fallback if the target folder is unavailable.
    """
    return [
        "Python is a high-level programming language known for readability.",
        "FAISS is a library for efficient similarity search on dense vectors.",
        "Embeddings convert text into numeric vectors that capture semantic meaning.",
        "RAG combines retrieval with generation so answers can use external context.",
        "The OpenAI API can generate both embeddings and chat completions.",
    ]


def read_text_file(path: Path) -> str:
    """Reads plain text files such as .txt and .md."""
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf_file(path: Path) -> str:
    """Extracts text from a PDF file using pypdf."""
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def load_documents_from_folder(folder_path: str) -> List[str]:
    """
    Loads documents from the given folder.
    Supports .txt, .md, and .pdf files.
    """
    root = Path(folder_path)
    if not root.exists() or not root.is_dir():
        print(f"Folder not found: {folder_path}")
        return []

    files: List[Path] = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(root.glob(f"*{ext}"))
    files = sorted(files)[:MAX_DOCS]

    documents: List[str] = []
    for file_path in files:
        try:
            if file_path.suffix.lower() == ".pdf":
                content = read_pdf_file(file_path)
            else:
                content = read_text_file(file_path)

            # Keep each document short for a lightweight beginner demo.
            content = content.strip()[:MAX_CHARS_PER_DOC]
            if content:
                documents.append(f"Source: {file_path.name}\n{content}")
        except Exception as exc:
            print(f"Skipping {file_path.name} (read error: {exc})")

    return documents


def get_openai_client() -> OpenAI:
    """
    Reads OPENAI_API_KEY from environment and creates an OpenAI client.
    We do NOT hardcode keys in source code.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing OPENAI_API_KEY. Set it in your environment first.\n"
            "Linux/macOS example: export OPENAI_API_KEY='your_api_key_here'\n"
            "Windows PowerShell example: $env:OPENAI_API_KEY='your_api_key_here'"
        )

    return OpenAI(api_key=api_key)


def embed_texts(client: OpenAI, texts: List[str], model: str) -> np.ndarray:
    """
    Creates embeddings for multiple text strings using OpenAI.
    Returns a 2D numpy array with shape: [num_texts, embedding_dim].
    """
    response = client.embeddings.create(model=model, input=texts)
    vectors = [item.embedding for item in response.data]
    return np.array(vectors, dtype="float32")


def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    """
    Builds a FAISS index for cosine similarity search.
    We normalize vectors then use inner product (dot product).
    """
    faiss.normalize_L2(vectors)
    embedding_dim = vectors.shape[1]
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(vectors)
    return index


def retrieve_top_k(
    index: faiss.Index,
    documents: List[str],
    query_vector: np.ndarray,
    top_k: int = 2,
) -> List[Tuple[int, float, str]]:
    """
    Retrieves top-k documents using the FAISS index.
    Returns (doc_index, score, doc_text) tuples.
    """
    query = np.array(query_vector, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(query)

    safe_top_k = min(top_k, len(documents))
    scores, indices = index.search(query, safe_top_k)
    results: List[Tuple[int, float, str]] = []
    for idx, score in zip(indices[0], scores[0]):
        results.append((int(idx), float(score), documents[int(idx)]))
    return results


def generate_answer(
    client: OpenAI,
    question: str,
    retrieved_docs: List[Tuple[int, float, str]],
    model: str,
) -> str:
    """
    Generates a final answer with chat completion, grounded in retrieved context.
    """
    context_blocks = []
    for doc_index, score, doc_text in retrieved_docs:
        context_blocks.append(
            f"[Document {doc_index} | similarity={score:.4f}]\n{doc_text}"
        )
    context = "\n\n".join(context_blocks)

    system_prompt = (
        "You are a helpful assistant. Use only the provided context to answer. "
        "If the context is insufficient, say what is missing."
    )
    user_prompt = (
        f"Question: {question}\n\n"
        f"Retrieved context:\n{context}\n\n"
        "Answer clearly for a beginner."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content or ""


def main() -> None:
    embedding_model = "text-embedding-3-small"
    chat_model = "gpt-4o-mini"

    # -------------------------------------------------
    # STEP 1: Document setup
    # -------------------------------------------------
    print_section("STEP 1: DOCUMENT SETUP")
    docs_dir = os.getenv("RAG_DOCS_DIR", DEFAULT_DOCS_DIR)
    print(f"Trying to load documents from: {docs_dir}")
    documents = load_documents_from_folder(docs_dir)

    if not documents:
        print("No readable files found in folder. Using fallback sample documents.")
        documents = build_sample_documents()

    print(f"Total documents loaded: {len(documents)}")
    for i, doc in enumerate(documents[:5]):
        preview = doc.replace("\n", " ")[:140]
        print(f"Document {i}: {preview}...")
    if len(documents) > 5:
        print("... (showing first 5 documents only)")

    question = "Summarize the most important ideas from these papers."
    print(f"\nQuestion: {question}")

    client = get_openai_client()

    # -------------------------------------------------
    # STEP 2: Embeddings
    # -------------------------------------------------
    print_section("STEP 2: EMBEDDINGS")
    print(f"Creating embeddings with model: {embedding_model}")
    doc_vectors = embed_texts(client, documents, embedding_model)
    print(f"Document embeddings shape: {doc_vectors.shape}")

    query_vector = embed_texts(client, [question], embedding_model)[0]
    print(f"Query embedding length: {query_vector.shape[0]}")

    # -------------------------------------------------
    # STEP 3: Retrieval
    # -------------------------------------------------
    print_section("STEP 3: RETRIEVAL")
    index = build_faiss_index(doc_vectors)
    top_matches = retrieve_top_k(index, documents, query_vector, top_k=3)
    print("Top retrieved documents:")
    for rank, (doc_index, score, doc_text) in enumerate(top_matches, start=1):
        preview = doc_text.replace("\n", " ")[:180]
        print(f"{rank}. doc_id={doc_index}, similarity={score:.4f}")
        print(f"   {preview}...")

    # -------------------------------------------------
    # STEP 4: Generation
    # -------------------------------------------------
    print_section("STEP 4: GENERATION")
    print(f"Generating final answer with model: {chat_model}")
    answer = generate_answer(client, question, top_matches, chat_model)
    print("\nFinal Answer:")
    print(answer)


if __name__ == "__main__":
    main()
