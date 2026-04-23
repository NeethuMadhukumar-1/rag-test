"""
mini_rag_1.py

An updated beginner-friendly Retrieval-Augmented Generation (RAG) example.

Pipeline stages:
1) document setup
2) embeddings
3) retrieval
4) generation (RAG answer)
5) web-search fallback (only when RAG looks insufficient)

It uses:
- OpenAI Embeddings API for vector creation
- FAISS for nearest-neighbor retrieval
- OpenAI Chat Completions API for RAG-grounded answers
- OpenAI Responses API + web_search_preview tool for fallback answers
"""

import os
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

# Your requested Windows folder location.
# You can also override this with environment variable RAG_DOCS_DIR.
DEFAULT_DOCS_DIR = r"C:\Users\mad177\OneDrive - CSIRO\Documents\Important Papers"
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}
MAX_CHARS_PER_FILE = 12000
MAX_DOCS = 30
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

# If the best retrieved score is below this threshold,
# we treat local RAG context as weak and try web-search fallback.
RAG_SCORE_THRESHOLD = 0.35


def load_environment_variables() -> None:
    """
    Loads environment variables from a local .env file (if present).
    This lets beginners keep secrets out of source code.
    """
    load_dotenv()


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


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Splits long text into small overlapping chunks.
    Overlap helps preserve context between neighboring chunks.
    """
    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    chunks: List[str] = []
    step = max(1, chunk_size - overlap)
    start = 0
    while start < len(cleaned):
        end = start + chunk_size
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def load_documents_from_folder(folder_path: str) -> Tuple[List[str], int]:
    """
    Loads documents from the given folder.
    Supports .txt, .md, and .pdf files, then chunks each long document.
    """
    root = Path(folder_path)
    if not root.exists() or not root.is_dir():
        print(f"Folder not found: {folder_path}")
        return [], 0

    files: List[Path] = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(root.glob(f"*{ext}"))
    files = sorted(files)[:MAX_DOCS]

    documents: List[str] = []
    loaded_file_count = 0
    for file_path in files:
        try:
            if file_path.suffix.lower() == ".pdf":
                content = read_pdf_file(file_path)
            else:
                content = read_text_file(file_path)

            # Cap file size to keep this example lightweight.
            content = content.strip()[:MAX_CHARS_PER_FILE]
            if content:
                loaded_file_count += 1
                chunks = chunk_text(content)
                for chunk_idx, chunk in enumerate(chunks, start=1):
                    documents.append(
                        f"Source: {file_path.name} | Chunk {chunk_idx}/{len(chunks)}\n{chunk}"
                    )
        except Exception as exc:
            print(f"Skipping {file_path.name} (read error: {exc})")

    return documents, loaded_file_count


def get_openai_client() -> OpenAI:
    """
    Reads OPENAI_API_KEY from environment (including .env) and creates an OpenAI client.
    We do NOT hardcode keys in source code.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing OPENAI_API_KEY.\n"
            "Create a .env file with: OPENAI_API_KEY=your_api_key_here\n"
            "or set it in your environment directly.\n"
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
    top_k: int = 3,
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


def generate_answer_from_rag(
    client: OpenAI,
    question: str,
    retrieved_docs: List[Tuple[int, float, str]],
    model: str,
    chat_history: List[Tuple[str, str]],
) -> str:
    """
    Generates answer from local retrieved chunks only.
    """
    context_blocks = []
    for doc_index, score, doc_text in retrieved_docs:
        context_blocks.append(
            f"[Document {doc_index} | similarity={score:.4f}]\n{doc_text}"
        )
    context = "\n\n".join(context_blocks)

    system_prompt = (
        "You are a helpful assistant. Use only the provided context to answer. "
        "If the context is insufficient, clearly say the context is insufficient."
    )
    recent_history = chat_history[-4:]
    history_text = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in recent_history])
    user_prompt = (
        f"Conversation so far (most recent turns):\n{history_text}\n\n"
        f"Current question: {question}\n\n"
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


def _extract_responses_text(response: object) -> str:
    """
    Safely extracts text from Responses API objects.
    Different SDK versions can shape response objects differently.
    """
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    # Generic fallback parsing:
    pieces: List[str] = []
    output_items = getattr(response, "output", []) or []
    for item in output_items:
        content_items = getattr(item, "content", []) or []
        for content in content_items:
            text_value = getattr(content, "text", None)
            if isinstance(text_value, str) and text_value.strip():
                pieces.append(text_value.strip())

    return "\n".join(pieces).strip()


def generate_answer_with_web_search(
    client: OpenAI,
    question: str,
    model: str,
    chat_history: List[Tuple[str, str]],
) -> str:
    """
    Uses OpenAI Responses API with web search enabled.
    This is called only when local RAG context appears insufficient.
    """
    if not hasattr(client, "responses"):
        raise RuntimeError(
            "Your openai package does not expose Responses API. "
            "Please run: pip install -U openai"
        )

    recent_history = chat_history[-4:]
    history_text = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in recent_history])
    prompt = (
        "You are a helpful assistant. Use web search to answer the question accurately.\n"
        "Keep the answer beginner-friendly and concise. "
        "Include key source links when possible.\n\n"
        f"Conversation so far (most recent turns):\n{history_text}\n\n"
        f"Current question: {question}"
    )

    response = client.responses.create(
        model=model,
        tools=[{"type": "web_search_preview"}],
        input=prompt,
    )
    return _extract_responses_text(response)


def run_retrieval_for_question(
    client: OpenAI,
    question: str,
    embedding_model: str,
    index: faiss.Index,
    documents: List[str],
) -> List[Tuple[int, float, str]]:
    """Embeds the question and returns top matching document chunks."""
    query_vector = embed_texts(client, [question], embedding_model)[0]
    print(f"Query embedding length: {query_vector.shape[0]}")
    top_matches = retrieve_top_k(index, documents, query_vector, top_k=3)
    print("Top retrieved chunks:")
    for rank, (doc_index, score, doc_text) in enumerate(top_matches, start=1):
        preview = doc_text.replace("\n", " ")[:180]
        print(f"{rank}. doc_id={doc_index}, similarity={score:.4f}")
        print(f"   {preview}...")
    return top_matches


def should_use_web_fallback(
    retrieved_docs: List[Tuple[int, float, str]],
    rag_answer: str,
) -> bool:
    """
    Decides whether to call web-search fallback.
    We fallback when:
    - No retrieved chunks exist
    - Best similarity score is low
    - RAG answer explicitly says context is insufficient
    """
    if not retrieved_docs:
        return True

    best_score = retrieved_docs[0][1]
    if best_score < RAG_SCORE_THRESHOLD:
        return True

    lowered = rag_answer.lower()
    insufficient_hints = [
        "context is insufficient",
        "insufficient context",
        "not enough context",
        "missing context",
        "i do not have enough information",
    ]
    return any(hint in lowered for hint in insufficient_hints)


def chat_loop(
    client: OpenAI,
    embedding_model: str,
    chat_model: str,
    web_search_model: str,
    index: faiss.Index,
    documents: List[str],
) -> None:
    """
    Runs an interactive chat loop.
    Each user question first tries local RAG.
    If local context seems weak, it automatically falls back to web search.
    """
    print_section("RAG CHAT MODE")
    print("Ask questions about your documents.")
    print("Type 'exit', 'quit', or 'q' to stop.\n")

    history: List[Tuple[str, str]] = []
    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break

        print_section("STEP 3: RETRIEVAL")
        top_matches = run_retrieval_for_question(
            client=client,
            question=question,
            embedding_model=embedding_model,
            index=index,
            documents=documents,
        )

        print_section("STEP 4: GENERATION (RAG)")
        print(f"Generating answer with model: {chat_model}")
        rag_answer = generate_answer_from_rag(
            client=client,
            question=question,
            retrieved_docs=top_matches,
            model=chat_model,
            chat_history=history,
        )

        final_answer = rag_answer
        if should_use_web_fallback(top_matches, rag_answer):
            print_section("STEP 5: WEB SEARCH FALLBACK")
            print(
                "RAG context appears insufficient. "
                "Trying OpenAI web search for this question..."
            )
            try:
                web_answer = generate_answer_with_web_search(
                    client=client,
                    question=question,
                    model=web_search_model,
                    chat_history=history,
                )
                if web_answer:
                    final_answer = web_answer
            except Exception as exc:
                print(f"Web search fallback failed: {exc}")
                final_answer = (
                    rag_answer
                    + "\n\n(Note: Web fallback failed, so this is the local RAG answer.)"
                )

        print(f"\nAssistant: {final_answer}\n")
        history.append((question, final_answer))


def main() -> None:
    embedding_model = "text-embedding-3-small"
    chat_model = "gpt-4o-mini"
    load_environment_variables()
    web_search_model = os.getenv("WEB_SEARCH_MODEL", "gpt-4.1-mini")

    # -------------------------------------------------
    # STEP 1: Document setup
    # -------------------------------------------------
    print_section("STEP 1: DOCUMENT SETUP")
    docs_dir = os.getenv("RAG_DOCS_DIR", DEFAULT_DOCS_DIR)
    print(f"Trying to load documents from: {docs_dir}")
    documents, file_count = load_documents_from_folder(docs_dir)

    if not documents:
        print("No readable files found in folder. Using fallback sample documents.")
        documents = build_sample_documents()
        file_count = len(documents)

    print(f"Files loaded: {file_count}")
    print(f"Total chunks/documents for embedding: {len(documents)}")
    for i, doc in enumerate(documents[:5]):
        preview = doc.replace("\n", " ")[:140]
        print(f"Document {i}: {preview}...")
    if len(documents) > 5:
        print("... (showing first 5 documents only)")

    client = get_openai_client()

    # -------------------------------------------------
    # STEP 2: Embeddings
    # -------------------------------------------------
    print_section("STEP 2: EMBEDDINGS")
    print(f"Creating embeddings with model: {embedding_model}")
    doc_vectors = embed_texts(client, documents, embedding_model)
    print(f"Document embeddings shape: {doc_vectors.shape}")

    index = build_faiss_index(doc_vectors)
    print("Vector index is ready.\n")

    # -------------------------------------------------
    # STEP 3 + STEP 4 + STEP 5 (for each chat question)
    # -------------------------------------------------
    chat_loop(
        client=client,
        embedding_model=embedding_model,
        chat_model=chat_model,
        web_search_model=web_search_model,
        index=index,
        documents=documents,
    )


if __name__ == "__main__":
    main()
