from typing import List, Dict, Any, Optional
import os

# Import the chunking/extraction helper from your existing module
from scrape_pdf import extract_metadata_from_pdf

# Optional imports for sentence-transformers
try:
    from sentence_transformers import SentenceTransformer, util
    import torch
except Exception as e:
    SentenceTransformer = None  # type: ignore
    util = None  # type: ignore
    torch = None  # type: ignore


def semantic_search_pdf(
    pdf_path: str,
    user_query: str,
    top_k: int = 5,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    names_to_ignore: Optional[List[str]] = None,
    device: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Load text chunks from `pdf_path` (using extract_metadata_from_pdf),
    encode chunks and the user query with the specified sentence-transformers model,
    and return the top_k matching chunks ordered by cosine similarity score.

    Returns a list of dicts:
      [
        {"rank": 1, "chunk": "<text>", "score": 0.9123, "chunk_index": 12},
        ...
      ]

    Notes:
    - `names_to_ignore` is forwarded to extract_metadata_from_pdf (default: []).
    - If sentence-transformers is not installed, raises RuntimeError with a helpful message.
    - `device` can be e.g. "cuda" or "cpu". If None, SentenceTransformer chooses automatically.
    """
    if names_to_ignore is None:
        names_to_ignore = []

    # Ensure the PDF exists
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Get chunks from the PDF (uses your existing chunking/extraction logic)
    chunks = extract_metadata_from_pdf(pdf_path, names_to_ignore=names_to_ignore)
    if not chunks:
        return []

    # Ensure sentence-transformers is available
    if SentenceTransformer is None or util is None:
        raise RuntimeError(
            "sentence-transformers is not installed or failed to import. "
            "Install with `pip install sentence-transformers` and try again."
        )

    # Load model (device selection: let SentenceTransformer handle default if device is None)
    model = SentenceTransformer(model_name)
    if device:
        try:
            model.to(device)
        except Exception:
            # Ignore if device move fails; SentenceTransformer may manage it itself
            pass

    # Encode chunks and query (convert_to_tensor=True yields torch tensors for fast similarity)
    chunk_embeddings = model.encode(chunks, convert_to_tensor=True, show_progress_bar=False)
    query_embedding = model.encode(user_query, convert_to_tensor=True)

    # Compute cosine similarities
    cos_scores = util.cos_sim(query_embedding, chunk_embeddings)[0]  # shape: (num_chunks,)

    # Get top_k indices (handle case top_k > num_chunks)
    k = min(top_k, len(chunks))
    if isinstance(cos_scores, torch.Tensor):
        topk = torch.topk(cos_scores, k=k)
        top_indices = topk.indices.cpu().tolist()
        top_scores = topk.values.cpu().tolist()
    else:
        # Fallback if not a torch tensor (shouldn't happen)
        import numpy as np

        arr = np.array(cos_scores)
        top_indices = arr.argsort()[::-1][:k].tolist()
        top_scores = arr[top_indices].tolist()

    results: List[Dict[str, Any]] = []
    for rank, (idx, score) in enumerate(zip(top_indices, top_scores), start=1):
        results.append(
            {
                "rank": rank,
                "chunk_index": int(idx),
                "chunk": chunks[int(idx)],
                "score": float(score),
            }
        )

    return results


if __name__ == "__main__":
    # Example usage
    pdf_path = "./pdfs/Class 8 Science CH 9.pdf"
    user_query = "What is a solution?"
    results = semantic_search_pdf(pdf_path, user_query)
    for res in results:
        print(f"Rank: {res['rank']}, Score: {res['score']}, Chunk: {res['chunk']}")
        print("\n" * 5)