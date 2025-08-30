from typing import List, Dict, Any, Optional
import os
import pickle
import hashlib
from pathlib import Path
import numpy as np
import tempfile

# Import your helper
from scrape_pdf import extract_metadata_from_pdf

# sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None  # type: ignore

# optional acceleration: FAISS CPU/GPU
try:
    import faiss  # type: ignore
    _HAS_FAISS = True
    _HAS_FAISS_GPU = hasattr(faiss, "StandardGpuResources")
except Exception:
    faiss = None  # type: ignore
    _HAS_FAISS = False
    _HAS_FAISS_GPU = False


def _safe_model_name(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_")


def _pdf_fingerprint(pdf: str) -> Dict[str, Any]:
    st = os.stat(pdf)
    return {"size": st.st_size, "mtime": st.st_mtime}


def _chunks_hash(text_chunks: List[str]) -> str:
    sep = "\n\u241F\n"
    return hashlib.sha256(sep.join(text_chunks).encode("utf-8", errors="ignore")).hexdigest()


def _query_hash(q: str) -> str:
    return hashlib.sha256(q.encode("utf-8", errors="ignore")).hexdigest()


def _get_cache_dir() -> Path:
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "semantic_search"
    return Path.home() / ".cache" / "semantic_search"


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=str(path.parent), delete=False) as tf:
        tf.write(data)
        tmp = Path(tf.name)
    tmp.replace(path)


def semantic_search_pdf(
    pdf_path: str,
    user_query: str,
    top_k: int = 5,
    model: Optional[SentenceTransformer] = None,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    names_to_ignore: Optional[List[str]] = None,
    device: Optional[str] = None,
    enable_query_cache: bool = True,
    use_faiss: bool = True,
    use_faiss_gpu: bool = True,
) -> List[Dict[str, Any]]:
    """
    Semantic search over PDF content using FAISS with optional GPU acceleration.

    - Uses SentenceTransformers for embeddings (normalize_embeddings=True so dot product == cosine).
    - Builds & persists a FAISS CPU index and metadata; optionally moves index to GPU at runtime.
    - Falls back to CPU FAISS or numpy dot-product if faiss is not installed.
    - Query embeddings can be cached.

    Parameters:
    - use_faiss: attempt to use FAISS if available.
    - use_faiss_gpu: if FAISS has GPU support, move index to GPU for search.
    """

    if names_to_ignore is None:
        names_to_ignore = []

    pdf_path = os.path.abspath(pdf_path)
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # If FAISS GPU desired, prefer model on CUDA for fastest pipeline
    if model is None:
        if SentenceTransformer is None:
            raise RuntimeError("Install sentence-transformers with `pip install sentence-transformers`")
        desired_device = device
        if use_faiss and use_faiss_gpu and _HAS_FAISS_GPU:
            # if GPU is available for faiss and user requested it, set model to cuda if not specified
            desired_device = device or "cuda"
        model = SentenceTransformer(model_name, device=desired_device)

    cache_dir = _get_cache_dir()
    safe_model = _safe_model_name(model_name)
    stem = Path(pdf_path).stem
    cache_dir.mkdir(parents=True, exist_ok=True)

    embeddings_cache_file = cache_dir / f"{stem}.embeddings.{safe_model}.pkl"
    faiss_index_file = cache_dir / f"{stem}.faiss.{safe_model}.index"
    faiss_meta_file = cache_dir / f"{stem}.faiss.{safe_model}.meta.pkl"
    query_cache_file = cache_dir / f"{stem}.queries.{safe_model}.pkl"

    # --- Load chunks ---
    chunks = extract_metadata_from_pdf(pdf_path, names_to_ignore=names_to_ignore)
    if not chunks:
        return []

    fingerprint = _pdf_fingerprint(pdf_path)
    chunks_digest = _chunks_hash(chunks)

    # --- Try loading cached chunk embeddings ---
    chunk_embeddings_np: Optional[np.ndarray] = None
    cached_chunks: Optional[List[str]] = None
    if embeddings_cache_file.is_file():
        try:
            with open(embeddings_cache_file, "rb") as f:
                data = pickle.load(f)
            if (
                isinstance(data, dict)
                and data.get("model_name") == model_name
                and data.get("pdf_path") == pdf_path
                and data.get("pdf_fingerprint") == fingerprint
                and data.get("chunks_hash") == chunks_digest
            ):
                cached_chunks = data["chunks"]
                emb = data["embeddings"]
                chunk_embeddings_np = np.ascontiguousarray(np.asarray(emb, dtype=np.float32))
                chunks = cached_chunks
        except Exception:
            chunk_embeddings_np = None

    # --- If no cache, compute embeddings ---
    if chunk_embeddings_np is None:
        chunk_embeddings = model.encode(
            chunks,
            convert_to_tensor=False,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        chunk_embeddings_np = np.ascontiguousarray(np.asarray(chunk_embeddings, dtype=np.float32))
        try:
            payload = {
                "model_name": model_name,
                "pdf_path": pdf_path,
                "pdf_fingerprint": fingerprint,
                "chunks_hash": chunks_digest,
                "chunks": chunks,
                "embeddings": chunk_embeddings_np,
            }
            raw = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
            _atomic_write_bytes(embeddings_cache_file, raw)
        except Exception:
            pass

    num_chunks = chunk_embeddings_np.shape[0]
    if num_chunks == 0:
        return []

    # --- FAISS index loading/building ---
    faiss_index = None
    faiss_on_gpu = False

    if use_faiss and _HAS_FAISS:
        # Try to load persisted index and metadata
        try:
            if faiss_index_file.is_file() and faiss_meta_file.is_file():
                with open(faiss_meta_file, "rb") as fm:
                    meta = pickle.load(fm)
                if (
                    isinstance(meta, dict)
                    and meta.get("model_name") == model_name
                    and meta.get("pdf_path") == pdf_path
                    and meta.get("pdf_fingerprint") == fingerprint
                    and meta.get("chunks_hash") == chunks_digest
                ):
                    # load CPU index and optionally transfer to GPU
                    cpu_index = faiss.read_index(str(faiss_index_file))
                    if use_faiss_gpu and _HAS_FAISS_GPU:
                        try:
                            res = faiss.StandardGpuResources()
                            # select device 0; more advanced options could be added
                            faiss_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                            faiss_on_gpu = True
                        except Exception:
                            faiss_index = cpu_index
                            faiss_on_gpu = False
                    else:
                        faiss_index = cpu_index
        except Exception:
            faiss_index = None

        # Build a new index if necessary
        if faiss_index is None:
            try:
                dim = int(chunk_embeddings_np.shape[1])
                cpu_index = faiss.IndexFlatIP(dim)  # inner product for normalized embeddings
                # Ensure contiguous float32
                vectors = np.ascontiguousarray(chunk_embeddings_np.astype(np.float32))
                cpu_index.add(vectors)
                # persist cpu index to disk
                try:
                    faiss.write_index(cpu_index, str(faiss_index_file))
                    meta = {
                        "model_name": model_name,
                        "pdf_path": pdf_path,
                        "pdf_fingerprint": fingerprint,
                        "chunks_hash": chunks_digest,
                        "num_chunks": num_chunks,
                    }
                    raw_meta = pickle.dumps(meta, protocol=pickle.HIGHEST_PROTOCOL)
                    _atomic_write_bytes(faiss_meta_file, raw_meta)
                except Exception:
                    pass
                # optionally move to GPU
                if use_faiss_gpu and _HAS_FAISS_GPU:
                    try:
                        res = faiss.StandardGpuResources()
                        faiss_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                        faiss_on_gpu = True
                    except Exception:
                        faiss_index = cpu_index
                        faiss_on_gpu = False
                else:
                    faiss_index = cpu_index
                    faiss_on_gpu = False
            except Exception:
                faiss_index = None

    # --- Query caching (by hash) ---
    query_embedding_np: Optional[np.ndarray] = None
    query_cache: Dict[str, np.ndarray] = {}
    qhash = _query_hash(user_query)

    if enable_query_cache and query_cache_file.is_file():
        try:
            with open(query_cache_file, "rb") as f:
                query_cache = pickle.load(f)
            if qhash in query_cache:
                query_embedding_np = np.ascontiguousarray(np.asarray(query_cache[qhash], dtype=np.float32))
        except Exception:
            query_cache = {}

    if query_embedding_np is None:
        query_embedding = model.encode(
            user_query,
            convert_to_tensor=False,
            normalize_embeddings=True,
        )
        query_embedding_np = np.ascontiguousarray(np.asarray(query_embedding, dtype=np.float32))
        if enable_query_cache:
            try:
                query_cache[qhash] = query_embedding_np
                raw = pickle.dumps(query_cache, protocol=pickle.HIGHEST_PROTOCOL)
                _atomic_write_bytes(query_cache_file, raw)
            except Exception:
                pass

    k = min(top_k, num_chunks)

    # --- Search using FAISS when available, else numpy path ---
    if faiss_index is not None:
        try:
            qv = np.ascontiguousarray(query_embedding_np.reshape(1, -1).astype(np.float32))
            distances, indices = faiss_index.search(qv, k)
            top_indices = indices[0].tolist()
            top_scores = [float(d) for d in distances[0].tolist()]
        except Exception:
            # fallback to numpy
            cos_scores = np.dot(chunk_embeddings_np, query_embedding_np)
            if k == 1:
                top_idx = int(np.argmax(cos_scores))
                top_indices = [top_idx]
                top_scores = [float(cos_scores[top_idx])]
            else:
                part = np.argpartition(-cos_scores, k - 1)[:k]
                top_sorted = part[np.argsort(-cos_scores[part])]
                top_indices = top_sorted.tolist()
                top_scores = [float(cos_scores[i]) for i in top_indices]
    else:
        # FAISS not available: do exact dot-product
        cos_scores = np.dot(chunk_embeddings_np, query_embedding_np)
        if k == 1:
            top_idx = int(np.argmax(cos_scores))
            top_indices = [top_idx]
            top_scores = [float(cos_scores[top_idx])]
        else:
            part = np.argpartition(-cos_scores, k - 1)[:k]
            top_sorted = part[np.argsort(-cos_scores[part])]
            top_indices = top_sorted.tolist()
            top_scores = [float(cos_scores[i]) for i in top_indices]

    results: List[Dict[str, Any]] = []
    for rank, (idx, score) in enumerate(zip(top_indices, top_scores), start=1):
        results.append({"rank": rank, "chunk_index": int(idx), "chunk": chunks[int(idx)], "score": float(score)})

    return results


if __name__ == "__main__":
    pdf_path = "./pdfs/Class 8 Science CH 9.pdf"
    user_query = "In Gulab Jamun chashni, which component is the solvent and which is the solute?"

    # Load model once and reuse (attempt GPU)
    model = None
    if SentenceTransformer is not None:
        try:
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
        except Exception:
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    results = semantic_search_pdf(pdf_path, user_query, model=model, use_faiss=True, use_faiss_gpu=True)
    for res in results:
        print(f"Rank: {res['rank']}, Score: {res['score']:.4f}, Chunk: {res['chunk']}\n")