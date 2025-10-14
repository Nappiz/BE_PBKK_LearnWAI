import os, shutil, json, numpy as np
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import httpx
from fastapi import HTTPException
import urllib.parse
import asyncio  # pakai asyncio.sleep untuk backoff non-blocking

VECTOR_DIR = os.getenv("VECTOR_DIR", "./data/vectorstore")

def clear_store():
    if os.path.exists(VECTOR_DIR):
        shutil.rmtree(VECTOR_DIR)
    os.makedirs(VECTOR_DIR, exist_ok=True)

MAX_HF_RETRIES = int(os.getenv("HF_MAX_RETRIES", "4"))
HF_BACKOFF_S   = float(os.getenv("HF_BACKOFF_S", "1.5"))

TIMEOUT = int(os.getenv("REQUEST_TIMEOUT_S", "120"))

def _cfg():
    return {
        "EMB_BACKEND": os.getenv("EMB_BACKEND", "hf").lower(),
        "HF_MODEL": os.getenv("HF_EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        "OPENAI_EMB_MODEL": os.getenv("OPENAI_EMB_MODEL", "text-embedding-3-small"),
        "LOCAL_MODEL": os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        "EMB_BATCH_SIZE": int(os.getenv("EMB_BATCH_SIZE", "64")),
    }

def _ensure_dir():
    os.makedirs(VECTOR_DIR, exist_ok=True)

def _raise_502(msg: str):
    raise HTTPException(status_code=502, detail=msg)

def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    return normalize(arr, norm="l2", axis=1)

def _resolve_hf_model_id() -> str:
    m = (
        os.getenv("HF_EMB_MODEL")
        or os.getenv("EMB_MODEL")
        or "sentence-transformers/all-MiniLM-L6-v2"
    )
    m = (m or "").strip().strip('"').strip("'")
    if m.endswith(":latest"):
        m = m[:-7]
    return m

async def embed_texts(texts: List[str]) -> np.ndarray:
    cfg = _cfg()
    backend = cfg["EMB_BACKEND"]

    if backend == "hf":
        key = (os.getenv("HF_API_KEY") or "").strip()
        if not key:
            _raise_502("Missing HF_API_KEY for embeddings")

        model_id = _resolve_hf_model_id()
        model_id_enc = urllib.parse.quote(model_id, safe='-_/')

        # Pakai endpoint umum /models (lebih toleran, jarang 404)
        url = f"https://api-inference.huggingface.co/models/{model_id_enc}"
        headers = {
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        payload_inputs = texts[0] if len(texts) == 1 else texts

        # Log supaya kelihatan PERSIS model dan URL-nya
        print(f"[HF-EMB] backend=hf | model='{model_id}' | url={url}", flush=True)

        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            r = await client.post(
                url,
                headers=headers,
                json={
                    "inputs": payload_inputs,
                    # memaksa server siapin model; hindari 404/empty di load pertama
                    "options": {"wait_for_model": True}
                },
            )

        if r.status_code >= 400:
            # Bubble-up pesan server lengkap agar tahu akar masalahnya
            _raise_502(f"HF embeddings: {r.status_code} {r.text.strip()} (model='{model_id}')")

        data = r.json()

        # Normalisasi output:
        # - bisa [D] untuk single, atau [N,D], atau [N,T,D] (token-level)
        arr = np.array(data, dtype=np.float32)

        if arr.ndim == 1:
            arr = arr[None, :]
        elif arr.ndim == 3:
            # pooling mean token → sentence embedding
            arr = arr.mean(axis=1)

        if arr.ndim != 2:
            _raise_502(f"HF embeddings: unexpected shape {arr.shape} (model='{model_id}')")

        return _l2_normalize(arr).astype(np.float32)

    elif backend == "openai":
        key = os.getenv("OPENAI_API_KEY", "")
        if not key:
            _raise_502("Missing OPENAI_API_KEY for embeddings")
        url = "https://api.openai.com/v1/embeddings"
        payload = {"input": texts, "model": cfg["OPENAI_EMB_MODEL"]}
        headers = {"Authorization": f"Bearer {key}"}
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            r = await client.post(url, json=payload, headers=headers)
        if r.status_code >= 400:
            _raise_502(f"OpenAI embeddings: {r.text}")
        data = r.json()
        vecs = [item["embedding"] for item in data["data"]]
        arr = np.array(vecs, dtype=np.float32)
        return _l2_normalize(arr)

    elif backend == "local":
        try:
            from sentence_transformers import SentenceTransformer  # lazy import
        except Exception as e:
            _raise_502(f"Local embeddings import failed: {e}")
        try:
            model = SentenceTransformer(cfg["LOCAL_MODEL"])
        except Exception as e:
            _raise_502(f"Local embeddings model load failed: {e}")
        arr = model.encode(
            texts,
            batch_size=cfg["EMB_BATCH_SIZE"],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return arr.astype(np.float32)

    else:
        _raise_502(f"Unknown EMB_BACKEND={backend}")

def _paths():
    _ensure_dir()
    return (
        os.path.join(VECTOR_DIR, "embeddings.npy"),
        os.path.join(VECTOR_DIR, "texts.jsonl"),
    )

def save_vectors(embs: np.ndarray, texts: List[str]) -> None:
    p_emb, p_txt = _paths()
    np.save(p_emb, embs)
    with open(p_txt, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")

def load_vectors() -> Tuple[np.ndarray, List[str]]:
    p_emb, p_txt = _paths()
    if not (os.path.exists(p_emb) and os.path.exists(p_txt)):
        raise HTTPException(status_code=400, detail="Belum ada dokumen terindeks. Silakan upload dulu.")
    embs = np.load(p_emb)
    texts: List[str] = []
    with open(p_txt, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])
    return embs, texts

async def add_texts(chunks: List[str]) -> None:
    new_embs = await embed_texts(chunks)
    try:
        embs, texts = load_vectors()
        embs = np.vstack([embs, new_embs])
        texts = texts + chunks
    except HTTPException:
        embs, texts = new_embs, chunks
    save_vectors(embs, texts)

async def search(query: str, top_k: int = 4) -> List[Tuple[str, float]]:
    embs, texts = load_vectors()
    q_emb = await embed_texts([query])
    sims = cosine_similarity(q_emb, embs)[0]
    idxs = sims.argsort()[::-1][:top_k]
    return [(texts[i], float(sims[i])) for i in idxs]
