import os, json, io, numpy as np
from typing import List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import httpx
from fastapi import HTTPException
import urllib.parse

from supabase import create_client, Client

# =========================
#   Supabase client
# =========================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY/SUPABASE_KEY for vectorstore")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)  # type: ignore

# Bucket untuk vectorstore
VECTOR_BUCKET = os.getenv("VECTOR_BUCKET", "vectors")

MAX_HF_RETRIES = int(os.getenv("HF_MAX_RETRIES", "4"))
HF_BACKOFF_S   = float(os.getenv("HF_BACKOFF_S", "1.5"))
TIMEOUT        = int(os.getenv("REQUEST_TIMEOUT_S", "120"))


def _raise_502(msg: str):
    raise HTTPException(status_code=502, detail=msg)


def _cfg():
    return {
        "EMB_BACKEND": os.getenv("EMB_BACKEND", "hf").lower(),
        "HF_MODEL": os.getenv("HF_EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        "OPENAI_EMB_MODEL": os.getenv("OPENAI_EMB_MODEL", "text-embedding-3-small"),
        "LOCAL_MODEL": os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        "EMB_BATCH_SIZE": int(os.getenv("EMB_BATCH_SIZE", "64")),
    }


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
        url = (
            "https://router.huggingface.co/"
            f"hf-inference/models/{model_id_enc}/pipeline/feature-extraction"
        )
        headers = {
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        payload_inputs = texts[0] if len(texts) == 1 else texts
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            r = await client.post(url, headers=headers, json={
                "inputs": payload_inputs,
                "options": {"wait_for_model": True}
            })

        if r.status_code >= 400:
            _raise_502(f"HF embeddings: {r.status_code} {r.text.strip()} (model='{model_id}')")

        data = r.json()
        arr = np.array(data, dtype=np.float32)

        if arr.ndim == 1:
            arr = arr[None, :]
        elif arr.ndim == 3:
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
            from sentence_transformers import SentenceTransformer
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


# =========================
#   Supabase Storage paths
# =========================
def _paths(doc_id: str) -> Tuple[str, str]:
    if not doc_id:
        raise HTTPException(status_code=400, detail="Missing doc_id for vectorstore")
    doc_id = doc_id.strip()
    return (
        f"{doc_id}/embeddings.npy",
        f"{doc_id}/texts.jsonl",
    )


def clear_store(doc_id: Optional[str] = None):
    """
    - Jika doc_id diberikan: hapus semua file vector milik dokumen tsb di bucket 'vectors'.
    - Jika None: NO-OP (hindari nge-wipe seluruh bucket dari kode).
    """
    if not doc_id:
        return
    try:
        # list semua file di "folder" doc_id
        files = supabase.storage.from_(VECTOR_BUCKET).list(doc_id)
        if not files:
            return
        paths = [f"{doc_id}/{f['name']}" for f in files if f.get("name")]
        if paths:
            supabase.storage.from_(VECTOR_BUCKET).remove(paths)
    except Exception as e:
        # jangan bikin request gagal cuma karena bersihin vector gagal
        print(f"[vectorstore] clear_store({doc_id}) failed: {e}", flush=True)


def save_vectors(embs: np.ndarray, texts: List[str], doc_id: str) -> None:
    if embs.ndim != 2:
        raise HTTPException(status_code=500, detail=f"Invalid embedding shape: {embs.shape}")
    if len(texts) != embs.shape[0]:
        n = min(len(texts), embs.shape[0])
        embs = embs[:n]
        texts = texts[:n]

    p_emb, p_txt = _paths(doc_id)

    # ---- upload embeddings.npy ----
    buf = io.BytesIO()
    np.save(buf, embs.astype(np.float32))
    buf.seek(0)
    try:
        res1 = supabase.storage.from_(VECTOR_BUCKET).upload(
            p_emb,
            buf.read(),
            {
                "content-type": "application/octet-stream",
                "cache-control": "no-cache",
                "upsert": "true",   # <- STRING, bukan bool
            },
        )
        if isinstance(res1, dict) and res1.get("error"):
            raise RuntimeError(res1["error"])
    except Exception as e:
        _raise_502(f"Failed to upload embeddings to storage: {e}")

    # ---- upload texts.jsonl ----
    lines = "\n".join(json.dumps({"text": t}, ensure_ascii=False) for t in texts)
    try:
        res2 = supabase.storage.from_(VECTOR_BUCKET).upload(
            p_txt,
            lines.encode("utf-8"),
            {
                "content-type": "application/json",
                "cache-control": "no-cache",
                "upsert": "true",   # <- sama di sini
            },
        )
        if isinstance(res2, dict) and res2.get("error"):
            raise RuntimeError(res2["error"])
    except Exception as e:
        _raise_502(f"Failed to upload texts to storage: {e}")

def load_vectors(doc_id: str) -> Tuple[np.ndarray, List[str]]:
    p_emb, p_txt = _paths(doc_id)
    try:
        raw_emb = supabase.storage.from_(VECTOR_BUCKET).download(p_emb)
        raw_txt = supabase.storage.from_(VECTOR_BUCKET).download(p_txt)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Belum ada dokumen terindeks. Silakan upload dulu."
        )

    try:
        embs = np.load(io.BytesIO(raw_emb))
    except Exception as e:
        _raise_502(f"Failed to load embeddings.npy from storage: {e}")

    texts: List[str] = []
    try:
        for line in raw_txt.decode("utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            texts.append(obj["text"])
    except Exception as e:
        _raise_502(f"Failed to parse texts.jsonl from storage: {e}")

    if embs.shape[0] != len(texts):
        n = min(embs.shape[0], len(texts))
        embs = embs[:n]
        texts = texts[:n]

    if embs.size == 0 or not texts:
        raise HTTPException(
            status_code=400,
            detail="Belum ada dokumen terindeks. Silakan upload dulu."
        )

    return embs, texts


async def add_texts(chunks: List[str], doc_id: str) -> None:
    """
    - Embed chunks baru
    - Jika sudah ada vectorstore doc_id, append
    - Simpan balik ke Supabase Storage
    """
    new_embs = await embed_texts(chunks)
    try:
        embs, texts = load_vectors(doc_id)
        embs = np.vstack([embs, new_embs])
        texts = texts + chunks
    except HTTPException:
        embs, texts = new_embs, chunks
    save_vectors(embs, texts, doc_id)


async def search(query: str, doc_id: str, top_k: int = 4) -> List[Tuple[str, float]]:
    embs, texts = load_vectors(doc_id)
    q_emb = await embed_texts([query])
    sims = cosine_similarity(q_emb, embs)[0]
    idxs = sims.argsort()[::-1][:top_k]
    return [(texts[i], float(sims[i])) for i in idxs]
