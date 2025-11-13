from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()

import os, re, io, asyncio, logging, secrets, bcrypt, unicodedata
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from zoneinfo import ZoneInfo
from uuid import uuid4
from urllib.parse import urlparse
import httpx

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field
from PyPDF2 import PdfReader

from supabase import create_client, Client

# ==== schemas & providers & vectorstore ====
from schemas import (
    SummarizeRequest, SummarizeResponse, RagasScores,
    QARequest, QAResponse,
    FlashcardsRequest, Flashcard, FlashcardsResponse,
)
from vectorstore import add_texts, search, clear_store
import providers

RAGAS_ENABLED = os.getenv("RAGAS_ENABLED", "true").lower() == "true"
try:
    if RAGAS_ENABLED:
        from evaluation import run_ragas_eval
    else:
        run_ragas_eval = None
except Exception as e:
    print(f"[RAGAS] disabled because import failed: {e}", flush=True)
    run_ragas_eval = None
    RAGAS_ENABLED = False

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
if not SUPABASE_URL or not (SUPABASE_SERVICE_KEY or SUPABASE_KEY):
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY/SUPABASE_KEY")
_EFF_KEY = SUPABASE_SERVICE_KEY or SUPABASE_KEY
supabase: Client = create_client(SUPABASE_URL, _EFF_KEY)  # type: ignore

# ---- CORS ----
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "").strip()
DEFAULT_REGEX = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$|^https://.*\.vercel\.app$"
CORS_ORIGIN_REGEX = os.getenv("CORS_ORIGIN_REGEX", DEFAULT_REGEX)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
MIN_CHARS_PER_CHUNK = int(os.getenv("MIN_CHARS_PER_CHUNK", "300"))

GEN_MAX_TOKENS = int(os.getenv("GEN_MAX_TOKENS", "2048"))

origins = [o.strip() for o in CORS_ORIGINS.split(",") if o.strip()]
logger = logging.getLogger("uvicorn.error")

APP_VERSION = "1.5.3"

app = FastAPI(title="LearnWAI RAG App", version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins else [],
    allow_origin_regex=None if origins else CORS_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,
)

# ======== auth payloads ========
class RegisterIn(BaseModel):
    name: str
    email: EmailStr
    password: str

class LoginIn(BaseModel):
    email: EmailStr
    password: str

# ======== update payloads ========
class UpdateDocumentIn(BaseModel):
    title: Optional[str] = Field(default=None, min_length=1, max_length=300)

# ======== in-memory job status ========
STATUS: Dict[str, dict] = {}

def new_job_id() -> str:
    return secrets.token_hex(16)

def _upsert_job(job: dict):
    try:
        supabase.table("jobs").upsert(job, on_conflict="job_id").execute()
    except Exception as e:
        try:
            supabase.table("jobs").update(job).eq("job_id", job["job_id"]).execute()
        except Exception:
            try:
                supabase.table("jobs").insert(job).execute()
            except Exception as e2:
                print(f"[jobs] persist failed: {e} | {e2}", flush=True)

def _set_status(job_id: str, stage: str, progress: int, message: str, ok: bool = True, doc_id: Optional[str] = None):
    s = {"job_id": job_id, "stage": stage, "progress": int(progress), "message": message, "ok": bool(ok)}
    STATUS[job_id] = s
    line = f"[STATUS] {job_id} | {stage} | {progress}% | {message}"
    print(line, flush=True); logger.info(line)
    payload = {
        "job_id": job_id,
        "stage": stage,
        "progress": int(progress),
        "message": message,
        "ok": bool(ok),
    }
    if doc_id:
        payload["doc_id"] = doc_id
    _upsert_job(payload)

# ======== storage helpers ========

BUCKET_NAME = "documents"

def _bucket_name_of(b) -> Optional[str]:
    if isinstance(b, dict):
        return b.get("name") or b.get("id")
    return getattr(b, "name", None) or getattr(b, "id", None)

def ensure_bucket(name: str = BUCKET_NAME, public: bool = True):
    try:
        buckets = supabase.storage.list_buckets()
        if any((_bucket_name_of(b) == name) for b in buckets):
            return
    except Exception:
        pass

    try:
        supabase.storage.create_bucket(name, public=public)  # type: ignore[arg-type]
        print(f"[storage] bucket '{name}' created via SDK(A) public={public}", flush=True)
        return
    except TypeError:
        pass
    except Exception as e:
        if "already exists" in str(e) or "409" in str(e):
            return

    try:
        options = {"name": name, "public": bool(public)}
        supabase.storage.create_bucket(name, options)  # type: ignore[arg-type]
        print(f"[storage] bucket '{name}' created via SDK(B) public={public}", flush=True)
        return
    except Exception as e:
        if "already exists" in str(e) or "409" in str(e):
            return
        print(f"[storage] SDK create_bucket failed: {e}", flush=True)

    key = SUPABASE_SERVICE_KEY or SUPABASE_KEY
    if not key:
        raise HTTPException(status_code=500, detail="Missing service key for bucket creation fallback.")

    url = f"{SUPABASE_URL.rstrip('/')}/storage/v1/bucket"
    headers = {
        "Authorization": f"Bearer {key}",
        "apikey": key,
        "Content-Type": "application/json",
    }
    payload = {"name": name, "public": bool(public)}
    try:
        r = httpx.post(url, headers=headers, json=payload, timeout=30.0)
        if r.status_code in (200, 201):
            print(f"[storage] bucket '{name}' created via REST public={public}", flush=True)
            return
        if r.status_code == 409:
            return
        raise HTTPException(status_code=502, detail=f"Create bucket failed: {r.text}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Create bucket failed: {e}")

# ======== small helpers ========
def _extract_pdf_text(data: bytes) -> str:
    with io.BytesIO(data) as bio:
        reader = PdfReader(bio)
        parts: List[str] = []
        for p in reader.pages:
            try:
                t = p.extract_text() or ""
            except Exception:
                t = ""
            if t: parts.append(t)
        return "\n".join(parts)

VISION_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT_S", "120"))


def _extract_text_generic(data):
    """
    Helper kecil buat narik teks dari berbagai bentuk respons:
    - string langsung
    - {"response": "..."} / {"text": "..."} / {"output": "..."}
    - fallback: str(data)
    """
    if data is None:
        return ""
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        for k in ("response", "text", "output"):
            v = data.get(k)
            if isinstance(v, str):
                return v
        return str(data)
    return str(data)


async def _extract_vision_from_pdf(raw_pdf: bytes) -> str:
    """
    Kirim PDF ke Senopati /vision/pdf untuk baca konten visual (gambar/tabel).
    Return: string deskripsi (bisa kosong kalau gagal).
    """
    base = os.getenv("SENOPATI_BASE_URL", "").rstrip("/")
    if not base:
        return ""

    model = os.getenv("SENOPATI_VISION_MODEL", "qwen2.5vl:7b")
    prompt = os.getenv(
        "SENOPATI_VISION_PROMPT",
        "Describe the content of this document",
    )

    try:
        temperature = float(os.getenv("VISION_TEMPERATURE", "0.7"))
    except Exception:
        temperature = 0.7

    try:
        max_tokens = int(os.getenv("VISION_MAX_TOKENS", "0") or "0")
    except Exception:
        max_tokens = 0

    dpi = int(os.getenv("VISION_DPI", "150"))
    max_image_size = int(os.getenv("VISION_MAX_IMAGE_SIZE", "1024"))
    image_quality = int(os.getenv("VISION_IMAGE_QUALITY", "85"))
    save_images = os.getenv("VISION_SAVE_IMAGES", "true").lower() == "true"

    max_pages_env = os.getenv("VISION_MAX_PAGES", "").strip()
    max_pages = int(max_pages_env) if max_pages_env else None

    params = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "dpi": dpi,
        "max_image_size": max_image_size,
        "image_quality": image_quality,
        "save_images": save_images,
    }
    if max_tokens > 0:
        params["max_tokens"] = max_tokens
    if max_pages is not None:
        params["max_pages"] = max_pages

    files = {
        "pdf": ("document.pdf", raw_pdf, "application/pdf"),
    }

    url = f"{base}/vision/pdf"

    try:
        async with httpx.AsyncClient(timeout=VISION_TIMEOUT) as client:
            r = await client.post(url, params=params, files=files)
    except Exception as e:
        print(f"[vision] request failed: {e}", flush=True)
        return ""

    if r.status_code >= 400:
        print(f"[vision] bad status {r.status_code}: {r.text}", flush=True)
        return ""

    try:
        data = r.json()
    except Exception:
        return r.text

    return _extract_text_generic(data)

# ==== PDF -> image previews untuk tampilan di FE ====
def _generate_page_previews(doc_id: str, raw_pdf: bytes) -> list[str]:
    """
    Render setiap halaman PDF menjadi PNG, upload ke Supabase storage,
    dan return list public URL-nya.

    Kalau gagal (pdf2image/poppler gak ada, dsb) -> return [].
    """
    try:
        from pdf2image import convert_from_bytes
    except Exception as e:
        print(f"[pdf-previews] pdf2image import failed: {e}", flush=True)
        return []

    try:
        dpi = int(os.getenv("PDF_IMG_DPI", "120"))
        pages = convert_from_bytes(raw_pdf, dpi=dpi)
    except Exception as e:
        print(f"[pdf-previews] convert_from_bytes failed: {e}", flush=True)
        return []

    urls: list[str] = []
    for idx, page in enumerate(pages, start=1):
        try:
            buf = io.BytesIO()
            page.save(buf, format="PNG")
            data = buf.getvalue()
        except Exception as e:
            print(f"[pdf-previews] save page {idx} failed: {e}", flush=True)
            continue

        path = f"previews/{doc_id}/page-{idx}.png"

        try:
            res = supabase.storage.from_(BUCKET_NAME).upload(
                path,
                data,
                {
                    "content-type": "image/png",
                    "cache-control": "public, max-age=31536000",
                    "upsert": "true",
                },
            )
            if isinstance(res, dict) and res.get("error"):
                print(f"[pdf-previews] upload error for {path}: {res['error']}", flush=True)
                continue

            url = supabase.storage.from_(BUCKET_NAME).get_public_url(path)
            urls.append(url)
        except Exception as e:
            print(f"[pdf-previews] upload failed for {path}: {e}", flush=True)
            continue

    return urls

def normalize_text(s: str) -> str:
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def chunk_text(text: str, chunk_size: int, overlap: int, min_chars: int) -> List[str]:
    text = text.strip()
    if not text: return []
    chunks: List[str] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        if len(chunk) >= min_chars: chunks.append(chunk)
        if end >= n: break
        start = max(0, end - overlap)
    if not chunks and text:
        chunks = [text[:chunk_size]]
    return chunks

def _dedup(seq: List[str]) -> List[str]:
    seen, out = set(), []
    for s in seq:
        k = s.strip()
        if not k or k in seen: continue
        seen.add(k); out.append(s)
    return out

def _ragas_or_none(question: str, answer: str, contexts: List[str]):
    if not RAGAS_ENABLED or run_ragas_eval is None:
        return {"context_relevancy": None, "context_recall": None, "answer_correctness": None, "faithfulness": None}
    return None

def slugify(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower() or "document"
    return s

def _parse_flashcards(text: str) -> List[Flashcard]:
    cards: List[Flashcard] = []
    q, a = None, None
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.lower().startswith("q:"):
            if q and a:
                cards.append(Flashcard(question=q, answer=a))
            q, a = s[2:].strip(" :"), None
        elif s.lower().startswith("a:") and q:
            a = s[2:].strip(" :")
            cards.append(Flashcard(question=q, answer=a))
            q, a = None, None
    if q and a:
        cards.append(Flashcard(question=q, answer=a))
    return cards

def _dedup_cards(cards: List[Flashcard]) -> List[Flashcard]:
    seen = set()
    out: List[Flashcard] = []
    for c in cards:
        key = re.sub(r"\s+", " ", c.question.strip().lower())
        if not key or key in seen:
            continue
        seen.add(key); out.append(c)
    return out

# ==== Document lookup & fallbacks ====
def _doc_by_id(doc_id: str):
    res = supabase.table("documents").select("*").eq("id", doc_id).limit(1).execute()
    return res.data[0] if res.data else None

def _doc_by_slug(slug: str):
    res = supabase.table("documents").select("*").eq("slug", slug).limit(1).execute()
    return res.data[0] if res.data else None

def _latest_ready_doc(user_id: Optional[str] = None):
    q = supabase.table("documents").select("*").eq("status", "ready").order("created_at", desc=True)
    if user_id:
        q = q.eq("user_id", user_id)
    res = q.limit(1).execute()
    return res.data[0] if res.data else None

def _resolve_doc_id(doc_id: Optional[str], slug: Optional[str], allow_not_ready: bool = False, user_id: Optional[str] = None) -> str:
    d = None
    if doc_id:
        d = _doc_by_id(doc_id)
    elif slug:
        d = _doc_by_slug(slug)
    else:
        d = _latest_ready_doc(user_id=user_id)

    if not d:
        raise HTTPException(status_code=400, detail="No ready document found. Upload or specify doc_id/slug.")
    if not allow_not_ready and (d.get("status") or "") != "ready":
        raise HTTPException(status_code=409, detail="Document is not ready yet")
    return d["id"]

# Extract storage path from public URL
def _extract_path_from_public_url(public_url: Optional[str]) -> Optional[str]:
    """
    Supabase public URL example:
    https://<proj>.supabase.co/storage/v1/object/public/documents/<folder>/<doc_id>.pdf
    We need '<folder>/<doc_id>.pdf'
    """
    if not public_url:
        return None
    try:
        u = urlparse(public_url)
        # find "/object/public/<bucket>/" then take the tail
        m = re.search(r"/object/public/([^/]+)/(.+)$", u.path)
        if not m:
            # older style: .../storage/v1/object/public/<bucket>/<path>
            # already handled by regex; fallback try by bucket name
            idx = u.path.find(f"/{BUCKET_NAME}/")
            if idx != -1:
                return u.path[idx + len(BUCKET_NAME) + 2:]  # skip "/<bucket>/"
            return None
        bucket = m.group(1)
        tail = m.group(2)
        if bucket != BUCKET_NAME:
            return None
        return tail
    except Exception:
        return None

# =====================================
#               AUTH
# =====================================
@app.post("/auth/register")
def auth_register(req: RegisterIn):
    sel = supabase.table("users").select("*").eq("email", req.email).limit(1).execute()
    if sel.data:
        raise HTTPException(status_code=409, detail="Email already registered")
    pw_hash = bcrypt.hashpw(req.password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    ins = supabase.table("users").insert({
        "id": str(uuid4()),
        "name": req.name,
        "email": req.email,
        "password_hash": pw_hash,
        "created_at": datetime.utcnow().isoformat(),
    }).execute()
    if not ins.data:
        raise HTTPException(status_code=500, detail="Failed to create user")
    u = ins.data[0]
    return {"user": {"id": u["id"], "name": u["name"], "email": u["email"], "created_at": u.get("created_at")}}

@app.post("/auth/login")
def auth_login(req: LoginIn):
    sel = supabase.table("users").select("*").eq("email", req.email).limit(1).execute()
    if not sel.data:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    u = sel.data[0]
    pw_hash = u.get("password_hash")
    if not pw_hash or not bcrypt.checkpw(req.password.encode("utf-8"), pw_hash.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return {"user": {"id": u["id"], "name": u["name"], "email": u["email"], "created_at": u.get("created_at")}}

# =====================================
#        DOCUMENTS + PIPELINE
# =====================================
@app.post("/documents/upload")
async def documents_upload(
    file: UploadFile = File(...),
    user_id: Optional[str] = None,
    background_tasks: BackgroundTasks = None,
):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF is allowed")

    ensure_bucket(BUCKET_NAME, public=True)

    raw = await file.read()
    size = len(raw)
    title = file.filename or "document.pdf"

    folder = user_id or "anonymous"
    doc_id = str(uuid4())
    path = f"{folder}/{doc_id}.pdf"

    try:
        res = supabase.storage.from_(BUCKET_NAME).upload(
            path, raw, {"content-type": "application/pdf"}
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Storage upload failed: {e}")
    if isinstance(res, dict) and res.get("error"):
        raise HTTPException(status_code=500, detail=f"Storage error: {res['error']}")

    url = supabase.storage.from_(BUCKET_NAME).get_public_url(path)

    try:
        pages = len(PdfReader(io.BytesIO(raw)).pages)
    except Exception:
        pages = None

    base_slug = slugify(os.path.splitext(title)[0])
    unique_slug = f"{base_slug}-{uuid4().hex[:6]}"

    ins = supabase.table("documents").insert({
        "id": doc_id,
        "title": title,
        "url": url,
        "size": size,
        "page_count": pages,
        "status": "processing",
        "user_id": user_id,
        "created_at": datetime.utcnow().isoformat(),
        "slug": unique_slug,
    }).execute()
    if not ins.data:
        raise HTTPException(status_code=500, detail="Failed to insert document")

    job_id = new_job_id()
    if background_tasks is None:
        raise HTTPException(status_code=500, detail="Background tasks not available")

    _upsert_job({
        "job_id": job_id,
        "doc_id": doc_id,
        "stage": "received",
        "progress": 5,
        "message": f"File received: {title}",
        "ok": True,
    })

    _set_status(job_id, "received", 5, f"File received: {title}", ok=True, doc_id=doc_id)
    background_tasks.add_task(_pipeline_process, job_id, doc_id, title, raw)

    return {"job_id": job_id, "document": ins.data[0]}

@app.get("/api/status/{job_id}")
async def job_status(job_id: str):
    st = STATUS.get(job_id)
    if st:
        return st
    res = supabase.table("jobs").select("*").eq("job_id", job_id).limit(1).execute()
    if res.data:
        row = res.data[0]
        return {
            "job_id": row["job_id"],
            "stage": row["stage"],
            "progress": row["progress"],
            "message": row.get("message") or "",
            "ok": bool(row.get("ok", True)),
        }
    raise HTTPException(status_code=404, detail="job_id not found")

async def _pipeline_process(job_id: str, doc_id: str, title: str, raw: bytes):
    try:
        clear_store(doc_id)
        _set_status(job_id, "extract", 10, "Extracting PDF…", doc_id=doc_id)
        text_plain = _extract_pdf_text(raw)

        _set_status(job_id, "normalize", 20, "Normalizing…", doc_id=doc_id)
        text_plain = normalize_text(text_plain)

        _set_status(job_id, "vision", 30, "Analyzing images…", doc_id=doc_id)
        vision_notes = await _extract_vision_from_pdf(raw) 

        if vision_notes:
            full_text = text_plain + "\n\n[CATATAN VISUAL]\n" + vision_notes
        else:
            full_text = text_plain

        _set_status(job_id, "split", 40, "Chunking…", doc_id=doc_id)
        chunks = chunk_text(full_text, CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHARS_PER_CHUNK)
        if not chunks:
            raise HTTPException(status_code=400, detail="Document too short after normalization")

        _set_status(job_id, "embedding", 60, "Embedding & indexing…", doc_id=doc_id)
        await asyncio.shield(add_texts(chunks, doc_id))

        _set_status(job_id, "images", 70, "Generating page previews…", doc_id=doc_id)
        image_urls = _generate_page_previews(doc_id, raw)

        _set_status(job_id, "embedding", 80, "Generating summary…", doc_id=doc_id)
        try:
            sum_res: SummarizeResponse = await summarize(
                SummarizeRequest(query="ringkas dokumen ini"),
                doc_id=doc_id,
                internal=True,  
            )
            summary_text = sum_res.text
        except Exception as e:
            summary_text = f"(summary failed: {e})"

        _set_status(job_id, "embedding", 88, "Generating flashcards…", doc_id=doc_id)
        try:
            fc_res: FlashcardsResponse = await flashcards(
                FlashcardsRequest(question_hint=None),
                doc_id=doc_id,
                internal=True,  
            )
            flashcards_json = [c.dict() for c in fc_res.cards]
        except Exception as e:
            flashcards_json = [{"question": "Generation failed", "answer": str(e)}]

        supabase.table("documents").update({
            "status": "ready",
            "summary": summary_text,
            "flashcards": flashcards_json,
            "image_urls": image_urls,  
        }).eq("id", doc_id).execute()

        _set_status(job_id, "done", 100, "Ready", ok=True, doc_id=doc_id)

    except Exception as e:
        _set_status(job_id, "error", 100, str(e), ok=False, doc_id=doc_id)
        try:
            supabase.table("documents").update({"status": "error"}).eq("id", doc_id).execute()
        except Exception:
            pass

@app.get("/documents")
def list_documents(user_id: Optional[str] = None):
    q = supabase.table("documents").select("*").order("created_at", desc=True)
    if user_id:
        q = q.eq("user_id", user_id)
    res = q.execute()
    return {"data": res.data or []}

@app.get("/documents/{doc_id}")
def get_document(doc_id: str):
    res = supabase.table("documents").select("*").eq("id", doc_id).limit(1).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"data": res.data[0]}

@app.get("/documents/by-slug/{slug}")
def get_document_by_slug(slug: str):
    res = supabase.table("documents").select("*").eq("slug", slug).limit(1).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"data": res.data[0]}

# ===== NEW: update & delete =====

@app.patch("/documents/{doc_id}")
def update_document(doc_id: str, body: UpdateDocumentIn):
    row = _doc_by_id(doc_id)
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")
    updates: Dict[str, object] = {}
    if body.title is not None:
        title = body.title.strip()
        if not title:
            raise HTTPException(status_code=400, detail="Title cannot be empty")
        if len(title) > 300:
            raise HTTPException(status_code=400, detail="Title too long")
        updates["title"] = title
    if not updates:
        return {"ok": True, "data": row}
    res = supabase.table("documents").update(updates).eq("id", doc_id).execute()
    return {"ok": True, "data": (res.data[0] if res.data else {**row, **updates})}

@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    row = _doc_by_id(doc_id)
    if not row:
        raise HTTPException(status_code=404, detail="Document not found")

    p = _extract_path_from_public_url(row.get("url"))
    try:
        if p:
            supabase.storage.from_(BUCKET_NAME).remove([p])
    except Exception as e:
        print(f"[storage] remove failed for {p}: {e}", flush=True)

    try:
        clear_store(doc_id)
    except Exception as e:
        print(f"[vectorstore] clear_store({doc_id}) failed: {e}", flush=True)

    supabase.table("documents").delete().eq("id", doc_id).execute()

    try:
        supabase.table("jobs").delete().eq("doc_id", doc_id).execute()
    except Exception:
        pass

    return {"ok": True, "id": doc_id}

# ===== Helpers RAG =====
def _pick_contexts(
    results: List[tuple[str, float]],
    min_sim: float,
    max_chunks: int,
    char_budget: int,
) -> List[str]:
    cands = [(t, s) for t, s in results if s >= min_sim]
    cands.sort(key=lambda x: x[1], reverse=True)
    ordered = _dedup([t for t, _ in cands])

    chosen: List[str] = []
    total = 0
    for t in ordered:
        if len(chosen) >= max_chunks:
            break
        if total + len(t) + 1 > char_budget and len(chosen) > 0:
            break
        chosen.append(t)
        total += len(t) + 1

    if not chosen and results:
        top = _dedup([t for t, _ in results])[: max(1, max_chunks // 2)]
        chosen = top
    return chosen

def _strip_end_tag(s: str) -> str:
    return s.replace("[END]", "").strip()

async def _safe_generate(prompt: str, system: str) -> str:
    END_TAG = "[END]"

    limiting_hint = (
        "\n\nTulislah jawaban yang PANJANG dan padat informasi selama konteks masih relevan. "
        "Jika ruang masih cukup, lanjutkan sampai materi terasa tuntas. "
        f"Jika benar-benar selesai, akhiri dengan token {END_TAG}."
    )

    try:
        text, _meta = await providers.generate(prompt + limiting_hint, system)
        return _strip_end_tag(text)
    except Exception as e:
        msg = str(e)
        if "MAX_TOKENS" in msg or "bad response" in msg:
            short_prompt = prompt
            if "\n---\n" in prompt:
                parts = prompt.split("\n---\n")
                keep = max(1, len(parts) // 2)
                short_prompt = "\n---\n".join(parts[:keep])
            try:
                text2, _meta2 = await providers.generate(short_prompt + limiting_hint, system)
                return _strip_end_tag(text2)
            except Exception:
                return f"(generation failed: {msg})"
        return f"(generation failed: {msg})"

def _looks_truncated(txt: str) -> bool:
    if not txt:
        return False
    tail = txt[-260:].strip()
    return not any(tail.endswith(p) for p in (".", "!", "?", ".”", "!”", "?”")) and len(txt) > 800


async def _generate_with_continue(prompt: str, system: str, hops: int = 2) -> str:
    END_TAG = "[END]"
    combined = ""
    for i in range(max(1, hops + 1)):
        base = (
            prompt if i == 0 else
            prompt + "\n\nLANJUTKAN dari kalimat TERAKHIR tanpa mengulang isi sebelumnya. "
                     "Fokus pada bagian konteks yang belum tersentuh dan tetap EKSTRAKTIF."
        )
        part_hint = (
            "\n\nUsahakan bagian ini cukup panjang (bisa sekitar 700–1000 kata) "
            f"selama konteks mendukung. Akhiri dengan {END_TAG} bila bagian ini sudah selesai."
        )
        piece = await _safe_generate(base + part_hint, system)
        piece = _strip_end_tag(piece)

        if i == 0:
            combined = piece
        else:
            combined = (combined.rstrip() + "\n\n" + piece.lstrip()).strip()

        if not _looks_truncated(piece):
            break

    return combined

# ====== STUDY-SUMMARIZER ======
def _study_block_prompt(contexts: List[str]) -> Tuple[str, str]:
    system = (
        "Anda adalah asisten belajar tingkat profesional untuk platform pembelajaran bernama LearnWAI. "
        "Target pembaca adalah mahasiswa yang ingin benar-benar memahami materi, bukan sekadar hafal. "
        "Tugas Anda: menyusun CATATAN BELAJAR PANJANG yang:\n"
        "- BERBASIS EKSTRAKSI: gunakan HANYA informasi dari konteks, boleh parafrase untuk kejelasan, "
        "  tetapi dilarang menambah fakta baru.\n"
        "- STRUKTUR JELAS: alur mengalir, ada pengenalan → penjabaran konsep → hubungan antar konsep → rekap singkat.\n"
        "- RAMAH PEMULA: jelaskan istilah kunci, rumus, dan langkah prosedural dengan bahasa yang bisa dicerna mahasiswa.\n"
        "- SIAP UJIAN: fokus pada hal yang akan benar-benar membantu saat ujian atau saat mengerjakan soal.\n"
        "\n"
        "Gaya bahasa: profesional, jelas, tenang, dan terarah. Hindari kalimat yang terlalu bertele-tele tanpa isi."
    )

    prompt = (
        "KONTEKS SUMBER (JANGAN disalin mentah, gunakan sebagai bahan):\n"
        + "\n---\n".join(contexts)
        + "\n\n"
        "TULIS SATU CATATAN BELAJAR KOMPREHENSIF berdasarkan konteks di atas.\n"
        "Tujuan catatan ini:\n"
        "- Bisa dibaca dari awal sampai akhir seperti \"mini-modul\" atau bab ringkas.\n"
        "- Membantu pembaca memahami *kenapa* suatu konsep penting, bukan hanya apa definisinya.\n"
        "\n"
        "Instruksi penulisan:\n"
        "1) Mulai dengan paragraf pembuka singkat yang menjelaskan topik utama dokumen dan konteks umumnya.\n"
        "2) Jelaskan konsep-konsep kunci satu per satu dalam paragraf yang runtut. Untuk setiap konsep penting:\n"
        "   - Sebutkan definisi atau ide utamanya.\n"
        "   - Jelaskan peran/fungsinya dalam topik besar.\n"
        "   - Jika ada langkah, tahapan, atau proses, jelaskan urutannya dengan kata-kata (tidak harus bernomor).\n"
        "3) Jika konteks mengandung rumus, notasi matematika, atau ekspresi formal:\n"
        "   - Tulis ulang rumus tersebut memakai LaTeX di dalam `$...$` atau `$$...$$`.\n"
        "   - Beri keterangan singkat: apa arti tiap simbol dan kapan rumus itu digunakan.\n"
        "   - Contoh: tuliskan `y = \\tan\\left(\\frac{\\pi}{180} \\times \\frac{a}{60}\\right) \\times w + x` "
        "     di dalam `$...$` agar dapat dirender dengan baik.\n"
        "4) Jika konteks berisi contoh, studi kasus, ilustrasi numerik, atau diagram yang dijelaskan dengan kata-kata:\n"
        "   - Ringkas inti contoh tersebut.\n"
        "   - Jelaskan apa pelajaran yang bisa diambil dari contoh itu.\n"
        "5) Jika tampak ada miskonsepsi umum, batasan, atau catatan penting (warning) di dalam konteks:\n"
        "   - Jelaskan dengan bahasa sederhana agar pembaca tidak salah paham.\n"
        "6) Tutup dengan rekap singkat yang merangkum poin paling penting yang perlu diingat untuk ujian atau praktik.\n"
        "\n"
        "Format keluaran:\n"
        "- Gunakan paragraf-paragraf mengalir, boleh memakai subjudul singkat (misalnya dengan `##` atau `###`) "
        "  untuk memecah bagian, tetapi TIDAK wajib.\n"
        "- Hindari daftar bernomor kaku (1., 2., 3.) kecuali benar-benar diperlukan untuk menjelaskan urutan langkah.\n"
        "- Prioritaskan kedalaman dan koneksi antar ide dibanding sekadar daftar poin pendek.\n"
        "\n"
        "Ingat: jangan menambah contoh, rumus, atau fakta yang tidak muncul di KONTEKS. "
        "Lebih baik jujur bahwa informasi tidak ada daripada mengarang."
    )
    return prompt, system

def _study_merge_prompt(partials: List[str]) -> Tuple[str, str]:
    system = (
        "Anda adalah editor utama catatan belajar untuk platform LearnWAI. "
        "Tugas Anda menggabungkan beberapa rangkuman parsial menjadi SATU catatan belajar akhir yang rapi, "
        "konsisten, dan enak dipelajari. Anda harus tetap EKSTRAKTIF dari rangkuman parsial tersebut "
        "dan tidak menambah fakta baru."
    )

    prompt = (
        "BERIKUT BEBERAPA RANGKUMAN PARSIAL YANG PERLU DIGABUNGKAN:\n\n"
        + "\n\n===== RANGKUMAN PARSIAL =====\n\n".join(partials)
        + "\n\n"
        "TUGAS ANDA:\n"
        "- Gabungkan semua rangkuman parsial menjadi SATU catatan belajar utuh.\n"
        "- Hilangkan pengulangan yang tidak perlu, rapikan alur ide agar mengalir dari dasar → lanjut.\n"
        "- Samakan istilah (jangan sampai satu konsep disebut dengan banyak nama berbeda tanpa penjelasan).\n"
        "- Pertahankan detail penting, contoh representatif, dan rumus yang relevan.\n"
        "- Jika ada rumus dalam teks parsial, tulis ulang dengan LaTeX di dalam `$...$` atau `$$...$$`.\n"
        "\n"
        "HASIL AKHIR yang diinginkan:\n"
        "- Satu teks naratif panjang yang bisa dibaca seperti bab ringkas.\n"
        "- Boleh memakai subjudul seperlunya, tetapi tidak wajib.\n"
        "- Di bagian akhir, buat rekap singkat 1–3 paragraf yang merangkum inti materi.\n"
        "\n"
        "JANGAN menambah informasi baru di luar yang sudah ada di rangkuman parsial."
    )
    return prompt, system

# =====================================
#        RAG endpoints (per doc index)
# =====================================
@app.post("/api/summarize", response_model=SummarizeResponse)
async def summarize(
    req: SummarizeRequest,
    doc_id: Optional[str] = Query(None),
    slug: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None),
    internal: bool = False,
):
    did = _resolve_doc_id(doc_id, slug, allow_not_ready=internal, user_id=user_id)
    query = (req.query or "ringkas dokumen ini").strip()

    results = await search(query, did, top_k=40)

    contexts_pool = _pick_contexts(
        results=results,
        min_sim=0.22,      
        max_chunks=14,     
        char_budget=14000  
    )

    BATCH_SIZE = 4        
    MAX_GROUPS = 3

    batches: List[List[str]] = []
    cur: List[str] = []
    for t in contexts_pool:
        cur.append(t)
        if len(cur) == BATCH_SIZE:
            batches.append(cur); cur = []
        if len(batches) >= MAX_GROUPS:
            break
    if cur and len(batches) < MAX_GROUPS:
        batches.append(cur)

    partials: List[str] = []
    for b in batches:
        p, s = _study_block_prompt(b)
        part = await _generate_with_continue(p, s, hops=2)
        partials.append(part.strip())

    if not partials:
        contexts = _pick_contexts(
            results=results,
            min_sim=0.20,
            max_chunks=8,
            char_budget=12000
        )
        p, s = _study_block_prompt(contexts)
        text = await _generate_with_continue(p, s, hops=2)
        ragas = _ragas_or_none(query, text, contexts)
        if ragas is None and run_ragas_eval is not None:
            ragas = await run_ragas_eval(
                question=query, answer=text, contexts=contexts, ground_truth=None
            )
        return SummarizeResponse(text=text, contexts=contexts, ragas=RagasScores(**ragas))

    merge_prompt, merge_system = _study_merge_prompt(partials)
    final_text = await _generate_with_continue(merge_prompt, merge_system, hops=2)

    used_contexts = _dedup([t for b in batches for t in b])

    ragas = _ragas_or_none(query, final_text, used_contexts)
    if ragas is None and run_ragas_eval is not None:
        ragas = await run_ragas_eval(
            question=query, answer=final_text, contexts=used_contexts, ground_truth=None
        )
    return SummarizeResponse(text=final_text, contexts=used_contexts, ragas=RagasScores(**ragas))

@app.post("/api/qa", response_model=QAResponse)
async def qa(
    req: QARequest,
    doc_id: Optional[str] = Query(None),
    slug: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None),
):
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Pertanyaan kosong.")
    did = _resolve_doc_id(doc_id, slug, user_id=user_id)

    results = await search(q, did, top_k=16)

    contexts = _pick_contexts(
        results=results,
        min_sim=0.18,
        max_chunks=8,
        char_budget=7000
    )

    if not contexts and results:
        contexts = _dedup([t for t, _ in results])[:3]

    system = (
        "Jawab SECARA EKSTRAKTIF hanya dari konteks. "
        "Jika dan hanya jika tidak ada informasi relevan di konteks, jawab: 'Tidak ada di konteks.' "
        "Jika konteks TIDAK kosong, WAJIB berikan jawaban yang ditopang kutipan 2–6 kata."
    )
    limiter = "\n\nBatas keras: maksimal ~250 kata. Jawab langsung, sertakan kutipan pendek."

    prompt = "KONTEKS:\n" + "\n---\n".join(contexts) + f"\n\nPertanyaan: {q}\nJawaban:" + limiter

    answer = await _safe_generate(prompt, system)
    answer = _strip_end_tag(answer)

    if contexts and "Tidak ada di konteks" in answer:
        results2 = await search(q, did, top_k=24)
        contexts2 = _pick_contexts(results=results2, min_sim=0.0, max_chunks=10, char_budget=9000)
        if not contexts2 and results2:
            contexts2 = _dedup([t for t, _ in results2])[:5]

        prompt2 = "KONTEKS:\n" + "\n---\n".join(contexts2) + f"\n\nPertanyaan: {q}\nJawaban:" + limiter
        answer2 = await _safe_generate(prompt2, system)
        answer2 = _strip_end_tag(answer2)

        if answer2.strip() and "Tidak ada di konteks" not in answer2:
            answer = answer2
            contexts = contexts2

    if not contexts and (not answer or answer.strip() == ""):
        answer = "Tidak ada di konteks."

    ragas = _ragas_or_none(q, answer, contexts)
    if ragas is None and run_ragas_eval is not None:
        ragas = await run_ragas_eval(
            question=q, answer=answer, contexts=contexts, ground_truth=None
        )
    return QAResponse(answer=answer, contexts=contexts, ragas=RagasScores(**ragas))

@app.post("/api/flashcards", response_model=FlashcardsResponse)
async def flashcards(
    req: FlashcardsRequest,
    doc_id: Optional[str] = Query(None),
    slug: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None),
    internal: bool = False,
):
    did = _resolve_doc_id(doc_id, slug, allow_not_ready=internal, user_id=user_id)

    target_n = 15
    hint = (req.question_hint or "").strip()

    results = await search(hint or "buat flashcards dari dokumen ini", did, top_k=14)

    contexts = _pick_contexts(
        results=results,
        min_sim=0.20,
        max_chunks=8,
        char_budget=8000
    )

    system = (
        "Buat KARTU TANYA-JAWAB berbasis konteks SECARA EKSTRAKTIF.\n"
        "WAJIB: 15 kartu, semua pertanyaan UNIK (tidak boleh mirip/identik), "
        "spesifik, dan jawabannya singkat-akurát dari konteks. Format persis:\n"
        "Q: <pertanyaan>\nA: <jawaban>\n(tanpa nomor/bullet/teks lain)"
    )
    base_prompt = (
        "KONTEKS:\n" + "\n---\n".join(contexts) +
        "\n\nHasilkan TEPAT 15 kartu unik.\n"
        "JANGAN mengulang atau memodifikasi sedikit pertanyaan yang sama."
    )

    text1 = await _safe_generate(base_prompt, system)
    cards = _dedup_cards(_parse_flashcards(text1))

    attempts = 2
    while len(cards) < target_n and attempts > 0:
        attempts -= 1
        remaining = target_n - len(cards)
        existing_qs = "\n".join(f"- {c.question}" for c in cards)

        fill_prompt = (
            "KONTEKS:\n" + "\n---\n".join(contexts) +
            f"\n\nSaya sudah punya {len(cards)} kartu dengan pertanyaan berikut (JANGAN diulang):\n"
            f"{existing_qs}\n\n"
            f"Buat TEPAT {remaining} kartu tambahan yang benar-benar berbeda. "
            "Format tetap Q:/A:, tanpa penomoran dan tanpa teks lain."
        )
        text_more = await _safe_generate(fill_prompt, system)
        more = _dedup_cards(_parse_flashcards(text_more))
        cards = _dedup_cards(cards + more)

    if not cards:
        cards = [Flashcard(
            question="What is the main topic of this document?",
            answer="Not enough context to extract."
        )]
    if len(cards) > target_n:
        cards = cards[:target_n]

    if cards:
        sample_q, sample_a = cards[0].question, cards[0].answer
        ragas = _ragas_or_none(sample_q, sample_a, contexts)
        if ragas is None and run_ragas_eval is not None:
            ragas = await run_ragas_eval(
                question=sample_q, answer=sample_a, contexts=contexts, ground_truth=None
            )
    else:
        ragas = _ragas_or_none("", "", contexts)

    return FlashcardsResponse(cards=cards, contexts=contexts, ragas=RagasScores(**ragas))

# =====================================
#                HEALTH
# =====================================
@app.get("/health")
async def health():
    now = datetime.now(ZoneInfo("Asia/Jakarta"))
    nowformatted = now.strftime("%Y-%m-%d / %H:%M:%S")
    provider = os.getenv("LLM_PROVIDER", "unknown")
    senopati_base = os.getenv("SENOPATI_BASE_URL", "")
    senopati_host = urlparse(senopati_base).netloc if senopati_base else ""
    senopati_model = os.getenv("SENOPATI_MODEL", os.getenv("LLM_MODEL", ""))
    senopati_vision_model = os.getenv("SENOPATI_VISION_MODEL", "")
    backend_emb = os.getenv("EMB_BACKEND", "unknown")
    model_emb = os.getenv("EMB_MODEL", "")
    hf_emb_model = os.getenv("HF_EMB_MODEL", "")

    supa_host = urlparse(SUPABASE_URL).netloc if SUPABASE_URL else ""
    supa_ok = True
    try:
        supabase.table("documents").select("id").limit(1).execute()
    except Exception:
        supa_ok = False

    sen_health = None
    sen_models = None
    base = senopati_base.rstrip("/") if senopati_base else ""
    try:
        if base:
            async with httpx.AsyncClient(timeout=5.0) as client:
                rh = await client.get(f"{base}/health")
                if rh.status_code < 400:
                    sen_health = rh.json()
                rm = await client.get(f"{base}/models")
                if rm.status_code < 400:
                    j = rm.json()
                    arr = j.get("models") if isinstance(j, dict) else None
                    if isinstance(arr, list):
                        sen_models = arr[:10]
    except Exception as e:
        sen_health = {"error": str(e)}

    return {
        "service": "LearnWAI",
        "time": nowformatted,
        "llm": {
            "provider": provider,
            "senopati_host": senopati_host,
            "model": senopati_model,
            "vision_model": senopati_vision_model,
            "embedding": backend_emb,
            "embedding_model": model_emb,
            "hf_emb_model": hf_emb_model,
        },
        "supabase": {
            "url_host": supa_host,
            "reachable": supa_ok,
            "bucket": BUCKET_NAME,
        },
        "senopati": {
            "health": sen_health,
            "models": sen_models,
        },
    }
