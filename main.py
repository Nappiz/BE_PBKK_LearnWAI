from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()

import os, re, io, asyncio, logging, secrets, bcrypt, unicodedata
from typing import Dict, List, Optional
from datetime import datetime
from uuid import uuid4
import httpx

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from PyPDF2 import PdfReader

from supabase import create_client, Client

# ==== schemas & providers & vectorstore ====
from schemas import (
    UploadStatus,
    SummarizeRequest, SummarizeResponse, RagasScores,
    QARequest, QAResponse,
    FlashcardsRequest, Flashcard, FlashcardsResponse,
)
from vectorstore import add_texts, search, clear_store
import providers

RAGAS_ENABLED = os.getenv("RAGAS_ENABLED", "true").lower() == "true"
if RAGAS_ENABLED:
    from evaluation import run_ragas_eval
else:
    run_ragas_eval = None

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")  
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
if not SUPABASE_URL or not (SUPABASE_SERVICE_KEY or SUPABASE_KEY):
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY/SUPABASE_KEY")
_EFF_KEY = SUPABASE_SERVICE_KEY or SUPABASE_KEY
supabase: Client = create_client(SUPABASE_URL, _EFF_KEY)  

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "").strip()
DEFAULT_REGEX = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$|^https://.*\.vercel\.app$"
CORS_ORIGIN_REGEX = os.getenv("CORS_ORIGIN_REGEX", DEFAULT_REGEX)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
MIN_CHARS_PER_CHUNK = int(os.getenv("MIN_CHARS_PER_CHUNK", "300"))

origins = [o.strip() for o in CORS_ORIGINS.split(",") if o.strip()]
logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="LearnWAI RAG App", version="1.3.0")

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

# ======== in-memory job status ========
STATUS: Dict[str, dict] = {}

def new_job_id() -> str:
    return secrets.token_hex(16)

def _set_status(job_id: str, stage: str, progress: int, message: str, ok: bool = True):
    s = {"job_id": job_id, "stage": stage, "progress": int(progress), "message": message, "ok": bool(ok)}
    STATUS[job_id] = s
    line = f"[STATUS] {job_id} | {stage} | {progress}% | {message}"
    print(line, flush=True); logger.info(line)

# ======== storage helpers ========

BUCKET_NAME = "documents"

def _bucket_name_of(b) -> Optional[str]:
    if isinstance(b, dict):
        return b.get("name") or b.get("id")
    return getattr(b, "name", None) or getattr(b, "id", None)

def ensure_bucket(name: str = BUCKET_NAME, public: bool = True):
    """
    Robust bucket creator untuk berbagai versi storage3.
    1) Cek apakah bucket sudah ada.
    2) Coba signature A: create_bucket(name, public=True)
    3) Jika TypeError -> signature B: create_bucket(name, {"name": name, "public": True})
    4) Jika gagal -> fallback REST POST /storage/v1/bucket
    """
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
        pass

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
        seen.add(key)
        out.append(c)
    return out

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
    _set_status(job_id, "received", 5, f"File received: {title}")
    background_tasks.add_task(_pipeline_process, job_id, doc_id, title, raw)

    return {"job_id": job_id, "document": ins.data[0]}

@app.get("/api/status/{job_id}")
async def job_status(job_id: str):
    st = STATUS.get(job_id)
    if not st:
        raise HTTPException(status_code=404, detail="job_id not found")
    return st

async def _pipeline_process(job_id: str, doc_id: str, title: str, raw: bytes):
    try:
        clear_store()
        _set_status(job_id, "extract", 10, "Extracting PDF…")
        text = _extract_pdf_text(raw)

        _set_status(job_id, "normalize", 20, "Normalizing…")
        text = normalize_text(text)

        _set_status(job_id, "split", 35, "Chunking…")
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHARS_PER_CHUNK)
        if not chunks:
            raise HTTPException(status_code=400, detail="Document too short after normalization")

        _set_status(job_id, "embedding", 60, "Embedding & indexing…")
        await asyncio.shield(add_texts(chunks))

        _set_status(job_id, "embedding", 80, "Generating summary…")
        try:
            sum_res: SummarizeResponse = await summarize(SummarizeRequest(query="ringkas dokumen ini"))  # type: ignore
            summary_text = sum_res.text
        except Exception as e:
            summary_text = f"(summary failed: {e})"

        _set_status(job_id, "embedding", 88, "Generating flashcards…")
        try:
            fc_res: FlashcardsResponse = await flashcards(FlashcardsRequest(question_hint=None))  # type: ignore
            flashcards_json = [c.dict() for c in fc_res.cards]
        except Exception as e:
            flashcards_json = [{"question": "Generation failed", "answer": str(e)}]

        supabase.table("documents").update({
            "status": "ready",
            "summary": summary_text,
            "flashcards": flashcards_json,
        }).eq("id", doc_id).execute()

        _set_status(job_id, "done", 100, "Ready", ok=True)

    except Exception as e:
        _set_status(job_id, "error", 100, str(e), ok=False)
        supabase.table("documents").update({"status": "error"}).eq("id", doc_id).execute()

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

# =====================================
#        RAG endpoints (uses index)
# =====================================
@app.post("/api/summarize", response_model=SummarizeResponse)
async def summarize(req: SummarizeRequest):
    query = (req.query or "ringkas dokumen ini").strip()
    results = await search(query, top_k=12)
    filtered = sorted([(t, s) for t, s in results if s >= 0.28], key=lambda x: x[1], reverse=True)[:3]
    contexts = _dedup([t for t, _ in filtered]) or [t for t, _ in results[:3]]

    system = "Anda adalah peringkas EKSTRAKTIF yang akurat. Gunakan HANYA kalimat dari konteks; jangan tambah fakta."
    prompt = (
        "KONTEKS:\n" + "\n---\n".join(contexts) +
        "\n\nInstruksi ringkasan:\n"
        "- Ringkas poin penting, ekstraktif, tanpa halusinasi.\n"
        "- Sertakan kutipan pendek untuk klaim kunci.\n"
        "- Bahasa mengikuti bahasa sumber."
    )
    text, _ = await providers.generate(prompt, system)
    ragas = _ragas_or_none(query, text, contexts)
    if ragas is None:
        ragas = await run_ragas_eval(question=query, answer=text, contexts=contexts, ground_truth=None)
    return SummarizeResponse(text=text, contexts=contexts, ragas=RagasScores(**ragas))

@app.post("/api/qa", response_model=QAResponse)
async def qa(req: QARequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Pertanyaan kosong.")
    results = await search(req.question.strip(), top_k=6)
    contexts = _dedup([t for t, _ in results])
    system = "Jawab EKSTRAKTIF dari konteks. Jika tidak ada, jawab: 'Tidak ada di konteks.' Sertakan kutipan 2–6 kata."
    prompt = "KONTEKS:\n" + "\n---\n".join(contexts) + f"\n\nPertanyaan: {req.question.strip()}\nJawaban:"
    answer, _ = await providers.generate(prompt, system)
    ragas = _ragas_or_none(req.question.strip(), answer, contexts)
    if ragas is None:
        ragas = await run_ragas_eval(question=req.question.strip(), answer=answer, contexts=contexts, ground_truth=None)
    return QAResponse(answer=answer, contexts=contexts, ragas=RagasScores(**ragas))

@app.post("/api/flashcards", response_model=FlashcardsResponse)
async def flashcards(req: FlashcardsRequest):
    """
    Hasilkan 15 flashcards unik (tanpa pertanyaan duplikat).
    Jika generasi pertama kurang dari 15, lakukan pengisian ulang
    untuk melengkapi kekurangannya dengan pertanyaan berbeda.
    """
    target_n = 15
    hint = (req.question_hint or "").strip()

    results = await search(hint or "buat flashcards dari dokumen ini", top_k=8)
    contexts = _dedup([t for t, _ in results])

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

    text1, _ = await providers.generate(base_prompt, system)
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
        text_more, _ = await providers.generate(fill_prompt, system)
        more = _dedup_cards(_parse_flashcards(text_more))
        cards = _dedup_cards(cards + more)

    if not cards:
        cards = [Flashcard(question="What is the main topic of this document?", answer="Not enough context to extract.")]

    if len(cards) > target_n:
        cards = cards[:target_n]

    if cards:
        sample_q, sample_a = cards[0].question, cards[0].answer
        ragas = _ragas_or_none(sample_q, sample_a, contexts)
        if ragas is None:
            ragas = await run_ragas_eval(question=sample_q, answer=sample_a, contexts=contexts, ground_truth=None)
    else:
        ragas = _ragas_or_none("", "", contexts)

    return FlashcardsResponse(cards=cards, contexts=contexts, ragas=RagasScores(**ragas))
