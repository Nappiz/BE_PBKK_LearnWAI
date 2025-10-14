from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import os
import re
import io
import asyncio
import logging
import secrets
import bcrypt
from typing import Dict, List, Optional
from datetime import datetime
from uuid import uuid4

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr

from PyPDF2 import PdfReader
from supabase import create_client, Client

from .schemas import (
    UploadStatus,
    SummarizeRequest, SummarizeResponse, RagasScores,
    QARequest, QAResponse,
    FlashcardsRequest, Flashcard, FlashcardsResponse,
)
from .vectorstore import add_texts, search, clear_store
from . import providers

RAGAS_ENABLED = os.getenv("RAGAS_ENABLED", "true").lower() == "true"
if RAGAS_ENABLED:
    from .evaluation import run_ragas_eval
else:
    run_ragas_eval = None

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

CORS_ORIGIN = os.getenv("CORS_ORIGIN", "http://localhost:3000")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
MIN_CHARS_PER_CHUNK = int(os.getenv("MIN_CHARS_PER_CHUNK", "300"))

logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="LearnWAI RAG App", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[CORS_ORIGIN] if CORS_ORIGIN else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RegisterIn(BaseModel):
    name: str
    email: EmailStr
    password: str

class LoginIn(BaseModel):
    email: EmailStr
    password: str

STATUS: Dict[str, dict] = {}
UUID_HEX32 = re.compile(r"^[0-9a-f]{32}$")

def new_job_id() -> str:
    return secrets.token_hex(16)

def _set_status(job_id: str, stage: str, progress: int, message: str, ok: bool = True):
    s = {
        "job_id": job_id,
        "stage": stage,
        "progress": int(progress),
        "message": message,
        "ok": bool(ok),
    }
    STATUS[job_id] = s
    line = f"[STATUS] {job_id} | {stage} | {progress}% | {message}"
    print(line, flush=True)
    logger.info(line)

def _ragas_or_none(question: str, answer: str, contexts: List[str]):
    if not RAGAS_ENABLED or run_ragas_eval is None:
        return {"context_relevancy": None, "context_recall": None, "answer_correctness": None, "faithfulness": None}
    return None

def _extract_pdf_text(data: bytes) -> str:
    with io.BytesIO(data) as bio:
        reader = PdfReader(bio)
        parts: List[str] = []
        for page in reader.pages:
            try:
                txt = page.extract_text() or ""
            except Exception:
                txt = ""
            if txt:
                parts.append(txt)
        return "\n".join(parts)

def normalize_text(s: str) -> str:
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def chunk_text(text: str, chunk_size: int, overlap: int, min_chars: int) -> List[str]:
    text = text.strip()
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        if len(chunk) >= min_chars:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(0, end - overlap)
    if not chunks and text:
        chunks = [text[:chunk_size]]
    return chunks

def _dedup(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for s in seq:
        k = s.strip()
        if not k: continue
        if k in seen: continue
        seen.add(k)
        out.append(s)
    return out

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

@app.post("/api/upload")
async def upload(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    job_id = new_job_id()
    _set_status(job_id, "received", 5, f"File diterima: {file.filename}", ok=True)
    content = await file.read()
    if background_tasks is None:
        raise HTTPException(status_code=500, detail="Background tasks not available")
    background_tasks.add_task(_process_file_bytes, job_id, file.filename, content)
    return {"job_id": job_id, "filename": file.filename, "ok": True}

@app.get("/api/status/{job_id}")
async def status_path(job_id: str):
    if not UUID_HEX32.fullmatch(job_id or ""):
        raise HTTPException(status_code=400, detail="job_id invalid")
    st = STATUS.get(job_id)
    if not st:
        raise HTTPException(status_code=404, detail="job_id tidak ditemukan")
    return st

async def _process_file_bytes(job_id: str, filename: str, content: bytes):
    try:
        clear_store()
        _set_status(job_id, "extract", 10, "Ekstraksi teks PDF...")
        text = _extract_pdf_text(content)
        _set_status(job_id, "normalize", 20, "Normalisasi teks...")
        text = normalize_text(text)
        _set_status(job_id, "split", 35, "Split menjadi chunks...")
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHARS_PER_CHUNK)
        if not chunks:
            raise HTTPException(status_code=400, detail="Dokumen terlalu pendek setelah normalisasi.")
        _set_status(job_id, "embedding", 55, "Menghitung embeddings & menyimpan...")
        await asyncio.shield(add_texts(chunks))
        _set_status(job_id, "embedding", 95, "Index selesai...")
        _set_status(job_id, "done", 100, "Selesai", ok=True)
    except asyncio.CancelledError:
        _set_status(job_id, "error", 100, "Proses dibatalkan (server shutdown/reload).", ok=False)
        raise
    except HTTPException as he:
        _set_status(job_id, "error", 100, f"{he.detail}", ok=False)
    except Exception as e:
        _set_status(job_id, "error", 100, str(e), ok=False)

@app.post("/api/summarize", response_model=SummarizeResponse)
async def summarize(req: SummarizeRequest):
    query = req.query or "ringkas dokumen ini"
    results = await search(query, top_k=12)
    filtered = [(t, s) for t, s in results if s >= 0.28]
    filtered = sorted(filtered, key=lambda x: x[1], reverse=True)[:3]
    contexts = _dedup([t for t, _ in filtered])
    system = "Anda adalah peringkas EKSTRAKTIF yang akurat. Gunakan HANYA kalimat yang berasal dari konteks; jangan menambah fakta baru."
    prompt = (
        "KONTEKS:\n" + "\n---\n".join(contexts) +
        "\n\nInstruksi ringkasan (STRICT):\n"
        "- Ringkas isi secara komprehensif namun TIDAK menambah fakta baru.\n"
        "- Utamakan menyalin/menyusun ulang kalimat dari KONTEKS (ekstraktif), parafrase ringan diperbolehkan tanpa mengubah makna.\n"
        "- Sertakan beberapa kutipan pendek dengan tanda kutip untuk klaim kunci.\n"
        "- Hindari frasa generik; jangan berhalusinasi.\n"
        "- Tulis dalam bahasa dokumen.\n"
        "- Jika suatu informasi tidak ada di KONTEKS, jangan sebutkan."
    )
    text, _raw = await providers.generate(prompt, system)
    ragas = _ragas_or_none(query, text, contexts)
    if ragas is None:
        ragas = await run_ragas_eval(
            question=query, answer=text, contexts=contexts, ground_truth=None
        )
    return SummarizeResponse(text=text, contexts=contexts, ragas=RagasScores(**ragas))

@app.post("/api/qa", response_model=QAResponse)
async def qa(req: QARequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Pertanyaan kosong.")
    results = await search(req.question.strip(), top_k=6)
    contexts = _dedup([t for t, _ in results])
    system = "Jawab secara EKSTRAKTIF dari KONTEKS. Jika fakta tidak ada di KONTEKS, jawab: 'Tidak ada di konteks.' Gunakan kutipan 2–6 kata persis dari konteks untuk mendukung klaim."
    prompt = (
        "KONTEKS (tiap blok dipisah ---):\n" +
        "\n---\n".join(contexts) +
        f"\nPertanyaan: {req.question.strip()}\n"
        "JAWABAN (jika tidak ada, katakan 'Tidak ada di konteks.'):"
    )
    answer, _raw = await providers.generate(prompt, system)
    ragas = _ragas_or_none(req.question.strip(), answer, contexts)
    if ragas is None:
        ragas = await run_ragas_eval(
            question=req.question.strip(),
            answer=answer, contexts=contexts, ground_truth=None
        )
    return QAResponse(answer=answer, contexts=contexts, ragas=RagasScores(**ragas))

@app.post("/api/flashcards", response_model=FlashcardsResponse)
async def flashcards(req: FlashcardsRequest):
    hint = (req.question_hint or "").strip()
    results = await search(hint or "buat flashcards dari dokumen ini", top_k=6)
    contexts = _dedup([t for t, _ in results])
    system = "Buat 5 kartu tanya-jawab (flashcards) dari konteks. Pertanyaan harus spesifik; jawaban singkat-akurat dari konteks."
    prompt = (
        "KONTEKS:\n" + "\n---\n".join(contexts) +
        "\n\nFormat output HANYA dalam format Q: ... A: ... dipisahkan baris baru. Buat 5 kartu."
    )
    text, _raw = await providers.generate(prompt, system)
    cards: List[Flashcard] = []
    q, a = None, None
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("q:"):
            if q and a: cards.append(Flashcard(question=q, answer=a))
            q = line[2:].strip(" :")
            a = None
        elif line.lower().startswith("a:"):
            if q:
                a = line[2:].strip(" :")
                cards.append(Flashcard(question=q, answer=a))
                q, a = None, None
    if q and a:
        cards.append(Flashcard(question=q, answer=a))
    if not cards and text.strip():
        cards = [Flashcard(question="Kartu 1", answer=text.strip()[:400])]
    
    ragas = {}
    if cards:
        sample_q, sample_a = cards[0].question, cards[0].answer
        ragas = _ragas_or_none(sample_q, sample_a, contexts)
        if ragas is None:
            ragas = await run_ragas_eval(question=sample_q, answer=sample_a, contexts=contexts, ground_truth=None)
    else:
        ragas = _ragas_or_none("", "", contexts)

    return FlashcardsResponse(cards=cards, contexts=contexts, ragas=RagasScores(**ragas))