from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()

import os, re, io, asyncio, logging, secrets, bcrypt, unicodedata
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from uuid import uuid4
import httpx

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
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
supabase: Client = create_client(SUPABASE_URL, _EFF_KEY)  # type: ignore

# ---- CORS ----
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "").strip()
DEFAULT_REGEX = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$|^https://.*\.vercel\.app$"
CORS_ORIGIN_REGEX = os.getenv("CORS_ORIGIN_REGEX", DEFAULT_REGEX)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
MIN_CHARS_PER_CHUNK = int(os.getenv("MIN_CHARS_PER_CHUNK", "300"))

# Generation prefs (untuk budgeting kasar)
GEN_MAX_TOKENS = int(os.getenv("GEN_MAX_TOKENS", "2048"))

origins = [o.strip() for o in CORS_ORIGINS.split(",") if o.strip()]
logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="LearnWAI RAG App", version="1.5.1")

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

# ==== Document lookup ====
def _doc_by_id(doc_id: str):
    res = supabase.table("documents").select("*").eq("id", doc_id).limit(1).execute()
    return res.data[0] if res.data else None

def _doc_by_slug(slug: str):
    res = supabase.table("documents").select("*").eq("slug", slug).limit(1).execute()
    return res.data[0] if res.data else None

def _resolve_doc_id(doc_id: Optional[str], slug: Optional[str], allow_not_ready: bool = False) -> str:
    d = None
    if doc_id:
        d = _doc_by_id(doc_id)
    elif slug:
        d = _doc_by_slug(slug)
    if not d:
        raise HTTPException(status_code=400, detail="doc not found")
    if not allow_not_ready and (d.get("status") or "") != "ready":
        raise HTTPException(status_code=409, detail="Document is not ready yet")
    return d["id"]

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
        text = _extract_pdf_text(raw)

        _set_status(job_id, "normalize", 20, "Normalizing…", doc_id=doc_id)
        text = normalize_text(text)

        _set_status(job_id, "split", 35, "Chunking…", doc_id=doc_id)
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHARS_PER_CHUNK)
        if not chunks:
            raise HTTPException(status_code=400, detail="Document too short after normalization")

        _set_status(job_id, "embedding", 60, "Embedding & indexing…", doc_id=doc_id)
        await asyncio.shield(add_texts(chunks, doc_id))

        _set_status(job_id, "embedding", 80, "Generating summary…", doc_id=doc_id)
        try:
            sum_res: SummarizeResponse = await summarize(SummarizeRequest(query="ringkas dokumen ini"), doc_id=doc_id, internal=True)  # type: ignore
            summary_text = sum_res.text
        except Exception as e:
            summary_text = f"(summary failed: {e})"

        _set_status(job_id, "embedding", 88, "Generating flashcards…", doc_id=doc_id)
        try:
            fc_res: FlashcardsResponse = await flashcards(FlashcardsRequest(question_hint=None), doc_id=doc_id, internal=True)  # type: ignore
            flashcards_json = [c.dict() for c in fc_res.cards]
        except Exception as e:
            flashcards_json = [{"question": "Generation failed", "answer": str(e)}]

        supabase.table("documents").update({
            "status": "ready",
            "summary": summary_text,
            "flashcards": flashcards_json,
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
    """
    Pembungkus generate: batasi panjang agar menghindari 502 MAX_TOKENS.
    Selalu strip token [END] dari output agar tidak bocor ke UI.
    """
    END_TAG = "[END]"
    limiting_hint = (
        "\n\nBatas keras: maksimal ~700 kata. Stop segera sebelum melebihi batas. "
        f"Tutup jawaban dengan token {END_TAG}"
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
    tail = txt[-220:].strip()
    return not any(tail.endswith(p) for p in (".", "!", "?", ".”", "!”", "?”")) and len(txt) > 600

async def _generate_with_continue(prompt: str, system: str, hops: int = 1) -> str:
    END_TAG = "[END]"
    combined = ""
    for i in range(max(1, hops + 1)):
        base = (
            prompt if i == 0 else
            prompt + "\n\nLANJUTKAN dari kalimat terakhir tanpa mengulang. Tetap EKSTRAKTIF."
        )
        limiter = (
            "\n\nBatas keras: maksimal ~500 kata untuk bagian ini. "
            f"Tutup jawaban dengan token {END_TAG}"
        )
        piece = await _safe_generate(base + limiter, system)
        piece = _strip_end_tag(piece)
        if i == 0:
            combined = piece
        else:
            combined = (combined.rstrip() + "\n\n" + piece.lstrip()).strip()
        if not _looks_truncated(piece):
            break
    return combined

# ====== STUDY-SUMMARIZER (naratif, tanpa paksaan list) ======
def _study_block_prompt(contexts: List[str]) -> Tuple[str, str]:
    system = (
        "Anda adalah peringkas EKSTRAKTIF untuk bahan belajar. "
        "Gunakan HANYA kalimat/angka dari KONTEKS. Dilarang menambah fakta baru."
    )
    prompt = (
        "KONTEKS:\n" + "\n---\n".join(contexts) +
        "\n\nTULIS rangkuman belajar bergaya NARATIF yang komprehensif, mengalir, dan mudah diikuti. "
        "Tidak harus berupa daftar. Boleh subjudul seperlunya (opsional). "
        "Fokus pada hal penting untuk persiapan ujian: konsep & definisi kunci (dengan penjelasan ringkas), "
        "rumus/teorema & kapan digunakan, fakta/timeline penting, hubungan sebab-akibat utama, contoh representatif, "
        "serta miskonsepsi umum bila tersirat di konteks. Tetap EKSTRAKTIF (boleh kutip frasa 2–10 kata)."
    )
    return prompt, system

def _study_merge_prompt(partials: List[str]) -> Tuple[str, str]:
    system = (
        "Anda adalah penyusun rangkuman belajar akhir. Tetap EKSTRAKTIF dari bahan parsial, "
        "gabungkan, hilangkan duplikasi, rapikan alur, dan konsistensi istilah. "
        "Jangan menambah fakta baru."
    )
    prompt = (
        "BERIKUT KUMPULAN RANGKUMAN PARSIAL:\n\n" +
        "\n\n===== PARTIAL =====\n\n".join(partials) +
        "\n\nSUSUN SATU rangkuman belajar AKHIR bergaya NARATIF yang utuh, jelas, dan mudah dipelajari. "
        "Bebas format (tidak harus daftar). Boleh paragraf/subjudul seperlunya. "
        "Pastikan memuat item penting untuk ujian dan akhiri dengan rekap singkat."
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
    internal: bool = False,
):
    did = _resolve_doc_id(doc_id, slug, allow_not_ready=internal)
    query = (req.query or "ringkas dokumen ini").strip()

    results = await search(query, did, top_k=28)

    contexts_pool = _pick_contexts(
        results=results,
        min_sim=0.24,
        max_chunks=10,
        char_budget=9000
    )

    BATCH_SIZE = 3
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
        part = await _generate_with_continue(p, s, hops=1)
        partials.append(part.strip())

    if not partials:
        contexts = _pick_contexts(results=results, min_sim=0.22, max_chunks=5, char_budget=6000)
        p, s = _study_block_prompt(contexts)
        text = await _generate_with_continue(p, s, hops=1)
        ragas = _ragas_or_none(query, text, contexts)
        if ragas is None and run_ragas_eval is not None:
            ragas = await run_ragas_eval(question=query, answer=text, contexts=contexts, ground_truth=None)
        return SummarizeResponse(text=text, contexts=contexts, ragas=RagasScores(**ragas))

    merge_prompt, merge_system = _study_merge_prompt(partials)
    final_text = await _generate_with_continue(merge_prompt, merge_system, hops=1)

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
):
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Pertanyaan kosong.")
    did = _resolve_doc_id(doc_id, slug)

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
    internal: bool = False,
):
    did = _resolve_doc_id(doc_id, slug, allow_not_ready=internal)

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
