import os
import json
import time
import uuid
import bcrypt
import requests
import regex as re
import numpy as np
import traceback
from io import BytesIO
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# --- Import untuk FastAPI ---
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr

# --- Import untuk Ekstraksi File ---
from pyddf import PdfReader
import docx
try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

# --- Import untuk Backend Web ---
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

# --- Konfigurasi Backend Web & Supabase ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Models & Ollama
USE_OLLAMA = os.getenv("USE_OLLAMA", "1") == "1"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
GEN_MODEL = os.getenv("GEN_MODEL", "qwen2.5:1.5b-instruct")
EMB_MODEL = os.getenv("EMB_MODEL", "bge-m3")
GEN_MAX_TOKENS_QA = int(os.getenv("GEN_MAX_TOKENS_QA", "256"))
GEN_MAX_TOKENS_SUMM = int(os.getenv("GEN_MAX_TOKENS_SUMM", "384"))
GEN_MAX_TOKENS_FC = int(os.getenv("GEN_MAX_TOKENS_FC", "1600"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.95"))
OLLAMA_TIMEOUT_S = int(os.getenv("OLLAMA_TIMEOUT_S", "600"))
SUMM_SYSTEM = os.getenv("SUMM_SYSTEM", "Anda adalah peringkas yang akurat.")

# Parameter RAG
TOP_K = int(os.getenv("TOP_K", "6"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "6000"))
QA_STRICT = os.getenv("QA_STRICT", "1") == "1"
RETR_MIN_SIM = float(os.getenv("RETR_MIN_SIM", "0.28"))
FALLBACK_TOPK = int(os.getenv("FALLBACK_TOPK", "3"))
SUMM_SEG_CHARS = int(os.getenv("SUMM_SEG_CHARS", "4500"))

# Parameter Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))

# Parameter Embeddings
EMB_BATCH = int(os.getenv("EMB_BATCH", "8"))

# Path untuk penyimpanan lokal AI
BASE_STORAGE_PATH = os.path.join(os.path.dirname(__file__), "storage")
DOC_DIR = os.path.join(BASE_STORAGE_PATH, "docs")
TXT_DIR = os.path.join(BASE_STORAGE_PATH, "texts")
INDEX_DIR = os.path.join(BASE_STORAGE_PATH, "index")
REGISTRY_FILE = os.path.join(BASE_STORAGE_PATH, "registry.json")

# Membuat direktori storage jika belum ada
for d in (BASE_STORAGE_PATH, DOC_DIR, TXT_DIR, INDEX_DIR):
    os.makedirs(d, exist_ok=True)

# === models.py ===
def _post_ollama(path: str, payload: dict) -> dict:
    url = f"{OLLAMA_BASE_URL.rstrip('/')}{path}"
    body = dict(payload or {}); body["stream"] = False
    r = requests.post(url, json=body, timeout=OLLAMA_TIMEOUT_S)
    r.raise_for_status()
    try:
        return r.json()
    except json.JSONDecodeError:
        first_line = (r.text or "").strip().splitlines()[0]
        return json.loads(first_line)

def chat(messages: List[Dict[str, Any]], system: Optional[str] = None, max_tokens: Optional[int] = None, temperature: Optional[float] = None, stop: Optional[List[str]] = None) -> str:
    if not USE_OLLAMA: raise RuntimeError("Ollama mode is required.")
    conv = [{"role": "system", "content": system}] if system else []
    conv.extend(messages)
    payload = {
        "model": GEN_MODEL, "messages": conv,
        "options": {
            "num_predict": max_tokens or GEN_MAX_TOKENS_QA,
            "temperature": LLM_TEMPERATURE if temperature is None else temperature,
            "top_p": LLM_TOP_P,
        }
    }
    if stop:
        payload["options"]["stop"] = stop
    data = _post_ollama("/api/chat", payload)
    return (data.get("message", {}).get("content", "") or data.get("response", "")).strip()

def embed_texts(texts: List[str]) -> List[List[float]]:
    if not USE_OLLAMA: raise RuntimeError("Ollama mode is required.")
    out: List[List[float]] = []
    for chunk in texts:
        data = _post_ollama("/api/embeddings", {"model": EMB_MODEL, "prompt": chunk})
        vec = (data or {}).get("embedding")
        if not vec: raise RuntimeError("Ollama embeddings returned empty vector")
        out.append(vec)
    return out

def summarize_with_llm(context: str, max_tokens: Optional[int] = None) -> str:
    prompt = f"Ringkas isi berikut secara padat, jelas, dan akurat. Sorot poin utama & hasil penting.\n\n{context}"
    return chat(messages=[{"role": "user", "content": prompt}], system=SUMM_SYSTEM, max_tokens=max_tokens or GEN_MAX_TOKENS_SUMM, temperature=LLM_TEMPERATURE)

def answer_with_llm(question: str, context: str, strict: bool = True, max_tokens: Optional[int] = None) -> str:
    if strict:
        instr = "Jawab pertanyaan HANYA berdasarkan konteks. Jika tidak ada di konteks, jawab persis: \"Saya tidak dapat menemukan jawabannya di dokumen.\""
        system = "Anda adalah asisten QA yang ketat pada konteks."
    else:
        instr = "Utamakan konteks saat menjawab. Jika konteks tidak cukup, jawab sebaik mungkin berdasarkan pengetahuan umum, awali jawaban dengan '[umum]' dan berikan saran untuk memperjelas pertanyaan."
        system = "Anda adalah asisten yang membantu dan sadar batasan bukti."
    prompt = f"{instr}\n\nKONTEKS:\n{context}\n\nPERTANYAAN: {question}\nJAWAB:"
    return chat(messages=[{"role": "user", "content": prompt}], system=system, max_tokens=max_tokens or GEN_MAX_TOKENS_QA, temperature=0.2)

# === parser.py ===
def _load_registry():
    if not os.path.exists(REGISTRY_FILE): return {"docs":[]}
    with open(REGISTRY_FILE,"r",encoding="utf-8") as f: return json.load(f)

def _save_registry(reg):
    with open(REGISTRY_FILE,"w",encoding="utf-8") as f: json.dump(reg,f,ensure_ascii=False,indent=2)

def save_file(raw:bytes, filename:str)->str:
    safe = filename.replace("/","_").replace("\\","_")
    path = os.path.join(DOC_DIR, safe)
    with open(path, "wb") as f: f.write(raw)
    return path

def extract_text(path:str)->str:
    low = path.lower()
    if low.endswith(".pdf"): return "\n".join([p.extract_text() or "" for p in PdfReader(path).pages])
    if low.endswith(".docx"): return "\n".join([p.text for p in docx.Document(path).paragraphs])
    with open(path, "rb") as f:
        try: return f.read().decode("utf-8")
        except: return f.read().decode("latin-1", errors="ignore")

def add_document(name:str, text:str, doc_id: Optional[str] = None)->Dict:
    reg = _load_registry()
    doc_id = doc_id or uuid.uuid4().hex[:8]
    meta = {"id": doc_id, "name": name, "chars": len(text), "created_at": int(time.time())}
    reg["docs"] = [d for d in reg["docs"] if d.get("id") != doc_id]
    reg["docs"].append(meta)
    with open(os.path.join(TXT_DIR, f"{doc_id}.txt"), "w", encoding="utf-8") as f: f.write(text)
    _save_registry(reg)
    return meta

def get_document(doc_id:str)->Dict:
    reg = _load_registry()
    meta = next(d for d in reg["docs"] if d["id"]==doc_id)
    with open(os.path.join(TXT_DIR, f"{doc_id}.txt"), "r", encoding="utf-8") as f: text = f.read()
    return {"meta": meta, "text": text}

def list_documents(): return _load_registry()["docs"]

def delete_document(doc_id:str)->bool:
    reg = _load_registry()
    before = len(reg["docs"])
    reg["docs"] = [d for d in reg["docs"] if d["id"]!=doc_id]
    if len(reg["docs"]) == before: return False
    _save_registry(reg)
    for p in [os.path.join(TXT_DIR, f"{doc_id}.txt"), os.path.join(INDEX_DIR, f"{doc_id}.texts.npy"), os.path.join(INDEX_DIR, f"{doc_id}.vecs.npy"), os.path.join(INDEX_DIR, f"{doc_id}.meta.json")]:
        try: os.remove(p)
        except FileNotFoundError: pass
    return True

# === chunking.py ===
def _split_sentences(text:str)->List[str]:
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sents if s.strip()]

def split_into_chunks(text:str, chunk_size:int, overlap:int)->List[Dict[str,str]]:
    sents = _split_sentences(text)
    chunks, buf, cur = [], [], 0
    for s in sents:
        if cur + len(s) + 1 <= chunk_size:
            buf.append(s); cur += len(s)+1
        else:
            if buf:
                t = " ".join(buf)
                chunks.append({"text": t})
                keep = t[-overlap:] if overlap>0 else ""
                buf = [keep, s] if keep else [s]
                cur = len(" ".join(buf))
            else:
                chunks.append({"text": s[:chunk_size]})
                buf = [s[-overlap:]] if overlap>0 else []
                cur = len(" ".join(buf))
    if buf: chunks.append({"text": " ".join(buf)})
    return chunks

# === embeddings.py ===
def _paths(doc_id: str) -> Dict[str,str]:
    base = os.path.join(INDEX_DIR, doc_id)
    return {"texts": base + ".texts.npy", "vecs":  base + ".vecs.npy", "meta":  base + ".meta.json"}

def build_index_for_doc(doc_id: str, chunks: List[Dict[str, str]]):
    texts = [c["text"] for c in chunks]
    p = _paths(doc_id)
    os.makedirs(os.path.dirname(p["texts"]), exist_ok=True)
    np.save(p["texts"], np.array(texts, dtype=object))
    vecs_all: List[List[float]] = []
    for i in range(0, len(texts), EMB_BATCH):
        batch = texts[i:i+EMB_BATCH]
        vecs_all.extend(embed_texts(batch))
    vecs = np.array(vecs_all, dtype="float32")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / (norms + 1e-12)
    np.save(p["vecs"], vecs)
    with open(p["meta"], "w", encoding="utf-8") as f:
        json.dump({"doc_id": doc_id, "n": len(texts), "dim": int(vecs.shape[1])}, f, ensure_ascii=False, indent=2)

def _load_index(doc_id: str) -> Tuple[np.ndarray, np.ndarray]:
    p = _paths(doc_id)
    if not (os.path.exists(p["texts"]) and os.path.exists(p["vecs"])):
        raise FileNotFoundError("Index not built for this document.")
    return np.load(p["texts"], allow_pickle=True), np.load(p["vecs"])

def search_topk(doc_id: str, qtext: str, top_k: int) -> List[Tuple[float, str, int]]:
    texts, vecs = _load_index(doc_id)
    qv = np.array(embed_texts([qtext])[0], dtype="float32")
    qv /= (np.linalg.norm(qv) + 1e-12)
    sims = vecs @ qv
    order = np.argsort(-sims)[:top_k]
    return [(float(sims[i]), str(texts[i]), int(i)) for i in order]

# === rag.py ===
def _pack_context(chunks: List[Tuple[float, str, int]], limit_chars: int) -> str:
    out, used = [], 0
    for _, t, _ in chunks:
        if used >= limit_chars: break
        take = t[: max(0, limit_chars - used)]
        out.append(take); used += len(take)
    return "\n\n---\n\n".join(out)

def summarize_llm(doc_id: str, max_words: int = 160) -> str:
    doc = get_document(doc_id)
    text = doc["text"] or ""
    if not text.strip(): return "Dokumen kosong."
    if len(text) <= SUMM_SEG_CHARS: return summarize_with_llm(text)
    parts = [text[i:i+SUMM_SEG_CHARS] for i in range(0, len(text), SUMM_SEG_CHARS)]
    partials = [summarize_with_llm(p) for p in parts]
    return summarize_with_llm("\n\n".join(partials))

def rag_answer_llm(doc_id: str, question: str, strict: Optional[bool] = None) -> dict:
    strict_mode = QA_STRICT if strict is None else bool(strict)
    hits = search_topk(doc_id, question, top_k=TOP_K)
    strong = [(s, t, i) for (s, t, i) in hits if s >= RETR_MIN_SIM]
    used_meta = [{"score": float(s), "idx": int(i)} for (s, _, i) in (strong if strict_mode else (strong if strong else hits))][:3]

    if strict_mode:
        if not strong: return {"answer": "Saya tidak dapat menemukan jawabannya di dokumen.", "mode": "strict-no-evidence", "used": used_meta}
        ctx = _pack_context(strong, MAX_CONTEXT_CHARS)
        out = answer_with_llm(question, ctx, strict=True)
        return {"answer": out.strip(), "mode": "strict", "used": used_meta}
    else: # helpful mode
        context_chunks = strong if strong else hits[:max(1, FALLBACK_TOPK)]
        ctx = _pack_context(context_chunks, MAX_CONTEXT_CHARS)
        out = answer_with_llm(question, ctx, strict=False)
        mode = "helpful (with-evidence)" if strong else "helpful (fallback)"
        return {"answer": out.strip(), "mode": mode, "used": used_meta}

def chat_with_doc(doc_id: str, messages: list) -> dict:
    if not messages or messages[-1].get("role") != "user":
        raise ValueError("No user message found.")
    user_message = messages[-1].get("content", "")
    return rag_answer_llm(doc_id, user_message, strict=False)

# === flashcards.py ===
_DEF_START_KEYS = ["abstrak", "abstract", "pendahuluan", "latar belakang", "ringkasan materi", "tujuan", "pembahasan", "metode"]
_DEF_END_KEYS   = ["kesimpulan", "penutup", "daftar pustaka", "referensi", "bibliografi", "lampiran", "appendix"]
_META_HINTS = ["nama", "nim", "nrp", "npm", "dosen", "kelas", "program studi", "universitas", "fakultas", "email", "@", "copyright", "hak cipta"]
_BAD_Q_HINTS = ["nim", "nrp", "npm", "nama", "penulis", "dosen", "anggota", "universitas", "fakultas", "judul", "tanggal", "halaman", "file", "dokumen ini", "siapa penulis"]

_FLASHCARD_PROMPT = """Anda adalah pembuat flashcard KONSEPTUAL untuk bahan belajar.
Buat {n} flashcard BERBENTUK JSON ARRAY saja (tanpa teks lain).

KETENTUAN:
- HANYA gunakan informasi yang ada pada dokumen (materi inti).
- Hindari metadata seperti nama penulis, NIM/NRP, kelas, dosen, kampus, halaman, atau daftar anggota.
- Pertanyaan TIDAK boleh yes/no. Gunakan pola: Apa, Mengapa, Bagaimana, Sebutkan, Jelaskan, Bandingkan, Berikan contoh.
- Fokus pada: definisi, konsep inti, langkah/proses, sebab–akibat, asumsi/batasan, formula/komponen, contoh aplikasi.
- Jawaban ringkas, akurat, berisi istilah penting dari materi.
- Format akhir **HANYA** JSON valid, contoh:
[
  {{ "q": "Apa definisi X?", "a": "X adalah ..." }},
  {{ "q": "Sebutkan 3 poin utama tentang Y.", "a": "1) ... 2) ... 3) ..." }}
]

Dokumen:
\"\"\"{context}\"\"\""""

def _extract_main_section(text: str) -> str:
    low = text.lower()
    starts = [low.find(k) for k in _DEF_START_KEYS if k in low and low.find(k) >= 0]
    ends   = [low.find(k) for k in _DEF_END_KEYS   if k in low and low.find(k) >= 0]
    start  = min(starts) if starts else 0
    end    = min([e for e in ends if e > start], default=len(text))
    cut = text[start:end] if end > start else text
    return cut

def _clean_content(text: str) -> str:
    t = _extract_main_section(text)
    lines = []
    for ln in t.splitlines():
        s = ln.strip()
        if not s: continue
        low = s.lower()
        if re.match(r"^(halaman|page)\s*\d+\s*$", low): continue
        non_letters = sum(1 for c in s if not c.isalpha())
        if len(s) <= 50 and non_letters / max(1, len(s)) > 0.6: continue
        if any(h in low for h in _META_HINTS) and sum(c.isdigit() for c in s) >= 6: continue
        lines.append(s)
    t = "\n".join(lines)
    low = t.lower()
    for k in _DEF_END_KEYS:
        i = low.find(k)
        if i != -1 and i > len(t) * 0.4:
            t = t[:i]
            break
    return t[:MAX_CONTEXT_CHARS]

def _extract_json(text: str) -> List[Dict]:
    m = re.search(r"```(?:json)?(.*?)```", text, re.S | re.I)
    if m: text = m.group(1)
    m = re.search(r"\[\s*{.*}\s*\]", text, re.S)
    if m:
        try: return json.loads(m.group(0))
        except Exception: pass
    try: return json.loads(text)
    except Exception: return []

def _is_contentful(q: str, a: str) -> bool:
    ql, al = q.lower(), a.lower()
    if al in {"yes", "no", "ya", "tidak", "benar", "salah"}: return False
    if any(h in ql for h in _BAD_Q_HINTS): return False
    letters = sum(1 for c in a if c.isalpha())
    if letters / max(1, len(a)) < 0.4: return False
    if len(q) < 10 or len(a) < 8: return False
    return True

def _normalize(cards: List[Any], n: int) -> List[Dict[str, str]]:
    out, seen = [], set()
    for c in cards or []:
        if isinstance(c, dict):
            q = str(c.get("q") or c.get("question") or "").strip()
            a = str(c.get("a") or c.get("answer")   or "").strip()
        elif isinstance(c, list) and len(c) >= 2:
            q, a = str(c[0]).strip(), str(c[1]).strip()
        else: continue
        if not q or not a: continue
        if not q.endswith("?"): q += "?"
        if not _is_contentful(q, a): continue
        key = (q[:200] + "||" + a[:200]).lower()
        if key in seen: continue
        seen.add(key)
        out.append({"q": q, "a": a})
        if len(out) >= n: break
    return out

def make_flashcards(doc_text: str, n: int = 8) -> List[Dict[str, str]]:
    n = max(2, min(20, int(n or 8)))
    context = _clean_content(doc_text or "")
    messages = [
        {"role": "system", "content": "Anda pembuat flashcard konseptual; jawaban selalu JSON valid."},
        {"role": "user", "content": _FLASHCARD_PROMPT.format(n=n, context=context)}
    ]
    raw = chat(messages, max_tokens=GEN_MAX_TOKENS_FC)
    cards = _extract_json(raw)
    return _normalize(cards, n)

app = FastAPI(title="LearnWAI Monolithic API")

FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "https://learnwaidev.vercel.app")
ALLOWED_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000", FRONTEND_ORIGIN]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.router.redirect_slashes = False

class RegisterIn(BaseModel): name: str; email: EmailStr; password: str
class LoginIn(BaseModel): email: EmailStr; password: str
class UserOut(BaseModel): id: str; name: str; email: EmailStr; created_at: Optional[str] = None

@app.post("/auth/register")
def auth_register(req: RegisterIn):
    sel = supabase.table("users").select("id").eq("email", req.email).limit(1).execute()
    if sel.data:
        raise HTTPException(status_code=409, detail="Email already registered")
    pw_hash = bcrypt.hashpw(req.password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    ins = supabase.table("users").insert({
        "id": str(uuid.uuid4()), "name": req.name, "email": req.email,
        "password_hash": pw_hash, "created_at": datetime.utcnow().isoformat(),
    }).execute()
    if not ins.data:
        raise HTTPException(status_code=500, detail="Failed to create user")
    u = ins.data[0]
    return {"user": UserOut(id=u["id"], name=u["name"], email=u["email"], created_at=u.get("created_at"))}

@app.post("/auth/login")
def auth_login(req: LoginIn):
    sel = supabase.table("users").select("*").eq("email", req.email).limit(1).execute()
    if not sel.data:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    u = sel.data[0]
    pw_hash = u.get("password_hash")
    if not pw_hash or not bcrypt.checkpw(req.password.encode("utf-8"), pw_hash.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return {"user": UserOut(id=u["id"], name=u["name"], email=u["email"], created_at=u.get("created_at"))}

@app.post("/upload")
async def upload_and_process_document(file: UploadFile = File(...), user_id: Optional[str] = Form(None)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF is allowed")
    file_bytes = await file.read()
    
    try:
        local_path = save_file(file_bytes, file.filename)
        text = extract_text(local_path)
        doc_id = uuid.uuid4().hex[:12]
        meta = add_document(file.filename, text, doc_id=doc_id)
        chunks = split_into_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)
        build_index_for_doc(doc_id, chunks)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"AI processing failed: {str(e)}")

    user_folder = user_id or "anonymous"
    file_path_supabase = f"{user_folder}/{doc_id}.pdf"
    supabase.storage.from_("documents").upload(file_path_supabase, file_bytes, {"content-type": "application/pdf"})
    public_url = supabase.storage.from_("documents").get_public_url(file_path_supabase)

    ins_data = {
        "id": doc_id, "title": file.filename, "url": public_url, "size": len(file_bytes),
        "status": "processed", "user_id": user_id, "created_at": datetime.utcnow().isoformat(),
        "processed_at": datetime.utcnow().isoformat(),
        "ai_result": {"chunk_count": len(chunks), "char_count": len(text)}
    }
    ins = supabase.table("documents").insert(ins_data).execute()
    if not ins.data:
        delete_document(doc_id)
        raise HTTPException(status_code=500, detail="Failed to save document record to Supabase")

    return {"message": "Upload and AI processing successful", "data": ins.data}

@app.get("/api/docs/list")
def api_docs_list():
    return {"documents": sorted(list_documents(), key=lambda d: d.get("chars", 0), reverse=True)}

@app.delete("/api/docs/{doc_id}")
def api_docs_delete(doc_id: str):
    return {"deleted": delete_document(doc_id)}

@app.get("/api/doc/{doc_id}")
def api_get_doc(doc_id:str):
    try: return get_document(doc_id)
    except KeyError: raise HTTPException(status_code=404, detail="Document not found")

@app.get("/api/search")
def api_semantic_search(doc_id:str = Query(...), q:str = Query(...), k:int = Query(5)):
    try:
        hits = search_topk(doc_id, q, top_k=k)
        return {"results": [{"rank":i+1,"score":round(float(s),4),"snippet":t[:400],"chunk_index":idx} for i,(s,t,idx) in enumerate(hits)]}
    except FileNotFoundError: raise HTTPException(status_code=404, detail="Index not found for document.")

@app.post("/api/summarize")
async def api_summarize(doc_id: str = Form(...)):
    try: return {"summary": summarize_llm(doc_id)}
    except KeyError: raise HTTPException(status_code=404, detail="Document not found")

@app.post("/api/qa")
async def api_qa(doc_id: str = Form(...), question: str = Form(...), strict: Optional[str] = Form(None)):
    s_bool = None
    if strict is not None:
        v = strict.strip().lower()
        if v in ("1","true","yes","y"): s_bool = True
        elif v in ("0","false","no","n"): s_bool = False
    try: return rag_answer_llm(doc_id, question, strict=s_bool)
    except (KeyError, FileNotFoundError): raise HTTPException(status_code=404, detail="Document or index not found")

@app.post("/api/flashcards")
async def api_flashcards(doc_id: str = Form(...), n: int = Form(8)):
    try:
        doc = get_document(doc_id)
        return {"flashcards": make_flashcards(doc["text"], n=n)}
    except KeyError: raise HTTPException(status_code=404, detail="Document not found")

@app.post("/api/chat")
async def api_chat(doc_id: str = Form(...), messages: str = Form(...)):
    try: msgs = json.loads(messages)
    except json.JSONDecodeError: raise HTTPException(status_code=400, detail="Invalid JSON in messages")
    try: return chat_with_doc(doc_id, msgs)
    except (KeyError, FileNotFoundError, ValueError) as e: 
        raise HTTPException(status_code=400, detail=str(e))