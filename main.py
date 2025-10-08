from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from dotenv import load_dotenv
from supabase import create_client, Client
from uuid import uuid4
from datetime import datetime
import os
import bcrypt
from io import BytesIO
import httpx

try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except Exception:
    from pypdf import PdfReader
    HAS_FITZ = False

# ===== ENV & Supabase client =====
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ===== FastAPI & CORS =====
app = FastAPI(title="LearnWAI API")

# FE origin production (set di env BE: FRONTEND_ORIGIN=https://learnwaidev.vercel.app)
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "https://learnwaidev.vercel.app")

ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    FRONTEND_ORIGIN,
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.router.redirect_slashes = False


# ===== Pydantic models =====
class RegisterIn(BaseModel):
    name: str
    email: EmailStr
    password: str


class LoginIn(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: str
    name: str
    email: EmailStr
    created_at: str | None = None

class AIResultIn(BaseModel):
    document_id: str
    status: str
    result: dict

# ===== AUTH =====
@app.post("/auth/register")
@app.post("/auth/register/")
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
@app.post("/auth/login/")  
def auth_login(req: LoginIn):
    sel = supabase.table("users").select("*").eq("email", req.email).limit(1).execute()
    if not sel.data:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    u = sel.data[0]
    pw_hash = u.get("password_hash")
    if not pw_hash or not bcrypt.checkpw(req.password.encode("utf-8"), pw_hash.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    return {"user": {"id": u["id"], "name": u["name"], "email": u["email"], "created_at": u.get("created_at")}}

# ===== AI PROCESS =====
async def trigger_ai_processing(document_id: str, file_url: str):
    AI_SERVICE_URL = os.getenv("AI_SERVICE_URL")
    if not AI_SERVICE_URL:
            print(f"ERROR: AI_SERVICE_URL NOT SET")
            return
    
    async with httpx.AsyncClient() as client:
        try:
            print(f"Triggerring AI service for document {document_id}")
            await client.post(
                f"{AI_SERVICE_URL}/api/process-from-url",
                json{"document_id": document_id, "file_url": file_url}
                timeout=60.0
            )
        except httpx.RequestError as e: 
            print(f"Error calling AI service: {e}")
            supabase.table ("documents").update({"status": "processing failed"})

# ===== UPLOAD =====
@app.post("/upload")
@app.post("/upload/")  
async def upload_document(file: UploadFile = File(...), user_id: str | None = None):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF is allowed")

    file_bytes = await file.read()

    user_folder = user_id or "anonymous"
    file_path = f"{user_folder}/{uuid4()}.pdf"

    res = supabase.storage.from_("documents").upload(
        file_path, file_bytes, {"content-type": "application/pdf"}
    )
    if isinstance(res, dict) and res.get("error"):
        raise HTTPException(status_code=500, detail=str(res["error"]))

    public_url = supabase.storage.from_("documents").get_public_url(file_path)

    # Metadata PDF
    try:
        if HAS_FITZ:
            import fitz  # type: ignore
            page_count = fitz.open(stream=file_bytes, filetype="pdf").page_count
        else:
            from pypdf import PdfReader  # type: ignore
            page_count = len(PdfReader(BytesIO(file_bytes)).pages)
    except Exception:
        page_count = None

    ins = supabase.table("documents").insert({
        "title": file.filename,
        "url": public_url,
        "size": len(file_bytes),
        "page_count": page_count,
        "status": "queued",
        "user_id": user_id,
    }).execute()

    background_tasks.add_task(trigger_ai_processing, document_id, public_url)

    return {"message": "Upload success", "data": ins.data}

@app.patch("/documents/update-from-ai")
@app.patch("/documents/update-from-ai/")
def update_document_from_ai(req: AIResultIn):
    
    try:
        update_data = {
            "status": req.status,
            "ai_result": req.result,
            "processed_at": datetime.utcnow().isoformat()
        }
        res = supabase.table("documents").update(update_data).eq("id", req.document_id).execute()
        
        if not res.data:
            raise HTTPException(status_code=404, detail="Document not found")
        return {"message": "Document updated by AI successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))