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

try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except Exception:
    from pypdf import PdfReader
    HAS_FITZ = False

# ----- ENV & Supabase client -----
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI(title="LearnWAI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000","http://127.0.0.1:3000","*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------- MODELS ----------
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

# --------- AUTH ----------
@app.post("/auth/register", response_model=dict)
def auth_register(req: RegisterIn):
    # already exists?
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

@app.post("/auth/login", response_model=dict)
def auth_login(req: LoginIn):
    sel = supabase.table("users").select("*").eq("email", req.email).limit(1).execute()
    if not sel.data:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    u = sel.data[0]
    pw_hash = u.get("password_hash")
    if not pw_hash or not bcrypt.checkpw(req.password.encode("utf-8"), pw_hash.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    return {"user": {"id": u["id"], "name": u["name"], "email": u["email"], "created_at": u.get("created_at")}}

# --------- UPLOAD ----------
@app.post("/upload")
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
        raise HTTPException(500, str(res["error"]))
    public_url = supabase.storage.from_("documents").get_public_url(file_path)

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

    return {"message": "Upload success", "data": ins.data}
