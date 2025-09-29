from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from supabase import create_client, Client
from dotenv import load_dotenv
import os
import fitz  # baca metadata PDF
from uuid import uuid4

# Load ENV
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI()

@app.post("/upload")
async def upload_document(file: UploadFile = File(...), user_id: str = None):
    try:
        file_bytes = await file.read()
        file_id = str(uuid4())
        file_path = f"{user_id}/{file_id}.pdf"
        
        res = supabase.storage.from_("documents").upload(file_path, file_bytes, {"content-type": "application/pdf"})
        if "error" in res:
            raise Exception(res["error"])

        public_url = supabase.storage.from_("documents").get_public_url(file_path)

        doc = fitz.open(stream=file_bytes, filetype="pdf")
        page_count = doc.page_count
        size = len(file_bytes)

        insert_res = supabase.table("documents").insert({
            "title": file.filename,
            "url": public_url,
            "size": size,
            "page_count": page_count,
            "status": "queued",
            "user_id": user_id
        }).execute()

        return {"message": "Upload success", "data": insert_res.data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
