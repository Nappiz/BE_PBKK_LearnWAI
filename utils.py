import re
import uuid
from typing import List

def new_job_id() -> str:
    return uuid.uuid4().hex

_ws_re = re.compile(r"[ \t]+")

def normalize_text(s: str) -> str:
    s = s.replace("\r", "")
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = _ws_re.sub(" ", s)
    return s.strip()

def chunk_text(text: str, chunk_size: int, overlap: int, min_chars: int) -> List[str]:
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = min(len(words), start + chunk_size // 6)  # approx 6 chars/word
        chunk = " ".join(words[start:end]).strip()
        if len(chunk) >= min_chars:
            chunks.append(chunk)
        if end == len(words):
            break
        # move with overlap (approx by words)
        back = max(0, overlap // 6)
        start = max(0, end - back)
    return chunks
