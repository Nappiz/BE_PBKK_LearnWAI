import os
import httpx
from typing import Optional, Tuple
from fastapi import HTTPException

PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

TEMP = float(os.getenv("GEN_TEMPERATURE", "0.2"))
TOP_P = float(os.getenv("GEN_TOP_P", "0.95"))
TIMEOUT = float(os.getenv("REQUEST_TIMEOUT_S", "120"))
SEED = int(os.getenv("GEN_SEED", "42"))  # untuk stabilitas hasil

def _raise_502(msg: str):
    raise HTTPException(status_code=502, detail=msg)

def _system_user_messages(system: Optional[str], prompt: str):
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    return msgs

def _max_tokens() -> Optional[int]:
    try:
        v = int(os.getenv("GEN_MAX_TOKENS", "2048"))
        return v if v > 0 else None
    except Exception:
        return 2048

async def generate(prompt: str, system: Optional[str]) -> Tuple[str, dict]:
    """
    Return (text, raw_json). No retry/fallback. Any non-2xx -> HTTP 502.
    """
    try:
        mx = _max_tokens()

        if PROVIDER == "openai":
            key = os.getenv("OPENAI_API_KEY", "")
            if not key:
                _raise_502("Missing OPENAI_API_KEY")

            url = "https://api.openai.com/v1/chat/completions"
            payload = {
                "model": MODEL,
                "messages": _system_user_messages(system, prompt),
                "temperature": TEMP,
                "top_p": TOP_P,
                "seed": SEED,  # OpenAI mendukung seed
            }
            if mx is not None:
                payload["max_tokens"] = mx

            headers = {"Authorization": f"Bearer {key}"}
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                r = await client.post(url, json=payload, headers=headers)
            if r.status_code >= 400:
                _raise_502(f"openai: {r.text}")

            data = r.json()
            text = data["choices"][0]["message"]["content"]
            return text, data

        elif PROVIDER == "gemini":
            key = os.getenv("GEMINI_API_KEY", "")
            if not key:
                _raise_502("Missing GEMINI_API_KEY")

            base = "https://generativelanguage.googleapis.com/v1beta"
            url = f"{base}/models/{MODEL}:generateContent?key={key}"

            gen_cfg = {
                "temperature": TEMP,
                "topP": TOP_P,
                "candidateCount": 1,  # bantu konsistensi
            }
            if mx is not None:
                gen_cfg["maxOutputTokens"] = mx

            payload = {
                "contents": [
                    {"parts": [{"text": (system + "\n\n" if system else "") + prompt}]}
                ],
                "generationConfig": gen_cfg,
            }

            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                r = await client.post(url, json=payload)
            if r.status_code >= 400:
                _raise_502(f"gemini: {r.text}")

            data = r.json()
            try:
                text = data["candidates"][0]["content"]["parts"][0]["text"]
            except Exception:
                _raise_502(f"gemini bad response: {data}")
            return text, data

        elif PROVIDER == "ollama":
            base = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
            url = f"{base}/api/chat"

            # Opsi deterministik untuk Ollama
            options = {
                "temperature": TEMP,
                "top_p": TOP_P,
                "seed": SEED,
            }
            if mx is not None:
                options["num_predict"] = mx

            payload = {
                "model": MODEL,
                "messages": _system_user_messages(system, prompt),
                "stream": False,
                "options": options,
            }

            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                r = await client.post(url, json=payload)
            if r.status_code >= 400:
                _raise_502(f"ollama: {r.text}")

            data = r.json()
            text = data.get("message", {}).get("content", "")
            return text, data

        else:
            _raise_502(f"Unknown LLM_PROVIDER={PROVIDER}")

    except HTTPException:
        raise
    except Exception as e:
        _raise_502(str(e))
