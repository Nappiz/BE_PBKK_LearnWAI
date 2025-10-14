# backend/app/evaluation.py
import os
import sys
import re
import math
import json
import asyncio
import hashlib
import pickle
import traceback
from typing import Dict, List, Optional

from sklearn.metrics.pairwise import cosine_similarity

# --------------------------- utils: safe floats ---------------------------

def _json_float(x: Optional[float]) -> Optional[float]:
    if x is None or isinstance(x, bool):
        return None
    try:
        if math.isnan(x) or math.isinf(x):
            return None
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return None

def _safe_scores(d: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    return {k: _json_float(v) for k, v in d.items()}

def _finalize_four(scores: Dict[str, Optional[float]]) -> Dict[str, float]:
    # jika relevancy tidak ada tapi precision tersedia → pakai precision
    if scores.get("context_relevancy") is None and scores.get("context_precision") is not None:
        scores["context_relevancy"] = scores["context_precision"]

    defaults = {
        "context_relevancy": 0.0,
        "context_recall": 0.0,
        "answer_correctness": 0.0,
        "faithfulness": 0.0,
    }
    out: Dict[str, float] = {}
    for k, dflt in defaults.items():
        v = scores.get(k, None)
        try:
            v = float(v)  # type: ignore
            if math.isnan(v) or math.isinf(v):
                v = dflt
        except Exception:
            v = dflt
        out[k] = max(0.0, min(1.0, v))
    return out

# --------------------------- utils: text canon & cache ---------------------------

def _canon(s: str) -> str:
    # Hilangkan variasi whitespace/linebreak agar scoring lebih stabil
    return " ".join((s or "").split())

_CACHE_DIR = os.getenv("RAGAS_CACHE_DIR", "./data/ragas_cache")
os.makedirs(_CACHE_DIR, exist_ok=True)

def _cache_key(action: str, q: str, ctxs: List[str], ans: str) -> str:
    raw = json.dumps({"a": action, "q": q, "c": ctxs, "ans": ans}, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

# --------------------------- LLM & Emb evaluator (no fallback) ---------------------------

def _build_ragas_llm_and_emb():
    """
    Build evaluator LLM + embeddings sesuai ENV.
    Tanpa fallback: jika provider/back-end tak didukung → raise.
    """
    provider = os.getenv("RAGAS_EVAL_PROVIDER", os.getenv("LLM_PROVIDER", "ollama")).lower()
    model = os.getenv("RAGAS_EVAL_MODEL", os.getenv("LLM_MODEL", "llama3.1:8b-instruct"))
    temp = float(os.getenv("GEN_TEMPERATURE", "0.2"))
    seed = int(os.getenv("RAGAS_EVAL_SEED", "1337"))

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        llm = ChatOllama(
            model=model,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            temperature=temp,
            model_kwargs={"seed": seed},  # stabilkan evaluator
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=model, temperature=temp)
    else:
        raise RuntimeError(f"Unsupported RAGAS_EVAL_PROVIDER={provider}")

    emb_backend = os.getenv("RAGAS_EVAL_EMB_BACKEND", "local").lower()
    emb_model = os.getenv("RAGAS_EVAL_EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    if emb_backend in ("local", "hf"):
        from langchain_huggingface import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name=emb_model)
    elif emb_backend == "openai":
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(model=emb_model)
    else:
        raise RuntimeError(f"Unsupported RAGAS_EVAL_EMB_BACKEND={emb_backend}")

    return llm, embeddings

# --------------------------- auto-reference (synth GT) ---------------------------

def _make_auto_reference_prompt(question: str, contexts: List[str]) -> str:
    ctx = "\n\n".join(f"- {c}" for c in contexts if c)
    q = (question or "Summarize the key information.").strip()
    return (
        "You are a grader assistant. Using ONLY the provided contexts, write a concise, factual, self-contained gold answer.\n"
        "- Use 3–6 sentences, one paragraph.\n"
        "- No external facts.\n"
        "- Do NOT output JSON.\n\n"
        f"Question:\n{q}\n\nContexts:\n{ctx}\n\nGold answer:"
    )

def _auto_reference_sync(llm, question: str, contexts: List[str]) -> Optional[str]:
    try:
        prompt = _make_auto_reference_prompt(question, contexts) + (
            "\n\nReturn plain text only. Do NOT return JSON, lists, or keys. "
            "No headings. One paragraph."
        )
        out = llm.invoke(prompt)
        raw = getattr(out, "content", None) or (str(out) if out is not None else "")
        txt = None

        if isinstance(raw, str):
            raw_s = raw.strip()
            if raw_s.startswith("{") or raw_s.startswith("["):
                try:
                    obj = json.loads(raw_s)
                    if isinstance(obj, dict):
                        for k in ("text", "answer", "output", "response"):
                            if isinstance(obj.get(k), str) and obj[k].strip():
                                txt = obj[k].strip()
                                break
                        if txt is None and isinstance(obj.get("statements"), list):
                            txt = " ".join(str(x) for x in obj["statements"]).strip()
                    elif isinstance(obj, list):
                        txt = " ".join(str(x) for x in obj).strip()
                except Exception:
                    pass
            if txt is None:
                txt = raw_s
        else:
            txt = str(raw).strip()

        return txt[:1200] if txt else None
    except Exception:
        return None

# --------------------------- ekstraktif reference ---------------------------

def _split_sentences(txt: str) -> List[str]:
    parts = re.split(r'(?<=[.!?])\s+', (txt or "").strip())
    return [p.strip() for p in parts if len(p.strip()) > 0]

def _build_extractive_reference(question: str, answer: str, contexts: List[str], emb) -> Optional[str]:
    # 1) pecah semua contexts ke kalimat
    sents: List[str] = []
    for ctx in contexts:
        sents.extend(_split_sentences(ctx))
    sents = sents[:800]  # guard
    if not sents:
        return None

    # 2) embed query = question + answer
    q = (question or "") + " " + (answer or "")
    qv = emb.embed_query(q)
    sv = emb.embed_documents(sents)

    # 3) ambil top-N kalimat paling mirip
    sims = cosine_similarity([qv], sv)[0]
    top_idx = sims.argsort()[::-1][:6]
    top_sents = [sents[i] for i in top_idx]

    # 4) reference = gabungan kalimat
    ref = " ".join(top_sents)[:1200].strip()
    return ref or None

# --------------------------- main entry ---------------------------

async def run_ragas_eval(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: Optional[str],
) -> Dict[str, float]:
    """
    Jalankan 4 metrik RAGAS (context_relevancy/precision, context_recall,
    answer_correctness, faithfulness) dan SELALU kembalikan 4 angka 0..1.
    """
    try:
        from ragas.metrics import faithfulness
        try:
            from ragas.metrics import context_relevancy as m_context_rel
        except Exception:
            from ragas.metrics import context_precision as m_context_rel
        try:
            from ragas.metrics import context_recall as m_context_recall
        except Exception:
            m_context_recall = None
        try:
            from ragas.metrics import answer_correctness as m_answer_correctness
        except Exception:
            m_answer_correctness = None

        from ragas import evaluate
        from datasets import Dataset

        # ==== trimming & canonization ====
        max_k = int(os.getenv("RAGAS_EVAL_TOPK", "2"))
        max_ctx_chars = int(os.getenv("RAGAS_EVAL_MAX_CHARS", "800"))
        max_ans_chars = int(os.getenv("RAGAS_EVAL_MAX_ANS_CHARS", "1200"))

        question = _canon(question or "")
        answer = _canon((answer or "")[:max_ans_chars])
        contexts = [_canon(c)[:max_ctx_chars] for c in (contexts or [])[:max_k]]

        # ==== cache ====
        action = os.getenv("RAGAS_ACTION", "generic")
        key = _cache_key(action, question, contexts, answer)
        cpath = os.path.join(_CACHE_DIR, key + ".pkl")
        if os.path.exists(cpath):
            with open(cpath, "rb") as f:
                return pickle.load(f)

        # ==== evaluator backends ====
        llm, embeddings = _build_ragas_llm_and_emb()

        # ==== Auto-reference (GT → ekstraktif → generatif → fallback) ====
        reference = (ground_truth or "").strip() or None
        if reference is None:
            try:
                reference = _build_extractive_reference(question, answer, contexts, embeddings)
            except Exception:
                reference = None
        if reference is None:
            reference = _auto_reference_sync(llm, question, contexts)
        if reference is None:
            reference = (" ".join(contexts)).strip()[:800] or (question or "reference")

        # ==== dataset ====
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
            "ground_truth": [reference],
            "reference": [reference],
        }
        ds = Dataset.from_dict(data)

        # ==== metrik (4) ====
        metrics_list = [faithfulness]
        if m_context_recall is not None:
            metrics_list.append(m_context_recall)
        if m_answer_correctness is not None:
            metrics_list.append(m_answer_correctness)
        metrics_list.append(m_context_rel)

        # ==== jalankan eval (timeout adaptif) ====
        base = int(os.getenv("RAGAS_EVAL_TIMEOUT_S", "60"))
        size_hint = len(answer) + sum(len(c) for c in contexts)
        timeout_s = max(base, min(300, 30 + 0.01 * size_hint))

        loop = asyncio.get_event_loop()
        def _eval_sync():
            return evaluate(ds, metrics=metrics_list, llm=llm, embeddings=embeddings)
        res = await asyncio.wait_for(loop.run_in_executor(None, _eval_sync), timeout=timeout_s)

        # ==== hasil ====
        vals = res.to_pandas().iloc[0]
        raw = {
            "context_relevancy": float(vals.get("context_relevancy", float("nan")))
                                 if "context_relevancy" in vals
                                 else float(vals.get("context_precision", float("nan"))),
            "context_precision": float(vals.get("context_precision", float("nan"))),
            "context_recall": float(vals.get("context_recall", float("nan"))),
            "answer_correctness": float(vals.get("answer_correctness", float("nan"))),
            "faithfulness": float(vals.get("faithfulness", float("nan"))),
        }
        result = _finalize_four(raw)

        # ==== cache out ====
        try:
            with open(cpath, "wb") as f:
                pickle.dump(result, f)
        except Exception:
            pass

        return result

    except Exception as e:
        print("[RAGAS] evaluation failed:", repr(e), file=sys.stderr)
        traceback.print_exc()
        return _finalize_four({
            "context_relevancy": None,
            "context_recall": None,
            "answer_correctness": None,
            "faithfulness": None,
        })
