from typing import List, Optional, Literal
from pydantic import BaseModel, Field

Stage = Literal["received","extract","normalize","split","embedding","done","error"]

class UploadStatus(BaseModel):
    job_id: str
    stage: Stage
    progress: int = Field(ge=0, le=100)
    message: str = ""
    ok: bool = True

class SummarizeRequest(BaseModel):
    query: Optional[str] = None

class RagasScores(BaseModel):
    context_relevancy: Optional[float] = None
    context_recall: Optional[float] = None
    answer_correctness: Optional[float] = None
    faithfulness: Optional[float] = None

class SummarizeResponse(BaseModel):
    text: str
    contexts: List[str]
    ragas: RagasScores

class QARequest(BaseModel):
    question: str

class QAResponse(BaseModel):
    answer: str
    contexts: List[str]
    ragas: RagasScores

class Flashcard(BaseModel):
    question: str
    answer: str

class FlashcardsRequest(BaseModel):
    question_hint: Optional[str] = None

class FlashcardsResponse(BaseModel):
    cards: List[Flashcard]
    contexts: List[str]
    ragas: RagasScores
