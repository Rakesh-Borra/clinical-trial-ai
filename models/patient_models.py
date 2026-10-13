
from typing import List, Optional

from pydantic import BaseModel, Field


class PatientCitation(BaseModel):
    """A protocol fact cited in a patient-facing answer."""

    category: str
    statement: str
    source_page: Optional[int] = None
    verification_status: str
    plain_language: Optional[str] = None


class PatientAnswer(BaseModel):
    """Structured response for patient-friendly protocol explanations."""

    answer_text: str
    intent: str
    citations: List[PatientCitation] = Field(default_factory=list)
    confidence: str
    limitations: List[str] = Field(default_factory=list)
    disclaimers: List[str] = Field(default_factory=list)
    audience: str = "patient"
