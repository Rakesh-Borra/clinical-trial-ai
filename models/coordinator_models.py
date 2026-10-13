
from typing import List, Optional

from pydantic import BaseModel, Field


class CoordinatorCitation(BaseModel):
    """A protocol fact cited in a coordinator answer."""

    category: str
    statement: str
    source_page: Optional[int] = None
    verification_status: str


class CoordinatorAnswer(BaseModel):
    """Structured response for coordinator protocol lookup."""

    answer_text: str
    intent: str
    citations: List[CoordinatorCitation] = Field(default_factory=list)
    confidence: str
    limitations: List[str] = Field(default_factory=list)
    disclaimers: List[str] = Field(default_factory=list)
