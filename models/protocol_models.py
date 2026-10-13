from typing import List, Optional
from pydantic import BaseModel, Field


class EligibilityCriterion(BaseModel):
    description: str
    source_page: Optional[int] = None


class TrialProcedure(BaseModel):
    name: str
    description: Optional[str] = None
    source_page: Optional[int] = None


class TrialVisit(BaseModel):
    visit_name: str
    timing: Optional[str] = None
    procedures: List[str] = Field(default_factory=list)
    source_page: Optional[int] = None


class SafetyInformation(BaseModel):
    description: str
    source_page: Optional[int] = None


class ProtocolAnalysis(BaseModel):
    trial_title: Optional[str] = None
    protocol_id: Optional[str] = None

    inclusion_criteria: List[EligibilityCriterion] = Field(default_factory=list)
    exclusion_criteria: List[EligibilityCriterion] = Field(default_factory=list)

    procedures: List[TrialProcedure] = Field(default_factory=list)
    visits: List[TrialVisit] = Field(default_factory=list)
    safety_information: List[SafetyInformation] = Field(default_factory=list)