from typing import List, Literal, Optional
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


PatientInfoCategory = Literal[
    "blood_collection",
    "interventions",
    "risks",
    "consent_withdrawal",
    "study_purpose",
]


class PatientInfoItem(BaseModel):
    """Patient-relevant fact outside the original five lists."""

    category: PatientInfoCategory
    description: str
    source_page: Optional[int] = None


class ProtocolAnalysis(BaseModel):
    trial_title: Optional[str] = None
    protocol_id: Optional[str] = None
    analysis_complete: Optional[bool] = None
    # Separate from analysis_complete (which means "ran to completion"):
    # False when deterministic checks found problems with extracted content.
    verification_passed: Optional[bool] = None

    inclusion_criteria: List[EligibilityCriterion] = Field(default_factory=list)
    exclusion_criteria: List[EligibilityCriterion] = Field(default_factory=list)

    procedures: List[TrialProcedure] = Field(default_factory=list)
    visits: List[TrialVisit] = Field(default_factory=list)
    safety_information: List[SafetyInformation] = Field(default_factory=list)
    # Added later; defaults keep stored analyses and cache entries loadable.
    patient_information: List[PatientInfoItem] = Field(default_factory=list)
