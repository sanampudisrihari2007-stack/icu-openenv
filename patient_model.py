from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class Severity(str, Enum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


class Diagnosis(str, Enum):
    SEPSIS = "sepsis"
    PNEUMONIA = "pneumonia"
    CARDIAC_ARREST = "cardiac_arrest"
    POST_SURGERY = "post_surgery"
    RESPIRATORY_FAILURE = "respiratory_failure"


class PatientVitals(BaseModel):
    heart_rate: float = Field(..., description="Heart rate in bpm (normal: 60-100)")
    systolic_bp: float = Field(..., description="Systolic blood pressure mmHg (normal: 90-120)")
    diastolic_bp: float = Field(..., description="Diastolic blood pressure mmHg (normal: 60-80)")
    spo2: float = Field(..., description="Oxygen saturation % (normal: 95-100)")
    temperature: float = Field(..., description="Body temperature Celsius (normal: 36.5-37.5)")
    blood_glucose: float = Field(..., description="Blood glucose mg/dL (normal: 70-140)")
    respiratory_rate: float = Field(..., description="Breaths per minute (normal: 12-20)")
    creatinine: float = Field(..., description="Kidney function mg/dL (normal: 0.6-1.2)")


class PatientState(BaseModel):
    patient_id: str
    step: int
    max_steps: int
    vitals: PatientVitals
    diagnosis: Diagnosis
    severity: Severity
    survival_probability: float = Field(..., ge=0.0, le=1.0)
    treatment_history: List[str] = []
    done: bool = False
    info: dict = {}


class TreatmentAction(BaseModel):
    action: str = Field(..., description="Treatment action to perform")
    dose: Optional[float] = Field(None, description="Dose amount if applicable")
    duration: Optional[int] = Field(None, description="Duration in minutes if applicable")


class StepResult(BaseModel):
    state: PatientState
    reward: float = Field(..., ge=0.0, le=1.0)
    done: bool
    info: dict


class GradeRequest(BaseModel):
    task_id: int
    episode_log: List[dict]


class GradeResult(BaseModel):
    task_id: int
    score: float = Field(..., ge=0.0, le=1.0)
    details: dict