from pydantic import BaseModel, Field, validator
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
    heart_rate: float
    systolic_bp: float
    diastolic_bp: float
    spo2: float
    temperature: float
    blood_glucose: float
    respiratory_rate: float
    creatinine: float


class PatientState(BaseModel):
    patient_id: str
    step: int
    max_steps: int
    vitals: PatientVitals
    diagnosis: Diagnosis
    severity: Severity
    survival_probability: float
    treatment_history: List[str] = []
    done: bool = False
    info: dict = {}

    @validator("survival_probability")
    def clamp_survival(cls, v):
        return round(max(0.001, min(0.999, float(v))), 4)


class TreatmentAction(BaseModel):
    action: str
    dose: Optional[float] = None
    duration: Optional[int] = None


class StepResult(BaseModel):
    state: PatientState
    reward: float
    done: bool
    info: dict

    @validator("reward")
    def clamp_reward(cls, v):
        return round(max(0.001, min(0.999, float(v))), 4)


class GradeRequest(BaseModel):
    task_id: int
    episode_log: List[dict]


class GradeResult(BaseModel):
    task_id: int
    score: float
    details: dict

    @validator("score")
    def clamp_score(cls, v):
        return round(max(0.001, min(0.999, float(v))), 4)