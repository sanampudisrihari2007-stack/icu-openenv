import random
import numpy as np
from patient_model import PatientVitals, PatientState, Diagnosis, Severity


# ── Normal ranges ──────────────────────────────────────────────────────────────
NORMAL = {
    "heart_rate":       (60,   100),
    "systolic_bp":      (90,   120),
    "diastolic_bp":     (60,    80),
    "spo2":             (95,   100),
    "temperature":      (36.5, 37.5),
    "blood_glucose":    (70,   140),
    "respiratory_rate": (12,    20),
    "creatinine":       (0.6,   1.2),
}

# ── Action definitions: each action nudges specific vitals ─────────────────────
ACTION_EFFECTS = {
    "increase_vasopressor": {"systolic_bp": +12, "diastolic_bp": +6,  "heart_rate": +5},
    "decrease_vasopressor": {"systolic_bp": -12, "diastolic_bp": -6,  "heart_rate": -5},
    "give_antibiotics":     {"temperature": -0.4, "heart_rate": -12,  "creatinine": +0.05},
    "increase_insulin":     {"blood_glucose": -20, "heart_rate": +3},
    "decrease_insulin":     {"blood_glucose": +20},
    "give_iv_fluids":       {"systolic_bp": +8,  "diastolic_bp": +4,  "creatinine": -0.1},
    "increase_oxygen":      {"spo2": +3,  "respiratory_rate": -2},
    "decrease_oxygen":      {"spo2": -3,  "respiratory_rate": +2},
    "increase_peep":        {"spo2": +4,  "systolic_bp": -5},
    "decrease_peep":        {"spo2": -4,  "systolic_bp": +5},
    "order_labs":           {},   # observation action — no direct effect
    "call_specialist":      {"survival_probability": +0.03},
    "do_nothing":           {},
}

VALID_ACTIONS = list(ACTION_EFFECTS.keys())


def _clamp(value, lo, hi):
    return max(lo, min(hi, value))


def _random_vitals(diagnosis: Diagnosis, severity: Severity) -> PatientVitals:
    """Generate abnormal starting vitals based on diagnosis & severity."""
    severity_multiplier = {"mild": 0.5, "moderate": 1.0, "severe": 1.5, "critical": 2.0}[severity]

    base = {
        "heart_rate":       random.uniform(100, 130) * severity_multiplier,
        "systolic_bp":      random.uniform(60,   85),
        "diastolic_bp":     random.uniform(40,   60),
        "spo2":             random.uniform(80,   92),
        "temperature":      random.uniform(38.0, 39.5),
        "blood_glucose":    random.uniform(160,  250),
        "respiratory_rate": random.uniform(22,   30),
        "creatinine":       random.uniform(1.3,  2.5),
    }

    if diagnosis == Diagnosis.CARDIAC_ARREST:
        base["heart_rate"] = random.uniform(30, 50)
        base["systolic_bp"] = random.uniform(50, 70)
    elif diagnosis == Diagnosis.RESPIRATORY_FAILURE:
        base["spo2"] = random.uniform(70, 85)
        base["respiratory_rate"] = random.uniform(28, 40)
    elif diagnosis == Diagnosis.POST_SURGERY:
        base["blood_glucose"] = random.uniform(180, 300)
        base["temperature"] = random.uniform(37.8, 38.5)

    # Clamp to plausible physiological limits
    return PatientVitals(
        heart_rate=       _clamp(base["heart_rate"],       20,  200),
        systolic_bp=      _clamp(base["systolic_bp"],      40,  200),
        diastolic_bp=     _clamp(base["diastolic_bp"],     20,  130),
        spo2=             _clamp(base["spo2"],              50,  100),
        temperature=      _clamp(base["temperature"],      34.0, 42.0),
        blood_glucose=    _clamp(base["blood_glucose"],    40,   600),
        respiratory_rate= _clamp(base["respiratory_rate"],  4,    60),
        creatinine=       _clamp(base["creatinine"],        0.1,  10.0),
    )


def _survival_probability(vitals: PatientVitals) -> float:
    """Estimate survival probability from current vitals (0.0–1.0)."""
    score = 1.0

    def penalty(value, lo, hi, weight):
        if value < lo:
            return weight * (lo - value) / lo
        if value > hi:
            return weight * (value - hi) / hi
        return 0.0

    score -= penalty(vitals.heart_rate,       60,  100, 0.15)
    score -= penalty(vitals.systolic_bp,       90,  120, 0.20)
    score -= penalty(vitals.spo2,              95,  100, 0.20)
    score -= penalty(vitals.temperature,       36.5, 37.5, 0.10)
    score -= penalty(vitals.blood_glucose,     70,  140, 0.10)
    score -= penalty(vitals.respiratory_rate,  12,   20, 0.10)
    score -= penalty(vitals.creatinine,         0.6,  1.2, 0.15)

    return _clamp(score, 0.0, 1.0)


def _add_noise(vitals: PatientVitals) -> PatientVitals:
    """Add small physiological noise each step."""
    return PatientVitals(
        heart_rate=       _clamp(vitals.heart_rate       + random.gauss(0, 1.5),  20,  200),
        systolic_bp=      _clamp(vitals.systolic_bp      + random.gauss(0, 1.5),  40,  200),
        diastolic_bp=     _clamp(vitals.diastolic_bp     + random.gauss(0, 1.0),  20,  130),
        spo2=             _clamp(vitals.spo2              + random.gauss(0, 0.5),  50,  100),
        temperature=      _clamp(vitals.temperature      + random.gauss(0, 0.05), 34.0, 42.0),
        blood_glucose=    _clamp(vitals.blood_glucose    + random.gauss(0, 3.0),  40,   600),
        respiratory_rate= _clamp(vitals.respiratory_rate + random.gauss(0, 0.5),   4,    60),
        creatinine=       _clamp(vitals.creatinine       + random.gauss(0, 0.02),  0.1,  10.0),
    )


class ICUEnvironment:
    def __init__(self, task_id: int = 1):
        self.task_id = task_id
        self.state: PatientState = None
        self._configure_task()

    def _configure_task(self):
        configs = {
            1: {"max_steps": 10, "diagnosis": Diagnosis.SEPSIS,            "severity": Severity.MODERATE},
            2: {"max_steps": 20, "diagnosis": Diagnosis.POST_SURGERY,       "severity": Severity.SEVERE},
            3: {"max_steps": 24, "diagnosis": Diagnosis.RESPIRATORY_FAILURE,"severity": Severity.CRITICAL},
        }
        self.config = configs.get(self.task_id, configs[1])

    def reset(self) -> PatientState:
        diagnosis = self.config["diagnosis"]
        severity  = self.config["severity"]
        vitals    = _random_vitals(diagnosis, severity)

        self.state = PatientState(
            patient_id=f"P-{random.randint(1000, 9999)}",
            step=0,
            max_steps=self.config["max_steps"],
            vitals=vitals,
            diagnosis=diagnosis,
            severity=severity,
            survival_probability=_survival_probability(vitals),
            treatment_history=[],
            done=False,
            info={"task_id": self.task_id, "valid_actions": VALID_ACTIONS},
        )
        return self.state

    def step(self, action: str, dose: float = None) -> tuple:
        if self.state is None:
            raise ValueError("Call reset() before step()")
        if self.state.done:
            raise ValueError("Episode is done. Call reset() to start a new episode.")
        if action not in VALID_ACTIONS:
            raise ValueError(f"Invalid action '{action}'. Valid: {VALID_ACTIONS}")

        prev_survival = self.state.survival_probability
        effects = ACTION_EFFECTS[action]
        v = self.state.vitals.dict()

        # Apply action effects
        for key, delta in effects.items():
            if key in v:
                lo, hi = (0.0, 1.0) if key == "survival_probability" else (
                    NORMAL[key][0] * 0.3, NORMAL[key][1] * 2.5
                )
                v[key] = _clamp(v[key] + delta, lo, hi)

        # Apply noise
        new_vitals = _add_noise(PatientVitals(**v))
        new_survival = _survival_probability(new_vitals)

        # Adjust survival_probability from call_specialist
        if action == "call_specialist":
            new_survival = _clamp(new_survival + 0.03, 0.0, 1.0)

        # Calculate reward
        reward = self._calculate_reward(prev_survival, new_survival, new_vitals, action)

        self.state.step += 1
        done = (self.state.step >= self.state.max_steps) or (new_survival < 0.05)

        self.state = PatientState(
            patient_id=self.state.patient_id,
            step=self.state.step,
            max_steps=self.state.max_steps,
            vitals=new_vitals,
            diagnosis=self.state.diagnosis,
            severity=self.state.severity,
            survival_probability=new_survival,
            treatment_history=self.state.treatment_history + [action],
            done=done,
            info={"reward_breakdown": {}, "task_id": self.task_id},
        )

        return self.state, reward, done, self.state.info

    def _calculate_reward(self, prev_survival, new_survival, vitals, action) -> float:
        reward = 0.0

        # Survival improvement
        delta = new_survival - prev_survival
        if delta > 0:
            reward += 0.4 * (delta / 0.1)
        else:
            reward += 0.2 * (delta / 0.1)

        # Vitals in normal range bonus
        normal_count = sum(
            1 for key, (lo, hi) in NORMAL.items()
            if lo <= getattr(vitals, key, lo) <= hi
        )
        reward += 0.3 * (normal_count / len(NORMAL))

        # Absolute survival bonus
        reward += 0.3 * new_survival

        # Penalise contradictory / harmful combos
        if action in ("increase_vasopressor", "decrease_vasopressor") and vitals.heart_rate > 150:
            reward -= 0.15
        if action == "increase_oxygen" and vitals.spo2 > 99:
            reward -= 0.10

        return float(_clamp(reward, 0.0, 1.0))

    def get_state(self) -> PatientState:
        if self.state is None:
            raise ValueError("Call reset() first.")
        return self.state