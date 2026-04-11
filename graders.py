from patient_model import PatientState, GradeResult
from environment import NORMAL


def _clamp(score: float) -> float:
    """Ensure score is strictly between 0 and 1 (exclusive)."""
    return round(max(0.01, min(0.99, float(score))), 4)


def _vital_score(vitals, key: str) -> float:
    lo, hi = NORMAL[key]
    value = getattr(vitals, key)
    if lo <= value <= hi:
        return 0.98
    mid = (lo + hi) / 2
    span = (hi - lo) / 2 + 1e-9
    return max(0.01, min(0.98, 1.0 - abs(value - mid) / (span * 3)))


def grade_task1(episode_log: list) -> GradeResult:
    if not episode_log:
        return GradeResult(task_id=1, score=0.01, details={"error": "empty log"})

    final = episode_log[-1]
    vitals = final.get("vitals", {})
    sbp = vitals.get("systolic_bp", 0)
    steps_taken = len(episode_log)

    bp_score = _vital_score(
        type("V", (), {"systolic_bp": sbp})(), "systolic_bp"
    )

    first_normal_step = None
    for i, entry in enumerate(episode_log):
        v = entry.get("vitals", {})
        if 90 <= v.get("systolic_bp", 0) <= 120:
            first_normal_step = i + 1
            break

    if first_normal_step is not None and bp_score >= 0.9:
        efficiency_bonus = max(0.0, (10 - first_normal_step) / 10) * 0.2
    else:
        efficiency_bonus = 0.0

    score = bp_score * 0.8 + efficiency_bonus

    return GradeResult(
        task_id=1,
        score=_clamp(score),
        details={
            "final_systolic_bp": sbp,
            "bp_normal": 90 <= sbp <= 120,
            "bp_score": round(bp_score, 4),
            "efficiency_bonus": round(efficiency_bonus, 4),
            "first_normal_step": first_normal_step,
            "steps_taken": steps_taken,
        },
    )


def grade_task2(episode_log: list) -> GradeResult:
    if not episode_log:
        return GradeResult(task_id=2, score=0.01, details={"error": "empty log"})

    final = episode_log[-1]
    vitals = final.get("vitals", {})

    targets = {
        "heart_rate":    vitals.get("heart_rate", 0),
        "spo2":          vitals.get("spo2", 0),
        "blood_glucose": vitals.get("blood_glucose", 0),
    }

    class _V:
        pass

    v = _V()
    for k, val in targets.items():
        setattr(v, k, val)

    scores = {}
    for k in targets:
        raw = _vital_score(v, k)
        if k == "heart_rate":
            hr = targets[k]
            if hr <= 100:
                raw = 0.98
            elif hr <= 120:
                raw = 0.8
            elif hr <= 140:
                raw = 0.6
            elif hr <= 160:
                raw = 0.4
            elif hr <= 180:
                raw = 0.2
            else:
                raw = 0.05
        scores[k] = raw

    weights = {"heart_rate": 0.3, "spo2": 0.4, "blood_glucose": 0.3}
    weighted = sum(scores[k] * weights[k] for k in scores)
    all_normal = all(scores[k] >= 0.9 for k in scores)
    bonus = 0.08 if all_normal else 0.0
    score = weighted + bonus

    return GradeResult(
        task_id=2,
        score=_clamp(score),
        details={
            "vital_scores": {k: round(v, 4) for k, v in scores.items()},
            "all_vitals_normal": all_normal,
            "bonus": bonus,
            "final_vitals": targets,
        },
    )


def grade_task3(episode_log: list) -> GradeResult:
    if not episode_log:
        return GradeResult(task_id=3, score=0.01, details={"error": "empty log"})

    survivals = [step.get("survival_probability", 0.0) for step in episode_log]
    final_survival = survivals[-1]
    avg_survival = sum(survivals) / len(survivals)

    final_vitals_raw = episode_log[-1].get("vitals", {})

    class _V:
        pass

    fv = _V()
    for k, val in final_vitals_raw.items():
        setattr(fv, k, val)

    vital_scores = [_vital_score(fv, k) for k in NORMAL]
    vitals_normalised = sum(1 for s in vital_scores if s >= 0.9) / len(vital_scores)

    actions = [step.get("action", "") for step in episode_log if step.get("action")]
    unique_actions = len(set(actions))
    diversity_score = min(0.98, unique_actions / 6)

    score = (
        0.40 * final_survival
        + 0.30 * avg_survival
        + 0.20 * vitals_normalised
        + 0.10 * diversity_score
    )

    return GradeResult(
        task_id=3,
        score=_clamp(score),
        details={
            "final_survival_probability": round(final_survival, 4),
            "average_survival":           round(avg_survival, 4),
            "vitals_normalised_ratio":    round(vitals_normalised, 4),
            "unique_actions_used":        unique_actions,
            "diversity_score":            round(diversity_score, 4),
            "total_steps":                len(episode_log),
        },
    )


def grade(task_id: int, episode_log: list) -> GradeResult:
    graders = {1: grade_task1, 2: grade_task2, 3: grade_task3}
    if task_id not in graders:
        raise ValueError(f"Unknown task_id {task_id}. Valid: {list(graders.keys())}")
    return graders[task_id](episode_log)