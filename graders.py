from patient_model import GradeResult
from environment import NORMAL


def _clamp(score: float) -> float:
    """Score must be strictly between 0 and 1 — never 0.0 or 1.0."""
    return float(max(0.01, min(0.99, round(float(score), 4))))


def _vital_score(value: float, lo: float, hi: float) -> float:
    if lo <= value <= hi:
        return 0.95
    mid = (lo + hi) / 2
    span = (hi - lo) / 2 + 1e-9
    raw = 1.0 - abs(value - mid) / (span * 3)
    return max(0.02, min(0.95, raw))


def grade_task1(episode_log: list) -> GradeResult:
    if not episode_log:
        return GradeResult(task_id=1, score=0.01, details={"error": "empty log"})

    final = episode_log[-1]
    vitals = final.get("vitals", {})
    sbp = vitals.get("systolic_bp", 60)
    steps_taken = len(episode_log)

    bp_score = _vital_score(sbp, 90, 120)

    first_normal_step = None
    for i, entry in enumerate(episode_log):
        v = entry.get("vitals", {})
        if 90 <= v.get("systolic_bp", 0) <= 120:
            first_normal_step = i + 1
            break

    if first_normal_step is not None and bp_score >= 0.8:
        efficiency_bonus = max(0.0, (10 - first_normal_step) / 10) * 0.15
    else:
        efficiency_bonus = 0.0

    score = bp_score * 0.75 + efficiency_bonus + 0.05

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

    hr = vitals.get("heart_rate", 150)
    spo2 = vitals.get("spo2", 80)
    bg = vitals.get("blood_glucose", 200)

    hr_score = _vital_score(hr, 60, 100)
    spo2_score = _vital_score(spo2, 95, 100)
    bg_score = _vital_score(bg, 70, 140)

    # Heart rate partial credit
    if hr <= 100:
        hr_score = 0.95
    elif hr <= 120:
        hr_score = 0.75
    elif hr <= 140:
        hr_score = 0.55
    elif hr <= 160:
        hr_score = 0.35
    elif hr <= 180:
        hr_score = 0.15
    else:
        hr_score = 0.05

    weighted = hr_score * 0.3 + spo2_score * 0.4 + bg_score * 0.3
    all_normal = hr_score >= 0.7 and spo2_score >= 0.8 and bg_score >= 0.8
    bonus = 0.07 if all_normal else 0.0
    score = weighted + bonus + 0.02

    return GradeResult(
        task_id=2,
        score=_clamp(score),
        details={
            "vital_scores": {
                "heart_rate": round(hr_score, 4),
                "spo2": round(spo2_score, 4),
                "blood_glucose": round(bg_score, 4),
            },
            "all_vitals_normal": all_normal,
            "bonus": bonus,
            "final_vitals": {"heart_rate": hr, "spo2": spo2, "blood_glucose": bg},
        },
    )


def grade_task3(episode_log: list) -> GradeResult:
    if not episode_log:
        return GradeResult(task_id=3, score=0.01, details={"error": "empty log"})

    survivals = [step.get("survival_probability", 0.5) for step in episode_log]
    final_survival = min(0.97, survivals[-1])
    avg_survival = min(0.97, sum(survivals) / len(survivals))

    final_vitals_raw = episode_log[-1].get("vitals", {})
    normal_ranges = {
        "heart_rate": (60, 100),
        "systolic_bp": (90, 120),
        "spo2": (95, 100),
        "temperature": (36.5, 37.5),
        "blood_glucose": (70, 140),
        "respiratory_rate": (12, 20),
        "creatinine": (0.6, 1.2),
    }
    normal_count = 0
    total_count = 0
    for key, (lo, hi) in normal_ranges.items():
        val = final_vitals_raw.get(key, 0)
        if val > 0:
            total_count += 1
            if lo <= val <= hi:
                normal_count += 1
    vitals_normalised = normal_count / max(total_count, 1)

    actions = [step.get("action", "") for step in episode_log if step.get("action")]
    unique_actions = len(set(actions))
    diversity_score = min(0.95, unique_actions / 6)

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
            "average_survival": round(avg_survival, 4),
            "vitals_normalised_ratio": round(vitals_normalised, 4),
            "unique_actions_used": unique_actions,
            "diversity_score": round(diversity_score, 4),
            "total_steps": len(episode_log),
        },
    )


def grade(task_id: int, episode_log: list) -> GradeResult:
    graders = {1: grade_task1, 2: grade_task2, 3: grade_task3}
    if task_id not in graders:
        raise ValueError(f"Unknown task_id {task_id}. Valid: {list(graders.keys())}")
    return graders[task_id](episode_log)