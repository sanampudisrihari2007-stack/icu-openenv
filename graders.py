from patient_model import GradeResult


def _safe(score: float) -> float:
    """Always return strictly between 0 and 1 (exclusive)."""
    try:
        v = float(score)
    except Exception:
        return 0.05
    if v != v:  # NaN check
        return 0.05
    # Clamp strictly between 0.001 and 0.990
    v = max(0.001, min(0.990, v))
    return round(v, 4)


def _vital_score(value: float, lo: float, hi: float) -> float:
    """Score a vital sign — never returns 0.0 or 1.0."""
    try:
        value = float(value)
    except Exception:
        return 0.05
    if lo <= value <= hi:
        return 0.88
    mid = (lo + hi) / 2.0
    span = max((hi - lo) / 2.0, 1e-6)
    raw = 1.0 - abs(value - mid) / (span * 3.0)
    return _safe(raw)


def grade_task1(episode_log: list) -> GradeResult:
    if not episode_log:
        return GradeResult(task_id=1, score=0.05, details={"error": "empty log"})

    final  = episode_log[-1]
    vitals = final.get("vitals", {})
    sbp    = float(vitals.get("systolic_bp", 60))

    bp_score = _vital_score(sbp, 90, 120)

    first_normal_step = None
    for i, entry in enumerate(episode_log):
        v = entry.get("vitals", {})
        if 90 <= float(v.get("systolic_bp", 0)) <= 120:
            first_normal_step = i + 1
            break

    if first_normal_step and bp_score >= 0.7:
        efficiency_bonus = max(0.0, (10 - first_normal_step) / 10.0) * 0.15
    else:
        efficiency_bonus = 0.0

    score = bp_score * 0.75 + efficiency_bonus + 0.05
    return GradeResult(
        task_id=1,
        score=_safe(score),
        details={
            "final_systolic_bp": sbp,
            "bp_normal": bool(90 <= sbp <= 120),
            "bp_score": round(bp_score, 4),
            "efficiency_bonus": round(efficiency_bonus, 4),
            "first_normal_step": first_normal_step,
            "steps_taken": len(episode_log),
        },
    )


def grade_task2(episode_log: list) -> GradeResult:
    if not episode_log:
        return GradeResult(task_id=2, score=0.05, details={"error": "empty log"})

    final  = episode_log[-1]
    vitals = final.get("vitals", {})

    hr   = float(vitals.get("heart_rate",    150))
    spo2 = float(vitals.get("spo2",           80))
    bg   = float(vitals.get("blood_glucose", 200))

    if hr <= 100:   hr_score = 0.85
    elif hr <= 120: hr_score = 0.65
    elif hr <= 140: hr_score = 0.45
    elif hr <= 160: hr_score = 0.25
    elif hr <= 180: hr_score = 0.12
    else:           hr_score = 0.05

    spo2_score = _vital_score(spo2, 95, 100)
    bg_score   = _vital_score(bg,   70, 140)

    weighted   = hr_score * 0.3 + spo2_score * 0.4 + bg_score * 0.3
    all_normal = (hr_score >= 0.6 and spo2_score >= 0.7 and bg_score >= 0.7)
    bonus      = 0.06 if all_normal else 0.0
    score      = weighted + bonus + 0.02

    return GradeResult(
        task_id=2,
        score=_safe(score),
        details={
            "vital_scores": {
                "heart_rate":    round(hr_score,   4),
                "spo2":          round(spo2_score, 4),
                "blood_glucose": round(bg_score,   4),
            },
            "all_vitals_normal": bool(all_normal),
            "bonus": bonus,
            "final_vitals": {
                "heart_rate": hr,
                "spo2": spo2,
                "blood_glucose": bg,
            },
        },
    )


def grade_task3(episode_log: list) -> GradeResult:
    if not episode_log:
        return GradeResult(task_id=3, score=0.05, details={"error": "empty log"})

    survivals = []
    for s in episode_log:
        sv = s.get("survival_probability", 0.5)
        survivals.append(_safe(float(sv) if sv else 0.05))

    final_survival = min(0.95, max(0.05, survivals[-1]))
    avg_survival   = min(0.95, max(0.05, sum(survivals) / max(len(survivals), 1)))

    fv = episode_log[-1].get("vitals", {})
    normal_ranges = {
        "heart_rate":       (60,   100),
        "systolic_bp":      (90,   120),
        "spo2":             (95,   100),
        "temperature":      (36.5, 37.5),
        "blood_glucose":    (70,   140),
        "respiratory_rate": (12,    20),
        "creatinine":       (0.6,   1.2),
    }
    normal_count = sum(
        1 for key, (lo, hi) in normal_ranges.items()
        if lo <= float(fv.get(key, 0)) <= hi
    )
    # Never let vitals_ratio be exactly 0.0
    vitals_ratio = max(0.001, normal_count / len(normal_ranges))

    actions        = [s.get("action", "") for s in episode_log if s.get("action")]
    unique_actions = len(set(actions))
    # Never let diversity be exactly 0.0
    diversity      = max(0.001, min(0.88, unique_actions / 6.0))

    score = (
        0.40 * final_survival
        + 0.30 * avg_survival
        + 0.20 * vitals_ratio
        + 0.10 * diversity
    )

    return GradeResult(
        task_id=3,
        score=_safe(score),
        details={
            "final_survival_probability": round(final_survival, 4),
            "average_survival":           round(avg_survival,   4),
            "vitals_normalised_ratio":    round(vitals_ratio,   4),
            "unique_actions_used":        unique_actions,
            "diversity_score":            round(diversity,      4),
            "total_steps":                len(episode_log),
        },
    )


def grade(task_id: int, episode_log: list) -> GradeResult:
    graders = {1: grade_task1, 2: grade_task2, 3: grade_task3}
    if task_id not in graders:
        raise ValueError(f"Unknown task_id {task_id}. Valid: {list(graders.keys())}")
    result = graders[task_id](episode_log)
    # Final safety net — absolutely cannot be 0.0 or 1.0
    result.score = _safe(result.score)
    return result