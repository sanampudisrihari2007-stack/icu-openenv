"""
Graders for ICU Treatment Optimizer.
All scores are STRICTLY between 0.0 and 1.0 (exclusive).
Minimum: 0.001, Maximum: 0.999
"""
from patient_model import GradeResult


def _bound(x: float) -> float:
    """Strictly between 0 and 1 — guaranteed."""
    try:
        v = float(x)
    except Exception:
        return 0.5
    if v != v:  # NaN check
        return 0.5
    # Hard clamp
    if v <= 0.0:
        return 0.001
    if v >= 1.0:
        return 0.999
    return round(v, 6)


def grade_task1(episode_log: list) -> GradeResult:
    """Task 1: Blood Pressure Stabilisation"""
    if not episode_log:
        return GradeResult(task_id=1, score=0.001, details={"note": "empty"})

    vitals = episode_log[-1].get("vitals", {})
    sbp = vitals.get("systolic_bp", 0)
    try:
        sbp = float(sbp) if sbp else 0.0
    except Exception:
        sbp = 0.0

    # Score based on how close BP is to normal (90-120)
    if 90.0 < sbp < 120.0:
        score = 0.85
    elif 80.0 <= sbp <= 90.0 or 120.0 <= sbp <= 130.0:
        score = 0.55
    elif 70.0 <= sbp < 80.0 or 130.0 < sbp <= 145.0:
        score = 0.35
    elif sbp > 0:
        score = 0.15
    else:
        score = 0.001

    return GradeResult(
        task_id=1,
        score=_bound(score),
        details={"sbp": sbp, "steps": len(episode_log)}
    )


def grade_task2(episode_log: list) -> GradeResult:
    """Task 2: Multi-Vital Balancing"""
    if not episode_log:
        return GradeResult(task_id=2, score=0.001, details={"note": "empty"})

    vitals = episode_log[-1].get("vitals", {})

    try:
        hr = float(vitals.get("heart_rate", 150) or 150)
    except Exception:
        hr = 150.0
    try:
        spo2 = float(vitals.get("spo2", 80) or 80)
    except Exception:
        spo2 = 80.0
    try:
        bg = float(vitals.get("blood_glucose", 200) or 200)
    except Exception:
        bg = 200.0

    hr_ok   = 60.0 < hr   < 100.0
    spo2_ok = 95.0 < spo2 < 100.0
    bg_ok   = 70.0 < bg   < 140.0

    count = sum([hr_ok, spo2_ok, bg_ok])
    score_map = {0: 0.15, 1: 0.40, 2: 0.65, 3: 0.90}
    score = score_map[count]

    return GradeResult(
        task_id=2,
        score=_bound(score),
        details={"hr_ok": hr_ok, "spo2_ok": spo2_ok, "bg_ok": bg_ok, "count": count}
    )


def grade_task3(episode_log: list) -> GradeResult:
    """Task 3: Full ICU Management"""
    if not episode_log:
        return GradeResult(task_id=3, score=0.001, details={"note": "empty"})

    # Collect survival probabilities
    survivals = []
    for entry in episode_log:
        try:
            sp = float(entry.get("survival_probability", 0) or 0)
            if 0.0 < sp < 1.0:
                survivals.append(sp)
            elif sp >= 1.0:
                survivals.append(0.999)
            elif sp <= 0.0:
                survivals.append(0.001)
        except Exception:
            survivals.append(0.5)

    if not survivals:
        survivals = [0.5]

    avg = sum(survivals) / len(survivals)
    final = survivals[-1]

    # Count unique actions for diversity
    actions = set()
    for entry in episode_log:
        a = entry.get("action", "")
        if a:
            actions.add(a)
    diversity = min(0.9, len(actions) / 7.0)

    score = 0.5 * final + 0.3 * avg + 0.2 * diversity

    return GradeResult(
        task_id=3,
        score=_bound(score),
        details={
            "final_survival": round(final, 4),
            "avg_survival": round(avg, 4),
            "unique_actions": len(actions),
            "steps": len(episode_log)
        }
    )


def grade(task_id: int, episode_log: list) -> GradeResult:
    """Route to correct grader and ensure score is valid."""
    graders = {1: grade_task1, 2: grade_task2, 3: grade_task3}
    if task_id not in graders:
        raise ValueError(f"Unknown task_id {task_id}")

    result = graders[task_id](episode_log)

    # Final safety net - absolutely guaranteed
    result.score = _bound(result.score)

    # Double check
    assert 0.0 < result.score < 1.0, f"Score {result.score} out of range!"

    return result