from patient_model import GradeResult


def _safe(x):
    """
    Strictly between 0 and 1.
    Min 0.05, Max 0.95 - so f'{v:.2f}' never gives '0.00' or '1.00'.
    """
    try:
        v = float(x)
    except:
        return 0.50
    if v != v:   return 0.50  # NaN
    if v <= 0:   return 0.05
    if v >= 1:   return 0.95
    if v < 0.05: return 0.05  # prevents 0.00 when formatted
    if v > 0.95: return 0.95  # prevents 1.00 when formatted
    return round(v, 2)


def grade_task1(episode_log):
    if not episode_log:
        return GradeResult(task_id=1, score=0.05, details={"note": "empty"})

    vitals = episode_log[-1].get("vitals", {})
    try:
        sbp = float(vitals.get("systolic_bp", 0) or 0)
    except:
        sbp = 0.0

    if 90 < sbp < 120:   score = 0.85
    elif 80 <= sbp <= 90 or 120 <= sbp <= 130: score = 0.55
    elif 70 <= sbp < 80 or 130 < sbp <= 145:   score = 0.35
    elif sbp > 0:                               score = 0.15
    else:                                       score = 0.05

    return GradeResult(task_id=1, score=_safe(score), details={"sbp": sbp, "steps": len(episode_log)})


def grade_task2(episode_log):
    if not episode_log:
        return GradeResult(task_id=2, score=0.05, details={"note": "empty"})

    vitals = episode_log[-1].get("vitals", {})
    try: hr   = float(vitals.get("heart_rate",    150) or 150)
    except: hr = 150.0
    try: spo2 = float(vitals.get("spo2",           80) or 80)
    except: spo2 = 80.0
    try: bg   = float(vitals.get("blood_glucose", 200) or 200)
    except: bg = 200.0

    n = sum([60 < hr < 100, 95 < spo2 < 100, 70 < bg < 140])
    score = {0: 0.15, 1: 0.40, 2: 0.65, 3: 0.90}[n]

    return GradeResult(task_id=2, score=_safe(score), details={"vitals_normal": n})


def grade_task3(episode_log):
    if not episode_log:
        return GradeResult(task_id=3, score=0.05, details={"note": "empty"})

    survivals = []
    for e in episode_log:
        try:
            sp = float(e.get("survival_probability", 0.5) or 0.5)
            survivals.append(_safe(sp))
        except:
            survivals.append(0.50)

    if not survivals:
        survivals = [0.50]

    final = survivals[-1]
    avg   = sum(survivals) / len(survivals)
    acts  = {e.get("action","") for e in episode_log if e.get("action")}
    div   = min(0.90, len(acts) / 7.0)

    score = 0.5 * final + 0.3 * avg + 0.2 * div
    return GradeResult(task_id=3, score=_safe(score), details={"final": final, "avg": avg, "div": div})


def grade(task_id, episode_log):
    fns = {1: grade_task1, 2: grade_task2, 3: grade_task3}
    if task_id not in fns:
        raise ValueError(f"Unknown task_id {task_id}")
    result = fns[task_id](episode_log)
    result.score = _safe(result.score)
    return result