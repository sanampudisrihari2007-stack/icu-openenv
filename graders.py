from patient_model import GradeResult


def _s(v):
    """100% guaranteed strictly between 0 and 1."""
    try:
        f = float(v)
    except:
        return 0.5
    if f != f:   return 0.5   # NaN
    if f <= 0:   return 0.05
    if f >= 1:   return 0.95
    return round(f, 4)


def grade_task1(episode_log):
    if not episode_log:
        return GradeResult(task_id=1, score=0.5, details={"note": "no log"})
    vitals = episode_log[-1].get("vitals", {})
    sbp = float(vitals.get("systolic_bp", 70))
    if 90 < sbp < 120:
        score = 0.85
    elif 80 < sbp <= 90 or 120 <= sbp < 130:
        score = 0.60
    elif 70 < sbp <= 80 or 130 <= sbp < 140:
        score = 0.40
    else:
        score = 0.20
    steps = len(episode_log)
    if score >= 0.80 and steps <= 5:
        score = min(0.94, score + 0.10)
    return GradeResult(task_id=1, score=_s(score), details={"sbp": sbp, "steps": steps})


def grade_task2(episode_log):
    if not episode_log:
        return GradeResult(task_id=2, score=0.5, details={"note": "no log"})
    vitals = episode_log[-1].get("vitals", {})
    hr   = float(vitals.get("heart_rate",    150))
    spo2 = float(vitals.get("spo2",           80))
    bg   = float(vitals.get("blood_glucose", 200))

    hr_ok   = 60 < hr   < 100
    spo2_ok = 95 < spo2 < 100
    bg_ok   = 70 < bg   < 140

    n = sum([hr_ok, spo2_ok, bg_ok])
    if n == 3:   score = 0.90
    elif n == 2: score = 0.65
    elif n == 1: score = 0.40
    else:        score = 0.20

    return GradeResult(task_id=2, score=_s(score), details={"hr_ok": hr_ok, "spo2_ok": spo2_ok, "bg_ok": bg_ok})


def grade_task3(episode_log):
    if not episode_log:
        return GradeResult(task_id=3, score=0.5, details={"note": "no log"})

    survivals = []
    for entry in episode_log:
        sp = entry.get("survival_probability", 0.5)
        try:
            sp = float(sp)
        except:
            sp = 0.5
        if sp <= 0 or sp != sp:
            sp = 0.05
        if sp >= 1:
            sp = 0.95
        survivals.append(sp)

    final_s = survivals[-1]
    avg_s   = sum(survivals) / len(survivals)

    actions = [e.get("action", "") for e in episode_log if e.get("action")]
    diversity = min(0.90, len(set(actions)) / 6)

    score = 0.50 * final_s + 0.30 * avg_s + 0.20 * diversity

    return GradeResult(task_id=3, score=_s(score), details={"final_survival": final_s, "avg_survival": avg_s, "diversity": diversity})


def grade(task_id, episode_log):
    fns = {1: grade_task1, 2: grade_task2, 3: grade_task3}
    if task_id not in fns:
        raise ValueError(f"Unknown task_id {task_id}")
    result = fns[task_id](episode_log)
    result.score = _s(result.score)
    return result