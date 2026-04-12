from patient_model import GradeResult


def grade_task1(episode_log: list) -> GradeResult:
    score = 0.50
    if episode_log:
        vitals = episode_log[-1].get("vitals", {})
        sbp = float(vitals.get("systolic_bp", 70) or 70)
        if 90 < sbp < 120:
            score = 0.85
        elif 75 < sbp <= 90 or 120 <= sbp < 135:
            score = 0.55
        else:
            score = 0.25
    return GradeResult(task_id=1, score=score, details={"steps": len(episode_log)})


def grade_task2(episode_log: list) -> GradeResult:
    score = 0.50
    if episode_log:
        vitals = episode_log[-1].get("vitals", {})
        hr   = float(vitals.get("heart_rate",    150) or 150)
        spo2 = float(vitals.get("spo2",           80) or 80)
        bg   = float(vitals.get("blood_glucose", 200) or 200)
        n = sum([60 < hr < 100, 95 < spo2 < 100, 70 < bg < 140])
        score = [0.20, 0.45, 0.70, 0.90][n]
    return GradeResult(task_id=2, score=score, details={"steps": len(episode_log)})


def grade_task3(episode_log: list) -> GradeResult:
    score = 0.50
    if episode_log:
        total = 0.0
        count = 0
        for entry in episode_log:
            sp = entry.get("survival_probability", 0.5)
            try:
                sp = float(sp)
                if 0 < sp < 1:
                    total += sp
                    count += 1
            except:
                pass
        if count > 0:
            avg = total / count
            score = round(max(0.10, min(0.90, avg)), 4)
    return GradeResult(task_id=3, score=score, details={"steps": len(episode_log)})


def grade(task_id: int, episode_log: list) -> GradeResult:
    fns = {1: grade_task1, 2: grade_task2, 3: grade_task3}
    if task_id not in fns:
        raise ValueError(f"Unknown task_id {task_id}")
    result = fns[task_id](episode_log)
    # Absolute final safety net
    v = float(result.score)
    if v <= 0 or v >= 1 or v != v:
        result.score = 0.50
    return result