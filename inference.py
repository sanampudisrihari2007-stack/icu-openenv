"""
inference.py - ICU Treatment Optimizer
"""
import os
import json
import time
import sys
import requests

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "llama-3.1-8b-instant")
HF_TOKEN     = os.environ.get("HF_TOKEN",  "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")
ENV_URL      = os.environ.get("ENV_URL",      "https://har12334-icu-treatment-optimizer.hf.space")

TASKS = [1, 2, 3]


def rule_based_action(state: dict, task_id: int) -> str:
    vitals  = state.get("vitals", {})
    history = state.get("treatment_history", [])

    sbp  = vitals.get("systolic_bp", 100)
    spo2 = vitals.get("spo2", 95)
    hr   = vitals.get("heart_rate", 80)
    bg   = vitals.get("blood_glucose", 100)
    rr   = vitals.get("respiratory_rate", 15)

    antibiotic_count  = history.count("give_antibiotics")
    insulin_count     = history.count("increase_insulin")
    oxygen_count      = history.count("increase_oxygen")
    vasopressor_count = history.count("increase_vasopressor")

    if task_id == 1:
        if sbp < 100 and vasopressor_count < 4:
            return "increase_vasopressor"
        return "do_nothing"

    elif task_id == 2:
        if sbp < 90:
            return "increase_vasopressor"
        if bg < 75:
            return "decrease_insulin"
        if hr > 100 and antibiotic_count < 9:
            return "give_antibiotics"
        if spo2 < 95 and oxygen_count < 6:
            return "increase_oxygen"
        if bg > 140 and insulin_count < 5:
            return "increase_insulin"
        return "do_nothing"

    elif task_id == 3:
        if sbp < 90 and vasopressor_count < 3:
            return "increase_vasopressor"
        if sbp < 100 and history.count("give_iv_fluids") < 2:
            return "give_iv_fluids"
        if spo2 < 95 and oxygen_count < 8:
            return "increase_oxygen"
        if rr > 20 and history.count("increase_peep") < 2:
            return "increase_peep"
        if hr > 100 and antibiotic_count < 6:
            return "give_antibiotics"
        if bg > 140 and insulin_count < 5:
            return "increase_insulin"
        if bg < 75:
            return "decrease_insulin"
        if history.count("call_specialist") < 1:
            return "call_specialist"
        if history.count("order_labs") < 1:
            return "order_labs"
        return "do_nothing"

    return "do_nothing"


def run_episode(task_id: int) -> dict:
    task_name = f"task{task_id}"

    # [START] block — plain text format required by validator
    print(f"[START] task={task_name}", flush=True)

    try:
        state = requests.post(
            f"{ENV_URL}/reset",
            params={"task_id": task_id},
            timeout=60
        ).json()
    except Exception as e:
        print(f"[END] task={task_name} score=0.0 steps=0", flush=True)
        return {"task_id": task_id, "score": 0.0, "steps": 0}

    episode_log = []
    total_reward = 0.0
    step_num = 0

    while not state.get("done", False):
        action = rule_based_action(state, task_id)

        try:
            result = requests.post(
                f"{ENV_URL}/step",
                params={"task_id": task_id},
                json={"action": action},
                timeout=60,
            ).json()
        except Exception as e:
            break

        if "state" not in result:
            break

        new_state    = result["state"]
        reward       = result.get("reward", 0.0)
        done         = result.get("done", False)
        total_reward += reward
        step_num     = new_state.get("step", step_num + 1)

        log_entry = {
            "step":                 step_num,
            "action":               action,
            "vitals":               new_state["vitals"],
            "survival_probability": new_state["survival_probability"],
            "reward":               reward,
            "done":                 done,
        }
        episode_log.append(log_entry)

        # [STEP] block — plain text format required by validator
        print(
            f"[STEP] task={task_name} step={step_num} "
            f"action={action} reward={round(reward, 4)} "
            f"survival_prob={round(new_state['survival_probability'], 4)} "
            f"done={done}",
            flush=True
        )

        state = new_state
        if done:
            break

    try:
        grade_result = requests.post(
            f"{ENV_URL}/grade",
            json={"task_id": task_id, "episode_log": episode_log},
            timeout=60,
        ).json()
    except Exception:
        grade_result = {"score": 0.0, "details": {}}

    final_score = grade_result.get("score", 0.0)

    # [END] block — plain text format required by validator
    print(
        f"[END] task={task_name} score={round(final_score, 4)} steps={len(episode_log)}",
        flush=True
    )

    return {"task_id": task_id, "score": final_score, "steps": len(episode_log)}


def main():
    results = []
    for task_id in TASKS:
        result = run_episode(task_id)
        results.append(result)
        time.sleep(1)

    print("\n=== FINAL SCORES ===", flush=True)
    for r in results:
        print(f"Task {r['task_id']}: {r['score']:.4f}  ({r['steps']} steps)", flush=True)


if __name__ == "__main__":
    main()