"""
inference.py - ICU Treatment Optimizer
"""

import os
import json
import time
import requests
from openai import OpenAI

# Environment variables - NO hardcoded secrets
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "llama-3.1-8b-instant")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
ENV_URL      = os.environ.get("ENV_URL", "http://localhost:7860")

TASKS = [1, 2, 3]


def get_client():
    return OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN
    )


def get_llm_action(state: dict, task_id: int) -> str:
    vitals = state.get("vitals", {})
    hr = vitals.get("heart_rate", 100)
    bp = vitals.get("systolic_bp", 100)
    spo2 = vitals.get("spo2", 95)
    glucose = vitals.get("blood_glucose", 120)
    resp = vitals.get("respiratory_rate", 16)
    step = state.get("step", 0)
    max_steps = state.get("max_steps", 10)

    # Rule-based strategy (reliable, no LLM dependency)
    if task_id == 1:
        if bp < 90:
            return "increase_vasopressor"
        elif bp < 95 and step < 5:
            return "give_iv_fluids"
        else:
            return "do_nothing"

    elif task_id == 2:
        remaining = max_steps - step
        if hr > 120 and remaining > 10:
            return "give_antibiotics"
        elif spo2 < 95:
            return "increase_oxygen"
        elif glucose > 140:
            return "increase_insulin"
        elif glucose < 75:
            return "decrease_insulin"
        elif hr > 120:
            return "give_antibiotics"
        else:
            return "do_nothing"

    elif task_id == 3:
        remaining = max_steps - step
        if bp < 90:
            return "increase_vasopressor"
        elif bp < 100 and step < 5:
            return "give_iv_fluids"
        elif spo2 < 95:
            return "increase_oxygen"
        elif resp > 20 and step < 12:
            return "increase_peep"
        elif hr > 120 and remaining > 8:
            return "give_antibiotics"
        elif glucose > 140 and remaining > 5:
            return "increase_insulin"
        elif glucose < 75:
            return "decrease_insulin"
        elif step == max_steps - 3:
            return "call_specialist"
        elif step == max_steps - 2:
            return "order_labs"
        else:
            return "do_nothing"

    return "do_nothing"


def run_episode(task_id: int) -> dict:
    print(json.dumps({
        "event": "START",
        "task_id": task_id,
        "timestamp": time.time(),
    }), flush=True)

    try:
        state = requests.post(
            f"{ENV_URL}/reset",
            params={"task_id": task_id},
            timeout=60
        ).json()
    except Exception as e:
        print(f"Reset error: {e}", flush=True)
        print(json.dumps({
            "event": "END",
            "task_id": task_id,
            "total_steps": 0,
            "total_reward": 0.0,
            "final_score": 0.0,
            "details": {}
        }), flush=True)
        return {"task_id": task_id, "score": 0.0, "steps": 0}

    episode_log = []
    total_reward = 0.0

    while not state.get("done", False):
        action = get_llm_action(state, task_id)

        try:
            result = requests.post(
                f"{ENV_URL}/step",
                params={"task_id": task_id},
                json={"action": action},
                timeout=60,
            ).json()
        except Exception:
            break

        if "state" not in result:
            break

        new_state = result["state"]
        reward = result.get("reward", 0.0)
        done = result.get("done", False)
        total_reward += reward

        log_entry = {
            "step": new_state["step"],
            "action": action,
            "vitals": new_state["vitals"],
            "survival_probability": new_state["survival_probability"],
            "reward": reward,
            "done": done,
        }
        episode_log.append(log_entry)

        print(json.dumps({
            "event": "STEP",
            "task_id": task_id,
            "step": new_state["step"],
            "action": action,
            "reward": round(reward, 4),
            "survival_probability": round(new_state["survival_probability"], 4),
            "done": done,
        }), flush=True)

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

    print(json.dumps({
        "event": "END",
        "task_id": task_id,
        "total_steps": len(episode_log),
        "total_reward": round(total_reward, 4),
        "final_score": round(final_score, 4),
        "details": grade_result.get("details", {}),
    }), flush=True)

    return {
        "task_id": task_id,
        "score": final_score,
        "steps": len(episode_log)
    }


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