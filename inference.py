"""
inference.py - ICU Treatment Optimizer
"""
import os
import sys
import time
import requests
from openai import OpenAI

# Environment variables with defaults where required
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "llama-3.1-8b-instant")
HF_TOKEN     = os.getenv("HF_TOKEN")
ENV_URL      = os.getenv("ENV_URL",      "http://localhost:7860")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# Initialize OpenAI client
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

TASKS = [1, 2, 3]
TASK_NAMES = {1: "bp-stabilisation", 2: "multi-vital-balancing", 3: "icu-sepsis-management"}


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
    task_name = TASK_NAMES.get(task_id, f"task_{task_id}")

    # [START]
    print(f"[START] task={task_name} env=icu-optimizer model={MODEL_NAME}", flush=True)

    try:
        state = requests.post(
            f"{ENV_URL}/reset",
            params={"task_id": task_id},
            timeout=60
        ).json()
    except Exception as e:
        print(f"[END] success=false steps=0 rewards=", flush=True)
        return {"task_id": task_id, "score": 0.0, "steps": 0}

    rewards = []
    done = False
    step = 0
    last_error = None

    while not state.get("done", False):
        action = rule_based_action(state, task_id)

        try:
            result = requests.post(
                f"{ENV_URL}/step",
                params={"task_id": task_id},
                json={"action": action},
                timeout=60,
            ).json()
            last_error = None
        except Exception as e:
            last_error = str(e)
            print(f"[STEP] step={step+1} action={action} reward=0.00 done=false error={last_error}", flush=True)
            break

        if "state" not in result:
            last_error = "unexpected_response"
            break

        new_state = result["state"]
        reward    = result.get("reward", 0.0)
        done      = result.get("done", False)
        step      = new_state["step"]
        rewards.append(reward)

        # [STEP]
        print(
            f"[STEP] step={step} action={action} "
            f"reward={reward:.2f} done={'true' if done else 'false'} "
            f"error={'null' if last_error is None else last_error}",
            flush=True
        )

        state = new_state
        if done:
            break

    # Grade episode
    try:
        episode_log = []
        grade_result = requests.post(
            f"{ENV_URL}/grade",
            json={"task_id": task_id, "episode_log": episode_log},
            timeout=60,
        ).json()
        final_score = grade_result.get("score", 0.0)
        success = final_score > 0.0
    except Exception:
        final_score = 0.0
        success = False

    rewards_str = ",".join([f"{r:.2f}" for r in rewards])

    # [END]
    print(
        f"[END] success={'true' if success else 'false'} "
        f"steps={len(rewards)} rewards={rewards_str}",
        flush=True
    )

    return {"task_id": task_id, "score": final_score, "steps": len(rewards)}


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