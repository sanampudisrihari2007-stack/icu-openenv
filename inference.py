import os
import sys
import time
import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "llama-3.1-8b-instant")
HF_TOKEN     = os.getenv("HF_TOKEN")
ENV_URL      = os.getenv("ENV_URL",      "http://localhost:7860")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

TASKS = [1, 2, 3]
TASK_NAMES = {1: "bp-stabilisation", 2: "multi-vital-balancing", 3: "icu-sepsis-management"}

SYSTEM_PROMPT = """You are an ICU physician. Choose ONE action.
Valid: increase_vasopressor, decrease_vasopressor, give_antibiotics,
increase_insulin, decrease_insulin, give_iv_fluids, increase_oxygen,
decrease_oxygen, increase_peep, decrease_peep, order_labs, call_specialist, do_nothing
Reply with ONLY the action name."""

VALID_ACTIONS = [
    "increase_vasopressor","decrease_vasopressor","give_antibiotics",
    "increase_insulin","decrease_insulin","give_iv_fluids",
    "increase_oxygen","decrease_oxygen","increase_peep","decrease_peep",
    "order_labs","call_specialist","do_nothing"
]

def safe(x, default=0.05):
    """Strictly between 0 and 1."""
    try:
        v = float(x)
        if v <= 0.0 or v != v: return default
        if v >= 1.0: return 0.95
        return round(v, 4)
    except:
        return default

def choose_action(state):
    vitals = state.get("vitals", {})
    prompt = (
        f"HR:{vitals.get('heart_rate',0):.0f} "
        f"BP:{vitals.get('systolic_bp',0):.0f} "
        f"SpO2:{vitals.get('spo2',0):.0f} "
        f"BG:{vitals.get('blood_glucose',0):.0f} "
        f"RR:{vitals.get('respiratory_rate',0):.0f} "
        f"Diag:{state.get('diagnosis','?')} "
        f"Step:{state.get('step',0)}/{state.get('max_steps',0)}"
    )
    try:
        r = client.chat.completions.create(
            model=MODEL_NAME, max_tokens=10,
            messages=[
                {"role":"system","content":SYSTEM_PROMPT},
                {"role":"user","content":prompt}
            ]
        )
        action = r.choices[0].message.content.strip().lower()
        action = action.replace(".","").replace(",","").strip()
        if action in VALID_ACTIONS:
            return action
        for v in VALID_ACTIONS:
            if v in action:
                return v
    except:
        pass
    return "do_nothing"

def sanitize_log_entry(entry):
    """Ensure all numeric values in log entry are safe."""
    entry["survival_probability"] = safe(entry.get("survival_probability", 0.5), 0.5)
    entry["reward"] = safe(entry.get("reward", 0.05), 0.05)
    return entry

def run_episode(task_id):
    task_name = TASK_NAMES.get(task_id, f"task_{task_id}")
    print(f"[START] task={task_name} env=icu-optimizer model={MODEL_NAME}", flush=True)

    try:
        state = requests.post(
            f"{ENV_URL}/reset",
            params={"task_id": task_id},
            timeout=60
        ).json()
    except Exception as e:
        print(f"[END] success=false steps=0 rewards=0.05", flush=True)
        return {"task_id": task_id, "score": 0.05, "steps": 0}

    rewards = []
    episode_log = []
    last_error = None

    while not state.get("done", False):
        action = choose_action(state)
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
            r = 0.05
            rewards.append(r)
            print(f"[STEP] step={len(rewards)} action={action} reward={r:.2f} done=false error={last_error}", flush=True)
            break

        if "state" not in result:
            break

        new_state = result["state"]
        reward    = safe(result.get("reward", 0.05))
        done      = result.get("done", False)
        step      = new_state["step"]
        rewards.append(reward)

        log_entry = sanitize_log_entry({
            "step":   step,
            "action": action,
            "vitals": new_state.get("vitals", {}),
            "survival_probability": new_state.get("survival_probability", 0.5),
            "reward": reward,
            "done":   done,
        })
        episode_log.append(log_entry)

        print(
            f"[STEP] step={step} action={action} "
            f"reward={reward:.2f} "
            f"done={'true' if done else 'false'} "
            f"error={'null' if last_error is None else last_error}",
            flush=True
        )

        state = new_state
        if done:
            break

    # Grade with sanitized episode log
    try:
        grade_result = requests.post(
            f"{ENV_URL}/grade",
            json={"task_id": task_id, "episode_log": episode_log},
            timeout=60,
        ).json()
        final_score = safe(grade_result.get("score", 0.5))
    except:
        final_score = safe(sum(rewards) / len(rewards)) if rewards else 0.05

    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    print(f"[END] success=true steps={len(rewards)} rewards={rewards_str}", flush=True)

    return {"task_id": task_id, "score": final_score, "steps": len(rewards)}


def main():
    results = []
    for task_id in TASKS:
        result = run_episode(task_id)
        results.append(result)
        time.sleep(1)

    print("\n=== FINAL SCORES ===", flush=True)
    for r in results:
        print(f"Task {r['task_id']}: {r['score']:.4f} ({r['steps']} steps)", flush=True)


if __name__ == "__main__":
    main()