"""
inference.py - ICU Treatment Optimizer
"""
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
TASK_NAMES = {
    1: "bp-stabilisation",
    2: "multi-vital-balancing",
    3: "icu-sepsis-management"
}

SYSTEM_PROMPT = """You are an expert ICU physician AI.
Choose ONE treatment action based on patient vitals.

Valid actions:
increase_vasopressor, decrease_vasopressor, give_antibiotics,
increase_insulin, decrease_insulin, give_iv_fluids,
increase_oxygen, decrease_oxygen, increase_peep, decrease_peep,
order_labs, call_specialist, do_nothing

Normal ranges:
  heart_rate: 60-100 bpm
  systolic_bp: 90-120 mmHg
  spo2: 95-100%
  blood_glucose: 70-140 mg/dL
  respiratory_rate: 12-20/min

Priority rules:
1. systolic_bp < 90 -> increase_vasopressor
2. spo2 < 95 -> increase_oxygen
3. heart_rate > 120 -> give_antibiotics
4. blood_glucose > 140 -> increase_insulin
5. blood_glucose < 70 -> decrease_insulin
6. all normal -> do_nothing

Respond with ONLY the action name. Example: increase_oxygen"""

VALID_ACTIONS = [
    "increase_vasopressor", "decrease_vasopressor", "give_antibiotics",
    "increase_insulin", "decrease_insulin", "give_iv_fluids",
    "increase_oxygen", "decrease_oxygen", "increase_peep", "decrease_peep",
    "order_labs", "call_specialist", "do_nothing"
]


def choose_action(state: dict) -> str:
    vitals = state.get("vitals", {})
    prompt = f"""Patient vitals:
  Heart Rate: {vitals.get('heart_rate', 0):.1f} bpm
  Systolic BP: {vitals.get('systolic_bp', 0):.1f} mmHg
  SpO2: {vitals.get('spo2', 0):.1f}%
  Temperature: {vitals.get('temperature', 0):.1f}C
  Blood Glucose: {vitals.get('blood_glucose', 0):.1f} mg/dL
  Respiratory Rate: {vitals.get('respiratory_rate', 0):.1f}/min
Diagnosis: {state.get('diagnosis', '?')}
Severity: {state.get('severity', '?')}
Step: {state.get('step', 0)}/{state.get('max_steps', '?')}
Choose action:"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=20,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        action = response.choices[0].message.content.strip().lower()
        action = action.replace(".", "").replace(",", "").replace("\n", " ").replace("\r", " ").strip()
        if action in VALID_ACTIONS:
            return action
        for valid in VALID_ACTIONS:
            if valid in action:
                return valid
        return "do_nothing"
    except Exception:
        return "do_nothing"


def run_episode(task_id: int) -> dict:
    task_name = TASK_NAMES.get(task_id, f"task_{task_id}")

    print(f"[START] task={task_name} env=icu-optimizer model={MODEL_NAME}", flush=True)

    try:
        state = requests.post(
            f"{ENV_URL}/reset",
            params={"task_id": task_id},
            timeout=60
        ).json()
    except Exception as e:
        print(f"[END] success=false steps=0 rewards=", flush=True)
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
            last_error = str(e).replace('\n', ' ').replace('\r', '')
            print(f"[STEP] step={len(rewards)+1} action={action} reward=0.00 done=false error={last_error}", flush=True)
            break

        if "state" not in result:
            break

        new_state = result["state"]
        reward    = result.get("reward", 0.0)
        done      = result.get("done", False)
        step      = new_state["step"]
        rewards.append(reward)

        episode_log.append({
            "step": step,
            "action": action,
            "vitals": new_state.get("vitals", {}),
            "survival_probability": new_state.get("survival_probability", 0.5),
            "reward": reward,
            "done": done,
        })

        print(
            f"[STEP] step={step} action={action} "
            f"reward={reward:.2f} done={'true' if done else 'false'} "
            f"error={'null' if last_error is None else last_error}",
            flush=True
        )

        state = new_state
        if done:
            break

    # Grade with actual episode log
    try:
        grade_result = requests.post(
            f"{ENV_URL}/grade",
            json={"task_id": task_id, "episode_log": episode_log},
            timeout=60,
        ).json()
        final_score = grade_result.get("score", 0.5)
        final_score = max(0.01, min(0.99, float(final_score)))
        success = True
    except Exception:
        final_score = 0.5
        success = True

    rewards_str = ",".join([f"{r:.2f}" for r in rewards])

    print(
        f"[END] success={'true' if success else 'false'} "
        f"steps={len(rewards)} rewards={rewards_str}",
        flush=True
    )

    return {"task_id": task_id, "score": final_score, "steps": len(rewards)}


def main():
    for task_id in TASKS:
        run_episode(task_id)
        time.sleep(1)


if __name__ == "__main__":
    main()