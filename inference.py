"""
inference.py — ICU Treatment Optimizer
LLM-based agent that interacts with the OpenEnv environment.

Required env vars:
  API_BASE_URL  : LLM API base URL
  MODEL_NAME    : model identifier
  HF_TOKEN      : Hugging Face / API key
"""

import os
import json
import time
import requests
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")
ENV_URL      = os.environ.get("ENV_URL",      "http://localhost:7860")

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

TASKS = [1, 2, 3]

SYSTEM_PROMPT = """You are an expert ICU physician AI.
You will receive a patient's current vitals and must choose ONE treatment action.

Valid actions:
- increase_vasopressor   : raises blood pressure
- decrease_vasopressor   : lowers blood pressure
- give_antibiotics       : reduces fever and infection
- increase_insulin       : lowers blood glucose
- decrease_insulin       : raises blood glucose
- give_iv_fluids         : raises blood pressure, improves kidney function
- increase_oxygen        : raises SpO2
- decrease_oxygen        : lowers SpO2
- increase_peep          : improves lung function, slightly lowers BP
- decrease_peep          : reduces lung pressure
- order_labs             : observation only
- call_specialist        : small survival boost
- do_nothing             : take no action

Normal ranges:
  heart_rate: 60–100 bpm
  systolic_bp: 90–120 mmHg
  spo2: 95–100 %
  temperature: 36.5–37.5 °C
  blood_glucose: 70–140 mg/dL
  respiratory_rate: 12–20 /min
  creatinine: 0.6–1.2 mg/dL

Respond with ONLY a JSON object:
{"action": "<action_name>", "reasoning": "<one sentence>"}
"""


def call_env(method: str, path: str, payload: dict = None):
    url = f"{ENV_URL}{path}"
    if method == "GET":
        r = requests.get(url, params=payload or {}, timeout=30)
    else:
        r = requests.post(url, json=payload or {}, timeout=30)
    r.raise_for_status()
    return r.json()


def choose_action(state: dict) -> str:
    vitals = state.get("vitals", {})
    prompt = f"""
Patient vitals:
  Heart Rate:       {vitals.get('heart_rate', '?'):.1f} bpm
  Systolic BP:      {vitals.get('systolic_bp', '?'):.1f} mmHg
  Diastolic BP:     {vitals.get('diastolic_bp', '?'):.1f} mmHg
  SpO2:             {vitals.get('spo2', '?'):.1f} %
  Temperature:      {vitals.get('temperature', '?'):.1f} °C
  Blood Glucose:    {vitals.get('blood_glucose', '?'):.1f} mg/dL
  Respiratory Rate: {vitals.get('respiratory_rate', '?'):.1f} /min
  Creatinine:       {vitals.get('creatinine', '?'):.2f} mg/dL

Diagnosis: {state.get('diagnosis', '?')}
Severity:  {state.get('severity', '?')}
Step:      {state.get('step', 0)} / {state.get('max_steps', '?')}
Survival probability: {state.get('survival_probability', 0):.2f}

Choose the best treatment action.
"""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=150,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
    )
    raw = response.choices[0].message.content.strip()
    try:
        parsed = json.loads(raw)
        return parsed.get("action", "do_nothing")
    except json.JSONDecodeError:
        return "do_nothing"


def run_episode(task_id: int) -> dict:
    # ── [START] ──────────────────────────────────────────────────────────────
    print(json.dumps({
        "event":   "START",
        "task_id": task_id,
        "timestamp": time.time(),
    }))

    state = call_env("POST", "/reset", {"task_id": task_id})  # note: passed as query param
    # FastAPI reads task_id as query param for /reset
    state = requests.post(f"{ENV_URL}/reset?task_id={task_id}", timeout=30).json()

    episode_log = []
    total_reward = 0.0

    while not state.get("done", False):
        action = choose_action(state)
        result = requests.post(
            f"{ENV_URL}/step?task_id={task_id}",
            json={"action": action},
            timeout=30,
        ).json()

        new_state = result["state"]
        reward    = result["reward"]
        done      = result["done"]
        total_reward += reward

        log_entry = {
            "step":                 new_state["step"],
            "action":               action,
            "vitals":               new_state["vitals"],
            "survival_probability": new_state["survival_probability"],
            "reward":               reward,
            "done":                 done,
        }
        episode_log.append(log_entry)

        # ── [STEP] ────────────────────────────────────────────────────────
        print(json.dumps({
            "event":   "STEP",
            "task_id": task_id,
            "step":    new_state["step"],
            "action":  action,
            "reward":  round(reward, 4),
            "survival_probability": round(new_state["survival_probability"], 4),
            "done":    done,
        }))

        state = new_state
        if done:
            break

    # Grade the episode
    grade_result = requests.post(
        f"{ENV_URL}/grade",
        json={"task_id": task_id, "episode_log": episode_log},
        timeout=30,
    ).json()

    final_score = grade_result.get("score", 0.0)

    # ── [END] ─────────────────────────────────────────────────────────────
    print(json.dumps({
        "event":        "END",
        "task_id":      task_id,
        "total_steps":  len(episode_log),
        "total_reward": round(total_reward, 4),
        "final_score":  round(final_score, 4),
        "details":      grade_result.get("details", {}),
    }))

    return {"task_id": task_id, "score": final_score, "steps": len(episode_log)}


def main():
    results = []
    for task_id in TASKS:
        result = run_episode(task_id)
        results.append(result)
        time.sleep(1)

    print("\n=== FINAL SCORES ===")
    for r in results:
        print(f"Task {r['task_id']}: {r['score']:.4f}  ({r['steps']} steps)")


if __name__ == "__main__":
    main()