"""
inference.py - ICU Treatment Optimizer
"""

import os
import time
import requests
from openai import OpenAI

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
API_KEY      = os.environ.get("API_KEY", os.environ.get("HF_TOKEN", "dummy"))
ENV_URL      = os.environ.get("ENV_URL", "http://localhost:7860")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

TASKS = [1, 2, 3]
VALID_ACTIONS = ["increase_vasopressor","decrease_vasopressor","give_antibiotics","increase_insulin","decrease_insulin","give_iv_fluids","increase_oxygen","decrease_oxygen","increase_peep","decrease_peep","order_labs","call_specialist","do_nothing"]

def get_action(state, task_id):
    vitals = state.get("vitals", {})
    prompt = f"ICU task {task_id}. Vitals: BP={vitals.get('systolic_bp',100)}, HR={vitals.get('heart_rate',100)}, SpO2={vitals.get('spo2',95)}, Glucose={vitals.get('blood_glucose',120)}, RespRate={vitals.get('respiratory_rate',16)}, Step={state.get('step',0)}/{state.get('max_steps',10)}. Choose one action from: {','.join(VALID_ACTIONS)}. Reply with action name only."
    try:
        r = client.chat.completions.create(model=MODEL_NAME, messages=[{"role":"user","content":prompt}], temperature=0.1, max_tokens=20)
        action = r.choices[0].message.content.strip().lower().strip("'\" ")
        if action in VALID_ACTIONS:
            return action
    except Exception as e:
        print(f"LLM error: {e}", flush=True)
    bp = vitals.get("systolic_bp",100)
    spo2 = vitals.get("spo2",95)
    hr = vitals.get("heart_rate",100)
    glucose = vitals.get("blood_glucose",120)
    if bp < 90: return "increase_vasopressor"
    if spo2 < 95: return "increase_oxygen"
    if hr > 120: return "give_antibiotics"
    if glucose > 140: return "increase_insulin"
    if glucose < 75: return "decrease_insulin"
    return "do_nothing"

def run_episode(task_id):
    print(f"[START] task={task_id} timestamp={time.time()}", flush=True)
    try:
        state = requests.post(f"{ENV_URL}/reset", params={"task_id":task_id}, timeout=60).json()
    except Exception as e:
        print(f"[END] task={task_id} score=0.0 steps=0", flush=True)
        return {"task_id":task_id,"score":0.0,"steps":0}
    episode_log = []
    total_reward = 0.0
    while not state.get("done", False):
        action = get_action(state, task_id)
        try:
            result = requests.post(f"{ENV_URL}/step", params={"task_id":task_id}, json={"action":action}, timeout=60).json()
        except:
            break
        if "state" not in result:
            break
        new_state = result["state"]
        reward = result.get("reward", 0.0)
        done = result.get("done", False)
        total_reward += reward
        episode_log.append({"step":new_state["step"],"action":action,"vitals":new_state["vitals"],"survival_probability":new_state["survival_probability"],"reward":reward,"done":done})
        print(f"[STEP] task={task_id} step={new_state['step']} action={action} reward={round(reward,4)} survival={round(new_state['survival_probability'],4)} done={done}", flush=True)
        state = new_state
        if done:
            break
    try:
        grade_result = requests.post(f"{ENV_URL}/grade", json={"task_id":task_id,"episode_log":episode_log}, timeout=60).json()
    except:
        grade_result = {"score":0.0,"details":{}}
    final_score = grade_result.get("score", 0.0)
    print(f"[END] task={task_id} score={round(final_score,4)} steps={len(episode_log)}", flush=True)
    return {"task_id":task_id,"score":final_score,"steps":len(episode_log)}

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
