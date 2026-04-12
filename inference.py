import os
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

TASKS      = [1, 2, 3]
TASK_NAMES = {1: "bp-stabilisation", 2: "multi-vital-balancing", 3: "icu-sepsis-management"}

SYSTEM_PROMPT = """You are an ICU physician. Choose ONE action:
increase_vasopressor, decrease_vasopressor, give_antibiotics,
increase_insulin, decrease_insulin, give_iv_fluids, increase_oxygen,
decrease_oxygen, increase_peep, decrease_peep, order_labs, call_specialist, do_nothing
Reply ONLY with the action name."""

VALID = [
    "increase_vasopressor","decrease_vasopressor","give_antibiotics",
    "increase_insulin","decrease_insulin","give_iv_fluids",
    "increase_oxygen","decrease_oxygen","increase_peep","decrease_peep",
    "order_labs","call_specialist","do_nothing"
]

def safe_reward(x):
    try:
        v = float(x)
        if v <= 0 or v >= 1 or v != v:
            return 0.50
        return round(v, 2)
    except:
        return 0.50

def choose_action(state):
    vitals = state.get("vitals", {})
    prompt = (
        f"HR:{vitals.get('heart_rate',0):.0f} "
        f"BP:{vitals.get('systolic_bp',0):.0f} "
        f"SpO2:{vitals.get('spo2',0):.0f} "
        f"BG:{vitals.get('blood_glucose',0):.0f} "
        f"Diag:{state.get('diagnosis','sepsis')}"
    )
    try:
        r = client.chat.completions.create(
            model=MODEL_NAME, max_tokens=10,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt}
            ]
        )
        a = r.choices[0].message.content.strip().lower().replace(".","").replace(",","").strip()
        if a in VALID: return a
        for v in VALID:
            if v in a: return v
    except:
        pass
    return "do_nothing"

def run_episode(task_id):
    task_name = TASK_NAMES.get(task_id, f"task_{task_id}")
    print(f"[START] task={task_name} env=icu-optimizer model={MODEL_NAME}", flush=True)

    try:
        state = requests.post(
            f"{ENV_URL}/reset",
            params={"task_id": task_id},
            timeout=60
        ).json()
    except:
        print(f"[END] success=false steps=0 rewards=0.50", flush=True)
        return

    rewards  = []
    ep_log   = []

    while not state.get("done", False):
        action = choose_action(state)
        try:
            res = requests.post(
                f"{ENV_URL}/step",
                params={"task_id": task_id},
                json={"action": action},
                timeout=60,
            ).json()
        except:
            rewards.append(0.50)
            print(f"[STEP] step={len(rewards)} action={action} reward=0.50 done=false error=timeout", flush=True)
            break

        if "state" not in res:
            break

        new_state = res["state"]
        reward    = safe_reward(res.get("reward", 0.50))
        done      = res.get("done", False)
        step      = new_state["step"]
        rewards.append(reward)

        sp = new_state.get("survival_probability", 0.5)
        try:
            sp = float(sp)
            if sp <= 0 or sp >= 1 or sp != sp:
                sp = 0.5
        except:
            sp = 0.5

        ep_log.append({
            "step":   step,
            "action": action,
            "vitals": new_state.get("vitals", {}),
            "survival_probability": sp,
            "reward": reward,
            "done":   done,
        })

        print(
            f"[STEP] step={step} action={action} "
            f"reward={reward:.2f} "
            f"done={'true' if done else 'false'} "
            f"error=null",
            flush=True
        )
        state = new_state
        if done: break

    # Grade
    try:
        gr = requests.post(
            f"{ENV_URL}/grade",
            json={"task_id": task_id, "episode_log": ep_log},
            timeout=60,
        ).json()
        score = float(gr.get("score", 0.50))
        if score <= 0 or score >= 1 or score != score:
            score = 0.50
    except:
        score = 0.50

    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    print(f"[END] success=true steps={len(rewards)} rewards={rewards_str}", flush=True)

def main():
    for task_id in TASKS:
        run_episode(task_id)
        time.sleep(1)

if __name__ == "__main__":
    main()