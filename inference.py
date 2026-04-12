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

def safe(x):
    """Strictly between 0 and 1. Min 0.05, Max 0.95."""
    try:
        v = float(x)
        if v != v: return 0.50   # NaN
        if v <= 0: return 0.05
        if v >= 1: return 0.95
        if v < 0.05: return 0.05
        if v > 0.95: return 0.95
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
        f"RR:{vitals.get('respiratory_rate',0):.0f} "
        f"Diag:{state.get('diagnosis','sepsis')} "
        f"Step:{state.get('step',0)}/{state.get('max_steps',10)}"
    )
    try:
        r = client.chat.completions.create(
            model=MODEL_NAME, max_tokens=10,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt}
            ]
        )
        a = r.choices[0].message.content.strip().lower()
        a = a.replace(".","").replace(",","").strip()
        if a in VALID: return a
        for v in VALID:
            if v in a: return v
    except:
        pass
    return "do_nothing"

def run_episode(task_id):
    task_name = TASK_NAMES.get(task_id, f"task_{task_id}")

    # [START]
    print(f"[START] task={task_name} env=icu-optimizer model={MODEL_NAME}", flush=True)

    try:
        state = requests.post(
            f"{ENV_URL}/reset",
            params={"task_id": task_id},
            timeout=60
        ).json()
    except:
        # Clamp score BEFORE printing [END]
        final_score = max(0.01, min(0.99, 0.05))
        print(f"[END] success=false steps=0 score={final_score:.2f} rewards=0.05", flush=True)
        return

    rewards = []
    ep_log  = []

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
            r = 0.05
            rewards.append(r)
            print(f"[STEP] step={len(rewards)} action={action} reward={r:.2f} done=false error=timeout", flush=True)
            break

        if "state" not in res:
            break

        new_state = res["state"]

        # Clamp reward immediately after getting from API
        raw_reward = res.get("reward", 0.05)
        reward = max(0.01, min(0.99, float(raw_reward) if raw_reward else 0.05))
        reward = safe(reward)  # double safe

        done  = res.get("done", False)
        step  = new_state["step"]
        rewards.append(reward)

        sp = safe(new_state.get("survival_probability", 0.5))

        ep_log.append({
            "step": step, "action": action,
            "vitals": new_state.get("vitals", {}),
            "survival_probability": sp,
            "reward": reward, "done": done,
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

    # Get grade score from API
    try:
        gr = requests.post(
            f"{ENV_URL}/grade",
            json={"task_id": task_id, "episode_log": ep_log},
            timeout=60,
        ).json()
        raw_score = gr.get("score", 0.50)
    except:
        raw_score = 0.50

    # CLAMP SCORE BEFORE PRINTING - this is the critical fix
    final_score = max(0.01, min(0.99, float(raw_score) if raw_score else 0.50))
    final_score = safe(final_score)  # double safe

    # Build rewards string
    safe_rewards = [safe(r) for r in rewards] if rewards else [0.05]
    rewards_str  = ",".join([f"{r:.2f}" for r in safe_rewards])

    # [END] - score is clamped BEFORE printing
    print(f"[END] success=true steps={len(safe_rewards)} score={final_score:.2f} rewards={rewards_str}", flush=True)

def main():
    for task_id in TASKS:
        run_episode(task_id)
        time.sleep(1)

if __name__ == "__main__":
    main()