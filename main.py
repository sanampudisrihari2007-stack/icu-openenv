from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from patient_model import TreatmentAction, StepResult, GradeRequest, GradeResult, PatientState
from environment import ICUEnvironment, VALID_ACTIONS
from graders import grade
from typing import Optional
import uvicorn

app = FastAPI(
    title="ICU Treatment Optimizer — OpenEnv",
    description="RL environment for ICU patient treatment optimization.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_envs: dict[int, ICUEnvironment] = {
    1: ICUEnvironment(task_id=1),
    2: ICUEnvironment(task_id=2),
    3: ICUEnvironment(task_id=3),
}

_episode_logs: dict[int, list] = {1: [], 2: [], 3: []}


def _get_env(task_id: int) -> ICUEnvironment:
    if task_id not in _envs:
        raise HTTPException(status_code=400, detail=f"Invalid task_id {task_id}. Valid: [1, 2, 3]")
    return _envs[task_id]


def _safe_score(score: float) -> float:
    """Ensure score is strictly between 0 and 1."""
    return round(max(0.01, min(0.99, float(score))), 4)


@app.get("/")
def root():
    return {
        "status": "ok",
        "environment": "ICU Treatment Optimizer",
        "version": "1.0.0",
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/grade"],
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/reset", response_model=PatientState)
def reset(task_id: int = 1):
    env = _get_env(task_id)
    state = env.reset()
    _episode_logs[task_id] = []
    return state


@app.post("/step", response_model=StepResult)
def step(action: TreatmentAction, task_id: int = 1):
    env = _get_env(task_id)
    try:
        new_state, reward, done, info = env.step(action.action, action.dose)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Clamp reward
    reward = _safe_score(reward)

    log_entry = {
        "step": new_state.step,
        "action": action.action,
        "vitals": new_state.vitals.dict(),
        "survival_probability": new_state.survival_probability,
        "reward": reward,
        "done": done,
    }
    _episode_logs[task_id].append(log_entry)

    return StepResult(state=new_state, reward=reward, done=done, info=info)


@app.get("/state", response_model=PatientState)
def state(task_id: int = 1):
    env = _get_env(task_id)
    try:
        return env.get_state()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "task_id": 1,
                "name": "Blood Pressure Stabilisation",
                "difficulty": "easy",
                "max_steps": 10,
                "description": "Bring systolic BP into 90-120 mmHg range within 10 steps.",
                "target_vital": "systolic_bp",
                "scoring": "BP normal score + efficiency bonus",
                "reward_range": [0.01, 0.99],
            },
            {
                "task_id": 2,
                "name": "Multi-Vital Balancing",
                "difficulty": "medium",
                "max_steps": 20,
                "description": "Normalise heart_rate, spo2, and blood_glucose simultaneously.",
                "target_vitals": ["heart_rate", "spo2", "blood_glucose"],
                "scoring": "Weighted average of 3 vital scores + all-normal bonus",
                "reward_range": [0.01, 0.99],
            },
            {
                "task_id": 3,
                "name": "Full ICU / Sepsis Management",
                "difficulty": "hard",
                "max_steps": 24,
                "description": "Maximise patient survival probability over a 24-step episode.",
                "scoring": "Composite: final survival (40%), avg survival (30%), vitals (20%), diversity (10%)",
                "reward_range": [0.01, 0.99],
            },
        ],
        "valid_actions": VALID_ACTIONS,
    }


@app.post("/grade", response_model=GradeResult)
def grade_episode(request: GradeRequest):
    log = request.episode_log if request.episode_log else _episode_logs.get(request.task_id, [])
    try:
        result = grade(request.task_id, log)
        # Ensure score is strictly between 0 and 1
        result.score = _safe_score(result.score)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/grade/last/{task_id}", response_model=GradeResult)
def grade_last_episode(task_id: int):
    log = _episode_logs.get(task_id, [])
    if not log:
        raise HTTPException(status_code=404, detail="No episode log found. Run an episode first.")
    try:
        result = grade(task_id, log)
        result.score = _safe_score(result.score)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=False)