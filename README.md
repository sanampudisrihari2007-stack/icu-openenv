#  ICU Treatment Optimizer — OpenEnv

An RL environment where an AI agent learns to recommend treatment actions
for critically ill ICU patients, maximising survival probability and
normalising vital signs.

---

##  Environment Overview

The environment simulates an ICU patient whose vitals change in response
to treatment decisions. The agent observes 8 physiological parameters and
chooses from 13 treatment actions each step.

**Diagnoses supported:** Sepsis, Pneumonia, Cardiac Arrest, Post-Surgery, Respiratory Failure

---

##  Observation Space

| Vital | Unit | Normal Range |
|---|---|---|
| Heart Rate | bpm | 60–100 |
| Systolic BP | mmHg | 90–120 |
| Diastolic BP | mmHg | 60–80 |
| SpO2 | % | 95–100 |
| Temperature | °C | 36.5–37.5 |
| Blood Glucose | mg/dL | 70–140 |
| Respiratory Rate | /min | 12–20 |
| Creatinine | mg/dL | 0.6–1.2 |

---

##  Action Space

`increase_vasopressor`, `decrease_vasopressor`, `give_antibiotics`,
`increase_insulin`, `decrease_insulin`, `give_iv_fluids`,
`increase_oxygen`, `decrease_oxygen`, `increase_peep`, `decrease_peep`,
`order_labs`, `call_specialist`, `do_nothing`

---

##  Tasks

| # | Name | Difficulty | Steps |
|---|---|---|---|
| 1 | Blood Pressure Stabilisation | Easy | 10 |
| 2 | Multi-Vital Balancing | Medium | 20 |
| 3 | Full ICU / Sepsis Management | Hard | 24 |

---

##  API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/reset?task_id=1` | Start a new episode |
| POST | `/step?task_id=1` | Take a treatment action |
| GET | `/state?task_id=1` | Get current patient state |
| GET | `/tasks` | List all tasks + valid actions |
| POST | `/grade` | Grade a completed episode |

---

##  Setup & Run

### Local

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 7860
```

### Docker

```bash
docker build -t icu-openenv .
docker run -p 7860:7860 icu-openenv
```

### Run Inference

```bash
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=your_api_key
export ENV_URL=http://localhost:7860

python inference.py
```

---

##  Reward Function

Rewards are in the range `[0.0, 1.0]` and include partial progress signals:

- **+0.4** Survival probability improvement
- **+0.3** Vitals in normal range
- **+0.3** Absolute survival probability
- **-0.15** Dangerous treatment combination
- **-0.10** Over-treatment penalty

---

##  Real-World Impact

This environment models the clinical decision support problem actively
researched at DeepMind, Stanford, and MIT. Optimal ICU treatment planning
can meaningfully reduce mortality rates in sepsis, which affects 50 million
people globally each year.