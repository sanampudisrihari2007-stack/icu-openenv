---
title: ICU Treatment Optimizer
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# ICU Treatment Optimizer

An RL environment for ICU patient treatment optimization.
An AI agent learns to recommend treatments to maximize patient survival.

## API Endpoints

- POST /reset - Start a new patient episode
- POST /step - Take a treatment action
- GET /state - Get current patient state
- GET /tasks - List all tasks
- POST /grade - Grade a completed episode

## Tasks

- Task 1: Blood Pressure Stabilisation (easy, 10 steps)
- Task 2: Multi-Vital Balancing (medium, 20 steps)
- Task 3: Full ICU Management (hard, 24 steps)

## Setup

pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 7860