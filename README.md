---
title: Medical Triage OpenEnv
emoji: 🏥
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - healthcare
  - triage
  - medical
  - agent
license: mit
---

# 🏥 Medical Triage OpenEnv

[![OpenEnv Spec](https://img.shields.io/badge/OpenEnv-v1.0.0-blue.svg)](https://github.com/raun/openenv-course)
[![Healthcare AI](https://img.shields.io/badge/Domain-Healthcare-red.svg)]()
[![Complexity](https://img.shields.io/badge/Difficulty-Easy--to--Hard-green.svg)]()

A high-fidelity RL environment for **Emergency Medical Triage** based on the Emergency Severity Index (ESI). This environment challenges AI agents to assess complex patient presentations, assign acuity levels, and manage hospital resources in a dynamic, high-stakes setting.

## 🌟 Key Features

- **Clinical Fidelity**: Implements the standard 5-level ESI triage algorithm used in real emergency departments.
- **Dynamic Vital Decay**: A custom progression engine simulates patient deterioration over time if triage is delayed or incorrect.
- **Resource Management**: Agents must track ICU beds, ventilators, and staff availability while making routing decisions.
- **Explainable AI (XAI)**: Built-in clinical reasoning engine provides differential diagnoses and symptom weighting to aid agent transparency.
- **Multi-Objective Rewards**: Dense reward function covering accuracy, resource efficiency, and patient safety (undertriage penalties).

---

## 🚀 Getting Started

### 1. Requirements
Ensure you have Python 3.11+ and the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run the Environment Locally
```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```
Visit `http://localhost:7860` to view the interactive dashboard.

### 3. Run Inference
The environment supports the standard OpenEnv inference format.
```bash
# Set required variables
export HF_TOKEN="your_token"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export API_BASE_URL="https://router.huggingface.co/v1"

# Execute inference
python inference.py
```

---

## 📊 Task Hierarchy

| Task ID | Name | Difficulty | Description |
| :--- | :--- | :--- | :--- |
| `easy` | Textbook Triage | Easy | Clear-cut cases (e.g., minor laceration, obvious STEMI). |
| `medium` | Overlapping Signs | Medium | Presentations that require differentiating between similar conditions (e.g., Flu vs Sepsis). |
| `hard` | Subtle & Ambiguous | Hard | Complex cases with rare symptoms or rapid deterioration risk (e.g., Aortic Dissection). |

---

## 🛠 Action Space

Agents submit actions as JSON objects:

```json
{
  "esi_level": 2,
  "department": "Emergency",
  "resource_request": {
    "icu_bed": false,
    "er_bed": true,
    "cardiac_monitor": true
  },
  "reasoning": "Suspected ACS with stable vitals..."
}
```

---

## ⚖️ Evaluation Criteria

- **Real-world utility (30%)**: Modeled on actual ESI protocols.
- **Task & grader quality (25%)**: 3 difficulty levels with clinical graders.
- **Environment design (20%)**: Robust state management and dense rewards.
- **Code quality (15%)**: Strictly follows OpenEnv V1 spec.
- **Creativity (10%)**: Novel patient progression and XAI systems.

---

## 📜 License
MIT License. Developed by Rahul R gadgimata for the OpenEnv 2026 Hackathon.
