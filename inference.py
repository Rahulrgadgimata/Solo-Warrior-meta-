"""
Inference Script — Medical Triage OpenEnv
==========================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    ENV_URL        The URL of the deployed environment (default: localhost:7860)

- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import os
import json
import textwrap
import re
from typing import List, Optional

from openai import OpenAI
from client import MedicalTriageEnv
from models import TriageAction

# ── Environment variables ──────────────────────────────────────────────────────
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

BENCHMARK = "medical-triage-env"
TASK_IDS = ["easy", "medium", "hard"]
MAX_STEPS = 3
SUCCESS_SCORE_THRESHOLD = 0.85  # Clinical accuracy threshold

# ── System Prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert Emergency Physician performing medical triage.
    Your goal is to assess patients and provide accurate ESI levels and departments.
    
    ESI LEVELS:
    1: Immediate — Life-threatening, act NOW.
    2: Emergent — High risk, should not wait.
    3: Urgent — Stable but needs multiple resources.
    4: Less Urgent — Needs one resource only.
    5: Non-Urgent — No resources needed.
    
    VALID DEPARTMENTS: 
    Resuscitation, Emergency, Cardiology, Neurology, Trauma, Pediatrics, 
    Orthopedics, General, Psychiatry, Obstetrics, Gastroenterology, Pulmonology.
    
    RESPONSE FORMAT:
    You must respond only with a raw JSON object containing:
    {
      "esi_level": <int 1-5>,
      "department": "<department_name>",
      "resource_request": {
        "icu_bed": <bool>,
        "er_bed": <bool>,
        "ventilator": <bool>,
        "ct_scanner": <bool>,
        "cardiac_monitor": <bool>,
        "or_room": <bool>,
        "cath_lab": <bool>
      },
      "reasoning": "<short clinical reasoning>"
    }
""").strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def extract_json(raw: str) -> dict:
    """Robustly extract JSON from model output."""
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    
    # Try markdown code block extraction
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if match:
        try: return json.loads(match.group(1))
        except: pass
        
    # Try simple greedy match
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try: return json.loads(match.group())
        except: pass
        
    raise ValueError("No valid JSON found in model output")

def get_model_action(client: OpenAI, obs_text: str) -> dict:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": obs_text},
            ],
            temperature=0.1,
            max_tokens=500,
        )
        text = (completion.choices[0].message.content or "").strip()
        return extract_json(text)
    except Exception as exc:
        return {"esi_level": 3, "department": "Emergency", "reasoning": f"Fallback due to error: {exc}"}

async def run_task(client: OpenAI, env_url: str, task_id: str):
    env = await MedicalTriageEnv.from_url(env_url)
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    
    rewards = []
    steps_taken = 0
    success = False
    
    try:
        obs = await env.reset(task_id=task_id)
        
        for step in range(1, MAX_STEPS + 1):
            # Format observation for the model
            obs_text = (
                f"PATIENT PRESENTATION:\n{obs.presentation}\n\n"
                f"VITALS:\nBP: {obs.vitals['bp']}, HR: {obs.vitals['hr']}, O2: {obs.vitals['o2_sat']}%, "
                f"RR: {obs.vitals['rr']}, Temp: {obs.vitals['temp']}C\n\n"
            )
            if obs.feedback:
                obs_text += f"PREVIOUS FEEDBACK: {'; '.join(obs.feedback)}\n"
            
            action_dict = get_model_action(client, obs_text)
            
            # Extract basic action for logging
            action_str = f"esi={action_dict.get('esi_level')},dept={action_dict.get('department')}"
            
            # Submit step
            result = await env.step(action_dict)
            
            reward = result.reward.value
            done = result.done
            
            rewards.append(reward)
            steps_taken = step
            obs = result.observation
            
            log_step(step=step, action=action_str, reward=reward, done=done, error=None)
            
            if done:
                break
        
        max_reward = max(rewards) if rewards else 0.0
        success = max_reward >= SUCCESS_SCORE_THRESHOLD
        log_end(success=success, steps=steps_taken, score=max_reward, rewards=rewards)
        
    except Exception as e:
        log_end(success=False, steps=steps_taken, score=0.0, rewards=rewards)
        print(f"[DEBUG] Task error: {e}")
    finally:
        await env.close()

async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    for task_id in TASK_IDS:
        await run_task(client, ENV_URL, task_id)

if __name__ == "__main__":
    asyncio.run(main())
