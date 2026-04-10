import asyncio
import json
import os
from typing import Optional, List, Union
import requests
import websockets
from pydantic import BaseModel
from models import (
    TriageAction, TriageObservation, TriageReward,
    ResetRequest, ResetResponse, StepRequest, StepResponse
)

class MedicalTriageEnv:
    """
    Python wrapper for the Medical Triage OpenEnv environment.
    Supports both HTTP/REST and persistent WebSocket communication.
    """

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.ws_url = self.base_url.replace("https://", "wss://").replace("http://", "ws://") + "/ws"
        self._ws = None

    @classmethod
    async def from_url(cls, url: str) -> "MedicalTriageEnv":
        """Initialize the environment from a URL."""
        return cls(url)

    async def reset(self, task_id: str = "easy", use_procedural: bool = True) -> TriageObservation:
        """Start a new episode."""
        if self._ws is None:
            self._ws = await websockets.connect(self.ws_url)
        
        payload = {
            "type": "reset",
            "task_id": task_id,
            "use_procedural": use_procedural
        }
        await self._ws.send(json.dumps(payload))
        resp = json.loads(await self._ws.recv())
        
        if "error" in resp:
            raise Exception(resp["error"])
            
        return TriageObservation(**resp["observation"])

    async def step(self, action: Union[TriageAction, dict]) -> StepResponse:
        """Submit an action and receive reward and observation."""
        if self._ws is None:
            raise Exception("Environment not initialized. Call reset() first.")
            
        if isinstance(action, dict):
            action_data = action
        else:
            action_data = action.model_dump()
            
        payload = {
            "type": "step",
            "action": action_data
        }
        await self._ws.send(json.dumps(payload))
        resp = json.loads(await self._ws.recv())
        
        if "error" in resp:
            raise Exception(resp["error"])
            
        # The websocket returns a dict with observation, reward, done, info
        return StepResponse(
            observation=TriageObservation(**resp["observation"]),
            reward=TriageReward(**resp["reward"]),
            done=resp["done"],
            info=resp["info"]
        )

    async def close(self):
        """Close the connection."""
        if self._ws:
            await self._ws.close()
            self._ws = None

    async def get_state(self) -> dict:
        """Retrieve current episode state via HTTP."""
        resp = requests.get(f"{self.base_url}/state")
        resp.raise_for_status()
        return resp.json()["state"]

    @staticmethod
    def get_tasks() -> List[dict]:
        """List available tasks."""
        # Simple static return to avoid needing an instance
        return [
            {"id": "easy", "name": "Textbook Triage"},
            {"id": "medium", "name": "Overlapping Presentations"},
            {"id": "hard", "name": "Subtle & Ambiguous Cases"}
        ]
