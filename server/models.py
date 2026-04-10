from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union, Literal

class Action(BaseModel):
    action_type: Literal["restart", "scale", "update_config", "rollback", "noop"]
    service_name: Optional[str] = None
    params: Optional[Dict[str, Union[str, int]]] = None

class ServiceStatus(BaseModel):
    name: str
    status: Literal["Running", "CrashLoopBackOff", "Pending", "Terminating"]
    replicas: int
    available_replicas: int
    latency_ms: float
    cpu_usage_percent: float
    memory_usage_mb: int
    last_logs: List[str]

class Observation(BaseModel):
    services: List[ServiceStatus]
    system_metrics: Dict[str, float]
    current_time: str
    last_action_result: Optional[str] = None

class Reward(BaseModel):
    score: float = Field(..., ge=-1.0, le=1.0)
    reason: str
    partial_progress: float = Field(..., ge=0.0, le=1.0)

class State(BaseModel):
    step_count: int
    max_steps: int
    is_done: bool
    task_id: str
    task_description: str
    internal_cluster_state: Dict  # For debugging/grading
