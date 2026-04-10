from typing import List, Dict, Any, Tuple
from pydantic import BaseModel
from .models import ServiceStatus, Observation, Action

class TaskDefinition(BaseModel):
    id: str
    name: str
    description: str
    difficulty: str
    max_steps: int
    initial_services: Dict[str, Dict[str, Any]]

TASKS = {
    "task1_auth_outage": TaskDefinition(
        id="task1_auth_outage",
        name="Emergency Auth Recovery",
        description="The 'auth' service has crashed with 'CrashLoopBackOff'. Restore service stability.",
        difficulty="easy",
        max_steps=5,
        initial_services={
            "auth": {"status": "CrashLoopBackOff", "replicas": 1, "available": 0, "logs": ["Error: Memory Limit Exceeded", "Killed by OOM"]},
            "backend": {"status": "Running", "replicas": 2, "available": 2, "logs": ["Warn: Auth service unavailable"]},
            "frontend": {"status": "Running", "replicas": 2, "available": 2, "logs": ["502 Bad Gateway - Auth Service Down"]},
        }
    ),
    "task2_payment_scaling": TaskDefinition(
        id="task2_payment_scaling",
        name="Scaling for High Demand",
        description="The 'payment' service is reporting high latency (800ms). Increase capacity and check performance.",
        difficulty="medium",
        max_steps=8,
        initial_services={
            "auth": {"status": "Running", "replicas": 2, "available": 2, "logs": ["User authenticated"]},
            "payment": {"status": "Running", "replicas": 1, "available": 1, "logs": ["Warn: High request volume", "Conn pool at 95%"], "latency": 850.0},
            "database": {"status": "Running", "replicas": 1, "available": 1, "logs": ["Slow query detected"], "cpu": 85.0},
            "frontend": {"status": "Running", "replicas": 3, "available": 3, "logs": ["504 Gateway Timeout - Payment Slow"]},
        }
    ),
    "task3_backend_config_corruption": TaskDefinition(
        id="task3_backend_config_corruption",
        name="Configuration Recovery",
        description="A bad configuration change in 'backend' has caused an outage in 'frontend'. Fix the config and restore the system.",
        difficulty="hard",
        max_steps=12,
        initial_services={
            "auth": {"status": "Running", "replicas": 3, "available": 3, "logs": ["Healthy"]},
            "backend": {"status": "Running", "replicas": 2, "available": 2, "logs": ["Error Connecting to Redis: localhost:6379", "Retrying..."], "config": {"REDIS_URL": "localhost:6379"}},
            "redis": {"status": "Running", "replicas": 1, "available": 1, "logs": ["Ready for connections"], "host": "redis-cluster.prod:6379"},
            "frontend": {"status": "Running", "replicas": 3, "available": 1, "logs": ["500 Internal Server Error - Backend unavailable"]},
        }
    ),
    "task4_canary_rollout": TaskDefinition(
        id="task4_canary_rollout",
        name="Safe Canary Deployment",
        description="A new version of 'payment' (v2) is showing anomalies. Perform a safe rollback and stabilize v1.",
        difficulty="hard",
        max_steps=10,
        initial_services={
            "payment": {"status": "Running", "replicas": 4, "available": 4, "logs": ["Version: v2.0.1", "Error: NullPointerException in /execute"], "latency": 2500.0},
            "frontend": {"status": "Running", "replicas": 3, "available": 3, "logs": ["50% of payments failing"]},
        }
    ),
    "task5_resource_leak_diagnosis": TaskDefinition(
        id="task5_resource_leak_diagnosis",
        name="Memory Leak Investigation",
        description="'worker' is leaking memory and crashing. Identify the leak logs, scale up to prevent total downtime, and then restart to clear buffer.",
        difficulty="expert",
        max_steps=15,
        initial_services={
            "worker": {"status": "Running", "replicas": 1, "available": 1, "logs": ["JVM Memory Usage: 98%", "Warn: GC overhead limit exceeded"], "mem": 1024, "cpu": 95.0},
            "queue": {"status": "Running", "replicas": 1, "available": 1, "logs": ["Queue growth: +200/s"]},
        }
    )
}

def clip_score(score: float) -> float:
    """Ensure score is within [0, 1] range."""
    return max(0.0, min(1.0, float(score)))

def grade_task1(observation: Observation, action_history: List[Action]) -> float:
    # Easy: Auth should be Running and available_replicas > 0
    auth_service = next((s for s in observation.services if s.name == "auth"), None)
    if not auth_service:
        return 0.0
        
    if auth_service.status == "Running" and auth_service.available_replicas >= 1:
        return 1.0
    
    # Partial reward for trying to restart
    if any(a.action_type == "restart" and a.service_name == "auth" for a in action_history):
        return 0.3
        
    return 0.0

def grade_task2(observation: Observation, action_history: List[Action]) -> float:
    # Medium: Payment replicas >= 3 and latency < 250ms
    payment_service = next((s for s in observation.services if s.name == "payment"), None)
    if not payment_service:
        return 0.0
    
    progress = 0.0
    # Reward for scaling
    if payment_service.replicas >= 2:
        progress += 0.2
    if payment_service.replicas >= 3:
        progress += 0.3
        
    # Reward for latency improvement
    if payment_service.latency_ms < 500:
        progress += 0.2
    if payment_service.latency_ms < 250:
        progress += 0.3
        
    return clip_score(progress)

def grade_task3(observation: Observation, action_history: List[Action]) -> float:
    # Hard: Backend config corrected AND system stable
    backend_service = next((s for s in observation.services if s.name == "backend"), None)
    frontend_service = next((s for s in observation.services if s.name == "frontend"), None)
    
    if not backend_service or not frontend_service:
        return 0.0
    
    score = 0.0
    
    # 1. Action taken (partial credit for discovering and trying)
    if any(a.action_type == "update_config" and a.service_name == "backend" for a in action_history):
        score += 0.2
        
    # 2. Config actually fixed (evidenced by logs)
    if any("Connection established to Redis" in log for log in backend_service.last_logs):
        score += 0.4
    
    # 3. System dependency stabilized
    if frontend_service.status == "Running" and frontend_service.available_replicas >= 1:
        score += 0.2
    if frontend_service.available_replicas >= 2:
        score += 0.2
    
    return clip_score(score)

def grade_task4(observation: Observation, action_history: List[Action]) -> float:
    # Canary Rollout: Payment should be rolled back and latency should be < 100ms
    payment_service = next((s for s in observation.services if s.name == "payment"), None)
    if not payment_service:
        return 0.0
    
    score = 0.0
    if any(a.action_type == "rollback" and a.service_name == "payment" for a in action_history):
        score += 0.5
    if payment_service.latency_ms < 100.0:
        score += 0.5
    return clip_score(score)

def grade_task5(observation: Observation, action_history: List[Action]) -> float:
    # Memory Leak: Worker should be scaled up AND restarted
    worker_service = next((s for s in observation.services if s.name == "worker"), None)
    if not worker_service:
        return 0.0
    
    score = 0.0
    # Scaled up to handle load?
    if worker_service.replicas >= 2:
        score += 0.3
    # Restarted to clear memory?
    if any(a.action_type == "restart" and a.service_name == "worker" for a in action_history):
        score += 0.4
    # Status healthy?
    if worker_service.status == "Running" and worker_service.cpu_usage_percent < 50.0:
        score += 0.3
    return clip_score(score)

GRADERS = {
    "task1_auth_outage": grade_task1,
    "task2_payment_scaling": grade_task2,
    "task3_backend_config_corruption": grade_task3,
    "task4_canary_rollout": grade_task4,
    "task5_resource_leak_diagnosis": grade_task5
}
