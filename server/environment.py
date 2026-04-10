import random
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List
from .models import Action, Observation, Reward, State, ServiceStatus
from .tasks import TASKS

class CloudSREEnv:
    def __init__(self, task_id: str = "task1_auth_outage"):
        self.task_id = task_id
        self.task_def = TASKS[task_id]
        self.current_step = 0
        self.max_steps = self.task_def.max_steps
        self.done = False
        self.services: Dict[str, Any] = {}
        self.configs: Dict[str, Dict[str, str]] = {}
        self.last_action_result: Optional[str] = None
        self.action_history: List[Action] = []
        self._reset_internal(task_id)

    def _reset_internal(self, task_id: str):
        self.task_id = task_id
        self.task_def = TASKS[task_id]
        self.current_step = 0
        self.done = False
        self.last_action_result = "Environment reset."
        self.action_history = []
        # Copy initial services state
        self.services = {}
        for s_name, s_data in self.task_def.initial_services.items():
            self.services[s_name] = {
                "status": s_data.get("status", "Running"),
                "replicas": s_data.get("replicas", 1),
                "available": s_data.get("available", 1),
                "latency": s_data.get("latency", 50.0),
                "cpu": s_data.get("cpu", 10.0),
                "mem": s_data.get("mem", 128),
                "logs": s_data.get("logs", ["System check passed."]),
                "config": s_data.get("config", {})
            }

    def reset(self) -> Observation:
        self._reset_internal(self.task_id)
        return self._generate_observation()

    def step(self, action_dict: Dict) -> Tuple[Observation, float, bool, Dict]:
        # Convert dict to Pydantic Action
        try:
            action = Action(**action_dict)
        except Exception:
            action = Action(action_type="noop")
            
        self.action_history.append(action)
        self.current_step += 1
        
        # Process Action
        self._process_action(action)
        
        # Advance simulation
        self._simulate_tick()
        
        obs = self._generate_observation()
        
        # Calculate Reward
        reward_obj = self._calculate_reward(action)
        reward_val = reward_obj.score
        
        if self.current_step >= self.max_steps:
            self.done = True
            
        # Info dictionary contains metadata for grading
        info = {
            "task_id": self.task_id,
            "reward_reason": reward_obj.reason,
            "partial_progress": reward_obj.partial_progress,
            "action_history_len": len(self.action_history)
        }
        
        return obs, reward_val, self.done, info

    def state(self) -> State:
        return State(
            step_count=self.current_step,
            max_steps=self.max_steps,
            is_done=self.done,
            task_id=self.task_id,
            task_description=self.task_def.description,
            internal_cluster_state=self.services
        )

    def _generate_observation(self) -> Observation:
        service_statuses = []
        for name, data in self.services.items():
            service_statuses.append(ServiceStatus(
                name=name,
                status=data["status"],
                replicas=data["replicas"],
                available_replicas=data["available"],
                latency_ms=data["latency"],
                cpu_usage_percent=data["cpu"],
                memory_usage_mb=data["mem"],
                last_logs=data["logs"][-5:]  # Give last 5 log lines
            ))
            
        return Observation(
            services=service_statuses,
            system_metrics={
                "cluster_load": sum(s["cpu"] for s in self.services.values()) / (100 * len(self.services)),
                "error_rate": self._calculate_error_rate()
            },
            current_time=datetime.now().strftime("%H:%M:%S"),
            last_action_result=self.last_action_result
        )

    def _process_action(self, action: Action):
        self.last_action_result = f"Action {action.action_type} executed."
        
        if action.action_type == "noop":
            return
            
        if not action.service_name or action.service_name not in self.services:
            self.last_action_result = f"Error: Service '{action.service_name}' not found."
            return

        service = self.services[action.service_name]
        
        if action.action_type == "restart":
            # Restarting fixes CrashLoopBackOff if conditions met
            self.last_action_result = f"Restarting service {action.service_name}..."
            service["status"] = "Pending"
            service["available"] = 0
            service["logs"].append(f"Lifecycle: SIGTERM received, restarting...")
            
        elif action.action_type == "scale":
            new_replicas = int(action.params.get("replicas", 1))
            self.last_action_result = f"Scaling {action.service_name} to {new_replicas} replicas."
            service["replicas"] = new_replicas
            service["logs"].append(f"Lifecycle: Scale event to {new_replicas}")
            
        elif action.action_type == "update_config":
            for key, val in action.params.items():
                service["config"][key] = val
                service["logs"].append(f"Config change: {key}={val}")
            self.last_action_result = f"Updated config for {action.service_name}."
            
        elif action.action_type == "rollback":
            self.last_action_result = f"Rolling back {action.service_name} to previous revision."
            service["status"] = "Pending"
            service["available"] = 0

    def _simulate_tick(self):
        # Update Pending services to Running
        for name, service in self.services.items():
            if service["status"] == "Pending":
                service["status"] = "Running"
                service["available"] = service["replicas"]
                service["logs"].append(f"Lifecycle: Pod ready and serving traffic.")

        # Logic for Task 1: Auth Outage
        if self.task_id == "task1_auth_outage":
            auth = self.services.get("auth")
            if auth and auth["status"] == "Running":
                auth["available"] = auth["replicas"]
                auth["logs"].append("Health check passed.")
            elif auth and auth["status"] == "CrashLoopBackOff":
                auth["available"] = 0
                auth["logs"].append("Error: Fatal exception - restart required.")

        # Logic for Task 2: Scaling
        if self.task_id == "task2_payment_scaling":
            payment = self.services.get("payment")
            if payment:
                if payment["replicas"] >= 3:
                    payment["latency"] = max(40.0, payment["latency"] * 0.5)
                    payment["logs"].append("Latency improving due to scale.")
                else:
                    payment["latency"] = min(900.0, payment["latency"] + random.uniform(10, 50))
                    payment["logs"].append("Warn: Latency threshold exceeded.")

        # Logic for Task 3: Config
        if self.task_id == "task3_backend_config_corruption":
            backend = self.services.get("backend")
            if backend:
                if backend["config"].get("REDIS_URL") == "redis-cluster.prod:6379":
                    if "Connection established to Redis" not in backend["logs"][-1]:
                        backend["logs"].append("Connection established to Redis. Ready.")
                        # Fix dependencies
                        frontend = self.services.get("frontend")
                        if frontend:
                            frontend["status"] = "Running"
                            frontend["available"] = frontend["replicas"]
                else:
                    backend["logs"].append("Error Connecting to Redis: localhost:6379")

    def _calculate_error_rate(self) -> float:
        total_errors = 0
        for name, service in self.services.items():
            if service["status"] != "Running" or service["available"] == 0:
                total_errors += 1
            if service["latency"] > 500:
                total_errors += 0.5
        return total_errors / len(self.services)

    def _calculate_reward(self, last_action: Action) -> Reward:
        from .tasks import GRADERS
        grader = GRADERS.get(self.task_id)
        if not grader:
            return Reward(score=0.0, reason="No grader found", partial_progress=0.0)
            
        current_obs = self._generate_observation()
        score = grader(current_obs, self.action_history)
        
        # Penalize noops or useless actions slightly to prevent infinite loops
        # only if the task isn't already perfectly solved
        if last_action.action_type == "noop" and score < 1.0:
            score -= 0.05
            
        return Reward(
            score=max(-1.0, score),
            reason=f"Current performance score: {score:.2f}",
            partial_progress=min(1.0, score)
        )
