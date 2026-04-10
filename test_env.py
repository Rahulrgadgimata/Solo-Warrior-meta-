from environment import CloudSREEnv
import json

def test():
    env = CloudSREEnv(task_id="task1_auth_outage")
    
    print("--- TEST: task1_auth_outage ---")
    obs = env.reset()
    print(f"Initial State: {obs.services[0].name} is {obs.services[0].status}")
    
    # Action: restart auth
    action = {"action_type": "restart", "service_name": "auth"}
    obs, reward, done, info = env.step(action)
    print(f"Step 1 Reward: {reward}")
    
    # Tick again (it takes a tick to become Running)
    obs, reward, done, info = env.step({"action_type": "noop"})
    print(f"Step 2 State: {obs.services[0].name} is {obs.services[0].status}")
    print(f"Step 2 Reward: {reward}")
    
    if reward >= 1.0:
        print("Task 1 PASS")
    else:
        print("Task 1 FAIL")

    print("\n--- TEST: task2_payment_scaling ---")
    env = CloudSREEnv(task_id="task2_payment_scaling")
    env.reset()
    # Action: scale payment to 3
    obs, reward, done, info = env.step({"action_type": "scale", "service_name": "payment", "params": {"replicas": 3}})
    print(f"Step 1 Reward: {reward}")
    
    # Tick
    obs, reward, done, info = env.step({"action_type": "noop"})
    print(f"Step 2 Reward: {reward}")
    
    if reward >= 1.0:
        print("Task 2 PASS")
    else:
        print("Task 2 FAIL")

    print("\n--- TEST: task3_backend_config_corruption ---")
    env = CloudSREEnv(task_id="task3_backend_config_corruption")
    env.reset()
    # Action: update config REDIS_URL
    obs, reward, done, info = env.step({"action_type": "update_config", "service_name": "backend", "params": {"REDIS_URL": "redis-cluster.prod:6379"}})
    print(f"Step 1 Reward: {reward}")
    
    # Tick
    obs, reward, done, info = env.step({"action_type": "noop"})
    print(f"Step 2 Progress Result: {obs.services[0].name} logs: {obs.services[0].last_logs[-1]}")
    print(f"Step 2 Reward: {reward}")
    
    if reward >= 1.0:
        print("Task 3 PASS")
    else:
        print("Task 3 FAIL")

if __name__ == "__main__":
    test()
