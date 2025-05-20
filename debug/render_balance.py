import vmas
import time
import numpy as np
import torch

env = vmas.make_env(
    scenario="balance",
    num_envs=1,
    device="cpu",
    continuous_actions=True,
    max_steps=100,
    seed=42
)

obs = env.reset()
done = False
total_reward = 0

while not done:
    actions = [torch.rand(1, 2) * 2 - 1 for _ in range(env.n_agents)]  # Random actions
    obs, rewards, done, info = env.step(actions)
    total_reward += sum(rewards)
    
    env.render(mode="human", visualize_when_rgb=True)
    time.sleep(0.05)  # Slow down visualization

print(f"Episode finished with total reward: {total_reward}")
env.close()