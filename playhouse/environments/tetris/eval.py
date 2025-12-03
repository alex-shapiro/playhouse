"""Evaluate latest Tetris model"""

import numpy as np
import torch

from playhouse import io
from playhouse.environments.tetris.tetris import Tetris
from playhouse.models.mini_policy import MiniPolicy


def load_latest_model(env: Tetris, device: str) -> MiniPolicy:
    """Load the most recently trained model if it exists."""
    state_dict = io.load_weights_for_inference("tetris", device)
    assert state_dict is not None, "no model to evaluate"
    hidden_size = state_dict["encoder.encoder.0.weight"].shape[0]
    policy = MiniPolicy(env, hidden_size=hidden_size)
    policy.load_state_dict(state_dict)
    policy = policy.to(device)
    policy.eval()
    return policy


def main():
    num_envs = 1
    num_episodes = 100

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = Tetris(num_envs=num_envs)
    obs, _ = env.reset(seed=np.random.randint(0, 1000000))

    policy = load_latest_model(env, device)

    episode_rewards: list[float] = []
    episode_steps: list[int] = []
    current_rewards = np.zeros(num_envs)
    current_steps = np.zeros(num_envs, dtype=np.int32)

    while len(episode_rewards) < num_episodes:
        obs_tensor = torch.as_tensor(obs, device=device)
        with torch.no_grad():
            logits, _ = policy.forward_eval(obs_tensor, {})
            actions = logits.argmax(dim=-1).cpu().numpy()

        obs, rewards, terminals, truncations, info = env.step(actions)

        current_rewards += rewards
        current_steps += 1

        # Check for completed episodes
        dones = terminals | truncations
        for i in np.where(dones)[0]:
            if len(episode_rewards) < num_episodes:
                episode_rewards.append(current_rewards[i])
                episode_steps.append(current_steps[i])
                print(
                    f"Ep {len(episode_rewards)} ended after {current_steps[i]} steps, "
                    f"reward: {current_rewards[i]:.2f}"
                )
            current_rewards[i] = 0.0
            current_steps[i] = 0

    print(f"\nEval Episodes: {len(episode_rewards)}")
    print(f"Avg Reward: {np.mean(episode_rewards):.2f}")
    print(f"Avg Steps: {np.mean(episode_steps):.1f}")


if __name__ == "__main__":
    main()
