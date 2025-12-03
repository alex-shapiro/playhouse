"""Interactive Tetris player with rendering."""

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
    """Evaluate a single Tetris environment with trained or random policy."""
    print("Controls:")
    print("  ESC - Exit")
    print("  TAB - Toggle fullscreen")
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = Tetris(num_envs=1)
    obs, _ = env.reset(seed=np.random.randint(0, 1000000))
    num_episodes = 100

    # Try to load trained model
    policy = load_latest_model(env, device)
    step_count = 0
    episode_count = 0
    total_reward = 0.0

    truncations = np.zeros(1)

    while episode_count < num_episodes:
        obs_tensor = torch.as_tensor(obs, device=device)
        with torch.no_grad():
            logits, _ = policy.forward_eval(obs_tensor, {})
            action = logits.argmax(dim=-1).cpu().numpy()
            obs, rewards, terminals, truncations, info = env.step(action)

            # Track stats
            step_count += 1
            total_reward += rewards[0]

            # Check if episode ended
            if terminals[0] or truncations[0]:
                episode_count += 1
                print(f"Ep {episode_count} ended after {step_count} steps")
                print(f"Ep reward: {total_reward:.2f}")

                # Reset for next episode
                obs, _ = env.reset(seed=np.random.randint(0, 1000000))
                step_count = 0
                total_reward = 0.0

    print(f"Eval Episodes: {episode_count}")
    print(f"Avg Reward: {total_reward / episode_count}")


if __name__ == "__main__":
    main()
