"""Interactive Tetris player with rendering."""

import glob
import time
from pathlib import Path

import numpy as np
import torch

from playhouse.environments.tetris.tetris import Tetris
from playhouse.models.mini_policy import MiniPolicy

DATA_DIR = Path("data/tetris")


def load_latest_model(env: Tetris, device: str) -> MiniPolicy | None:
    """Load the most recently trained model if it exists."""
    pattern = str(DATA_DIR / "tetris_*.pt")
    model_files = glob.glob(pattern)
    if not model_files:
        return None

    # Get most recent by modification time
    latest = max(model_files, key=lambda f: Path(f).stat().st_mtime)
    print(f"Loading model: {latest}")

    state_dict = torch.load(latest, map_location=device, weights_only=True)

    # Infer hidden size from encoder weight shape
    hidden_size = state_dict["encoder.encoder.0.weight"].shape[0]

    policy = MiniPolicy(env, hidden_size=hidden_size)
    policy.load_state_dict(state_dict)
    policy = policy.to(device)
    policy.eval()

    return policy


def main():
    """Render a single Tetris environment with trained or random policy."""
    print("Controls:")
    print("  ESC - Exit")
    print("  TAB - Toggle fullscreen")
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = Tetris(num_envs=1, n_init_garbage=0)
    obs, _ = env.reset(seed=np.random.randint(0, 1000000))

    # Try to load trained model
    policy = load_latest_model(env, device)
    if policy is None:
        print("No trained model found, using random policy")
    else:
        print("Using trained policy")

    # Run until the window is closed
    step_count = 0
    episode_count = 0
    total_reward = 0.0

    truncations = np.zeros(1)

    while truncations[0] == 0:
        if policy is not None:
            obs_tensor = torch.as_tensor(obs, device=device)
            with torch.no_grad():
                logits, _ = policy.forward_eval(obs_tensor, {})
                action = logits.argmax(dim=-1).cpu().numpy()
        else:
            action = np.array([env.action_space.sample()])
        obs, rewards, terminals, truncations, info = env.step(action)

        # Render
        env.render()

        # Track stats
        step_count += 1
        total_reward += rewards[0]

        # Check if episode ended
        if terminals[0] or truncations[0]:
            episode_count += 1
            print(f"Episode {episode_count} ended after {step_count} steps")
            print(f"  Total reward: {total_reward:.2f}")

            # Print info if available
            if "score" in info:
                print(f"  Score: {info.get('score', 0):.0f}")
                print(f"  Lines: {info.get('lines_deleted', 0):.0f}")
                print(f"  Level: {info.get('game_level', 0):.0f}")

            # Reset for next episode
            obs, _ = env.reset(seed=np.random.randint(0, 1000000))
            step_count = 0
            total_reward = 0.0

        # Small delay to make it watchable (30 FPS)
        time.sleep(1.0 / 30.0)


if __name__ == "__main__":
    main()
