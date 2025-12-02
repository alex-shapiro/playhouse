"""Interactive Tetris player with rendering."""

import time

import numpy as np

from playhouse.environments.tetris.tetris import Tetris


def main():
    """Render a single Tetris environment with random policy"""
    print("Controls:")
    print("  ESC - Exit")
    print("  TAB - Toggle fullscreen")
    print()

    env = Tetris(num_envs=1, n_init_garbage=0)
    env.reset(seed=np.random.randint(0, 1000000))

    # Run until the window is closed
    step_count = 0
    episode_count = 0
    total_reward = 0.0

    truncations = np.zeros(1)

    while truncations == 0:
        action = np.array([env.action_space.sample()])
        print(action)
        _, rewards, terminals, truncations, info = env.step(action)

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
            if "log" in info:
                log = info["log"]
                print(f"  Score: {log.get('score', 0):.0f}")
                print(f"  Lines: {log.get('lines_deleted', 0):.0f}")
                print(f"  Level: {log.get('game_level', 0):.0f}")

            # Reset for next episode
            obs, _ = env.reset(seed=np.random.randint(0, 1000000))
            step_count = 0
            total_reward = 0.0

        # Small delay to make it watchable (30 FPS)
        time.sleep(1.0 / 30.0)

    # except KeyboardInterrupt:
    #     print("\nExiting...")


if __name__ == "__main__":
    main()
