import time

import numpy as np

from playhouse.environments.tetris.tetris import Tetris

TIME = 10
num_envs = 4096
env = Tetris(num_envs=num_envs)
actions = [[env.action_space.sample() for _ in range(num_envs)] for _ in range(1000)]
obs, _ = env.reset(seed=np.random.randint(0, 1000))


start = time.time()
end = start + 10
tick = 0

while time.time() < end:
    action = np.array(actions[tick % 1000])
    obs, _, _, _, _ = env.step(action)
    tick += 1

sps = (tick * num_envs) / (time.time() - start)
print(f"SPS: {sps:.3f}")
