from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class CheckpointInfo:
    """Information about a saved checkpoint"""

    model_path: Path
    state_path: Path
    epoch: int
    global_step: int
    run_id: str


def load_checkpoint(
    checkpoint: CheckpointInfo,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    """Load model and optimizer state from a checkpoint

    Args:
        env_name: Environment name
        checkpoint: Checkpoint information
        policy: Policy network to load weights into
        optimizer: Optimizer to load state into
    """
    policy.load_state_dict(torch.load(checkpoint.model_path, weights_only=True))
    state = torch.load(checkpoint.state_path, weights_only=True)
    optimizer.load_state_dict(state["optimizer_state_dict"])


def latest_checkpoint(env_name: str) -> CheckpointInfo | None:
    """Find the latest checkpoint in the data directory"""
    # Look for checkpoint directories (format: tetris_{run_id})
    data_dir = data_directory(env_name)
    checkpoint_dirs = list(data_dir.glob(f"{env_name}_*"))
    if not checkpoint_dirs:
        return None

    # Find directories with trainer_state.pt
    valid_checkpoints: list[CheckpointInfo] = []
    for checkpoint_dir in checkpoint_dirs:
        if not checkpoint_dir.is_dir():
            continue

        state_path = checkpoint_dir / "trainer_state.pt"
        if not state_path.exists():
            continue

        try:
            state = torch.load(state_path, weights_only=True)
            model_name = state["model_name"]
            model_path = checkpoint_dir / model_name

            if not model_path.exists():
                continue

            valid_checkpoints.append(
                CheckpointInfo(
                    model_path=model_path,
                    state_path=state_path,
                    epoch=state["epoch"],
                    global_step=state["global_step"],
                    run_id=state["run_id"],
                )
            )
        except (KeyError, RuntimeError) as e:
            print(f"Warning: Could not load checkpoint from {checkpoint_dir}: {e}")
            continue

    if not valid_checkpoints:
        return None

    # Return the checkpoint with the highest epoch
    return max(valid_checkpoints, key=lambda c: c.epoch)


def data_directory(env_name: str) -> Path:
    return Path(f"data/{env_name}")
