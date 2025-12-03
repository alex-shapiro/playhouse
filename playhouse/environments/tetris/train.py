"""
Tetris PPO training script.

Usage:
    python -m playhouse.environments.tetris.train

This script:
1. Loads existing hyperparameters if available, otherwise runs Protein sweeper
2. Resumes from the latest checkpoint if available
3. Trains a PPO agent for the specified number of epochs, saving checkpoints
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch

from playhouse.environments.tetris.tetris import Tetris
from playhouse.logger import Logger
from playhouse.logger.neptune import NeptuneConfig, NeptuneLogger
from playhouse.logger.noop import NoopLogger
from playhouse.logger.wandb import WandbConfig, WandbLogger
from playhouse.models.mini_policy import MiniPolicy
from playhouse.ppo import RLConfig, Trainer
from playhouse.sweep.config import ParamSpaceConfig, SweepConfig
from playhouse.sweep.protein import Protein, ProteinConfig, ProteinState

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DATA_DIR = Path("data/tetris")
HYPERPARAMS_FILE = DATA_DIR / "hyperparameters.json"


@dataclass
class TrainConfig:
    """Training configuration."""

    # Training
    num_epochs: int = 5000
    num_envs: int = 1024
    batch_size: int = 65536
    bptt_horizon: int = 64
    checkpoint_interval: int = 50

    # Sweep
    run_sweep: bool = False
    sweep_trials: int = 200
    sweep_timesteps: int = 1_000_000

    # Logging
    logger: Literal["noop", "wandb", "neptune"] = "noop"
    wandb_project: str = "tetris"
    wandb_group: str = "ppo"
    neptune_name: str = "tetris"
    neptune_project: str = "ppo"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0


@dataclass
class TetrisHyperparameters:
    """Hyperparameters for Tetris PPO training."""

    learning_rate: float = 0.012
    gamma: float = 0.995
    gae_lambda: float = 0.55
    clip_coef: float = 0.1
    vf_coef: float = 4.74
    vf_clip_coef: float = 1.5
    ent_coef: float = 0.02
    max_grad_norm: float = 5.0
    update_epochs: int = 4
    hidden_size: int = 256

    # Adam optimizer
    adam_beta1: float = 0.95
    adam_beta2: float = 0.9999
    adam_eps: float = 1e-10

    # V-trace importance sampling
    vtrace_rho_clip: float = 0.70
    vtrace_c_clip: float = 1.29

    # Prioritized experience replay
    prio_alpha: float = 0.99
    prio_beta0: float = 0.91


# -----------------------------------------------------------------------------
# Hyperparameter Sweeping
# -----------------------------------------------------------------------------


def create_sweep_config(device: str) -> SweepConfig:
    """Create sweep configuration for Tetris hyperparameters."""
    return SweepConfig(
        device=device,
        metric="score",
        goal="maximize",
        params={
            "gae_lambda": ParamSpaceConfig(
                distribution="logit_normal",
                min=0.01,
                max=0.995,
                mean=0.6,
            ),
            "clip_coef": ParamSpaceConfig(
                distribution="uniform",
                min=0.01,
                max=1.0,
                mean=0.1,
            ),
            "adam_beta1": ParamSpaceConfig(
                distribution="logit_normal",
                min=0.5,
                max=0.999,
                mean=0.95,
            ),
        },
    )


def run_sweep_trial(
    hypers: dict[str, float | int],
    config: TrainConfig,
) -> tuple[float, float]:
    """Run a single sweep trial and return (score, cost).

    Args:
        hypers: Hyperparameters to evaluate
        config: Training configuration

    Returns:
        Tuple of (best_score, total_timesteps)
    """
    # Create environment
    env = Tetris(num_envs=config.num_envs, seed=config.seed)

    # Create policy with specified hidden size
    hidden_size = int(hypers.get("hidden_size", 128))
    policy = MiniPolicy(env, hidden_size=hidden_size)
    policy = policy.to(config.device)

    # Create RL config from hyperparameters
    rl_config = RLConfig(
        env_name="tetris_sweep",
        seed=config.seed,
        total_timesteps=config.sweep_timesteps,
        device=config.device,
        batch_size=config.batch_size,
        bptt_horizon=config.bptt_horizon,
        learning_rate=float(hypers.get("learning_rate", 2.5e-4)),
        gamma=float(hypers.get("gamma", 0.99)),
        gae_lambda=float(hypers.get("gae_lambda", 0.95)),
        clip_coef=float(hypers.get("clip_coef", 0.1)),
        vf_coef=float(hypers.get("vf_coef", 0.5)),
        ent_coef=float(hypers.get("ent_coef", 0.01)),
        max_grad_norm=float(hypers.get("max_grad_norm", 0.5)),
        update_epochs=int(hypers.get("update_epochs", 4)),
        save_checkpoints=False,
    )

    # Create trainer (no logger for sweep trials)
    trainer = Trainer(config=rl_config, env=env, policy=policy, logger=None)

    # Train and track best ep_return
    best_ep_return = float("-inf")
    while trainer.state.global_step < config.sweep_timesteps:
        trainer.evaluate()
        logs = trainer.train()
        if logs is not None and "environment/ep_return" in logs:
            ep_return = logs["environment/ep_return"]
            best_ep_return = max(best_ep_return, float(ep_return))

    cost = float(trainer.state.global_step)
    trainer.close()

    if best_ep_return == float("-inf"):
        raise RuntimeError("No ep_return logged during sweep trial")

    return best_ep_return, cost


def run_hyperparameter_sweep(config: TrainConfig) -> TetrisHyperparameters:
    """Run Bayesian hyperparameter optimization using Protein.

    Args:
        config: Training configuration

    Returns:
        Best hyperparameters found
    """
    print(f"Running hyperparameter sweep ({config.sweep_trials} trials)...")

    sweep_config = create_sweep_config(config.device)
    protein = Protein(sweep_config, ProteinConfig())
    state = ProteinState()

    best_hypers: dict[str, float | int] = {}
    best_score = float("-inf")

    for trial in range(config.sweep_trials):
        # Get suggestion from Protein
        suggestion, state = protein.suggest(state)
        hypers = suggestion.params

        print(f"\nTrial {trial + 1}/{config.sweep_trials}")
        print(f"  Hyperparameters: {hypers}")

        # Run trial
        try:
            score, cost = run_sweep_trial(hypers, config)
            is_failure = False
            print(f"  Score: {score:.2f}, Cost: {cost:.0f}")
        except Exception as e:
            print(f"  Trial failed: {e}")
            score = 0.0
            cost = config.sweep_timesteps
            is_failure = True

        # Record observation
        state = protein.observe(state, hypers, score, cost, is_failure)

        # Track best
        if score > best_score:
            best_score = score
            best_hypers = hypers
            print(f"  New best! Score: {best_score:.2f}")

    print(f"\nBest hyperparameters (score={best_score:.2f}):")
    for k, v in best_hypers.items():
        print(f"  {k}: {v}")

    return TetrisHyperparameters(**best_hypers)  # pyright: ignore[reportArgumentType]


# -----------------------------------------------------------------------------
# Hyperparameter Persistence
# -----------------------------------------------------------------------------


def load_hyperparameters() -> TetrisHyperparameters | None:
    """Load hyperparameters from disk if they exist."""
    if not HYPERPARAMS_FILE.exists():
        return None

    try:
        with open(HYPERPARAMS_FILE) as f:
            data = json.load(f)
        print(f"Loaded hyperparameters from {HYPERPARAMS_FILE}")
        return TetrisHyperparameters(**data)
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Failed to load hyperparameters: {e}")
        return None


def save_hyperparameters(hypers: TetrisHyperparameters) -> None:
    """Save hyperparameters to disk."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(HYPERPARAMS_FILE, "w") as f:
        json.dump(asdict(hypers), f, indent=2)
    print(f"Saved hyperparameters to {HYPERPARAMS_FILE}")


# -----------------------------------------------------------------------------
# Checkpoint Loading
# -----------------------------------------------------------------------------


@dataclass
class CheckpointInfo:
    """Information about a saved checkpoint."""

    model_path: Path
    state_path: Path
    epoch: int
    global_step: int
    run_id: str


def find_latest_checkpoint() -> CheckpointInfo | None:
    """Find the latest checkpoint in the data directory.

    Returns:
        CheckpointInfo if a valid checkpoint exists, None otherwise.
    """
    # Look for checkpoint directories (format: tetris_{run_id})
    checkpoint_dirs = list(DATA_DIR.glob("tetris_*"))
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


def load_checkpoint(
    checkpoint: CheckpointInfo,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    """Load model and optimizer state from a checkpoint.

    Args:
        checkpoint: Checkpoint information
        policy: Policy network to load weights into
        optimizer: Optimizer to load state into
    """
    # Load model weights
    policy.load_state_dict(torch.load(checkpoint.model_path, weights_only=True))

    # Load optimizer state
    state = torch.load(checkpoint.state_path, weights_only=True)
    optimizer.load_state_dict(state["optimizer_state_dict"])


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------


def train(config: TrainConfig, hypers: TetrisHyperparameters) -> str:
    """Train PPO agent on Tetris.

    Args:
        config: Training configuration
        hypers: Hyperparameters to use

    Returns:
        Path to final model checkpoint
    """
    print("\nStarting training...")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Device: {config.device}")

    # Create environment
    env = Tetris(num_envs=config.num_envs, seed=config.seed)

    # Create policy
    policy = MiniPolicy(env, hidden_size=hypers.hidden_size)
    policy = policy.to(config.device)

    # Calculate total timesteps
    total_timesteps = config.num_epochs * config.batch_size

    # Create RL config
    rl_config = RLConfig(
        env_name="tetris",
        seed=config.seed,
        total_timesteps=total_timesteps,
        device=config.device,
        data_dir=str(DATA_DIR),
        batch_size=config.batch_size,
        bptt_horizon=config.bptt_horizon,
        learning_rate=hypers.learning_rate,
        gamma=hypers.gamma,
        gae_lambda=hypers.gae_lambda,
        clip_coef=hypers.clip_coef,
        vf_coef=hypers.vf_coef,
        vf_clip_coef=hypers.vf_clip_coef,
        ent_coef=hypers.ent_coef,
        max_grad_norm=hypers.max_grad_norm,
        update_epochs=hypers.update_epochs,
        checkpoint_interval=config.checkpoint_interval,
        minibatch_size=config.batch_size,
        max_minibatch_size=config.batch_size,
        adam_beta1=hypers.adam_beta1,
        adam_beta2=hypers.adam_beta2,
        adam_eps=hypers.adam_eps,
        vtrace_rho_clip=hypers.vtrace_rho_clip,
        vtrace_c_clip=hypers.vtrace_c_clip,
        prio_alpha=hypers.prio_alpha,
        prio_beta0=hypers.prio_beta0,
    )

    # Create logger
    logger: Logger
    match config.logger:
        case "wandb":
            wandb_cfg = WandbConfig(
                wandb_project=config.wandb_project,
                wandb_group=config.wandb_group,
            )
            logger = WandbLogger(wandb_cfg)
            print(f"  Logging to W&B run: {logger.run_id}")
        case "neptune":
            neptune_cfg = NeptuneConfig(
                neptune_name=config.neptune_name,
                neptune_project=config.neptune_project,
            )
            logger = NeptuneLogger(neptune_cfg)
            print(f"  Logging to Neptune run: {logger.run_id}")
        case "noop":
            logger = NoopLogger()
            print(f"  Logging disabled (run_id: {logger.run_id})")

    # Create trainer
    trainer = Trainer(config=rl_config, env=env, policy=policy, logger=logger)

    # Try to load from checkpoint
    checkpoint = find_latest_checkpoint()
    if checkpoint is not None:
        print(f"\nFound checkpoint at epoch {checkpoint.epoch}")
        load_checkpoint(checkpoint, trainer.uncompiled_policy, trainer.optimizer)
        trainer.state.epoch = checkpoint.epoch
        trainer.state.global_step = checkpoint.global_step
        print(
            f"  Resumed from epoch {checkpoint.epoch}, step {checkpoint.global_step:,}"
        )

        # Check if we've already reached the target
        if checkpoint.epoch >= config.num_epochs:
            print(
                f"\nAlready trained for {checkpoint.epoch} epochs (target: {config.num_epochs})"
            )
            print("Increase num_epochs in TrainConfig to continue training.")
            trainer.close()
            return str(checkpoint.model_path)
    else:
        print("\nNo checkpoint found, starting from scratch")

    remaining_epochs = config.num_epochs - trainer.state.epoch
    print(
        f"Training for {remaining_epochs} more epochs ({total_timesteps:,} total timesteps)..."
    )

    # Training loop
    while trainer.state.global_step < total_timesteps:
        trainer.evaluate()
        logs = trainer.train()

        if logs is not None:
            epoch = trainer.state.epoch
            sps = logs["SPS"]
            step = logs["agent_steps"]

            if epoch % 10 == 0:
                msg = f"Epoch {epoch:4d} | Step {step:10,d} | SPS {sps:6.0f}"
                if "environment/ep_return" in logs:
                    ep_return = logs["environment/ep_return"]
                    score = logs["environment/score"]
                    msg += f" | Return {ep_return:.2f} | Score {score:.1f}"
                else:
                    msg += " | Return N/A | Score N/A"
                print(msg)

    # Close and return final model path
    model_path = trainer.close()
    print(f"\nTraining complete! Model saved to: {model_path}")
    return model_path


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    """Main entry point."""
    config = TrainConfig()

    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load or find hyperparameters
    hypers = load_hyperparameters()
    if hypers is None:
        print("No existing hyperparameters found. Running sweep...")
        hypers = run_hyperparameter_sweep(config)
        save_hyperparameters(hypers)
    else:
        print("Using existing hyperparameters:")
        for k, v in asdict(hypers).items():
            print(f"  {k}: {v}")

    # Train
    train(config, hypers)


if __name__ == "__main__":
    main()
