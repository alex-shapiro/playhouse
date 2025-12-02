"""
Tetris PPO training script with hyperparameter sweeping.

Usage:
    python -m playhouse.environments.tetris.train [--sweep] [--epochs N]

This script:
1. Loads existing hyperparameters if available
2. If not (or --sweep is passed), runs Protein sweeper to find optimal hyperparameters
3. Trains a PPO agent for the specified number of epochs, saving checkpoints
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from playhouse.environments.tetris.tetris import Tetris
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
    num_epochs: int = 1000
    num_envs: int = 64
    batch_size: int = 65536
    bptt_horizon: int = 16
    checkpoint_interval: int = 100

    # Sweep
    run_sweep: bool = False
    sweep_trials: int = 50
    sweep_timesteps: int = 1_000_000

    # Logging
    wandb_project: str = "tetris"
    wandb_group: str = "ppo"
    use_wandb: bool = True

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0


@dataclass
class TetrisHyperparameters:
    """Hyperparameters for Tetris PPO training."""

    learning_rate: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.1
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    hidden_size: int = 128

    def to_dict(self) -> dict[str, float | int]:
        return {
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_coef": self.clip_coef,
            "vf_coef": self.vf_coef,
            "ent_coef": self.ent_coef,
            "max_grad_norm": self.max_grad_norm,
            "update_epochs": self.update_epochs,
            "hidden_size": self.hidden_size,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TetrisHyperparameters:
        return cls(
            learning_rate=float(d.get("learning_rate", 2.5e-4)),
            gamma=float(d.get("gamma", 0.99)),
            gae_lambda=float(d.get("gae_lambda", 0.95)),
            clip_coef=float(d.get("clip_coef", 0.1)),
            vf_coef=float(d.get("vf_coef", 0.5)),
            ent_coef=float(d.get("ent_coef", 0.01)),
            max_grad_norm=float(d.get("max_grad_norm", 0.5)),
            update_epochs=int(d.get("update_epochs", 4)),
            hidden_size=int(d.get("hidden_size", 128)),
        )


# -----------------------------------------------------------------------------
# Hyperparameter Sweeping
# -----------------------------------------------------------------------------


def create_sweep_config(device: str) -> SweepConfig:
    """Create sweep configuration for Tetris hyperparameters."""
    return SweepConfig(
        device=device,
        metric="ep_return",
        goal="maximize",
        params={
            "learning_rate": ParamSpaceConfig(
                distribution="log_normal",
                min=1e-5,
                max=1e-2,
                mean=2.5e-4,
            ),
            "gamma": ParamSpaceConfig(
                distribution="uniform",
                min=0.9,
                max=0.999,
                mean=0.99,
            ),
            "gae_lambda": ParamSpaceConfig(
                distribution="uniform",
                min=0.8,
                max=0.99,
                mean=0.95,
            ),
            "clip_coef": ParamSpaceConfig(
                distribution="uniform",
                min=0.05,
                max=0.3,
                mean=0.1,
            ),
            "vf_coef": ParamSpaceConfig(
                distribution="uniform",
                min=0.1,
                max=1.0,
                mean=0.5,
            ),
            "ent_coef": ParamSpaceConfig(
                distribution="log_normal",
                min=1e-4,
                max=0.1,
                mean=0.01,
            ),
            "max_grad_norm": ParamSpaceConfig(
                distribution="uniform",
                min=0.1,
                max=1.0,
                mean=0.5,
            ),
            "update_epochs": ParamSpaceConfig(
                distribution="int_uniform",
                min=1,
                max=10,
                mean=4,
            ),
            "hidden_size": ParamSpaceConfig(
                distribution="uniform_pow2",
                min=64,
                max=512,
                mean=128,
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
        checkpoint_interval=1000000,  # Don't checkpoint during sweep
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

    return TetrisHyperparameters.from_dict(best_hypers)


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
        return TetrisHyperparameters.from_dict(data)
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Failed to load hyperparameters: {e}")
        return None


def save_hyperparameters(hypers: TetrisHyperparameters) -> None:
    """Save hyperparameters to disk."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(HYPERPARAMS_FILE, "w") as f:
        json.dump(hypers.to_dict(), f, indent=2)
    print(f"Saved hyperparameters to {HYPERPARAMS_FILE}")


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
        ent_coef=hypers.ent_coef,
        max_grad_norm=hypers.max_grad_norm,
        update_epochs=hypers.update_epochs,
        checkpoint_interval=config.checkpoint_interval,
    )

    # Create logger
    logger = None
    if config.use_wandb:
        try:
            wandb_config = WandbConfig(
                wandb_project=config.wandb_project,
                wandb_group=config.wandb_group,
            )
            logger = WandbLogger(wandb_config)
            print(f"  Logging to W&B run: {logger.run_id}")
        except Exception as e:
            print(f"  W&B logging disabled: {e}")

    # Create trainer
    trainer = Trainer(config=rl_config, env=env, policy=policy, logger=logger)

    print(f"\nTraining for {total_timesteps:,} timesteps...")

    # Training loop
    epoch = 0
    while trainer.state.global_step < total_timesteps:
        trainer.evaluate()
        logs = trainer.train()

        if logs is not None:
            epoch += 1
            sps = logs["SPS"]
            step = logs["agent_steps"]

            if "environment/ep_return" in logs:
                ep_return = logs["environment/ep_return"]
                score = logs["environment/score"]

                if epoch % 10 == 0:
                    print(
                        f"Epoch {epoch:4d} | "
                        f"Step {step:10,d} | "
                        f"SPS {sps:6.0f} | "
                        f"Return {ep_return:.2f} | "
                        f"Score {score:.1f}"
                    )

    # Close and return final model path
    model_path = trainer.close()
    print(f"\nTraining complete! Model saved to: {model_path}")
    return model_path


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def parse_args() -> TrainConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train PPO agent on Tetris")
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run hyperparameter sweep even if saved hyperparameters exist",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Number of training epochs (default: 1000)",
    )
    parser.add_argument(
        "--sweep-trials",
        type=int,
        default=50,
        help="Number of sweep trials (default: 50)",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=64,
        help="Number of parallel environments (default: 64)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=65536,
        help="Batch size (default: 65536)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)",
    )

    args = parser.parse_args()

    return TrainConfig(
        num_epochs=args.epochs,
        run_sweep=args.sweep,
        sweep_trials=args.sweep_trials,
        num_envs=args.num_envs,
        batch_size=args.batch_size,
        use_wandb=not args.no_wandb,
        seed=args.seed,
    )


def main() -> None:
    """Main entry point."""
    config = parse_args()

    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load or find hyperparameters
    hypers = None
    if not config.run_sweep:
        hypers = load_hyperparameters()

    if hypers is None:
        print("No existing hyperparameters found. Running sweep...")
        hypers = run_hyperparameter_sweep(config)
        save_hyperparameters(hypers)
    else:
        print("Using existing hyperparameters:")
        for k, v in hypers.to_dict().items():
            print(f"  {k}: {v}")

    # Train
    train(config, hypers)


if __name__ == "__main__":
    main()
