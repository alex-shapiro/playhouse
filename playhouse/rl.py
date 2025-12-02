"""
Reinforcement learning trainer with PPO and V-trace.

Ported from PufferLib's pufferl.py with full type annotations and
strongly typed configuration.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from threading import Thread
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol, final, override

import numpy as np
import psutil
import torch
import torch.distributed
import torch.nn as nn
from torch import Tensor

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray


# -----------------------------------------------------------------------------
# Type Definitions
# -----------------------------------------------------------------------------


class Logger(Protocol):
    """Protocol for loggers (WandbLogger, NeptuneLogger, NoopLogger)."""

    run_id: str

    def log(self, logs: dict[str, Any], step: int) -> None: ...
    def close(self, model_path: str) -> None: ...


class Policy(Protocol):
    """Protocol for policy networks."""

    hidden_size: int

    def parameters(self) -> Iterator[nn.Parameter]: ...
    def __call__(self, obs: Tensor, state: dict[str, Any]) -> tuple[Tensor, Tensor]: ...
    def forward_eval(
        self, obs: Tensor, state: dict[str, Any]
    ) -> tuple[Tensor, Tensor]: ...
    def state_dict(self) -> dict[str, Any]: ...


class Environment(Protocol):
    """Protocol for environments.

    In playhouse, environments are already multithreaded, so we expect
    a simpler interface than PufferLib's vecenv.
    """

    observation_space: Any
    action_space: Any

    def reset(self, seed: int | None = None) -> tuple[NDArray[Any], dict[str, Any]]: ...
    def step(
        self, action: NDArray[Any]
    ) -> tuple[
        NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any], dict[str, Any]
    ]: ...
    def close(self) -> None: ...

    @property
    def num_envs(self) -> int: ...


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RLConfig:
    """Configuration for the RL trainer."""

    # Environment
    env_name: str = "unknown"
    seed: int = 0
    total_timesteps: int = 10_000_000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir: str = "data"

    # Experience collection
    batch_size: int = 65536
    bptt_horizon: int = 16

    # PPO hyperparameters
    update_epochs: int = 1
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.1
    vf_clip_coef: float = 0.1
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5

    # V-trace importance sampling
    vtrace_rho_clip: float = 1.0
    vtrace_c_clip: float = 1.0

    # Prioritized experience replay
    prio_alpha: float = 0.0
    prio_beta0: float = 0.4

    # Minibatching
    minibatch_size: int = 8192
    max_minibatch_size: int = 8192

    # Optimizer
    optimizer: Literal["adam", "muon"] = "adam"
    learning_rate: float = 2.5e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-5
    anneal_lr: bool = True
    min_lr_ratio: float = 0.0

    # LSTM
    use_rnn: bool = False

    # Torch settings
    torch_deterministic: bool = False
    compile: bool = False
    compile_mode: str = "default"
    precision: Literal["float32", "bfloat16"] = "float32"
    amp: bool = True
    cpu_offload: bool = False

    # Checkpointing
    checkpoint_interval: int = 100


# -----------------------------------------------------------------------------
# Profiling
# -----------------------------------------------------------------------------


@dataclass
class ProfileEntry:
    """Single profiling entry."""

    start: float = 0.0
    delta: float = 0.0
    elapsed: float = 0.0
    buffer: float = 0.0


@final
class Profile:
    """Hierarchical profiler for training loop."""

    profiles: dict[str, ProfileEntry]
    frequency: int
    stack: list[str]

    def __init__(self, frequency: int = 5) -> None:
        self.profiles = defaultdict(ProfileEntry)
        self.frequency = frequency
        self.stack = []

    def __iter__(self) -> Iterator[tuple[str, ProfileEntry]]:
        return iter(self.profiles.items())

    def __getitem__(self, name: str) -> ProfileEntry:
        return self.profiles[name]

    def __call__(self, name: str, epoch: int, *, nest: bool = False) -> None:
        """Start timing a section."""
        # Skip profiling the first few epochs, which are noisy due to setup
        if (epoch + 1) % self.frequency != 0:
            return

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        tick = time.time()
        if len(self.stack) != 0 and not nest:
            self._pop(tick)

        self.stack.append(name)
        self.profiles[name].start = tick

    def _pop(self, end: float) -> None:
        profile = self.profiles[self.stack.pop()]
        delta = end - profile.start
        profile.delta += delta
        # Multiply delta by freq to account for skipped epochs
        profile.elapsed += delta * self.frequency

    def end(self) -> None:
        """End all active profiling sections."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end = time.time()
        for _ in range(len(self.stack)):
            self._pop(end)

    def clear(self) -> None:
        """Clear accumulated times."""
        for prof in self.profiles.values():
            if prof.delta > 0:
                prof.buffer = prof.delta
                prof.delta = 0


# -----------------------------------------------------------------------------
# System Utilization
# -----------------------------------------------------------------------------


@final
class Utilization(Thread):
    """Background thread monitoring CPU/GPU utilization."""

    cpu_mem: deque[float]
    cpu_util: deque[float]
    gpu_util: deque[float]
    gpu_mem: deque[float]
    stopped: bool
    delay: float

    def __init__(self, delay: float = 1.0, maxlen: int = 20) -> None:
        super().__init__(daemon=True)
        self.cpu_mem = deque([0.0], maxlen=maxlen)
        self.cpu_util = deque([0.0], maxlen=maxlen)
        self.gpu_util = deque([0.0], maxlen=maxlen)
        self.gpu_mem = deque([0.0], maxlen=maxlen)
        self.stopped = False
        self.delay = delay
        self.start()

    @override
    def run(self) -> None:
        while not self.stopped:
            cpu_count = psutil.cpu_count() or 1
            self.cpu_util.append(100 * psutil.cpu_percent() / cpu_count)
            mem = psutil.virtual_memory()
            self.cpu_mem.append(100 * mem.active / mem.total)

            if torch.cuda.is_available():
                # Monitoring in distributed crashes nvml
                if torch.distributed.is_initialized():
                    time.sleep(self.delay)
                    continue

                self.gpu_util.append(float(torch.cuda.utilization()))
                free, total = torch.cuda.mem_get_info()
                self.gpu_mem.append(100 * (total - free) / total)
            else:
                self.gpu_util.append(0.0)
                self.gpu_mem.append(0.0)

            time.sleep(self.delay)

    def stop(self) -> None:
        self.stopped = True


# -----------------------------------------------------------------------------
# Advantage Computation
# -----------------------------------------------------------------------------


def compute_gae_advantage(
    values: Tensor,
    rewards: Tensor,
    terminals: Tensor,
    ratio: Tensor,
    gamma: float,
    gae_lambda: float,
    vtrace_rho_clip: float,
    vtrace_c_clip: float,
) -> Tensor:
    """Compute GAE advantage with V-trace importance sampling correction.

    This is a pure PyTorch implementation that replaces the CUDA kernel.

    Args:
        values: Value estimates [segments, horizon]
        rewards: Rewards [segments, horizon]
        terminals: Terminal flags [segments, horizon]
        ratio: Importance sampling ratios [segments, horizon]
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        vtrace_rho_clip: V-trace rho clipping threshold
        vtrace_c_clip: V-trace c clipping threshold

    Returns:
        Advantages [segments, horizon]
    """
    device = values.device
    segments, horizon = values.shape

    # Clip importance sampling ratios for V-trace
    rho = torch.clamp(ratio, max=vtrace_rho_clip)
    c = torch.clamp(ratio, max=vtrace_c_clip)

    advantages = torch.zeros_like(values)
    last_gae = torch.zeros(segments, device=device)

    # Compute advantages backwards through time
    for t in reversed(range(horizon)):
        if t == horizon - 1:
            next_value = torch.zeros(segments, device=device)
        else:
            next_value = values[:, t + 1]

        not_terminal = 1.0 - terminals[:, t]
        delta = rho[:, t] * (
            rewards[:, t] + gamma * next_value * not_terminal - values[:, t]
        )
        last_gae = delta + gamma * gae_lambda * not_terminal * c[:, t] * last_gae
        advantages[:, t] = last_gae

    return advantages


# -----------------------------------------------------------------------------
# Sampling Utilities
# -----------------------------------------------------------------------------


def sample_logits(
    logits: Tensor,
    action: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Sample actions from logits and compute log probabilities.

    Args:
        logits: Action logits from policy [batch, num_actions]
        action: Optional pre-specified actions to compute log probs for

    Returns:
        Tuple of (actions, log_probs, entropy)
    """
    probs = torch.softmax(logits, dim=-1)
    log_probs_all = torch.log_softmax(logits, dim=-1)

    if action is None:
        action = torch.multinomial(probs, num_samples=1).squeeze(-1)

    # Gather log probs for selected actions
    log_prob = log_probs_all.gather(-1, action.unsqueeze(-1).long()).squeeze(-1)

    # Compute entropy
    entropy = -(probs * log_probs_all).sum(dim=-1)

    return action, log_prob, entropy


# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------


NUMPY_TO_TORCH_DTYPE: Final[dict[Any, torch.dtype]] = {
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.uint8: torch.uint8,
    np.dtype("float32"): torch.float32,
    np.dtype("float64"): torch.float64,
    np.dtype("int32"): torch.int32,
    np.dtype("int64"): torch.int64,
    np.dtype("uint8"): torch.uint8,
}


@dataclass
class TrainState:
    """Mutable training state."""

    epoch: int = 0
    global_step: int = 0
    last_log_step: int = 0
    last_log_time: float = field(default_factory=time.time)
    start_time: float = field(default_factory=time.time)


@final
class Trainer:
    """PPO trainer with V-trace importance sampling.

    This is a simplified version of PufferLib's PuffeRL that assumes
    the environment is already multithreaded (no vectorization needed).
    """

    config: RLConfig
    env: Environment
    device: torch.device
    num_envs: int
    segments: int
    horizon: int
    observations: Tensor
    actions: Tensor
    values: Tensor
    logprobs: Tensor
    rewards: Tensor
    terminals: Tensor
    ratio: Tensor
    lstm_h: Tensor | None
    lstm_c: Tensor | None
    minibatch_size: int
    accumulate_minibatches: int
    total_minibatches: int
    minibatch_segments: int
    uncompiled_policy: Policy
    policy: Policy
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR
    total_epochs: int
    amp_context: contextlib.AbstractContextManager[None] | torch.amp.autocast
    logger: Logger
    state: TrainState
    stats: dict[str, list[float]]
    losses: dict[str, float]
    utilization: Utilization
    profile: Profile
    model_size: int

    def __init__(
        self,
        config: RLConfig,
        env: Environment,
        policy: Policy,
        logger: Logger | None = None,
    ) -> None:
        # Backend perf optimization
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.deterministic = config.torch_deterministic
        torch.backends.cudnn.benchmark = True

        self.config = config
        self.env = env
        self.device = torch.device(config.device)

        # Environment info
        obs_space = env.observation_space
        atn_space = env.action_space
        num_envs = env.num_envs

        # Compute segments and horizon
        horizon = config.bptt_horizon
        batch_size = config.batch_size
        segments = batch_size // horizon

        if segments * horizon != batch_size:
            raise ValueError(
                f"batch_size {batch_size} must be divisible by bptt_horizon {horizon}"
            )

        self.num_envs = num_envs
        self.segments = segments
        self.horizon = horizon

        # Experience buffers
        obs_dtype = NUMPY_TO_TORCH_DTYPE.get(obs_space.dtype, torch.float32)
        atn_dtype = NUMPY_TO_TORCH_DTYPE.get(atn_space.dtype, torch.int64)

        pin_memory = config.device == "cuda" and config.cpu_offload
        buffer_device = "cpu" if config.cpu_offload else self.device

        self.observations = torch.zeros(
            segments,
            horizon,
            *obs_space.shape,
            dtype=obs_dtype,
            pin_memory=pin_memory,
            device=buffer_device,
        )
        self.actions = torch.zeros(
            segments, horizon, *atn_space.shape, device=self.device, dtype=atn_dtype
        )
        self.values = torch.zeros(segments, horizon, device=self.device)
        self.logprobs = torch.zeros(segments, horizon, device=self.device)
        self.rewards = torch.zeros(segments, horizon, device=self.device)
        self.terminals = torch.zeros(segments, horizon, device=self.device)
        self.ratio = torch.ones(segments, horizon, device=self.device)

        # LSTM state
        self.lstm_h: Tensor | None = None
        self.lstm_c: Tensor | None = None
        if config.use_rnn:
            h = policy.hidden_size
            self.lstm_h = torch.zeros(num_envs, h, device=self.device)
            self.lstm_c = torch.zeros(num_envs, h, device=self.device)

        # Minibatching
        minibatch_size = min(config.minibatch_size, config.max_minibatch_size)
        self.minibatch_size = minibatch_size

        if batch_size < minibatch_size:
            raise ValueError(
                f"batch_size {batch_size} must be >= minibatch_size {minibatch_size}"
            )

        self.accumulate_minibatches = max(
            1, config.minibatch_size // config.max_minibatch_size
        )
        self.total_minibatches = int(config.update_epochs * batch_size / minibatch_size)
        self.minibatch_segments = minibatch_size // horizon

        if self.minibatch_segments * horizon != minibatch_size:
            raise ValueError(
                f"minibatch_size {minibatch_size} must be divisible by bptt_horizon {horizon}"
            )

        # Policy and compilation
        self.uncompiled_policy = policy
        self.policy = policy
        if config.compile:
            self.policy = torch.compile(policy, mode=config.compile_mode)  # pyright: ignore[reportAttributeAccessIssue]

        # Optimizer
        if config.optimizer == "adam":
            self.optimizer: torch.optim.Optimizer = torch.optim.Adam(
                self.policy.parameters(),
                lr=config.learning_rate,
                betas=(config.adam_beta1, config.adam_beta2),
                eps=config.adam_eps,
            )
        elif config.optimizer == "muon":
            try:
                import heavyball  # pyright: ignore[reportMissingImports]

                ForeachMuon = heavyball.ForeachMuon
            except ImportError as e:
                raise ImportError(
                    "Muon optimizer requires heavyball package: pip install heavyball"
                ) from e

            self.optimizer = ForeachMuon(
                self.policy.parameters(),
                lr=config.learning_rate,
                betas=(config.adam_beta1, config.adam_beta2),
                eps=config.adam_eps,
                heavyball_momentum=True,
            )
        else:
            raise ValueError(f"Unknown optimizer: {config.optimizer}")

        # Learning rate scheduler
        total_epochs = config.total_timesteps // config.batch_size
        eta_min = config.learning_rate * config.min_lr_ratio
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_epochs, eta_min=eta_min
        )
        self.total_epochs = total_epochs

        # AMP context
        self.amp_context = contextlib.nullcontext()
        if config.amp and config.device == "cuda":
            dtype = torch.bfloat16 if config.precision == "bfloat16" else torch.float32
            self.amp_context = torch.amp.autocast(device_type="cuda", dtype=dtype)

        if config.precision not in ("float32", "bfloat16"):
            raise ValueError(
                f"Invalid precision: {config.precision}: use float32 or bfloat16"
            )

        # Logger
        from playhouse.logger.noop import NoopLogger

        self.logger: Logger = logger if logger is not None else NoopLogger()

        # Training state
        self.state = TrainState()
        self.stats: dict[str, list[float]] = defaultdict(list)
        self.losses: dict[str, float] = {}

        # Monitoring
        self.utilization = Utilization()
        self.profile = Profile()

        # Model info
        self.model_size = sum(p.numel() for p in policy.parameters() if p.requires_grad)

    @property
    def uptime(self) -> float:
        return time.time() - self.state.start_time

    @property
    def sps(self) -> float:
        """Samples per second."""
        if self.state.global_step == self.state.last_log_step:
            return 0.0

        elapsed = time.time() - self.state.last_log_time
        return (self.state.global_step - self.state.last_log_step) / elapsed

    def evaluate(self) -> dict[str, list[float]]:
        """Run environment rollouts to collect experience."""
        profile = self.profile
        epoch = self.state.epoch
        config = self.config
        device = self.device

        profile("eval", epoch)
        profile("eval_misc", epoch, nest=True)

        # Reset LSTM state at start of episode
        if config.use_rnn and self.lstm_h is not None and self.lstm_c is not None:
            self.lstm_h.zero_()
            self.lstm_c.zero_()

        # Reset environment
        profile("env", epoch)
        obs, info = self.env.reset(seed=config.seed + epoch)

        # Collect experience
        segment_idx = 0
        step_in_segment = 0

        while segment_idx < self.segments:
            profile("eval_copy", epoch)
            obs_tensor = torch.as_tensor(obs, device=device)

            profile("eval_forward", epoch)
            with torch.no_grad(), self.amp_context:
                state: dict[str, Any] = {}
                if config.use_rnn:
                    state["lstm_h"] = self.lstm_h
                    state["lstm_c"] = self.lstm_c

                logits, value = self.policy.forward_eval(obs_tensor, state)
                action, logprob, _ = sample_logits(logits)

                if config.use_rnn:
                    self.lstm_h = state.get("lstm_h")
                    self.lstm_c = state.get("lstm_c")

            profile("env", epoch)
            action_np = action.cpu().numpy()
            next_obs, reward, terminated, truncated, info = self.env.step(action_np)

            profile("eval_copy", epoch)
            # Clamp rewards
            reward_tensor = torch.as_tensor(reward, device=device).clamp(-1, 1)
            done_tensor = torch.as_tensor(
                terminated | truncated, dtype=torch.float32, device=device
            )

            # Store experience
            # For simplicity, we assume num_envs == segments for now
            # A more complete implementation would handle the mapping
            env_segment = min(segment_idx, self.segments - 1)

            if config.cpu_offload:
                self.observations[env_segment, step_in_segment] = torch.as_tensor(obs)
            else:
                self.observations[env_segment, step_in_segment] = obs_tensor

            self.actions[env_segment, step_in_segment] = action
            self.logprobs[env_segment, step_in_segment] = logprob
            self.rewards[env_segment, step_in_segment] = reward_tensor.mean()
            self.terminals[env_segment, step_in_segment] = done_tensor.mean()
            self.values[env_segment, step_in_segment] = value.flatten().mean()

            self.state.global_step += self.num_envs

            # Advance counters
            step_in_segment += 1
            if step_in_segment >= self.horizon:
                step_in_segment = 0
                segment_idx += 1

            obs = next_obs

            # Collect stats from info
            profile("eval_misc", epoch)
            if isinstance(info, dict):
                for k, v in info.items():
                    if isinstance(v, (int, float)):
                        self.stats[k].append(float(v))
                    elif isinstance(v, np.ndarray):
                        self.stats[k].extend(v.tolist())

        profile.end()
        return self.stats

    def train(self) -> dict[str, Any] | None:
        """Train on collected experience."""
        profile = self.profile
        epoch = self.state.epoch
        config = self.config
        device = self.device

        profile("train", epoch)
        profile("train_misc", epoch, nest=True)

        losses: dict[str, float] = defaultdict(float)

        # Prioritized experience replay parameters
        b0 = config.prio_beta0
        a = config.prio_alpha
        clip_coef = config.clip_coef
        vf_clip = config.vf_clip_coef
        anneal_beta = b0 + (1 - b0) * a * epoch / self.total_epochs

        self.ratio[:] = 1

        # Initialize advantages (will be computed in first iteration)
        advantages = compute_gae_advantage(
            self.values,
            self.rewards,
            self.terminals,
            self.ratio,
            config.gamma,
            config.gae_lambda,
            config.vtrace_rho_clip,
            config.vtrace_c_clip,
        )

        for mb in range(self.total_minibatches):
            profile("train_misc", epoch)

            # Recompute advantages with updated ratios
            advantages = compute_gae_advantage(
                self.values,
                self.rewards,
                self.terminals,
                self.ratio,
                config.gamma,
                config.gae_lambda,
                config.vtrace_rho_clip,
                config.vtrace_c_clip,
            )

            # Prioritize experience by advantage magnitude
            adv = advantages.abs().sum(dim=1)
            prio_weights = torch.nan_to_num(adv**a, 0, 0, 0)
            prio_probs = (prio_weights + 1e-6) / (prio_weights.sum() + 1e-6)
            idx = torch.multinomial(prio_probs, self.minibatch_segments)
            mb_prio = (self.segments * prio_probs[idx, None]) ** -anneal_beta

            profile("train_copy", epoch)
            mb_obs = self.observations[idx]
            mb_actions = self.actions[idx]
            mb_logprobs = self.logprobs[idx]
            mb_values = self.values[idx]
            mb_returns = advantages[idx] + mb_values
            mb_advantages = advantages[idx]

            profile("train_forward", epoch)
            with self.amp_context:
                # Reshape for non-RNN forward pass
                if not config.use_rnn:
                    obs_shape = self.env.observation_space.shape
                    mb_obs = mb_obs.reshape(-1, *obs_shape)

                state: dict[str, Any] = {
                    "action": mb_actions,
                    "lstm_h": None,
                    "lstm_c": None,
                }
                logits, newvalue = self.policy(mb_obs.to(device), state)
                _, newlogprob, entropy = sample_logits(
                    logits, action=mb_actions.reshape(-1)
                )

            profile("train_misc", epoch)
            newlogprob = newlogprob.reshape(mb_logprobs.shape)
            logratio = newlogprob - mb_logprobs
            ratio = logratio.exp()
            self.ratio[idx] = ratio.detach()

            with torch.no_grad():
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean()

            # Weight advantages by priority and normalize
            adv_mb = mb_advantages
            adv_mb = mb_prio * (adv_mb - adv_mb.mean()) / (adv_mb.std() + 1e-8)

            # Policy loss
            pg_loss1 = -adv_mb * ratio
            pg_loss2 = -adv_mb * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            newvalue = newvalue.view(mb_returns.shape)
            v_clipped = mb_values + torch.clamp(newvalue - mb_values, -vf_clip, vf_clip)
            v_loss_unclipped = (newvalue - mb_returns) ** 2
            v_loss_clipped = (v_clipped - mb_returns) ** 2
            v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

            # Entropy loss
            entropy_loss = entropy.mean()

            # Total loss
            loss = pg_loss + config.vf_coef * v_loss - config.ent_coef * entropy_loss

            # Update stored values for next iteration
            self.values[idx] = newvalue.detach().float()

            # Accumulate losses for logging
            profile("train_misc", epoch)
            losses["policy_loss"] += pg_loss.item() / self.total_minibatches
            losses["value_loss"] += v_loss.item() / self.total_minibatches
            losses["entropy"] += entropy_loss.item() / self.total_minibatches
            losses["old_approx_kl"] += old_approx_kl.item() / self.total_minibatches
            losses["approx_kl"] += approx_kl.item() / self.total_minibatches
            losses["clipfrac"] += clipfrac.item() / self.total_minibatches
            losses["importance"] += ratio.mean().item() / self.total_minibatches

            # Backward pass and optimization
            profile("learn", epoch)
            loss.backward()
            if (mb + 1) % self.accumulate_minibatches == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), config.max_grad_norm
                )
                self.optimizer.step()
                self.optimizer.zero_grad()

        # Learning rate scheduling
        profile("train_misc", epoch)
        if config.anneal_lr:
            self.scheduler.step()

        # Compute explained variance
        y_pred = self.values.flatten()
        y_true = advantages.flatten() + self.values.flatten()
        var_y = y_true.var()
        explained_var = (
            float("nan") if var_y == 0 else (1 - (y_true - y_pred).var() / var_y).item()
        )
        losses["explained_variance"] = explained_var

        profile.end()

        # Logging
        self.state.epoch += 1
        done_training = self.state.global_step >= config.total_timesteps
        logs = None

        if (
            done_training
            or self.state.global_step == 0
            or time.time() > self.state.last_log_time + 0.25
        ):
            logs = self._log_stats(losses)
            self.losses = losses
            self.stats = defaultdict(list)
            self.state.last_log_time = time.time()
            self.state.last_log_step = self.state.global_step
            profile.clear()

        if self.state.epoch % config.checkpoint_interval == 0 or done_training:
            self.save_checkpoint()

        return logs

    def _log_stats(self, losses: dict[str, float]) -> dict[str, Any]:
        """Compute and log training statistics."""
        device = self.device

        # Average user stats
        stats_averaged: dict[str, float] = {}
        for k, v in self.stats.items():
            try:
                stats_averaged[k] = float(np.mean(v))
            except (TypeError, ValueError):
                continue

        agent_steps = _dist_sum(self.state.global_step, device)

        logs: dict[str, Any] = {
            "SPS": _dist_sum(self.sps, device),
            "agent_steps": int(agent_steps),
            "uptime": self.uptime,
            "epoch": int(_dist_sum(self.state.epoch, device)),
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            **{f"environment/{k}": v for k, v in stats_averaged.items()},
            **{f"losses/{k}": v for k, v in losses.items()},
            **{f"performance/{k}": v.elapsed for k, v in self.profile},
        }

        # Only log on rank 0 in distributed setting
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                return logs

        self.logger.log(logs, int(agent_steps))
        return logs

    def save_checkpoint(self) -> str:
        """Save model checkpoint."""
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                return ""

        config = self.config
        run_id = self.logger.run_id

        path = os.path.join(config.data_dir, f"{config.env_name}_{run_id}")
        if not os.path.exists(path):
            os.makedirs(path)

        model_name = f"model_{config.env_name}_{self.state.epoch:06d}.pt"
        model_path = os.path.join(path, model_name)

        if os.path.exists(model_path):
            return model_path

        torch.save(self.uncompiled_policy.state_dict(), model_path)

        state = {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.state.global_step,
            "epoch": self.state.epoch,
            "model_name": model_name,
            "run_id": run_id,
        }
        state_path = os.path.join(path, "trainer_state.pt")
        torch.save(state, state_path + ".tmp")
        os.replace(state_path + ".tmp", state_path)

        return model_path

    def close(self) -> str:
        """Clean up resources and save final checkpoint."""
        self.env.close()
        self.utilization.stop()
        model_path = self.save_checkpoint()

        config = self.config
        run_id = self.logger.run_id
        final_path = os.path.join(config.data_dir, f"{config.env_name}_{run_id}.pt")
        if model_path:
            shutil.copy(model_path, final_path)

        self.logger.close(final_path)
        return final_path


# -----------------------------------------------------------------------------
# Distributed Utilities
# -----------------------------------------------------------------------------


def _dist_sum(value: float | int, device: str | torch.device) -> float:
    """Sum a value across all distributed processes."""
    if not torch.distributed.is_initialized():
        return float(value)

    tensor = torch.tensor(value, device=device)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return tensor.item()


def _dist_mean(value: float | int, device: str | torch.device) -> float:
    """Average a value across all distributed processes."""
    if not torch.distributed.is_initialized():
        return float(value)

    return _dist_sum(value, device) / torch.distributed.get_world_size()
