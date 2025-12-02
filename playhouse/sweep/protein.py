"""
Protein: Pareto-based Bayesian optimization for hyperparameter sweeps.
Hyperparameter optimization using Gaussian Process regression and Pareto front analysis.
"""

from __future__ import annotations

import math
import random
import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import gpytorch
import numpy as np
import torch
from gpytorch.kernels import AdditiveKernel, MaternKernel, PolynomialKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from gpytorch.priors import LogNormalPrior
from scipy.spatial import KDTree
from scipy.stats.qmc import Sobol
from sklearn.linear_model import LogisticRegression

from playhouse.sweep.config import SweepConfig
from playhouse.sweep.hyperparameters import Hyperparameters

if TYPE_CHECKING:
    from numpy.typing import NDArray

EPSILON: float = 1e-6


@dataclass(frozen=True, slots=True)
class Observation:
    """A single observation from a hyperparameter evaluation."""

    input: NDArray[np.floating]
    output: float
    cost: float
    is_failure: bool = False


@dataclass(frozen=True, slots=True)
class SuggestionInfo:
    """Metadata about a suggested hyperparameter configuration."""

    cost: float = 0.0
    score: float = 0.0
    rating: float = 0.0
    score_loss: float = 0.0
    cost_loss: float = 0.0
    score_lengthscale: tuple[float, float] = (0.0, 0.0)
    cost_lengthscale: tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True, slots=True)
class Suggestion:
    """A hyperparameter suggestion with its metadata."""

    params: dict[str, float | int]
    info: SuggestionInfo


@dataclass(frozen=True, slots=True)
class ObservationResult:
    """Result of observing a hyperparameter evaluation."""

    success_observations: tuple[Observation, ...]
    failure_observations: tuple[Observation, ...]


@dataclass(frozen=True, slots=True)
class ScoreStats:
    """Statistics about observed scores and costs."""

    min_score: float = math.inf
    max_score: float = -math.inf
    log_c_min: float = math.inf
    log_c_max: float = -math.inf


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------


@contextmanager
def default_tensor_dtype(dtype: torch.dtype) -> Iterator[None]:
    """Context manager to temporarily set the default tensor dtype."""
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        yield
    finally:
        torch.set_default_dtype(old_dtype)


def compute_pareto_points(
    observations: Sequence[Observation],
) -> tuple[list[Observation], list[int]]:
    """Compute Pareto-optimal observations (non-dominated points).

    Returns observations where no other observation has both higher score
    AND lower cost.

    Args:
        observations: Sequence of observations to analyze.

    Returns:
        Tuple of (pareto_observations, pareto_indices).
    """
    if not observations:
        return [], []

    scores = np.array([obs.output for obs in observations])
    costs = np.array([obs.cost for obs in observations])

    sorted_indices = np.argsort(costs)

    pareto: list[Observation] = []
    pareto_idxs: list[int] = []
    max_score_so_far = -np.inf

    for idx in sorted_indices:
        if scores[idx] > max_score_so_far + EPSILON:
            pareto.append(observations[idx])
            pareto_idxs.append(int(idx))
            max_score_so_far = scores[idx]

    return pareto, pareto_idxs


def prune_pareto_front(
    pareto: Sequence[Observation],
    efficiency_threshold: float = 0.5,
    pruning_stop_score_fraction: float = 0.98,
) -> list[Observation]:
    """Prune inefficient high-cost tail of a Pareto front.

    Removes points like (score 0.99, cost 100), (score 0.991, cost 200)
    where the marginal score gain doesn't justify the cost increase.

    Args:
        pareto: Pareto-optimal observations sorted by cost.
        efficiency_threshold: Minimum efficiency ratio to keep a point.
        pruning_stop_score_fraction: Stop pruning below this fraction of max score.

    Returns:
        Pruned Pareto front as a new list.
    """
    if not pareto or len(pareto) < 2:
        return list(pareto)

    sorted_pareto = sorted(pareto, key=lambda x: x.cost)
    scores = np.array([obs.output for obs in sorted_pareto])
    costs = np.array([obs.cost for obs in sorted_pareto])
    score_range = max(scores.max() - scores.min(), EPSILON)
    cost_range = max(costs.max() - costs.min(), EPSILON)

    max_pareto_score = scores[-1] if scores.size > 0 else -np.inf

    result = list(sorted_pareto)
    for i in range(len(sorted_pareto) - 1, 0, -1):
        if scores[i] < pruning_stop_score_fraction * max_pareto_score:
            break

        norm_score_gain = (scores[i] - scores[i - 1]) / score_range
        norm_cost_increase = (costs[i] - costs[i - 1]) / cost_range
        efficiency = norm_score_gain / (norm_cost_increase + EPSILON)

        if efficiency < efficiency_threshold:
            result.pop(i)
        else:
            break

    return result


def filter_near_duplicates(
    inputs: NDArray[np.floating],
    duplicate_threshold: float = EPSILON,
) -> NDArray[np.intp]:
    """Filter out near-duplicate inputs, keeping the most recent.

    Args:
        inputs: Array of input vectors.
        duplicate_threshold: Distance threshold for considering duplicates.

    Returns:
        Indices of inputs to keep.
    """
    if len(inputs) < 2:
        return np.arange(len(inputs))

    tree = KDTree(inputs)
    to_keep = np.ones(len(inputs), dtype=bool)

    for i in range(len(inputs) - 1, -1, -1):
        if to_keep[i]:
            nearby_indices = tree.query_ball_point(inputs[i], r=duplicate_threshold)
            nearby_indices.remove(i)
            if nearby_indices:
                to_keep[nearby_indices] = False

    return np.where(to_keep)[0]


def compute_score_stats(observations: Sequence[Observation]) -> ScoreStats:
    """Compute statistics from observations.

    Args:
        observations: Sequence of observations.

    Returns:
        ScoreStats with min/max score and log cost bounds.
    """
    if not observations:
        return ScoreStats()

    scores = np.array([obs.output for obs in observations])
    costs = np.array([obs.cost for obs in observations])
    log_costs = np.log(np.maximum(costs, EPSILON))

    return ScoreStats(
        min_score=float(scores.min()),
        max_score=float(scores.max()),
        log_c_min=float(log_costs.min()),
        log_c_max=float(log_costs.max()),
    )


# -----------------------------------------------------------------------------
# GP Model
# -----------------------------------------------------------------------------


class ExactGPModel(ExactGP):
    """Exact Gaussian Process model with Matern + linear kernel."""

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: GaussianLikelihood,
        x_dim: int,
    ) -> None:
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        matern_kernel = MaternKernel(nu=1.5, ard_num_dims=x_dim)
        linear_kernel = PolynomialKernel(power=1)
        self.covar_module = ScaleKernel(AdditiveKernel(linear_kernel, matern_kernel))

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)  # pyright: ignore[reportArgumentType]

    @property
    def lengthscale_range(self) -> tuple[float, float]:
        """Get min/max lengthscale from the Matern kernel."""
        # Access the Matern kernel (second in the additive kernel) and get its lengthscale
        additive_kernel = self.covar_module.base_kernel
        matern_kernel = additive_kernel.kernels[1]  # pyright: ignore[reportIndexIssue]
        lengthscale = matern_kernel.lengthscale.tolist()[0]  # pyright: ignore[reportAttributeAccessIssue]
        return (float(min(lengthscale)), float(max(lengthscale)))


def train_gp_model(
    model: ExactGPModel,
    likelihood: GaussianLikelihood,
    mll: ExactMarginalLogLikelihood,
    optimizer: torch.optim.Optimizer,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    training_iter: int = 50,
) -> float:
    """Train a GP model.

    Args:
        model: The GP model to train.
        likelihood: The likelihood function.
        mll: Marginal log likelihood objective.
        optimizer: Optimizer for training.
        train_x: Training inputs.
        train_y: Training targets.
        training_iter: Number of training iterations.

    Returns:
        Final training loss value.
    """
    model.train()
    likelihood.train()
    model.set_train_data(inputs=train_x, targets=train_y, strict=False)

    final_loss: float = 0.0
    for _ in range(training_iter):
        try:
            optimizer.zero_grad()
            output = model(train_x)
            loss: torch.Tensor = -mll(output, train_y)  # pyright: ignore[reportOperatorIssue]
            loss.backward()
            optimizer.step()
            final_loss = loss.detach().item()
        except gpytorch.utils.errors.NotPSDError:
            break

    return final_loss


# -----------------------------------------------------------------------------
# GP Models Container
# -----------------------------------------------------------------------------


@dataclass
class GPModels:
    """Container for GP models and associated state.

    This is mutable by design as GP models need to be trained in place.
    """

    device: torch.device
    num_params: int
    gp_max_obs: int
    infer_batch_size: int
    gp_learning_rate: float

    # Models - initialized in __post_init__
    likelihood_score: GaussianLikelihood = field(init=False)
    gp_score: ExactGPModel = field(init=False)
    mll_score: ExactMarginalLogLikelihood = field(init=False)
    score_opt: torch.optim.Adam = field(init=False)

    likelihood_cost: GaussianLikelihood = field(init=False)
    gp_cost: ExactGPModel = field(init=False)
    mll_cost: ExactMarginalLogLikelihood = field(init=False)
    cost_opt: torch.optim.Adam = field(init=False)

    # Buffers
    gp_params_buffer: torch.Tensor = field(init=False)
    gp_score_buffer: torch.Tensor = field(init=False)
    gp_cost_buffer: torch.Tensor = field(init=False)
    infer_batch_buffer: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        with default_tensor_dtype(torch.float64):
            noise_prior = LogNormalPrior(math.log(1e-2), 0.5)

            dummy_x = torch.ones((1, self.num_params), device=self.device)
            dummy_y = torch.zeros(1, device=self.device)

            # Score GP
            self.likelihood_score = GaussianLikelihood(
                noise_prior=LogNormalPrior(math.log(1e-2), 0.5)
            ).to(self.device)
            self.gp_score = ExactGPModel(
                dummy_x, dummy_y, self.likelihood_score, self.num_params
            ).to(self.device)
            self.mll_score = ExactMarginalLogLikelihood(
                self.likelihood_score, self.gp_score
            ).to(self.device)
            self.score_opt = torch.optim.Adam(
                self.gp_score.parameters(), lr=self.gp_learning_rate, amsgrad=True
            )

            # Cost GP
            self.likelihood_cost = GaussianLikelihood(
                noise_prior=LogNormalPrior(math.log(1e-2), 0.5)
            ).to(self.device)
            self.gp_cost = ExactGPModel(
                dummy_x, dummy_y, self.likelihood_cost, self.num_params
            ).to(self.device)
            self.mll_cost = ExactMarginalLogLikelihood(
                self.likelihood_cost, self.gp_cost
            ).to(self.device)
            self.cost_opt = torch.optim.Adam(
                self.gp_cost.parameters(), lr=self.gp_learning_rate, amsgrad=True
            )

            # Buffers
            self.gp_params_buffer = torch.empty(
                self.gp_max_obs, self.num_params, device=self.device
            )
            self.gp_score_buffer = torch.empty(self.gp_max_obs, device=self.device)
            self.gp_cost_buffer = torch.empty(self.gp_max_obs, device=self.device)
            self.infer_batch_buffer = torch.empty(
                self.infer_batch_size, self.num_params, device=self.device
            )

    def reset_optimizers(self) -> None:
        """Reset both GP optimizers."""
        self.score_opt = torch.optim.Adam(
            self.gp_score.parameters(), lr=self.gp_learning_rate, amsgrad=True
        )
        self.cost_opt = torch.optim.Adam(
            self.gp_cost.parameters(), lr=self.gp_learning_rate, amsgrad=True
        )


# -----------------------------------------------------------------------------
# Protein State
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProteinState:
    """Immutable state for the Protein optimizer.

    All observation data is stored immutably. The GP models are stored
    separately in GPModels which is mutable by necessity.
    """

    success_observations: tuple[Observation, ...] = ()
    failure_observations: tuple[Observation, ...] = ()
    suggestion_idx: int = 0
    stats: ScoreStats = field(default_factory=ScoreStats)


# -----------------------------------------------------------------------------
# Protein Configuration
# -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProteinConfig:
    """Configuration for the Protein optimizer."""

    prune_pareto: bool = True
    max_suggestion_cost: int = 3600
    downsample: int = 1
    resample_freq: int = 0
    num_random_samples: int = 10
    global_search_scale: int = 1
    suggestions_per_pareto: int = 256
    expansion_rate: float = 0.25
    gp_training_iter: int = 50
    gp_learning_rate: float = 0.001
    gp_max_obs: int = 750
    infer_batch_size: int = 4096
    optimizer_reset_freq: int = 50
    cost_param: str = "train/total_timesteps"


# -----------------------------------------------------------------------------
# Protein Optimizer
# -----------------------------------------------------------------------------


class Protein:
    """
    Pareto-based Bayesian optimizer for hyperparameter sweeps.

    Uses Gaussian Process regression to model both the objective score
    and computational cost, then suggests hyperparameters that balance
    score improvement with cost efficiency along the Pareto front.
    """

    def __init__(
        self,
        sweep_config: SweepConfig,
        config: ProteinConfig = ProteinConfig(),
    ) -> None:
        """
        Initialize the Protein optimizer
        sweep_config: Configuration defining the hyperparameter search space.
        config: Optional Protein-specific configuration.
        """

        self.hyperparameters = Hyperparameters(sweep_config)
        self.device = torch.device(sweep_config.device)
        self.config = config

        # Sobol sequence for quasi-random exploration
        self.sobol = Sobol(d=self.hyperparameters.num, scramble=True)

        self.cost_param_idx = self.hyperparameters.index(config.cost_param)
        self.cost_random_suggestion: float | None = None
        if self.cost_param_idx is not None:
            self.cost_random_suggestion = float(
                self.hyperparameters.means[self.cost_param_idx]
            )

        self.use_success_prob = config.downsample == 1
        self.success_classifier = LogisticRegression(class_weight="balanced")

        # Initialize GP models
        self.gp_models = GPModels(
            device=self.device,
            num_params=self.hyperparameters.num,
            gp_max_obs=config.gp_max_obs,
            infer_batch_size=config.infer_batch_size,
            gp_learning_rate=config.gp_learning_rate,
        )

    def _sample_observations(
        self,
        state: ProteinState,
        max_size: int | None = None,
        recent_ratio: float = 0.5,
    ) -> tuple[list[Observation], ScoreStats]:
        """Sample observations for GP training with deduplication.

        Args:
            state: Current optimizer state.
            max_size: Maximum number of observations to return.
            recent_ratio: Fraction of samples to take from recent observations.

        Returns:
            Tuple of (sampled_observations, updated_stats).
        """
        if not state.success_observations:
            return [], state.stats

        observations = list(state.success_observations)

        # Compute stats from full data
        stats = compute_score_stats(observations)

        # When data is scarce, include failed observations
        if len(observations) < 100 and state.failure_observations:
            failure_obs = [
                replace(obs, output=stats.min_score)
                for obs in state.failure_observations
            ]
            observations = list(failure_obs) + observations

        # Deduplicate
        params = np.array(
            [np.append(obs.input, [obs.output, obs.cost]) for obs in observations]
        )
        dedup_indices = filter_near_duplicates(params)
        observations = [observations[i] for i in dedup_indices]

        if max_size is None:
            max_size = self.config.gp_max_obs

        if len(observations) <= max_size:
            return observations, stats

        recent_size = int(recent_ratio * max_size)
        recent_obs = observations[-recent_size:]
        older_obs = observations[:-recent_size]
        num_to_sample = max_size - recent_size
        random_sample_obs = random.sample(older_obs, num_to_sample)

        return random_sample_obs + recent_obs, stats

    def _train_gp_models(self, state: ProteinState) -> tuple[float, float, ScoreStats]:
        """Train GP models on current observations.

        Args:
            state: Current optimizer state.

        Returns:
            Tuple of (score_loss, cost_loss, updated_stats).
        """
        if not state.success_observations:
            return 0.0, 0.0, state.stats

        sampled_obs, stats = self._sample_observations(
            state, max_size=self.config.gp_max_obs
        )
        num_sampled = len(sampled_obs)

        # Prepare tensors
        params = np.array([obs.input for obs in sampled_obs])
        params_tensor = self.gp_models.gp_params_buffer[:num_sampled]
        params_tensor.copy_(torch.from_numpy(params))

        # Normalized scores
        y = np.array([obs.output for obs in sampled_obs])
        y_norm = (y - stats.min_score) / (
            np.abs(stats.max_score - stats.min_score) + EPSILON
        )
        y_norm_tensor = self.gp_models.gp_score_buffer[:num_sampled]
        y_norm_tensor.copy_(torch.from_numpy(y_norm))

        # Normalized log costs
        c = np.array([obs.cost for obs in sampled_obs])
        log_c = np.log(np.maximum(c, EPSILON))
        log_c_norm = (log_c - stats.log_c_min) / (
            stats.log_c_max - stats.log_c_min + EPSILON
        )
        log_c_norm_tensor = self.gp_models.gp_cost_buffer[:num_sampled]
        log_c_norm_tensor.copy_(torch.from_numpy(log_c_norm))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", gpytorch.utils.warnings.NumericalWarning)
            score_loss = train_gp_model(
                self.gp_models.gp_score,
                self.gp_models.likelihood_score,
                self.gp_models.mll_score,
                self.gp_models.score_opt,
                params_tensor,
                y_norm_tensor,
                training_iter=self.config.gp_training_iter,
            )
            cost_loss = train_gp_model(
                self.gp_models.gp_cost,
                self.gp_models.likelihood_cost,
                self.gp_models.mll_cost,
                self.gp_models.cost_opt,
                params_tensor,
                log_c_norm_tensor,
                training_iter=self.config.gp_training_iter,
            )

        return score_loss, cost_loss, stats

    def suggest(self, state: ProteinState) -> tuple[Suggestion, ProteinState]:
        """Suggest a hyperparameter configuration.

        Args:
            state: Current optimizer state.

        Returns:
            Tuple of (suggestion, new_state).
        """
        new_idx = state.suggestion_idx + 1
        new_state = replace(state, suggestion_idx=new_idx)

        # Random exploration phase
        if new_idx <= self.config.num_random_samples:
            zero_one = self.sobol.random(1)[0]
            suggestion = 2 * zero_one - 1  # Scale from [0, 1) to [-1, 1)
            if (
                self.cost_param_idx is not None
                and self.cost_random_suggestion is not None
            ):
                cost_suggestion = self.cost_random_suggestion + 0.1 * np.random.randn()
                suggestion[self.cost_param_idx] = np.clip(cost_suggestion, -1, 1)
            params = self.hyperparameters.to_dict(suggestion)
            return Suggestion(params=params, info=SuggestionInfo()), new_state

        # Resampling from Pareto front
        if self.config.resample_freq and new_idx % self.config.resample_freq == 0:
            candidates, _ = compute_pareto_points(list(state.success_observations))
            if candidates:
                suggestions = np.stack([obs.input for obs in candidates])
                best_idx = np.random.randint(0, len(candidates))
                params = self.hyperparameters.to_dict(suggestions[best_idx])
                return Suggestion(params=params, info=SuggestionInfo()), new_state

        # Train GP models
        score_loss, cost_loss, stats = self._train_gp_models(state)
        new_state = replace(new_state, stats=stats)

        # Reset optimizers periodically
        if (
            self.config.optimizer_reset_freq
            and new_idx % self.config.optimizer_reset_freq == 0
        ):
            self.gp_models.reset_optimizers()

        # Get Pareto candidates
        candidates, _ = compute_pareto_points(list(state.success_observations))
        if self.config.prune_pareto:
            candidates = prune_pareto_front(candidates)

        if not candidates:
            # Fallback to random
            zero_one = self.sobol.random(1)[0]
            suggestion = 2 * zero_one - 1
            params = self.hyperparameters.to_dict(suggestion)
            return Suggestion(params=params, info=SuggestionInfo()), new_state

        # Sample suggestions around Pareto points
        search_centers = np.stack([obs.input for obs in candidates])
        suggestions = self.hyperparameters.sample(
            len(candidates) * self.config.suggestions_per_pareto, mu=search_centers
        )

        dedup_indices = filter_near_duplicates(suggestions)
        suggestions = suggestions[dedup_indices]

        if len(suggestions) == 0:
            return self.suggest(new_state)

        # Predict scores and costs
        gp_y_norm_list: list[torch.Tensor] = []
        gp_log_c_norm_list: list[torch.Tensor] = []

        with (
            torch.no_grad(),
            gpytorch.settings.fast_pred_var(),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore", gpytorch.utils.warnings.NumericalWarning)

            for i in range(0, len(suggestions), self.config.infer_batch_size):
                batch_numpy = suggestions[i : i + self.config.infer_batch_size]
                current_batch_size = len(batch_numpy)

                batch_tensor = self.gp_models.infer_batch_buffer[:current_batch_size]
                batch_tensor.copy_(torch.from_numpy(batch_numpy))

                try:
                    pred_y_mean = self.gp_models.likelihood_score(
                        self.gp_models.gp_score(batch_tensor)
                    ).mean.cpu()
                    pred_c_mean = self.gp_models.likelihood_cost(
                        self.gp_models.gp_cost(batch_tensor)
                    ).mean.cpu()
                except RuntimeError:
                    pred_y_mean = torch.zeros(current_batch_size)
                    pred_c_mean = torch.zeros(current_batch_size)

                gp_y_norm_list.append(pred_y_mean)
                gp_log_c_norm_list.append(pred_c_mean)

        gp_y_norm = torch.cat(gp_y_norm_list).numpy()
        gp_log_c_norm = torch.cat(gp_log_c_norm_list).numpy()

        # Unnormalize predictions
        gp_y = gp_y_norm * (stats.max_score - stats.min_score) + stats.min_score
        gp_log_c = gp_log_c_norm * (stats.log_c_max - stats.log_c_min) + stats.log_c_min
        gp_c = np.exp(gp_log_c)

        # Score suggestions
        suggestion_scores = self.hyperparameters.optimize_direction * gp_y_norm

        # Apply cost constraints and weighting
        max_c_mask = gp_c < self.config.max_suggestion_cost
        target = (1 + self.config.expansion_rate) * np.random.rand()
        weight = 1 - abs(target - gp_log_c_norm)
        suggestion_scores = suggestion_scores * max_c_mask * weight

        # Consider success probability if enabled
        if (
            self.use_success_prob
            and len(state.success_observations) > 9
            and len(state.failure_observations) > 9
        ):
            success_params = np.array([obs.input for obs in state.success_observations])
            failure_params = np.array([obs.input for obs in state.failure_observations])
            X_train = np.vstack([success_params, failure_params])
            y_train = np.concatenate(
                [np.ones(len(success_params)), np.zeros(len(failure_params))]
            )
            if len(np.unique(y_train)) > 1:
                self.success_classifier.fit(X_train, y_train)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    p_success = self.success_classifier.predict_proba(suggestions)[:, 1]
                suggestion_scores = suggestion_scores * p_success

        best_idx = int(np.argmax(suggestion_scores))
        info = SuggestionInfo(
            cost=float(gp_c[best_idx]),
            score=float(gp_y[best_idx]),
            rating=float(suggestion_scores[best_idx]),
            score_loss=score_loss,
            cost_loss=cost_loss,
            score_lengthscale=self.gp_models.gp_score.lengthscale_range,
            cost_lengthscale=self.gp_models.gp_cost.lengthscale_range,
        )

        best = suggestions[best_idx]
        params = self.hyperparameters.to_dict(best)
        return Suggestion(params=params, info=info), new_state


def observe(
    hyperparameters: Hyperparameters,
    state: ProteinState,
    hypers: dict[str, float | int],
    score: float,
    cost: float,
    is_failure: bool = False,
    cost_param_idx: int | None = None,
) -> ProteinState:
    """
    Record an observation from a hyperparameter evaluation

    Args:
        hyperparameters: Hyperparameters object for conversion.
        state: Current optimizer state.
        hypers: The hyperparameter dictionary that was evaluated.
        score: The objective score achieved.
        cost: The computational cost incurred.
        is_failure: Whether the evaluation failed.
        cost_param_idx: Index of cost parameter (for filtering).

    Returns:
        New ProteinState with the observation recorded.
    """
    params = hyperparameters.from_dict(hypers)
    observation = Observation(
        input=params,
        output=score,
        cost=cost,
        is_failure=is_failure,
    )

    # Handle failures
    if is_failure or not np.isfinite(score) or np.isnan(score):
        observation = replace(observation, is_failure=True)
        return replace(
            state,
            failure_observations=state.failure_observations + (observation,),
        )

    # Check for near-duplicates in success observations
    if state.success_observations:
        success_params = np.stack([obs.input for obs in state.success_observations])
        dist = np.linalg.norm(params - success_params, axis=1)
        same = np.where(dist < EPSILON)[0]
        if len(same) > 0:
            # Replace existing observation
            idx = int(same[0])
            new_obs = (
                state.success_observations[:idx]
                + (observation,)
                + state.success_observations[idx + 1 :]
            )
            return replace(state, success_observations=new_obs)

    # Ignore observations below minimum cost
    if cost_param_idx is not None and params[cost_param_idx] <= -1:
        return state

    return replace(
        state,
        success_observations=state.success_observations + (observation,),
    )
