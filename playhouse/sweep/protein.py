import math

from scipy.stats.qmc import Sobol
from sklearn.linear_model import LogisticRegression

from playhouse.sweep.config import SweepConfig
from playhouse.sweep.hyperparameters import Hyperparameters


class Protein:
    def __init__(
        self,
        config: SweepConfig,
        resample_freq: int = 0,
        num_random_samples: int = 0,
        global_search_scale: int = 0,
        suggestions_per_pareto: int = 256,
        expansion_rate: float = 0.25,
        gp_training_iter: int = 50,
        gp_learning_rate: float = 0.001,
        gp_max_ops: int = 750,  # gp train time jumps after 800
        infer_batch_size: int = 4096,
        optimizer_reset_freq: int = 50,
        cost_param: str = "train/total_timesteps",
    ):
        self.hyperparameters = Hyperparameters(config)
        self.device = config.device
        self.prune_pareto = config.prune_pareto
        self.max_suggestion_cost = config.max_suggestion_cost
        self.resample_freq = resample_freq
        self.num_random_samples = num_random_samples
        self.global_search_scale = global_search_scale
        self.suggestions_per_pareto = suggestions_per_pareto
        self.expansion_rate = expansion_rate
        self.gp_training_iter = gp_training_iter
        self.gp_learning_rate = gp_learning_rate
        self.gp_max_ops = gp_max_ops
        self.infer_batch_size = infer_batch_size
        self.optimizer_reset_freq = optimizer_reset_freq

        self.success_observations = []
        self.failure_observations = []

        self.suggestion_idx = 0
        self.min_score, self.max_score = math.inf, -math.inf
        self.log_c_min, self.log_c_max = math.inf, -math.inf

        # sobol sequence for quasi-random exploration
        # TODO: test if this is better than np.random
        self.sobol = Sobol(d=self.hyperparameters.num, scramble=True)

        self.cost_param_idx = self.hyperparameters.index(cost_param)
        self.cost_random_suggestion = None
        if self.cost_param_idx is not None:
            self.cost_random_suggestion = self.hyperparameters.means[
                self.cost_param_idx
            ]

        self.use_success_prob = config.downsample == 1
        self.success_classifier = LogisticRegression(class_weight="balanced")
