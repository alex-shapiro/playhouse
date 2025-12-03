from typing import Any, Protocol

from playhouse.logger.neptune import NeptuneConfig, NeptuneLogger
from playhouse.logger.noop import NoopLogger
from playhouse.logger.wandb import WandbConfig, WandbLogger


class Logger(Protocol):
    """Protocol for loggers"""

    run_id: str

    def log(self, logs: dict[str, Any], step: int) -> None: ...
    def close(self, model_path: str) -> None: ...


def init_logger(config: WandbConfig | NeptuneConfig | None) -> Logger:
    """Initializes a logger from its config"""
    if isinstance(config, WandbConfig):
        logger = WandbLogger(config)
        print(f"  Logging to W&B run: {logger.run_id}")
    elif isinstance(config, NeptuneConfig):
        logger = NeptuneLogger(config)
        print(f"  Logging to Neptune run: {logger.run_id}")
    else:
        logger = NoopLogger()
        print(f"  Logging disabled (run_id: {logger.run_id})")
    return logger
