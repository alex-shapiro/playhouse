from .neptune import NeptuneConfig, NeptuneLogger
from .noop import NoopLogger
from .protocol import Logger
from .wandb import WandbConfig, WandbLogger


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
