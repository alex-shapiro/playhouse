from typing import Any, Protocol


class Logger(Protocol):
    """Protocol for loggers (WandbLogger, NeptuneLogger, NoopLogger)."""

    run_id: str

    def log(self, logs: dict[str, Any], step: int) -> None: ...
    def close(self, model_path: str) -> None: ...
