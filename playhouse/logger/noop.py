import time
from typing import Any

from .protocol import Logger


class NoopLogger(Logger):
    run_id: str

    def __init__(self) -> None:
        self.run_id = str(int(100 * time.time()))

    def log(self, logs: dict[str, Any], step: int) -> None:
        pass

    def close(self, model_path: str) -> None:
        pass
