from dataclasses import dataclass, field
from typing import Any, Literal

from neptune import Run


@dataclass
class NeptuneConfig:
    neptune_name: str
    neptune_project: str
    should_upload_model: bool = True
    tags: list[str] = field(default_factory=list)


class NeptuneLogger:
    def __init__(
        self,
        config: NeptuneConfig,
        load_id: str | None = None,
        mode: Literal["async", "sync", "offline", "read-only", "debug"] = "async",
    ) -> None:
        import neptune as nept

        neptune_name = config.neptune_name
        neptune_project = config.neptune_project
        neptune = nept.init_run(
            project=f"{neptune_name}/{neptune_project}",
            capture_hardware_metrics=False,
            capture_stdout=False,
            capture_stderr=False,
            capture_traceback=False,
            with_id=load_id,
            mode=mode,
            tags=config.tags,
        )
        self.run_id: str = neptune._sys_id  # pyright: ignore[reportPrivateUsage]
        self.neptune: Run = neptune
        self.should_upload_model: bool = config.should_upload_model

    def log(self, logs: dict[str, Any], step: int) -> None:
        for k, v in logs.items():
            self.neptune[k].append(v, step=step)

    def upload_model(self, model_path: str) -> None:
        self.neptune["model"].track_files(model_path)

    def close(self, model_path: str) -> None:
        if self.should_upload_model:
            self.upload_model(model_path)
        self.neptune.stop()

    def download(self) -> str:
        self.neptune["model"].download(destination="artifacts")
        return f"artifacts/{self.run_id}.pt"
