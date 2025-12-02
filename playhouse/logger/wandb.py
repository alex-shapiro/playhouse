import os
from dataclasses import dataclass, field
from typing import Any, Literal, final


@dataclass
class WandbConfig:
    wandb_project: str
    wandb_group: str
    tags: list[str] = field(default_factory=list)
    should_upload_model: bool = True


@final
class WandbLogger:
    def __init__(
        self,
        config: WandbConfig,
        load_id: str | None = None,
        resume: bool | Literal["allow", "never", "must", "auto"] | None = "allow",
    ) -> None:
        import wandb
        from wandb.util import generate_id

        self.run = wandb.init(
            id=load_id or generate_id(),
            project=config.wandb_project,
            group=config.wandb_group,
            allow_val_change=True,
            save_code=False,
            resume=resume,
            tags=config.tags,
            settings=wandb.Settings(console="off"),
        )
        self.wandb_mod = wandb
        self.run_id = self.run.id
        self.should_upload_model = config.should_upload_model

    def log(self, logs: dict[str, Any], step: int) -> None:
        self.run.log(logs, step=step)

    def upload_model(self, model_path: str) -> None:
        artifact = self.wandb_mod.Artifact(self.run_id, type="model")
        artifact.add_file(model_path)
        self.run.log_artifact(artifact)

    def close(self, model_path: str) -> None:
        if self.should_upload_model:
            self.upload_model(model_path)
        self.run.finish()

    def download(self) -> str:
        artifact = self.run.use_artifact(f"{self.run.id}:latest")
        data_dir: str = artifact.download()
        model_file = max(os.listdir(data_dir))
        return f"{data_dir}/{model_file}"
