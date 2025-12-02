import os
from dataclasses import dataclass
from typing import Literal

import wandb
from wandb.util import generate_id


@dataclass
class WandbConfig:
    wandb_project: str
    wandb_group: str
    tags: list[str] = []
    should_upload_model: bool = True


class WandbLogger:
    def __init__(
        self,
        config: WandbConfig,
        load_id=None,
        resume: bool | Literal["allow", "never", "must", "auto"] | None = "allow",
    ):
        wandb.init(
            id=load_id or generate_id(),
            project=config.wandb_project,
            group=config.wandb_group,
            allow_val_change=True,
            save_code=False,
            resume=resume,
            tags=config.tags,
            settings=wandb.Settings(console="off"),  # stop sending dashboard to wandb
        )
        self.wandb = wandb
        assert wandb.run is not None
        self.run_id = wandb.run.id
        self.should_upload_model = config.should_upload_model

    def log(self, logs, step):
        self.wandb.log(logs, step=step)

    def upload_model(self, model_path):
        artifact = self.wandb.Artifact(self.run_id, type="model")
        artifact.add_file(model_path)
        assert self.wandb.run is not None
        self.wandb.run.log_artifact(artifact)

    def close(self, model_path):
        if self.should_upload_model:
            self.upload_model(model_path)
        self.wandb.finish()

    def download(self):
        artifact = self.wandb.use_artifact(f"{self.run_id}:latest")
        data_dir = artifact.download()
        model_file = max(os.listdir(data_dir))
        return f"{data_dir}/{model_file}"
