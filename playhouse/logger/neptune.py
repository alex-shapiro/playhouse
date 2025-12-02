from typing_extensions import Literal


class NeptuneConfig:
    neptune_name: str
    neptune_project: str
    should_upload_model: bool = True
    tags: list[str] = []


class NeptuneLogger:
    def __init__(
        self,
        config: NeptuneConfig,
        load_id: str | None = None,
        mode: Literal["async", "sync", "offline", "read-only", "debug"] = "async",
    ):
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
        self.run_id = neptune._sys_id
        self.neptune = neptune
        self.should_upload_model = config.should_upload_model

    def log(self, logs, step):
        for k, v in logs.items():
            self.neptune[k].append(v, step=step)

    def upload_model(self, model_path):
        self.neptune["model"].track_files(model_path)

    def close(self, model_path):
        if self.should_upload_model:
            self.upload_model(model_path)
        self.neptune.stop()

    def download(self):
        self.neptune["model"].download(destination="artifacts")
        return f"artifacts/{self.run_id}.pt"
