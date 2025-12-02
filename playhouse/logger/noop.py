import time


class NoopLogger:
    def __init__(self, args):
        self.run_id = str(int(100 * time.time()))

    def log(self, logs, step):
        pass

    def close(self, model_path):
        pass
