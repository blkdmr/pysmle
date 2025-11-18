from datetime import datetime
import wandb
import os

from .args import ArgParser

class Logger():

    def __init__(self, arg_parser:ArgParser):

        self._logger_args = arg_parser.args["logger"]
        self._args = arg_parser.args

        os.makedirs(self._logger_args["dir"], exist_ok=True)

        self._export_dir = self._logger_args["dir"]
        self._filename = f"{self._args["project"]}.log"

        self._log_file = os.path.join(self._export_dir, self._filename)

        with open(self._log_file, "w") as f:
            f.write("")

        for k, v in self._flatten_dict(self._args).items():

            self.log(f"{k}: {v}")

        self._use_wandb = self._logger_args["use_wandb"]

        if self._use_wandb:

            self._start_wandb_session(
                key=self._args["wandb"]["key"],
                entity=self._args["wandb"]["entity"],
                project=self._args["wandb"]["project"],
                config=self._args["training"]
            )

    def __del__(self):
        if self._use_wandb:
            self._wandb_session.finish()

    def log(self, message):

        timestamp = str(datetime.now()).split(".")[0]

        with open(self._log_file, "a") as f:
            f.write(f"[{timestamp}] {message}\n")

        print(message)

    def _start_wandb_session(self,
                           key:str,
                           entity:str,
                           project:str,
                           config
                           ) -> None:

        os.environ['WANDB_API_KEY'] = key

        self._wandb_session = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity=entity,
            # Set the wandb project where this run will be logged.
            project=project,
            # Track hyperparameters and run metadata.
            config=config,
        )

    def _flatten_dict(self, d, parent_key='', sep='/'):
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
