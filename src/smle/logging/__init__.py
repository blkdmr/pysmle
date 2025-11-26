import builtins
import os
import re
from datetime import datetime
from pathlib import Path
import wandb
from colorama import Fore, Style
import sys
from typing import Any, Dict, Optional

from smle.args import Parser
from smle.secrets.keystore import KeyStore

class Logger:

    def __init__(self) -> None:

        self._original_print = builtins.print

        self._keystore = KeyStore()
        self._parser = Parser()

        self._args: Dict[str, Any] = None
        self._wandb_run: Optional[Any] = None

        # Regex to strip ANSI color codes for the log file
        self._ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    def system_log(self, message:str):
        self._original_print(message)

    def start(self) -> None:

        self._args = self._parser.args

        self._log_filepath = Path(self._args["logger"]["dir"]) / Path(self._args["project"])
        self._log_filename:str = f'{self._args["session_id"]}.log'
        self._log_file = self._log_filepath / self._log_filename

        os.makedirs(self._args["logger"]["dir"], exist_ok=True)
        os.makedirs(self._log_filepath, exist_ok=True)

        # Create/Wipe log file
        with open(self._log_file, "w", encoding="utf-8") as f:
            f.write("")

        print(f"{Fore.GREEN}[SMLE] Logging file {Fore.LIGHTYELLOW_EX}{self._log_filename}{Fore.GREEN} created in {Fore.LIGHTYELLOW_EX}{self._log_filepath}{Fore.GREEN} directory.{Style.RESET_ALL}")

        # Hijack print
        builtins.print = self._log_print

        # Init WandB
        if self._args.get("wandb"):
            self._init_wandb()

    def stop(self) -> None:
        builtins.print = self._original_print
        if self._wandb_run:
            self._wandb_run.finish()

    def _log_print(self, *objects: Any, sep: str = " ", end: str = "\n", file: Optional[Any] = None, flush: bool = False) -> None:
        message = sep.join(map(str, objects))

        # 1. Write Clean Text to File (Strip Colors)
        if self._log_file and "[SMLE]" not in message:
            clean_message = self._ansi_escape.sub('', message)
            timestamp = datetime.now().replace(microsecond=0).isoformat(sep=" ")
            with open(self._log_file, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] {clean_message}\n")

        # 2. Write Colored Text to Console
        self._original_print(*objects, sep=sep, end=end, file=file, flush=flush)

    def _init_wandb(self) -> None:
        """
        Initialize wandb session by reading entity from yaml configuration file
        and wandb key from .env file
        """

        os.environ["WANDB_SILENT"] = "true" # Silence WandB

        wandb_conf = self._args.get("wandb", {})

        try:
            wandb_key = self._keystore.get_key("WANDB_API_KEY")
        except Exception:
            self.system_log(f"{Fore.RED}[SMLE] No valid WANDB API key provided in .env{Style.RESET_ALL}")
            sys.exit(1)

        if not os.environ.get("WANDB_API_KEY"):
            os.environ["WANDB_API_KEY"] = wandb_key

        try:
            self._wandb_run = wandb.init(
                entity=wandb_conf.get("entity"),
                project=self._args.get("project"),
                config=self._args.get("training"),
                name=self._args.get("session_id")
            )

            print(f"{Fore.GREEN}[SMLE] Wandb session initialized.{Style.RESET_ALL}")

        except:
            print(f"{Fore.RED}[SMLE] Failed to start wandb session.{Style.RESET_ALL}")
            print(f"{Fore.RED}[SMLE] Please check your wandb configuration in your yaml file and .env file.{Style.RESET_ALL}")
            print(f"{Fore.RED}[SMLE] Please ensure your internet connection is up-and-running.{Style.RESET_ALL}")
            sys.exit(1)