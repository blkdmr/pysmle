import builtins
import os
import re
from datetime import datetime
from pathlib import Path
import wandb
from colorama import Fore, Style

from typing import Any, Dict, Optional

from fenn.args import Parser
from fenn.secrets.keystore import KeyStore


class Logger:
    """Singleton logging system for FENN."""

    _instance: Optional["Logger"] = None

    def __new__(cls) -> "Logger":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @staticmethod
    def get_instance() -> "Logger":
        return Logger()

    def __init__(self) -> None:
        if hasattr(self, "_initialized") and self._initialized:
            return

        self._original_print = builtins.print

        self._keystore = KeyStore()
        self._parser = Parser()

        self._args: Dict[str, Any] = None
        self._wandb_run: Optional[Any] = None
        self._log_file: Optional[Path] = None

        self._ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

        self._initialized = True

    # ==========================================================
    # SYSTEM LOGS — auto-tagged with [FENN]
    # ==========================================================

    def system_info(self, message: str) -> None:
        tag = f"{Fore.GREEN}[FENN][INFO]{Style.RESET_ALL}"
        self._log_print(f"{tag} {message}")

    def system_warning(self, message: str) -> None:
        tag = f"{Fore.YELLOW}[FENN][WARNING]{Style.RESET_ALL}"
        self._log_print(f"{tag} {message}")

    def system_exception(self, message: str) -> None:
        tag = f"{Fore.RED}[FENN][EXCEPTION]{Style.RESET_ALL}"
        self._log_print(f"{tag} {message}")

    # ==========================================================
    # USER LOGS — no tags, just printed normally
    # ==========================================================

    def user_info(self, message: str) -> None:
        self._log_print(message)

    def user_warning(self, message: str) -> None:
        self._log_print(message)

    def user_exception(self, message: str) -> None:
        self._log_print(message)

    # ==========================================================
    # LOGGER CONTROL
    # ==========================================================

    def start(self) -> None:
        self._args = self._parser.args

        self._log_filepath = (
            Path(self._args["logger"]["dir"]) / Path(self._args["project"])
        )
        self._log_filename: str = f'{self._args["session_id"]}.log'
        self._log_file = self._log_filepath / self._log_filename

        os.makedirs(self._args["logger"]["dir"], exist_ok=True)
        os.makedirs(self._log_filepath, exist_ok=True)

        with open(self._log_file, "w", encoding="utf-8") as f:
            f.write("")

        self.system_info(
            f"Logging file {self._log_filename} created in {self._log_filepath}"
        )

        builtins.print = self._log_print

        if self._args.get("wandb"):
            self._init_wandb()

    def stop(self) -> None:
        builtins.print = self._original_print
        if self._wandb_run:
            self._wandb_run.finish()

    # ==========================================================
    # INTERNAL PRINT HANDLER
    # ==========================================================

    def _log_print(
        self,
        *objects: Any,
        sep: str = " ",
        end: str = "\n",
        file: Optional[Any] = None,
        flush: bool = False,
    ) -> None:
        message = sep.join(map(str, objects))

        if self._log_file:
            clean_message = self._ansi_escape.sub("", message)
            timestamp = datetime.now().replace(microsecond=0).isoformat(" ")
            with open(self._log_file, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] {clean_message}\n")

        self._original_print(*objects, sep=sep, end=end, file=file, flush=flush)

    # ==========================================================
    # WANDB INITIALIZATION
    # ==========================================================

    def _init_wandb(self) -> None:
        os.environ["WANDB_SILENT"] = "true"

        wandb_conf = self._args.get("wandb", {})

        try:
            wandb_key = self._keystore.get_key("WANDB_API_KEY")
        except Exception as exc:
            self.system_exception("No valid WANDB API key provided in .env")
            raise RuntimeError("No valid WANDB API key provided in .env") from exc

        if not os.environ.get("WANDB_API_KEY"):
            os.environ["WANDB_API_KEY"] = wandb_key

        try:
            self._wandb_run = wandb.init(
                entity=wandb_conf.get("entity"),
                project=self._args.get("project"),
                config=self._args.get("training"),
                name=self._args.get("session_id"),
            )
            self.system_info("Wandb session initialized.")

        except Exception as exc:
            self.system_exception("Failed to start wandb session.")
            self.system_warning("Please check your yaml and .env configuration.")
            self.system_warning("Ensure internet connection is active.")
            raise RuntimeError(f"Failed to initialize wandb: {exc}") from exc
