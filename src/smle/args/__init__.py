import yaml
import os
from colorama import Fore, Style, init
from typing import Any, Dict
from smle.secrets.keystore import KeyStore

class Parser:

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:

        self._config_file: str = "smle.yaml"
        self._args: Dict[str, Any] = {}

        self._keystore: KeyStore = KeyStore()

        # Initialize colorama
        init(autoreset=True)

    def load_configuration(self) -> Any:
        """Loads the YAML configuration into the _args dictionary."""

        with open(self._config_file) as f:
            self._args = yaml.safe_load(f)

        default = "(default)" if self._config_file == "smle.yaml" else ""
        if not os.path.isfile(self._config_file):
            print(f"{Fore.RED}[SMLE] Configuration file {Fore.LIGHTYELLOW_EX}{self._config_file} {default}{Fore.RED} was not found.")
            print(f"{Fore.RED}[SMLE] Please use {Fore.LIGHTYELLOW_EX}smle create yaml{Fore.RED} to create it.{Style.RESET_ALL}")
            raise FileNotFoundError(
                0,
                f"Configuration file {self._config_file} was not found.",
                self._config_file,
            )
        else:
            print(f"{Fore.GREEN}[SMLE] Configuration file {Fore.LIGHTYELLOW_EX}{self._config_file} {default}{Fore.GREEN} loaded.{Style.RESET_ALL}")


        if self._args.get("wandb", {}).get("key"):
            self._keystore.set_key("WANDB_API_KEY", self._args.get("wandb", {}).get("key"))
            self._args.get("wandb", {}).pop("key")
            print((
                f"{Fore.RED}[SMLE] WANDB key in yaml file is deprecated. "
                f"Please use {Fore.LIGHTYELLOW_EX}.env{Fore.RED} file instead.{Style.RESET_ALL}"
            ))

        return self._args

    def print(self) -> None:
        """Public method to trigger the flattened print with colored paths."""
        # Define a color cycle for nesting levels
        colors = [Fore.LIGHTCYAN_EX, Fore.LIGHTBLUE_EX, Fore.LIGHTMAGENTA_EX, Fore.LIGHTGREEN_EX]

        # Flatten dictionary and print
        flat_config = self._flatten_dict(self._args)

        for k, v in flat_config.items():
            parts = k.split("/")
            colored_parts = []

            # Color each segment of the path based on its depth
            for i, part in enumerate(parts):
                color = colors[i % len(colors)]
                colored_parts.append(f"{color}{part}{Style.RESET_ALL}")

            # Join with plain slash and print
            print(f"{'/'.join(colored_parts)}: {v}")

    @property
    def config_file(self) -> str:
        return self._config_file

    @config_file.setter
    def config_file(self, config_file: str) -> None:
        self._config_file = config_file

    @property
    def args(self) -> Dict[str, Any]:
        return self._args

    @staticmethod
    def _flatten_dict(d: dict, parent_key: str = "", sep: str = "/") -> dict:
        """
        Recursively flattens a nested dictionary.
        Example: {'a': {'b': 1}} -> {'a/b': 1}
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(Parser._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)