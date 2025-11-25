import yaml
import os
import sys
from colorama import Fore, Style, init
from typing import Any


class Parser:

    def __init__(self):
        self._default = ""
        self._config_file: str = ""
        self._args = {}
        # Initialize colorama
        init(autoreset=True)

    @property
    def config_file(self) -> str:
        return self._config_file

    @config_file.setter
    def config_file(self, config_file: str) -> None:
        self._config_file = config_file
        if not os.path.isfile(self._config_file):
            print(f"{Fore.RED}[SMLE] Configuration file {Fore.LIGHTYELLOW_EX}{self._config_file} {self._default}{Fore.RED} was not found.")
            print(f"{Fore.RED}[SMLE] Please use {Fore.LIGHTYELLOW_EX}smle create yaml{Fore.RED} to create it.{Style.RESET_ALL}")
            raise FileNotFoundError(
                0,
                f"Configuration file {self._config_file} was not found.",
                self._config_file,
            )
        else:
            print(f"{Fore.GREEN}[SMLE] Configuration file {Fore.LIGHTYELLOW_EX}{self._config_file} {self._default}{Fore.GREEN} loaded.{Style.RESET_ALL}")

        self._default =  "(default)" if config_file == "smle.yaml" else ""

    def load_configuration(self) -> Any:
        """Loads the YAML configuration into the _args dictionary."""
        # If no config file is set, use the default
        if not self._config_file:
            self.config_file = "smle.yaml"
            
        with open(self._config_file) as f:
            self._args = yaml.safe_load(f)

        return self._args

    @property
    def args(self):
        return self._args

    def print(self):
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