import yaml
import argparse

class ArgParser():

    def __init__(self, prog:str, description:str):

        self._parser = argparse.ArgumentParser(
            prog=prog,
            description=description
        )

        self._parser.add_argument(
            "--cfg",
            type=str,
            default="config.yaml",
            help="Path to YAML configuration file (default: config.yaml)"
        )

        self._cli_args = self._parser.parse_args()

        with open(self._cli_args.cfg) as f:
            self._args = yaml.safe_load(f)

    @property
    def args(self):
        return self._args