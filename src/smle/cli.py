import argparse
from pathlib import Path

SMLE_EMPTY_CONFIG_TEMPLATE = """\
project: example

# Training configuration
training:
  device: cpu
  seed: 42
  epochs: 10
  learning_rate: 1e-4
  weight_decay: 1e-2
  train_batch: 32
  test_batch: 32

# Logging
logger:
  dir: logger
  use_wandb: False

# Weights & Biases configuration
# wandb:
#   key: [your_wandb_key]
#   entity: [your_wandb_account]
"""

EMPTY_SCRIPT_TEMPLATE = """\
from smle.utils import set_seed
from smle import smle


@smle
def main(args):


    set_seed(args["training"]["seed"])


    # TODO: add your code here



if __name__ == "__main__":
    main()
"""

def cmd_init(args: argparse.Namespace) -> None:
    root = Path(args.path).resolve()


    if root.exists() and any(root.iterdir()) and not args.force:
        raise SystemExit(
            f"Refusing to init non-empty directory: {root}. Use --force to override."
        )


    root.mkdir(parents=True, exist_ok=True)


    # Folders
    (root / "logger").mkdir(exist_ok=True)


    # Config
    config_path = root / "smle.yaml"
    if config_path.exists() and not args.force:
        raise SystemExit("smle.yaml already exists; use --force to overwrite.")
    config_path.write_text(SMLE_EMPTY_CONFIG_TEMPLATE)


    # Example training script
    train_path = root / "main.py"
    if not train_path.exists() or args.force:
        train_path.write_text(EMPTY_SCRIPT_TEMPLATE)


    print(f"Initialized smle project in {root}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="smle")
    subparsers = parser.add_subparsers(dest="command")


    # init subcommand
    p_init = subparsers.add_parser("init", help="Initialize an empty smle project")
    p_init.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Target directory (default: current directory)",
    )
    p_init.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files if needed",
    )
    p_init.set_defaults(func=cmd_init)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
