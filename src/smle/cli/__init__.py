import argparse
import smle.cli.init_command as init_command
import smle.cli.create_command as create_command

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="smle")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Level 1 ---
    p_init = subparsers.add_parser("init", help="Initialize a smle project")
    p_create = subparsers.add_parser("create", help="Create a file for a smle project")

    #create_subparsers = p_init.add_subparsers(dest="file", required=True, help="Project file")

    # --- Level 2 ---
    p_init.add_argument(
        "template",
        nargs="?",
        default="empty",
        help="Target template",
    )

    p_init.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Target directory",
    )

    p_init.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files if needed",
    )

    p_init.set_defaults(func=init_command.execute)

    p_create.add_argument(
        "filetype",
        choices=["yaml", "env"],
        nargs="?",
        help="Type of file",
    )

    p_create.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Target directory",
    )

    p_create.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files if needed",
    )

    p_create.set_defaults(func=create_command.execute)

    return parser

def main(argv=None):
    parser = build_parser()
    # parse_args will exit with error if commands are missing due to required=True
    args = parser.parse_args(argv)

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
