"""CLI entry point for CAP General."""

import argparse
import sys

from cap_general.core.agent import BaseAgent


def main():
    """Main entry point of the capcmd line interface."""

    parser = argparse.ArgumentParser("Cap Command Line Interface.")
    parser.add_argument(
        "subcommand",
        type=str,
        choices=[
            "server",
        ],
        help="Subcommand to run. (choices: %(choices)s)",
    )

    parsed = parser.parse_args(sys.argv[1:2])
    # pylint: disable=import-outside-toplevel
    if parsed.subcommand in ("server"):
        sub_parser = argparse.ArgumentParser(parsed.subcommand)
        sub_parser.add_argument("--config", required=True)
        sub_parser.add_argument("--transport", default="streamable-http")
        args = sub_parser.parse_args(sys.argv[2:])
        agent = BaseAgent.from_yaml(args.config)
        agent.serve(transport=args.transport)
    else:
        raise ValueError(f"Unknown subcommand: {parsed.subcommand}")
    # pylint: enable=import-outside-toplevel


if __name__ == "__main__":
    main()
