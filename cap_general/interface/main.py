"""CLI entry point for CAP General."""

import argparse
import os
import platform
import sys

from cap_general.frameworks import import_frameworks
from cap_general.core.scene import BaseScene


def _normalize_graphics_env() -> None:
    """Normalize graphics env vars before importing framework modules."""
    if platform.system() == "Darwin":
        if os.environ.get("MUJOCO_GL") == "egl":
            os.environ["MUJOCO_GL"] = "cgl"
        else:
            os.environ.setdefault("MUJOCO_GL", "cgl")
        os.environ.pop("PYOPENGL_PLATFORM", None)


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
    _normalize_graphics_env()
    import_frameworks()
    # pylint: disable=import-outside-toplevel
    if parsed.subcommand in ("server"):
        sub_parser = argparse.ArgumentParser(parsed.subcommand)
        sub_parser.add_argument("--config", required=True, help="Scene config path.")
        sub_parser.add_argument("--transport", default="streamable-http")
        args = sub_parser.parse_args(sys.argv[2:])
        scene = BaseScene.from_yaml(args.config)
        scene.serve(transport=args.transport)
    else:
        raise ValueError(f"Unknown subcommand: {parsed.subcommand}")
    # pylint: enable=import-outside-toplevel


if __name__ == "__main__":
    main()
