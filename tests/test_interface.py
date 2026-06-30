"""Tests for the capcmd command-line interface."""

from cap_general.interface.main import _parse_server_args


def test_parse_server_args_with_recursive_overrides():
    args, overrides = _parse_server_args(
        [
            "--config",
            "scene.yaml",
            "--transport",
            "stdio",
            "--server.port",
            "9001",
            "--agents[0].config.robot.seed=11",
        ]
    )

    assert args.config == "scene.yaml"
    assert args.transport == "stdio"
    assert overrides == ["server.port=9001", "agents[0].config.robot.seed=11"]
