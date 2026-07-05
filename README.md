# CAP General

CAP General is a framework-agnostic Code-as-Policy runtime for building, serving, and testing robot agents. A CAP scene can host one or more agents, route generated Python tasks to them, expose the scene through MCP, record execution history, and run configurable policy pipelines such as training jobs.

The repository currently integrates Genesis, LIBERO, and Robosuite. Genesis is one supported backend rather than the scope of the whole project.

## Features

- Scene-level routing for single-agent and multi-agent execution
- Configurable agents, robots, policies, operators, and computation graphs
- Ordered pipelines for training and other policy transformation jobs
- Asynchronous task dispatch with status monitoring
- MCP server tools for execution, retry, observation, reset, recording, and pipelines
- Scene-level request/response history and robot execution artifacts
- Recursive YAML overrides from the command line
- Framework integrations for Genesis, LIBERO, and Robosuite

## Architecture

```text
cap_general/
├── core/
│   ├── agent/       # Agent execution and policy ownership
│   ├── graph/       # CapGraph, CapNode, and CapData
│   ├── operator/    # Graph operators and model adapters
│   ├── pipeline/    # Ordered jobs such as training
│   ├── policy/      # Policy graph runtime
│   ├── robot/       # Framework-independent robot contract
│   ├── scene/       # Multi-agent routing, monitoring, MCP, and history
│   └── utils/
├── frameworks/
│   ├── genesis/     # GO2, drone, Franka/grasp, and multi-agent scenes
│   ├── libero/      # LIBERO VLA agent and training pipeline
│   └── robosuite/   # Robosuite agent integration
├── interface/       # capcmd CLI
└── skills/          # MCP-facing task, state, reset, and training skills
```

Runtime composition is configuration-driven:

```text
Scene
└── Agent(s)
    ├── Robot
    ├── Policy graph(s)
    │   └── Operator nodes
    └── Pipeline (optional)
        └── Job(s)
```

## Installation

Install the core package and test dependencies:

```bash
pip install -e ".[test]"
```

Framework dependencies are intentionally environment-specific:

- **Genesis:** install Genesis, PyTorch, and RSL-RL in a simulation environment.
- **LIBERO:** install the LIBERO repository and its Robosuite dependencies. The optional package dependencies can be installed with `pip install -e ".[libero]"`.
- **Robosuite:** install the version required by the target environment and model stack.

Paths to external repositories, checkpoints, and models are configured in `configs/`.

## Configuration

Ready-to-run configurations are grouped by framework:

```text
configs/
├── genesis/
│   ├── genesis_drone_agent.yaml
│   ├── genesis_franka_agent.yaml
│   ├── genesis_go2_agent.yaml
│   ├── genesis_grasp_agent.yaml
│   └── genesis_multi_agents.yaml
├── libero/libero_agent.yaml
└── robosuite/robosuite_agent.yaml
```

Configuration values can be overridden recursively when starting a server:

```bash
capcmd server \
  --config configs/genesis/genesis_grasp_agent.yaml \
  --show_viewer false \
  --agents[0].robot.num_envs 1
```

## MCP Server

Start a CAP scene as an MCP server:

```bash
capcmd server --config configs/genesis/genesis_grasp_agent.yaml
```

The scene exposes tools including:

- `agent_doc`
- `execute`
- `retry`
- `monitor`
- `get_obs`
- `reset`
- `record`
- `run_pipe` when a pipeline is configured
- `update_history`

Task calls support multiple agents. Requests are routed by configured name or alias, while responses use each agent's scene-visible mark.

## Testing

The test suite covers the general CAP runtime as well as framework-specific integration cases.

### Core tests

These tests do not require a simulator:

```bash
pytest \
  tests/test_core.py \
  tests/test_graph.py \
  tests/test_interface.py \
  tests/test_robot.py \
  tests/test_scene.py
```

Coverage includes registration, graph execution, operators, policies, robots, pipelines, configuration overrides, multi-agent scene routing, task monitoring, and history tracing.

### Genesis tests

Run these from an environment containing Genesis and RSL-RL:

```bash
python tests/genesis/test_genesis_go2_agent.py
python tests/genesis/test_genesis_drone_agent.py
python tests/genesis/test_genesis_grasp_agent.py
python tests/genesis/test_genesis_franka_agent.py
python tests/genesis/test_genesis_multi_agents.py
```

Training smoke tests use `--train_ep`:

```bash
python tests/genesis/test_genesis_grasp_agent.py --task-num 0 --train_ep 1
```

The multi-agent case supports round-robin execution by default and concurrent batches with `--parallel`:

```bash
python tests/genesis/test_genesis_multi_agents.py --task-num 3
python tests/genesis/test_genesis_multi_agents.py --task-num 1 --parallel
```

### LIBERO test

Run from the environment containing LIBERO and its matching Robosuite installation:

```bash
python tests/libero/test_libero_agent.py
```

### Robosuite test

Run from the environment containing the configured Robosuite stack:

```bash
python tests/robosuite/test_robosuite_agent.py
```

### Remote MCP mode

Framework tests also support `--remote`. Start the configured server first, then run the matching test:

```bash
capcmd server --config configs/genesis/genesis_go2_agent.yaml
python tests/genesis/test_genesis_go2_agent.py --remote
```

## Outputs

Each scene writes runtime data beneath its configured `record_dir`. Depending on the agent and trace level, outputs include:

- `history.json` with scene tool requests and responses
- per-agent execution metadata and generated code
- observations, images, and videos
- pipeline exports and training checkpoints beneath `export_dir`

## License

MIT License
