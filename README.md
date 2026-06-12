# Genesis CAP (Code-as-Policy)

A Genesis-native Code-as-Policy module that can execute a full local task loop from task prompt to generated Python policy execution.

## Overview

This project implements a Code-as-Policy framework with a modular architecture:

- **cap_general.core**: Framework-agnostic core components (APIs, executor, models, environment)
- **cap_general.genesis**: Genesis simulator-specific components (Franka robot API, tasks)

Key features:
- Model adapters for code generation
- API documentation extraction
- In-process code executor with persistent state
- CapEnv for managing task loops
- Genesis Franka cube-lift task example

## Installation

```bash
# Install dependencies
pip install numpy pytest

# Optional: For Hugging Face model support
pip install transformers torch

# Optional: For Genesis simulation
# Follow Genesis installation instructions
```

## Project Structure

```
cap_general/
├── cap_general/              # Main package
│   ├── core/                 # Core framework-agnostic components
│   │   ├── apis/
│   │   │   ├── __init__.py
│   │   │   └── base.py       # CapApiBase with doc extraction
│   │   ├── executor.py       # CodeExecutor with persistent state
│   │   ├── models.py         # Policy models (Static, Callable, HuggingFace)
│   │   ├── env.py            # CapEnv for task loops
│   │   └── result.py         # Result dataclasses
│   └── genesis/              # Genesis-specific components
│       ├── apis/
│       │   ├── __init__.py
│       │   └── franka.py     # Franka robot API
│       └── tasks/
│           ├── __init__.py
│           └── franka_cube.py # Franka cube lift task
├── tests/                    # Test files
│   └── genesis/
│       ├── test_core.py      # Core primitives tests
│       ├── test_env.py       # CapEnv tests
│       └── test_franka_api.py # Franka API tests
└── examples/                 # Example scripts
    └── genesis/
        └── franka_cube_task.py # Example usage
```

## Quick Start

### Run Tests

```bash
# Run all tests
pytest tests/genesis -v

# Run specific test file
pytest tests/genesis/test_core.py -v
```

### Run Example

```bash
# Run with static policy (no model required)
python examples/genesis/franka_cube_task.py

# Run with local Hugging Face model
python examples/genesis/franka_cube_task.py --model-path /path/to/model
```

## Usage Examples

### Basic Code Execution

```python
from cap_general.core.executor import CodeExecutor

executor = CodeExecutor()
result = executor.run("x = 10 + 20")
print(result.success)  # True
print(executor.globals['x'])  # 30
```

### Static Policy Model

```python
from cap_general.core.models import StaticPolicyModel

model = StaticPolicyModel(code="action = [1.0, 2.0, 3.0]")
result = model.generate("move robot")
print(result.code)  # "action = [1.0, 2.0, 3.0]"
```

### CapEnv Task Loop

```python
from cap_general.core.env import CapEnv
from cap_general.core.models import StaticPolicyModel

env = CapEnv(
    task_description="Lift the cube",
    api_docs="def grasp(): ...\ndef lift(): ...",
    policy_model=StaticPolicyModel("grasp()\nlift()\ndone = True"),
    max_steps=5
)

result = env.run()
print(f"Success: {result.success}")
print(f"Steps: {result.total_steps}")
```

### Franka Robot API

```python
from cap_general.genesis.apis.franka import GenesisFrankaApi

# Create API wrapper
franka_api = GenesisFrankaApi(robot=your_genesis_robot)

# Get API documentation for policy model
api_docs = franka_api._function_doc()

# Use API methods
franka_api.set_joint_positions([0.0, -0.3, 0.0, -2.0, 0.0, 1.5, 0.0])
franka_api.grasp()
```

## Architecture

### Core Components

1. **CapApiBase**: Base class for APIs with automatic documentation extraction
2. **CodeExecutor**: Executes Python code with persistent globals and output capture
3. **Policy Models**: Generate code from prompts (Static, Callable, HuggingFace)
4. **CapEnv**: Manages the task execution loop with multi-turn support
5. **Result Types**: Dataclasses for tracking execution results

### Design Principles

- **Test-first development**: All components have comprehensive tests
- **Modular architecture**: Each component is independently testable
- **Lazy loading**: HuggingFace models load only when needed
- **Persistent state**: Code executor maintains globals across executions
- **Flexible policies**: Support for static, callable, and AI-generated policies

## Testing

The project uses pytest for testing. All tests follow a TDD approach:

1. Write failing tests first
2. Implement the feature
3. Verify tests pass

```bash
# Run all tests
pytest tests/genesis -v

# Run with coverage
pytest tests/genesis --cov=genesis.cap
```

## Advanced Features

### Hugging Face Integration

```python
from cap_general.core.models import HuggingFacePolicyModel

# Load a code generation model
model = HuggingFacePolicyModel(
    model_path="Salesforce/codegen-350M-mono",
    device="cuda"  # or "cpu"
)

# Generate code from prompt
result = model.generate("Write code to lift a cube using Franka robot")
print(result.code)
```

### Custom Policy Models

```python
from cap_general.core.models import CallablePolicyModel

def my_generator(prompt: str) -> str:
    # Your custom logic here
    if "grasp" in prompt:
        return "grasp()\ndone = False"
    else:
        return "lift()\ndone = True"

model = CallablePolicyModel(generator_fn=my_generator)
```

## License

MIT License

## Contributing

Contributions are welcome! Please ensure all tests pass before submitting PRs.

```bash
pytest tests/genesis -v
```
