"""Tests for core CAP primitives: API docs, executor, and policy models."""

import pytest
from cap_general.core.agent import AgentBase, CodeExecutor, ExecutionResult
from cap_general.core.models import PolicyModel, StaticPolicyModel, CallablePolicyModel
from cap_general.core.robot import CapEnv, RobotBase


class SimpleApi(AgentBase):
    """A simple test API."""

    def add(self, a: int, b: int) -> int:
        """Add two numbers.

        Args:
            a: First number
            b: Second number

        Returns:
            Sum of a and b
        """
        return a + b

    def multiply(self, x: float, y: float) -> float:
        """Multiply two numbers."""
        return x * y


def test_cap_api_combined_doc():
    """Test that combined_doc extracts method signatures and docstrings."""
    api = SimpleApi()
    doc = api.combined_doc()

    assert "add" in doc
    assert "multiply" in doc
    assert "Add two numbers" in doc
    assert "Multiply two numbers" in doc
    assert "a: int" in doc or "a:int" in doc
    assert "b: int" in doc or "b:int" in doc


def test_code_executor_run_success():
    """Test successful code execution."""
    executor = CodeExecutor()
    result = executor.run("x = 10\ny = 20\nresult = x + y")

    assert result.success is True
    assert result.error is None
    assert executor.globals.get("result") == 30


def test_code_executor_run_failure():
    """Test failed code execution."""
    executor = CodeExecutor()
    result = executor.run("raise ValueError('test error')")

    assert result.success is False
    assert result.error is not None
    assert "ValueError" in result.error


def test_code_executor_persistent_globals():
    """Test that globals persist across executions."""
    executor = CodeExecutor()
    executor.run("counter = 0")
    executor.run("counter += 1")
    executor.run("counter += 2")

    assert executor.globals.get("counter") == 3


def test_code_executor_stdout_capture():
    """Test stdout capture."""
    executor = CodeExecutor()
    result = executor.run("print('hello world')")

    assert result.success is True
    assert "hello world" in result.stdout


def test_static_policy_model_generate():
    """Test static policy model returns fixed code."""
    fixed_code = "action = [1.0, 2.0, 3.0]"
    model = StaticPolicyModel(code=fixed_code)
    result = model.generate(prompt="test prompt")

    assert result.code == fixed_code
    assert result.model_name == "StaticPolicyModel"


def test_callable_policy_model_generate():
    """Test callable policy model uses a function to generate code."""

    def generator(prompt: str) -> str:
        return f"# Generated for: {prompt}\naction = [0.0, 0.0, 0.0]"

    model = CallablePolicyModel(generator_fn=generator)
    result = model.generate(prompt="lift cube")

    assert "Generated for: lift cube" in result.code
    assert "action = [0.0, 0.0, 0.0]" in result.code
    assert result.model_name == "CallablePolicyModel"


def test_policy_model_base_cannot_instantiate():
    """Test that base PolicyModel cannot be instantiated directly."""
    with pytest.raises(TypeError):
        PolicyModel()


def test_core_registries_include_common_components():
    """Test that common core implementations are registered by type."""
    assert AgentBase.get_registered_type("code_executor") is CodeExecutor
    assert PolicyModel.get_registered_type("static") is StaticPolicyModel
    assert PolicyModel.get_registered_type("callable") is CallablePolicyModel
    assert RobotBase.get_registered_type("cap_env") is CapEnv


def test_agent_register_decorator():
    """Test that agent subclasses can be registered by type."""

    @AgentBase.register()
    class RegisteredApi(AgentBase):
        name = "Registered API"

        @classmethod
        def agent_type(cls) -> str:
            return "registered_api"

    assert AgentBase.get_registered_type("registered_api") is RegisteredApi
