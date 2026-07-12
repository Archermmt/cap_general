---
name: {cap_id}_skills
description: Before interacting with any control in {available_names}, read this skill and the relevant child skill; only then call {cap_id}-prefixed tools.
metadata: {"nanobot":{"emoji":"🧭"}}
---

# cap_general Skills

This directory contains skills for working with `cap_general` CAP controls through scene-level MCP tools.
The tools support batches: selector-only methods use a `controls` list, while methods with per-control arguments use mappings keyed by control name. Available names and aliases are `{available_names}`.

## Available Skills

- `cap_state`: Read the current state with `{cap_id}_get_obs`, then display the returned `main_image` with the `media` image display tool. Use this for status/observation queries such as "what is {control_name} doing?", "show me {control_name}", "let me see {control_name}", "current state", "current view", or "what does {control_name} see?"
- `cap_reset`: Reset the control with `{cap_id}_reset`, then fetch and display the post-reset `main_image`.
- `cap_pipeline`: Run one or more pipeline jobs (train, eval, compile, or any combination) for one or more controls via `run_pipe`, poll async status every five seconds, and report every result to the user.
- `cap_execute`: Decompose a robot manipulation task, record LLM plans and result judgments, execute subtasks with automatic Control result tracing, retry failures, and record the final run.

## Mandatory Rules

1. **Select concrete controls.** Every selected name or alias must come from `{available_names}`. If the target is not explicit, use the active context or ask the user.
2. **Read the relevant skill before scene tool calls.** Follow its batch parameter and response format.
3. **Use the skill-specific tool sequence.** For example, use `cap_state` before calling `{cap_id}_get_obs`, `cap_reset` before calling `{cap_id}_reset`, `cap_pipeline` before calling `{cap_id}_run_pipe`, and `cap_execute` before calling `{cap_id}_execute` or `{cap_id}_retry`.
4. **Respect exact tool names and schemas.** Call only the MCP-registered tools documented by the relevant skill.
5. **Use batch routing.** Use `controls` for `control_doc`, `retry`, `monitor`, `get_obs`, and `record`; use the documented control-keyed mappings for `reset`, `execute`, and `update_history`.
6. **Do not use long-goal/long_task mode by default for robot tasks.** Unless the user explicitly says the robot task should run in the background, continue asynchronously, or be tracked as a sustained background objective, do not call `long_task` and do not enter long-goal mode for `{control_name}` robot operations. Use the normal skill workflow instead.

## Routing

- User asks what `{control_name}` is doing, asks to see `{control_name}`, asks for current status/state/view/observation, or asks "让我看看{control_name}在干什么": read `cap_state/SKILL.md`.
- Need the current observation or image: read `cap_state/SKILL.md`.
- Need to reset the environment or control: read `cap_reset/SKILL.md`.
- User asks a robot or control to train, evaluate, compile, or run any pipeline stage: read `cap_pipeline/SKILL.md`.
- Need to complete a manipulation task through code execution: read `cap_execute/SKILL.md`.
