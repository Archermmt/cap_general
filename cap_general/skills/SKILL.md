---
name: {cap_id}_skills
description: Use cap_general scene tools with one or more agents from {available_names}. Read the relevant skill before getting state, resetting agents, or executing tasks.
metadata: {"nanobot":{"emoji":"🧭"}}
---

# cap_general Skills

This directory contains skills for working with `cap_general` CAP agents through scene-level MCP tools.
The tools support batches: selector-only methods use an `agents` list, while methods with per-agent arguments use mappings keyed by agent name. Available names and aliases are `{available_names}`.

## Available Skills

- `get_state`: Read the current state with `{cap_id}_get_obs`, then display the returned `main_image` with the `media` image display tool. Use this for status/observation queries such as "what is {agent_name} doing?", "show me {agent_name}", "let me see {agent_name}", "current state", "current view", or "what does {agent_name} see?"
- `reset_agent`: Reset the agent with `{cap_id}_reset`, then fetch and display the post-reset `main_image`.
- `execute_task`: Decompose a robot manipulation task, execute subtasks with `{cap_id}_execute`, wait with `{cap_id}_monitor`, retry with `{cap_id}_retry`, update the plan, and record the final run.

## Mandatory Rules

1. **Select concrete agents.** Every selected name or alias must come from `{available_names}`. If the target is not explicit, use the active context or ask the user.
2. **Read the relevant skill before scene tool calls.** Follow its batch parameter and response format.
3. **Use the skill-specific tool sequence.** For example, use `get_state` before calling `{cap_id}_get_obs`, `reset_agent` before calling `{cap_id}_reset`, and `execute_task` before calling `{cap_id}_execute` or `{cap_id}_retry`.
4. **Respect exact tool names and schemas.** Call only the MCP-registered tools documented by the relevant skill.
5. **Use batch routing.** Use `agents` for `agent_doc`, `retry`, `monitor`, `get_obs`, and `record`; use the documented agent-keyed mappings for `reset`, `execute`, and `update_plan`.
6. **Do not use long-goal/long_task mode by default for robot tasks.** Unless the user explicitly says the robot task should run in the background, continue asynchronously, or be tracked as a sustained background objective, do not call `long_task` and do not enter long-goal mode for `{agent_name}` robot operations. Use the normal skill workflow instead.

## Routing

- User asks what `{agent_name}` is doing, asks to see `{agent_name}`, asks for current status/state/view/observation, or asks "让我看看{agent_name}在干什么": read `get_state/SKILL.md`.
- Need the current observation or image: read `get_state/SKILL.md`.
- Need to reset the environment or agent: read `reset_agent/SKILL.md`.
- Need to complete a manipulation task through code execution: read `execute_task/SKILL.md`.
