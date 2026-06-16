---
name: {agent_id}_skills
description: All cap_general skills are bound to {agent_name}; before calling any {agent_name} tool, read the relevant skill instructions first. Questions like "what is {agent_name} doing", "show me {agent_name}", or "current state/status/observation" should use get_state.
metadata: {"nanobot":{"emoji":"🧭"}}
---

# cap_general Skills

This directory contains skills for working with `cap_general` CAP agents. These skills are wrappers around agent-bound MCP tools, where the concrete tool name is parameterized by `{agent_name}`.

## Available Skills

- `get_state`: Read the current state with `{agent_id}_get_obs`, then display the returned `main_image` with the `media` image display tool. Use this for status/observation queries such as "what is {agent_name} doing?", "show me {agent_name}", "let me see {agent_name}", "current state", "current view", or "what does {agent_name} see?"
- `reset_agent`: Reset the agent with `{agent_id}_reset`, then fetch and display the post-reset `main_image`.
- `execute_task`: Decompose a robot manipulation task, execute subtasks with `{agent_id}_execute`, retry with `{agent_id}_retry`, update the plan, and record the final run.

## Mandatory Rules

1. **All skills are bound to `agent_name`.** Do not use any skill in this directory unless the user request clearly involves a concrete `agent_name` or the active context makes the target agent unambiguous.
2. **Read the relevant skill before agent-bound calls.** For any operation involving `{agent_name}`, read the matching skill first and follow its workflow. Do not directly call `{agent_name}` tools without reading the skill.
3. **Use the skill-specific tool sequence.** For example, use `get_state` before calling `{agent_id}_get_obs`, `reset_agent` before calling `{agent_id}_reset`, and `execute_task` before calling `{agent_id}_execute` or `{agent_id}_retry`.
4. **Respect exact tool names.** Replace `{agent_name}` with the active agent name and call only the MCP-registered tool names documented by the relevant skill.

## Routing

- User asks what `{agent_name}` is doing, asks to see `{agent_name}`, asks for current status/state/view/observation, or asks "让我看看{agent_name}在干什么": read `get_state/SKILL.md`.
- Need the current observation or image: read `get_state/SKILL.md`.
- Need to reset the environment or agent: read `reset_agent/SKILL.md`.
- Need to complete a manipulation task through code execution: read `execute_task/SKILL.md`.
