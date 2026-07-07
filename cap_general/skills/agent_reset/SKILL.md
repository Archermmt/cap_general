---
name: {cap_id}_agent_reset
description: Reset one or more CAP agents with the batched reset tool, then fetch and display the post-reset observation image. Available names are {available_names}. Read this skill before execute, only then call {cap_id}-prefixed tools.
metadata: {"nanobot":{"emoji":"🔄"}}
---

# Agent Reset Skill

Call `{cap_id}_agent_doc` to inspect reset options, call `{cap_id}_reset` to reset selected agents, then call `{cap_id}_get_obs` to fetch the post-reset observation and display the primary image using an image-capable tool.
The placeholder `{agent_name}` can be replaced by any available agent name or alias from this scene: `{available_names}`.

## Features

- Inspect reset options and supported parameters from `agent_doc` before resetting
- Reset one or more selected agents or environments
- Fetch the post-reset observation with `{cap_id}_get_obs` after reset
- Display the returned `main_image` using an image-capable tool

## Usage Examples

### Inspect Reset Options

Call `agent_doc` first to learn the supported reset parameters for the agent:

```json
{"name": "{cap_id}_agent_doc", "arguments": {"agents": ["{agent_name}"]}}
```

### Reset The Agent

Reset with options from `agent_doc`:

```json
{"name": "{cap_id}_reset", "arguments": {"agent_options": {"{agent_name}": {}}}}
```

### Fetch Post-Reset Observation

After reset, call `{cap_id}_get_obs` to get the current observation:

```json
{"name": "{cap_id}_get_obs", "arguments": {"agents": ["{agent_name}"]}}
```

Example response shape:

```json
{
  "<alias>(<agent_name>)": {
    "images": {
      "agentview_image": "outputs/step_0/trial_0/agentview_image_0.png"
    },
    "main_image": "outputs/step_0/trial_0/agentview_image_0.png"
  }
}
```

After `get_obs`, **use an image-capable tool to display the `main_image`** (convert to absolute path if needed). This is required — always show the post-reset visual state to the user.

If `main_image` is missing but `images` contains camera images, use the first useful camera image from `images`.

## Required Workflow

1. Call `{cap_id}_agent_doc` to inspect the supported reset options for the target agents.
2. Call `{cap_id}_reset` with an `agent_options` mapping constructed from `agent_doc`.
3. Call `{cap_id}_get_obs` with the same agent names to fetch the post-reset observation.
4. If `main_image` is present, convert it to an absolute path if needed.
5. **Use an image-capable tool to display the `main_image`** so the user can see the post-reset state.
6. If `main_image` is missing, use the first available path from the returned `images` dict.

## Important Rules

1. **ALWAYS call `{cap_id}_agent_doc` first** to learn the supported reset options before calling reset.
2. **ALWAYS use `{cap_id}_reset`** to reset agents. Do not call environment internals directly.
3. **ALWAYS call `{cap_id}_get_obs` after reset** to fetch the post-reset observation — reset does not return obs.
4. **ALWAYS display the primary image** using an image-capable tool — the user must be able to see the post-reset visual state.
5. **Use absolute paths when displaying images**. If `get_obs` returns a relative image path, convert it to an absolute local path first.
