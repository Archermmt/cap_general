---
name: {agent_name}_get_state
description: Get the current state and observation of the {agent_name} agent by calling its MCP-registered get_obs tool. Use this before planning, after actions, or whenever the current environment state is needed.
metadata: {"nanobot":{"emoji":"📍"}}
---

# Get State Skill

Get the current agent state by calling the `{agent_name}.get_obs` MCP tool. This returns the latest observation saved by the agent environment, including image paths when image observations are available and normalized state fields when the environment provides them.

## Features

- Fetch the current observation from the active `{agent_name}` agent
- Return saved observation image paths for visual state inspection
- Return normalized non-image state fields when supported by the environment
- Use the same MCP tool name in all contexts: `{agent_name}.get_obs`

## Tool Parameters

The get state skill uses the `{agent_name}.get_obs` tool.

### Required Parameters

None.

### Optional Parameters

None.

## Usage Examples

### Get Current State

Get the latest observation from the agent:

```json
{"name": "{agent_name}.get_obs", "arguments": {}}
```

### Inspect Returned Images

If the response contains an `images` object or `main_image`, use those paths for visual inspection:

```json
{"name": "{agent_name}.get_obs", "arguments": {}}
```

Example response shape:

```json
{
  "images": {
    "agentview_image": "outputs/libero/step_1/trial_1/agentview_image_0.png"
  },
  "main_image": "outputs/libero/step_1/trial_1/agentview_image_0.png",
  "robot0_eef_pos": [0.0, 0.0, 0.0]
}
```

## Important Rules

1. **ALWAYS use `{agent_name}.get_obs`** to get agent state. Do not call environment internals directly.
2. **Use MCP-registered method names only**. The tool name must be exactly `{agent_name}.get_obs`.
3. **Do not pass arguments**. `get_obs` does not take parameters.
4. **Treat returned image paths as local artifacts**. Use them for visual inspection when planning the next action.
5. **Call this after reset or execute when state matters** so the next decision uses the latest observation.
