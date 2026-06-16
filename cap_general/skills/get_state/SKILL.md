---
name: {agent_id}_get_state
description: Get the current state and observation of the {agent_name} agent by calling its MCP-registered get_obs tool. Use this before planning, after actions, or whenever the current environment state is needed.
metadata: {"nanobot":{"emoji":"📍"}}
---

# Get State Skill

Get the current agent state by calling the `{agent_id}_get_obs` MCP tool. This returns the latest observation saved by the agent environment, including image paths when image observations are available and normalized state fields when the environment provides them.

## Features

- Fetch the current observation from the active `{agent_name}` agent
- Return saved observation image paths for visual state inspection
- Display the primary returned image with the `media` tool when available
- Return normalized non-image state fields when supported by the environment
- Use the same MCP tool name in all contexts: `{agent_id}_get_obs`

## Tool Parameters

The get state skill uses the `{agent_id}_get_obs` tool.

### Required Parameters

None.

### Optional Parameters

None.

## Usage Examples

### Get Current State

Get the latest observation from the agent:

```json
{"name": "{agent_id}_get_obs", "arguments": {}}
```

### Inspect Returned Images

If the response contains an `images` object or `main_image`, use those paths for visual inspection. After calling `get_obs`, display the primary image with the `media` tool using `media_type="image"` and `mode="display"`.

```json
{"name": "{agent_id}_get_obs", "arguments": {}}
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

Display the returned `main_image`:

```json
{"name": "media", "arguments": {"media_type": "image", "mode": "display", "media_path": "<absolute path from main_image>", "prompt": "Current agent observation"}}
```

If `main_image` is missing but `images` contains camera images, display the first useful camera image from `images`.

## Important Rules

1. **ALWAYS use `{agent_id}_get_obs`** to get agent state. Do not call environment internals directly.
2. **Use MCP-registered method names only**. The tool name must be exactly `{agent_id}_get_obs`.
3. **Do not pass arguments**. `get_obs` does not take parameters.
4. **Display the returned primary image**. If `main_image` is present, call `media` with `media_type="image"`, `mode="display"`, and that image path.
5. **Use absolute paths for `media` display**. If `get_obs` returns a relative image path, convert it to an absolute local path before calling `media`.
6. **Fallback to `images` when needed**. If `main_image` is missing, display the first available path from the returned `images` dict.
7. **Treat returned image paths as local artifacts**. Use them for visual inspection when planning the next action.
8. **Call this after reset or execute when state matters** so the next decision uses the latest observation.
