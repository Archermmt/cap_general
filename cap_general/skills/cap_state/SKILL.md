---
name: cap_state
description: Get and display current observations for one or more controls from {available_names} with the batched get_obs tool. Read this skill before execute, only then call {cap_id}-prefixed tools.
metadata: {"nanobot":{"emoji":"📍"}}
---

# Control State Skill

Get current control states by calling `{cap_id}_get_obs` with a `controls` list. Each result key uses `alias(control_name)` when an alias exists, otherwise `agent_name`; each value contains that control's latest observation.
Selected names and aliases must come from `{available_names}`.

Use this skill for present-state questions, including:

- "What is `{control_name}` doing?"
- "Show me `{control_name}`."
- "Let me see what `{control_name}` is doing."
- "Current state/status/view/observation."
- "让我看看`{control_name}`在干什么。"

## Features

- Fetch current observations for one or more selected controls
- Return saved observation image paths for visual state inspection
- Display or analyze the primary returned image using an image-capable tool
- Return normalized non-image state fields when supported by the environment
- Use the same MCP tool name in all contexts: `{cap_id}_get_obs`

## Tool Parameters

The get state skill uses the `{cap_id}_get_obs` tool.

### Optional Parameters

- `controls` (array of strings): Target control names or aliases. Omit it to query all controls.

## Usage Examples

### Get Current State

Get the latest observation from the control:

```json
{"name": "{cap_id}_get_obs", "arguments": {"controls": ["{control_name}"]}}
```

### Inspect Returned Images

If the response contains an `images` object or `main_image`, use those paths for visual inspection. After calling `get_obs`, use an image-capable tool to analyze the primary image.

Example response shape:

```json
{
  "<alias>(<agent_name>)": {
    "images": {
      "controlview_image": "outputs/libero/step_1/trial_1/controlview_image_0.png"
    },
    "main_image": "outputs/libero/step_1/trial_1/controlview_image_0.png"
  }
}
```

Pass the `main_image` path (converted to absolute if needed) to an image-capable tool to display or analyze it.

If `main_image` is missing but `images` contains camera images, use the first useful camera image from `images`.

## Important Rules

1. **ALWAYS use `{cap_id}_get_obs`** to get control state. Do not call environment internals directly.
2. **Use MCP-registered method names only**. The tool name must be exactly `{cap_id}_get_obs`.
3. **Use the `controls` list** to select one or more controls, or omit it to query all controls. Do not pass a formatted `alias(control_name)` response key back as a control selector.
4. **Display or analyze each returned primary image**. For each keyed observation with `main_image`, use an image-capable tool to display or analyze that image path.
5. **Use absolute paths**. If `get_obs` returns a relative image path, convert it to an absolute local path before passing it to an image tool.
6. **Fallback to `images` when needed**. If `main_image` is missing, use the first available path from the returned `images` dict.
7. **Treat returned image paths as local artifacts**. Use them for visual inspection when planning the next action.
8. **Call this after reset or execute when state matters** so the next decision uses the latest observation.
