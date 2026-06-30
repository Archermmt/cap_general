---
name: {cap_id}_reset_agent
description: Reset one or more CAP agents with the batched reset tool, then display each returned obs.main_image. Available names are {available_names}.
metadata: {"nanobot":{"emoji":"🔄"}}
---

# Reset Agent Skill

Reset selected agents by calling `{cap_id}_reset` with an `agent_options` mapping, then use each keyed result's `obs` object to display the primary image with the `media` tool.
The placeholder `{agent_name}` can be replaced by any available agent name or alias from this scene: `{available_names}`.

## Features

- Reset one or more selected agents or environments
- Support reset options such as `reset_level` and `episode_idx`
- Read the post-reset observation from the `{cap_id}_reset` return value
- Display the returned `main_image` with `media_type="image"` and `mode="display"`

## Tool Parameters

The reset skill uses the `{cap_id}_reset` tool.

### Required Parameters

- `agent_options` (object): A mapping from agent name or alias to its reset options. Names must come from `{available_names}`.

### Optional Parameters

- Each mapping value is an options object passed to that agent's reset method.
  - `reset_level`: Reset scope. `0` resets robot pose, `1` resets environment, `2` resets full agent state. Defaults to full agent reset.
  - `episode_idx`: LIBERO initial state index. Defaults to `0` when used by LIBERO.

## Usage Examples

### Reset The Agent

Reset the agent with default options:

```json
{"name": "{cap_id}_reset", "arguments": {"agent_options": {"{agent_name}": {}}}}
```

For LIBERO-style episodes, reset to the first initial state:

```json
{"name": "{cap_id}_reset", "arguments": {"agent_options": {"{agent_name}": {"episode_idx": 0}}}}
```

### Use Returned Post-Reset State

Each reset response key uses `alias(agent_name)` when an alias exists, otherwise `agent_name`. Each value includes `obs`; use it directly instead of calling `{cap_id}_get_obs` again.

Example reset response shape:

```json
{
  "<alias>(<agent_name>)": {
    "ok": true,
    "obs": {
      "images": {
        "agentview_image": "outputs/libero/step_0/trial_0/agentview_image_0.png"
      },
      "main_image": "outputs/libero/step_0/trial_0/agentview_image_0.png"
    }
  }
}
```

Display the returned `obs.main_image`:

```json
{"name": "media", "arguments": {"media_type": "image", "mode": "display", "media_path": "<absolute path from obs.main_image>", "prompt": "Post-reset agent observation"}}
```

If `obs.main_image` is missing but `obs.images` contains camera images, display the first useful camera image from `obs.images`.

## Required Workflow

1. Call `{cap_id}_reset` with an `agent_options` mapping.
2. Read each agent's `obs` from its keyed reset response.
3. If `obs.main_image` is present, convert it to an absolute path if needed.
4. Call `media` with `media_type="image"`, `mode="display"`, and the `obs.main_image` path.
5. If `obs.main_image` is missing, display the first available path from `obs.images`.

## Important Rules

1. **ALWAYS use `{cap_id}_reset`** to reset agents. Do not call environment internals directly.
2. **Use the `obs` returned by reset**. Do not call `{cap_id}_get_obs` just to fetch the post-reset image.
3. **Display the returned primary image** using the `media` tool with `media_type="image"` and `mode="display"`.
4. **Use absolute paths for `media` display**. If reset returns a relative image path, convert it to an absolute local path before calling `media`.
5. **Do not pass unsupported reset options**. If unsure, use an empty options object or inspect `{cap_id}_agent_doc` with `agents=["{agent_name}"]` for reset rules.
