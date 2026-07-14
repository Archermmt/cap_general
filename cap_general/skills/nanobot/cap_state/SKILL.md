---
name: cap_state
description: Read this skill before state/observation requests for {available_names}; then call only documented {cap_id}-prefixed tools.
metadata: {"nanobot":{"emoji":"📍"}}
---

# Control State

Use for current state, view, image, or "what is the control doing" requests.

Available controls and aliases: `{available_names}`.

## Required Flow

1. Call `{cap_id}_get_obs` with `controls` set to selected control names or aliases. Omit `controls` only when querying all controls.
2. For each returned observation, display or inspect `main_image` with an image-capable tool. If `main_image` is missing, use the first useful path in `images`.
3. Convert relative image paths to absolute paths before display.

## Tool

```json
{"name": "{cap_id}_get_obs", "arguments": {"controls": ["{control_name}"]}}
```

## Rules

- Do not call tools before reading this document.
- Do not pass response keys like `alias(control)` back as selectors; use real names or aliases from `{available_names}`.
- Treat returned image paths as local artifacts for visual inspection.
