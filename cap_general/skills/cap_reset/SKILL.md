---
name: cap_reset
description: Read this skill before reset requests for {available_names}; then call only documented {cap_id}-prefixed tools.
metadata: {"nanobot":{"emoji":"🔄"}}
---

# Control Reset

Use to reset one or more controls/environments.

Available controls and aliases: `{available_names}`.

## Required Flow

1. Call `{cap_id}_control_doc` for the selected controls and read supported reset options.
2. Call `{cap_id}_reset` with `control_options` keyed by control name or alias.
3. Call `{cap_id}_get_obs` for the same controls.
4. Display or inspect each returned `main_image`; if missing, use the first useful path in `images`.

## Tools

```json
{"name": "{cap_id}_control_doc", "arguments": {"controls": ["{control_name}"]}}
```

```json
{"name": "{cap_id}_reset", "arguments": {"control_options": {"{control_name}": {}}}}
```

```json
{"name": "{cap_id}_get_obs", "arguments": {"controls": ["{control_name}"]}}
```

## Rules

- Do not call reset before reading this document and `control_doc`.
- Reset does not return observations; always call `get_obs` after reset.
- Convert relative image paths to absolute paths before display.
