---
name: {cap_id}_skills
description: Read this routing skill before using {available_names}; then read the relevant child skill before calling any {cap_id}-prefixed tool.
metadata: {"nanobot":{"emoji":"🧭"}}
---

# cap_general Skills

Use this routing file to choose the child skill before any scene tool call.

Available controls and aliases: `{available_names}`.

Batch rule: selector tools use `controls`; per-control tools use mappings keyed by control name or alias.

## Available Skills

- `cap_state`: Read before `{cap_id}_get_obs`; use for current state/view/observation.
- `cap_reset`: Read before reset; inspect `control_doc`, reset, then fetch and display observation.
- `cap_pipeline`: Read before `{cap_id}_run_pipe`; use only for train/eval/compile/policy jobs.
- `cap_execute`: Read before `{cap_id}_execute` or retry; use only for robot task execution.

## Mandatory Rules

1. Do not call `{cap_id}` tools until the relevant child skill has been read.
2. Select concrete controls from `{available_names}`; ask only when genuinely ambiguous.
3. Use only exact tool names and schemas documented by the child skill.
4. Do not use pipeline tools for task execution, or execute tools for pipeline jobs.
5. Do not use long-goal/long_task mode unless the user explicitly asks for background tracking.

## Routing

- State/view/image/current status: read `cap_state/SKILL.md`.
- Reset: read `cap_reset/SKILL.md`.
- Train/eval/compile/policy pipeline: read `cap_pipeline/SKILL.md`.
- Robot task execution/retry/record: read `cap_execute/SKILL.md`.
