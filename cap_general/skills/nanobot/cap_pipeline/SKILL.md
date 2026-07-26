---
name: cap_pipeline
description: Read this skill before policy pipeline work for {available_names}; then call only documented {cap_id}-prefixed tools. Pipeline runs synchronously; do not call monitor.
metadata: {"nanobot":{"emoji":"⚙️"}}
---

# Control Pipeline

Use only for train, eval, compile, or other policy pipeline jobs. Do not use for robot task execution.

Available controls and aliases: `{available_names}`.

## Required Flow

1. Call `{cap_id}_control_doc` for selected controls.
2. Read `policy_doc` and the `run_pipe` docs to choose `policy_name`, valid jobs, and job options.
3. Call `{cap_id}_run_pipe` once with `control_options` keyed by control name or alias. Chain requested jobs in `job_options` order.
4. Wait for `{cap_id}_run_pipe` to return. The response is final.
5. Message the user the complete final `result`, including error details and reports.

## Tools

```json
{"name": "{cap_id}_control_doc", "arguments": {"controls": ["{control_name}"]}}
```

```json
{
  "name": "{cap_id}_run_pipe",
  "arguments": {
    "control_options": {
      "{control_name}": {
        "policy_name": "runner",
        "job_options": [
          {"job": "train", "options": {"epoch": 100}},
          {"job": "eval", "options": {"episode_num": 10}}
        ]
      }
    }
  }
}
```

## Rules

- Do not call tools before reading this document and `control_doc`.
- Do not call `{cap_id}_monitor`; pipeline results are returned directly by `{cap_id}_run_pipe`.
- Do not call `{cap_id}_execute` for pipeline jobs.
- Do not split train+eval when the user requested one ordered pipeline; put both jobs in one `job_options` list.
- Do not modify YAML configuration files during a pipeline run.
