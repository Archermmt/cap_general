---
name: cap_pipeline
description: Run one or more pipeline jobs (train, eval, compile, or any combination) to operate on a control's policy via run_pipe. Use this skill only for policy pipeline work, not for task execution. Available names are {available_names}. Read this skill before execute, only then call {cap_id}-prefixed tools.
metadata: {"nanobot":{"emoji":"⚙️"}}
---

# Control Pipeline Skill

Use this skill only when the user wants to train, evaluate, or compile a control policy.

Run ordered pipeline jobs for selected controls with `{cap_id}_run_pipe`. A single call can chain multiple stages — for example train then eval — executed in sequence. Poll async status with `{cap_id}_monitor` when needed.

Do not use this skill for control task execution, subtasks, verification, or `execute` / `retry` / `record`. Those belong to the control execute skill.

Available control names and aliases: `{available_names}`.

## CAP Tools

- `{cap_id}_control_doc`
- `{cap_id}_run_pipe`
- `{cap_id}_monitor`

Selector tools use a `controls` list. `run_pipe` uses an `control_options` mapping keyed by control name or alias. Response keys use each control's scene-visible mark, usually `alias(control_name)` when an alias exists.

## Pipeline Request

Each control's entry in `control_options` requires:

- `policy_name`: Policy key to operate on. Read from `control_doc` → `policy_doc`.
- `job_options`: Ordered list of job entries. Each entry has:
  - `job`: Job group name (e.g. `"train"`, `"eval"`, `"compile"`).
  - `options`: Job-specific runtime options (see `control_doc` → `run_pipe` function doc for per-job options).

Jobs execute in the order listed. Multiple jobs can be chained in one call.

Example — train then eval:

```json
{
  "control_options": {
    "{control_name}": {
      "policy_name": "runner",
      "job_options": [
        {"job": "train", "options": {"epoch": 100}},
        {"job": "eval",  "options": {"episode_num": 10}}
      ]
    }
  }
}
```

Do not send `method`, `stage`, or other unsupported top-level fields to `{cap_id}_run_pipe`.

## Required Workflow

1. Identify the requested control and parse the job request from the user.
2. Call `{cap_id}_control_doc` to inspect `policy_doc` and the `run_pipe` function doc for available jobs and per-job options. Read `result["scene"]["async_task"]` and store it as `async_task`.
3. Select the policy and construct `job_options` from `control_doc`. Ask the user only when multiple valid policies remain genuinely ambiguous.
4. Call `{cap_id}_run_pipe` once with all independent target controls in one `control_options` mapping.
5. Use a messaging tool to notify the user that the pipeline has started, including the initial `run_pipe` response.
6. If `async_task=true`: repeatedly call `{cap_id}_monitor` with `wait_ms=5000` for controls that are still running. After every monitor call, use a messaging tool to send the user the complete status and result. Stop polling a control when its status has `running=false`.
   If `async_task=false`: the `run_pipe` response already contains the final result — skip monitor polling and use a messaging tool to send the result to the user.
7. Use a messaging tool to send the user a final message containing the complete `result`, including any error details when a job fails.
8. Do not use `{cap_id}_execute`, `{cap_id}_retry`, or `{cap_id}_record` in this skill.

## Monitor Loop

Only applicable when `async_task=true`. Use a five-second wait on every poll:

```json
{
  "name": "{cap_id}_monitor",
  "arguments": {
    "controls": ["{control_name}"],
    "wait_ms": 5000
  }
}
```

Interpret each keyed status independently:

- `running=true`: Use a messaging tool to send the user the current status and result, then poll again after another five-second monitor wait.
- `running=false`: Inspect `result`. A successful pipeline result contains `ok=true` and a per-job `report`. Use a messaging tool to send the user the complete result, then stop polling that control.
- A failed job reports `ok=false` and error details such as `error` inside `result`; do not expect a separate top-level `error` field.

Do not use `wait_ms=-1`; the user must receive progress at five-second intervals. Do not busy-poll with `wait_ms=0`.

## Example Sequence

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
          {"job": "eval",  "options": {"episode_num": 10}}
        ]
      }
    }
  }
}
```

Use a messaging tool to send the user the initial response, then repeat:

```json
{"name": "{cap_id}_monitor", "arguments": {"controls": ["{control_name}"], "wait_ms": 5000}}
```

Use a messaging tool to send the user each monitor response before the next poll.

## Important Rules

1. Trigger this skill whenever the user asks a control to train, evaluate, compile, or run any pipeline stage.
2. Read `{cap_id}_control_doc` before constructing the request; check `result["scene"]["async_task"]` and the `run_pipe` function doc for valid jobs and options.
3. Call `{cap_id}_run_pipe` only once per requested pipeline run.
4. If `async_task=true`: poll only with `{cap_id}_monitor(wait_ms=5000)` while the pipeline is running and use the messaging tool to send every monitor result to the user.
   If `async_task=false`: skip monitor entirely; the final result is already in the `run_pipe` response.
5. Multiple jobs can be chained in one call — do not split a train+eval request into two separate `run_pipe` calls.
6. Do not call `{cap_id}_execute` to run pipeline jobs.
7. Do not modify YAML configuration files during a pipeline run.
