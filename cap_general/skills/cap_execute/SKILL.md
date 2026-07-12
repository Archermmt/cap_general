---
name: cap_execute
description: Execute robot tasks with one or more CAP controls. Use this skill only for control task execution and verification, not for training or policy pipeline jobs. Available names are {available_names}. Read this skill before execute, only then call {cap_id}-prefixed tools.
metadata: {"nanobot":{"emoji":"🤖"}}
---

# Control Execute Skill

Use this skill only when the user wants a control to perform a task through `execute` / `retry` / `record`.

Decompose robot tasks into verifiable subtasks, execute independent controls concurrently, verify outcomes, retry failures, and retain concise reasoning history.

Do not use this skill for training, evaluation, compilation, or any policy pipeline job. Those belong to `{cap_id}_run_pipe` via the control pipeline skill.

Available control names and aliases: `{available_names}`.

## CAP Tools

- `{cap_id}_control_doc`
- `{cap_id}_execute`
- `{cap_id}_monitor`
- `{cap_id}_retry`
- `{cap_id}_get_obs`
- `{cap_id}_update_history`
- `{cap_id}_record`

Selector tools use a `controls` list. Tools with per-control values use mappings keyed by control name. Response keys use each control's scene-visible `mark`, typically `alias(control_name)` when an alias exists. Never pass a formatted response key back as a control selector.

## CRITICAL: Notify The User At Key Checkpoints

This skill executes many steps before the final response. You MUST use the messaging tool to notify the user at these checkpoints:

1. **After planning** — immediately after `update_history` with the plan, send the main task and numbered subtask list.
2. **After each verification** — immediately after `update_history` with the verify result, send the SUCCESS/FAIL result and brief notes.
3. **After final record** — send the returned `code` in a fenced Python block.

## Workflow

### 1. Inspect

Call the tools directly without history updates:

```json
{"name": "{cap_id}_control_doc", "arguments": {"controls": ["{control_name}"]}}
```

```json
{"name": "{cap_id}_get_obs", "arguments": {"controls": ["{control_name}"]}}
```

Treat `function_doc`, `execute_rules`, `policy_doc`, and `max_retry` from `control_doc` as authoritative. Always follow the constraints and task descriptions in `execute_rules` exactly as specified.

Read `result["scene"]["async_task"]` from the `control_doc` response. Store it as `async_task`.

### 2. Plan

Build ordered atomic subtasks. Then call `update_history` with `tool: plan`:

```json
{
  "control_messages": {
    "{control_name}": {
      "role": "user",
      "tool": "plan",
      "request": {
        "main_task": "<original task>",
        "sub_tasks": ["<subtask 1>", "<subtask 2>"]
      }
    }
  }
}
```

Immediately after, use a messaging tool to notify the user: `Task: <original task>\n\nSubtasks:\n1. <subtask 1>\n2. <subtask 2>`

### 3. Execute

Generate code using only functions from `function_doc`:

```json
{"name": "{cap_id}_execute", "arguments": {"control_codes": {"{control_name}": "<python code>"}}}
```

Independent controls may be sent in one `control_codes` mapping. Dependent subtasks must run in order.

### 4. Monitor

If `async_task=true`, wait for completion:

```json
{"name": "{cap_id}_monitor", "arguments": {"controls": ["{control_name}"], "wait_ms": -1}}
```

If `async_task=false`, the result from `{cap_id}_execute` or `{cap_id}_retry` is already final — skip this step.

Inspect the result's `ok`, `result`, `stdout`, `stderr`, `exec_cnt`, `trial_cnt`, and `obs.main_image`.

### 5. Analyze And Verify

When `obs.main_image` is present, use an image tool to analyze task outcome. Pass the absolute path from `obs.main_image` and a prompt asking whether the subtask succeeded.

Then record exactly one verification event using `update_history` with `tool: verify`:

```json
{
  "control_messages": {
    "{control_name}": {
      "role": "user",
      "tool": "verify",
      "response": {
        "subtask": "<subtask>",
        "success": true,
        "image": "<obs.main_image path>",
        "notes": "<brief visual or execution assessment>"
      }
    }
  }
}
```

Immediately after, use a messaging tool to notify the user: `Verification result (Exec <exec_cnt> Trial <trial_cnt>): <subtask>\nResult: <SUCCESS or FAIL>\nNotes: <brief assessment>`

### 6. Retry

If the image analysis indicates failure, call retry directly:

```json
{"name": "{cap_id}_retry", "arguments": {"controls": ["{control_name}"]}}
```

If `async_task=true`, go back to **Monitor → Analyze And Verify** using the new `exec_cnt` and `trial_cnt`. If `async_task=false`, the retry result is already final — go directly to **Analyze And Verify**. Do not manually record the retry result; Control auto trace handles it.

Retry no more than `max_retry`. Stop if the result contains `error: "max_retry_exceeded"`.

### 7. Final Record

After all subtasks, call record once:

```json
{"name": "{cap_id}_record", "arguments": {"controls": ["{control_name}"], "clean_frames": true}}
```

Use a messaging tool to send the executed code to the user in a fenced Python block.

Then use a video tool to display `main_video` path in the record result.

## Conceptual Pseudo-Code

```python
doc = control_doc(control)
async_task = doc["scene"]["async_task"]
obs = get_obs(control)

subtasks = decompose(doc, obs)
update_history(control, {"role": "user", "tool": "plan", "request": subtasks})
message(f"Task: ...\n\nSubtasks:\n1. ...")

for subtask in subtasks:
    execute(control, make_code(subtask))  # auto-traced by Agent
    if async_task:
        status = monitor(control, wait_ms=-1)
    judgment = analyze_image(status.main_image, prompt=subtask)  # use image-capable tool
    update_history(control, {
        "role": "user", "tool": "verify", "response": judgment,
    })
    message(f"Verification result (Exec {exec_cnt} Trial {trial_cnt}): {subtask}\nResult: ...")

    while not judgment.success and status.trial_cnt <= doc.max_retry:
        retry(control)  # auto-traced by Agent
        if async_task:
            status = monitor(control, wait_ms=-1)
        judgment = analyze_image(status.main_image, prompt=subtask)  # use image-capable tool
        update_history(control, {
            "role": "user", "tool": "verify", "response": judgment,
        })
        message(f"Verification result (Exec {exec_cnt} Trial {trial_cnt}): {subtask}\nResult: ...")

record_result = record(control, clean_frames=True)
message(f"Executed code:\n\n```python\n{record_result.code}\n```")
display_video(record_result.main_video)  # use image-capable tool in display mode
```

## Important Rules

1. Always use the messaging tool immediately after planning and after each verification — the user sees nothing otherwise.
2. Call `update_history` only for completed LLM planning (`tool: plan`) and verification (`tool: verify`).
3. Keep `role`, `tool`, and `request` or `response` at the top level of each history message.
4. Never duplicate auto-traced `execute`, `retry`, or `train` results.
5. Read `control_doc` before generating execution code.
6. Call `record` once after all subtasks.
7. Do not modify YAML configuration files while executing this skill.
8. Do not use this skill to train, evaluate, or compile policies.
