---
name: {cap_id}_execute_task
description: Execute robot tasks with one or more CAP agents, recording only LLM-generated plans and result judgments while agent task results are traced automatically. Available names are {available_names}. Read this skill before execute, only then call {cap_id}-prefixed tools.
metadata: {"nanobot":{"emoji":"🤖"}}
---

# Execute Task Skill

Decompose robot tasks into verifiable subtasks, execute independent agents concurrently, verify outcomes, retry failures, and retain concise reasoning history.

Available agent names and aliases: `{available_names}`.

## CAP Tools

- `{cap_id}_reset`
- `{cap_id}_agent_doc`
- `{cap_id}_execute`
- `{cap_id}_monitor`
- `{cap_id}_retry`
- `{cap_id}_get_obs`
- `{cap_id}_update_history`
- `{cap_id}_record`

Selector tools use an `agents` list. Tools with per-agent values use mappings keyed by agent name. Response keys use each agent's scene-visible `mark`, typically `alias(agent_name)` when an alias exists. Never pass a formatted response key back as an agent selector.

## CRITICAL: Use `message` For Mid-Task Notifications

This skill executes many steps before the final response. The user sees nothing during that time unless you proactively send updates. You MUST call `message` at these checkpoints:

1. **After planning** — immediately after `update_history` with the plan, send the main task and numbered subtask list.
2. **After each verification** — immediately after `update_history` with the verify result, send the SUCCESS/FAIL result and brief notes.
3. **After final record** — send the returned `code` in a fenced Python block.

```json
{
  "name": "message",
  "arguments": {
    "content": "<text to show the user>"
  }
}
```

## Workflow

### 1. Reset And Inspect

Call the tools directly without history updates:

```json
{"name": "{cap_id}_reset", "arguments": {"agent_options": {"{agent_name}": {}}}}
```

```json
{"name": "{cap_id}_agent_doc", "arguments": {"agents": ["{agent_name}"]}}
```

```json
{"name": "{cap_id}_get_obs", "arguments": {"agents": ["{agent_name}"]}}
```

Treat `function_doc`, `execute_rules`, `policy_doc`, and `max_retry` from `agent_doc` as authoritative. For LIBERO, use task descriptions from `execute_rules` verbatim.

Read `result["scene"]["async_task"]` from the `agent_doc` response. Store it as `async_task`. When `async_task=false`, calls to `{cap_id}_execute`, `{cap_id}_retry`, and `{cap_id}_run_pipe` return the final result immediately — skip `{cap_id}_monitor` in that case. When `async_task=true`, always call `{cap_id}_monitor` after each of those calls to wait for completion.

### 2. Plan

Build ordered atomic subtasks. Then call `update_history` with `tool: plan`:

```json
{
  "agent_messages": {
    "{agent_name}": {
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

Immediately after, send the plan with `message`:

```json
{
  "name": "message",
  "arguments": {
    "content": "Task: <original task>\n\nSubtasks:\n1. <subtask 1>\n2. <subtask 2>"
  }
}
```

### 3. Execute

Generate code using only functions from `function_doc`:

```json
{"name": "{cap_id}_execute", "arguments": {"agent_codes": {"{agent_name}": "<python code>"}}}
```

Independent agents may be sent in one `agent_codes` mapping. Dependent subtasks must run in order.

### 4. Monitor

If `async_task=true`, wait for completion:

```json
{"name": "{cap_id}_monitor", "arguments": {"agents": ["{agent_name}"], "wait_ms": -1}}
```

If `async_task=false`, the result from `{cap_id}_execute` or `{cap_id}_retry` is already final — skip this step.

Inspect the result's `ok`, `result`, `stdout`, `stderr`, `exec_cnt`, `trial_cnt`, and `obs.main_image`.

### 5. Analyze And Verify

When `obs.main_image` is present, use an image tool to analyze task outcome. Pass the absolute path from `obs.main_image` and a prompt asking whether the subtask succeeded.

Then record exactly one verification event using `update_history` with `tool: plan`:

```json
{
  "agent_messages": {
    "{agent_name}": {
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

Immediately after, send the result with `message`:

```json
{
  "name": "message",
  "arguments": {
    "content": "Verification result (Exec <exec_cnt> Trial <trial_cnt>): <subtask>\nResult: <SUCCESS or FAIL>\nNotes: <brief assessment>"
  }
}
```

### 6. Retry

If the image analysis indicates failure, call retry directly:

```json
{"name": "{cap_id}_retry", "arguments": {"agents": ["{agent_name}"]}}
```

If `async_task=true`, go back to **Monitor → Analyze And Verify** using the new `exec_cnt` and `trial_cnt`. If `async_task=false`, the retry result is already final — go directly to **Analyze And Verify**. Do not manually record the retry result; Agent auto trace handles it.

Retry no more than `max_retry`. Stop if the result contains `error: "max_retry_exceeded"`.

### 7. Final Record

After all subtasks, call record once:

```json
{"name": "{cap_id}_record", "arguments": {"agents": ["{agent_name}"], "clean_frames": true}}
```

Send the executed code with `message`:

```json
{
  "name": "message",
  "arguments": {
    "content": "Executed code:\n\n```python\n<record.code>\n```"
  }
}
```

Then use an video tool to display `main_video` path in the record result.

## Conceptual Pseudo-Code

```python
reset(agent)
doc = agent_doc(agent)
async_task = doc["scene"]["async_task"]
obs = get_obs(agent)

subtasks = decompose(doc, obs)
update_history(agent, {"role": "user", "tool": "plan", "request": subtasks})
message(f"Task: ...\n\nSubtasks:\n1. ...")

for subtask in subtasks:
    execute(agent, make_code(subtask))  # auto-traced by Agent
    if async_task:
        status = monitor(agent, wait_ms=-1)
    judgment = analyze_image(status.main_image, prompt=subtask)  # use image-capable tool
    update_history(agent, {
        "role": "user", "tool": "verify", "response": judgment,
    })
    message(f"Verification result (Exec {exec_cnt} Trial {trial_cnt}): {subtask}\nResult: ...")

    while not judgment.success and status.trial_cnt <= doc.max_retry:
        retry(agent)  # auto-traced by Agent
        if async_task:
            status = monitor(agent, wait_ms=-1)
        judgment = analyze_image(status.main_image, prompt=subtask)  # use image-capable tool
        update_history(agent, {
            "role": "user", "tool": "verify", "response": judgment,
        })
        message(f"Verification result (Exec {exec_cnt} Trial {trial_cnt}): {subtask}\nResult: ...")

record_result = record(agent, clean_frames=True)
message(f"Executed code:\n\n```python\n{record_result.code}\n```")
display_video(record_result.main_video)  # use image-capable tool in display mode
```

## Important Rules

1. Always call `message` immediately after planning and after each verification — the user sees nothing otherwise.
2. Call `update_history` only for completed LLM planning (`tool: plan`) and verification (`tool: verify`).
3. Keep `role`, `tool`, and `request` or `response` at the top level of each history message.
4. Never duplicate auto-traced `execute`, `retry`, or `train` results.
5. Read `agent_doc` before generating execution code.
6. Call `record` once after all subtasks.
7. Do not modify YAML configuration files while executing this skill.
