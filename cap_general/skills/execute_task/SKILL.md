---
name: {cap_id}_execute_task
description: Execute robot tasks with one or more CAP agents, recording only LLM-generated plans and result judgments while agent task results are traced automatically. Available names are {available_names}.
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

Use `media` with `mode="analyze"` for verification images; it displays the image automatically. Use `media` with `mode="display"` for final videos.

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

Wait for completion:

```json
{"name": "{cap_id}_monitor", "arguments": {"agents": ["{agent_name}"], "wait_ms": -1}}
```

Inspect the result's `ok`, `result`, `stdout`, `stderr`, `exec_cnt`, `trial_cnt`, and `obs.main_image`.

### 5. Analyze And Verify

When `obs.main_image` is present, call `media` in analyze mode to judge task outcome (this also displays the image automatically):

```json
{
  "name": "media",
  "arguments": {
    "media_type": "image",
    "mode": "analyze",
    "media_path": "<absolute path from obs.main_image>",
    "prompt": "The robot was attempting to: <subtask>. Did it succeed? Answer YES or NO and briefly explain what you see."
  }
}
```

Then record exactly one verification event using `tool: verify`:

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

Then go back to **Monitor → Analyze And Verify** using the new `exec_cnt` and `trial_cnt`. Do not manually record the retry result; Agent auto trace handles it.

Retry no more than `max_retry`. Stop if the result contains `error: "max_retry_exceeded"`.

### 7. Final Record

After all subtasks, call record once:

```json
{"name": "{cap_id}_record", "arguments": {"agents": ["{agent_name}"]}}
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

Then display each useful final video with `media`:

```json
{"name": "media", "arguments": {"media_type": "video", "mode": "display", "media_path": "<absolute path from main_video>", "prompt": "Full task execution video"}}
```

## Conceptual Pseudo-Code

```python
reset(agent)
doc = agent_doc(agent)
obs = get_obs(agent)

subtasks = decompose(doc, obs)
update_history(agent, {"role": "user", "tool": "plan", "request": subtasks})
message(f"Task: ...\n\nSubtasks:\n1. ...")

for subtask in subtasks:
    execute(agent, make_code(subtask))  # auto-traced by Agent
    status = monitor(agent, wait_ms=-1)
    judgment = media(media_type="image", mode="analyze", media_path=status.main_image, prompt=subtask)
    update_history(agent, {
        "role": "user", "tool": "verify", "response": judgment,
    })
    message(f"Verification result (Exec {exec_cnt} Trial {trial_cnt}): {subtask}\nResult: ...")

    while not judgment.success and status.trial_cnt <= doc.max_retry:
        retry(agent)  # auto-traced by Agent
        status = monitor(agent, wait_ms=-1)
        judgment = media(media_type="image", mode="analyze", media_path=status.main_image, prompt=subtask)
        update_history(agent, {
            "role": "user", "tool": "verify", "response": judgment,
        })
        message(f"Verification result (Exec {exec_cnt} Trial {trial_cnt}): {subtask}\nResult: ...")

record_result = record(agent)
message(f"Executed code:\n\n```python\n{record_result.code}\n```")
media(record_result.main_video)
```

## Important Rules

1. Always call `message` immediately after planning and after each verification — the user sees nothing otherwise.
2. Always call `media` with `media_type="image"` and `mode="analyze"` after monitor before deciding to retry or record verify.
3. Call `update_history` only for completed LLM planning (`tool: plan`) and verification (`tool: verify`).
4. Keep `role`, `tool`, and `request` or `response` at the top level of each history message.
5. Never duplicate auto-traced `execute`, `retry`, or `train` results.
6. Read `agent_doc` before generating execution code.
7. Use exact LIBERO task strings from `execute_rules`.
8. Use `monitor(..., wait_ms=-1)` for final execution results.
9. Call `record` once after all subtasks.
10. Do not modify YAML configuration files while executing this skill.
