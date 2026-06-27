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

## History Protocol

Call `{cap_id}_update_history` only after the LLM produces one of these reasoning results:

1. The decomposed subtask plan.
2. A success/failure judgment after inspecting an execution or retry result.

Do not record CAP tool requests. Do not manually record responses from `reset`, `agent_doc`, `execute`, `monitor`, `retry`, `get_obs`, `train`, or `record`.

When the MCP server starts, Scene enables Agent auto trace. Completed `execute`, `retry`, and `train` calls append their full results to history automatically. Never duplicate those results with `{cap_id}_update_history`.

History is append-only. Each manual update sends one LLM reasoning message:

```json
{
  "agent_messages": {
    "{agent_name}": {
      "role": "llm",
      "mark": "step_1_trail_1",
      "response": {
        "tool": "verification",
        "data": {"success": true, "notes": "<reasoning result>"}
      }
    }
  }
}
```

Use history marks with this exact spelling:

```text
step_{exec_cnt}_trail_{trial_cnt}
```

Use `step_0_trail_0` for planning. After execution, use the returned `exec_cnt` and `trial_cnt` as the authoritative verification mark. Use `trail`, not `trial`, in history marks.

## User Notifications

Use `message` immediately after planning and after every verification result. Use `media` to display verification images and final videos. These UI tools do not require history entries.

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

Build ordered atomic subtasks. Record the completed plan once under `step_0_trail_0`:

```json
{
  "agent_messages": {
    "{agent_name}": {
      "role": "llm",
      "mark": "step_0_trail_0",
      "response": {
        "tool": "planning",
        "data": {
          "main_task": "<original task>",
          "sub_tasks": ["<subtask 1>", "<subtask 2>"]
        }
      }
    }
  }
}
```

Send the plan with `message` before execution.

### 3. Execute

Generate code using only functions from `function_doc`:

```json
{"name": "{cap_id}_execute", "arguments": {"agent_codes": {"{agent_name}": "<python code>"}}}
```

Independent agents may be sent in one `agent_codes` mapping. Dependent subtasks must run in order. Do not call `update_history` around `execute`; Agent auto trace records its completed result.

### 4. Monitor And Verify

Wait for completion:

```json
{"name": "{cap_id}_monitor", "arguments": {"agents": ["{agent_name}"], "wait_ms": -1}}
```

Inspect each result's `ok`, `result`, `stdout`, `stderr`, `exec_cnt`, `trial_cnt`, and `obs.main_image`. Display `main_image` with `media` when present.

After making the LLM success/failure judgment, append exactly one verification event under the result's authoritative counters:

```json
{
  "agent_messages": {
    "{agent_name}": {
      "role": "llm",
      "mark": "step_1_trail_1",
      "response": {
        "tool": "verification",
        "data": {
          "subtask": "<subtask>",
          "success": true,
          "image": "<main_image>",
          "notes": "<LLM judgment and evidence>"
        }
      }
    }
  }
}
```

Send the same SUCCESS/FAIL judgment with `message`.

### 5. Retry

On failure, call retry directly:

```json
{"name": "{cap_id}_retry", "arguments": {"agents": ["{agent_name}"]}}
```

Monitor it, make a new LLM judgment, and record one new verification event using the returned `exec_cnt` and `trial_cnt`. Do not manually record the retry result; Agent auto trace handles it.

Retry no more than `max_retry`. Stop if the result contains `error: "max_retry_exceeded"`.

### 6. Final Record

After all subtasks, call record once without wrapping it in history updates:

```json
{"name": "{cap_id}_record", "arguments": {"agents": ["{agent_name}"]}}
```

The complete record contains concise LLM reasoning history, automatic Agent task results, execution code, `main_video`, and other videos. Send the executed code with `message`, then display each useful final video with `media`.

## Conceptual Pseudo-Code

```python
reset(agent)
doc = agent_doc(agent)
obs = get_obs(agent)

subtasks = plan_task(doc, obs)
update_history(agent, {
    "role": "llm",
    "mark": "step_0_trail_0",
    "response": {"tool": "planning", "data": subtasks},
})
message(subtasks)

for subtask in subtasks:
    execute(agent, make_code(subtask))  # Result is auto-traced by Agent.
    status = monitor(agent, wait_ms=-1)
    judgment = verify(status)
    mark = f"step_{status.exec_cnt}_trail_{status.trial_cnt}"
    update_history(agent, {
        "role": "llm",
        "mark": mark,
        "response": {"tool": "verification", "data": judgment},
    })
    message(judgment)

    while not judgment.success and status.trial_cnt <= doc.max_retry:
        retry(agent)  # Result is auto-traced by Agent.
        status = monitor(agent, wait_ms=-1)
        judgment = verify(status)
        mark = f"step_{status.exec_cnt}_trail_{status.trial_cnt}"
        update_history(agent, {
            "role": "llm",
            "mark": mark,
            "response": {"tool": "verification", "data": judgment},
        })
        message(judgment)

record(agent)
```

## Important Rules

1. Call `update_history` only for completed LLM planning and verification reasoning.
2. Never record CAP tool requests.
3. Never duplicate auto-traced `execute`, `retry`, or `train` results.
4. Use `step_{cnt}_trail_{cnt}` marks exactly, including the spelling `trail`.
5. Never overwrite or summarize earlier history; updates append.
6. Read `agent_doc` before generating execution code.
7. Use exact LIBERO task strings from `execute_rules`.
8. Use `monitor(..., wait_ms=-1)` for final execution results.
9. Call `record` once after all subtasks.
10. Do not modify YAML configuration files while executing this skill.
