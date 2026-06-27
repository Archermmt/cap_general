---
name: {cap_id}_execute_task
description: Execute robot tasks with one or more CAP agents, recording every CAP request and response in marked execution history. Available names are {available_names}.
metadata: {"nanobot":{"emoji":"🤖"}}
---

# Execute Task Skill

Decompose robot tasks into verifiable subtasks, execute independent agents concurrently, retry failures, and retain the complete request/response history.

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

Selector tools use an `agents` list. Tools with per-agent values use mappings keyed by agent name. Response keys use each agent's scene-visible `mark` (typically `alias(agent_name)` when an alias exists, otherwise `agent_name`). Never pass a formatted response key back as an agent selector.

## Mandatory History Protocol

Every non-`update_history` CAP tool request and response MUST be recorded. Calls to `update_history` itself are excluded to avoid infinite recursion.

History is append-only. Each `update_history` call sends a mapping from agent selector to one transcript message:

```json
{
  "agent_messages": {
    "{agent_name}": {
      "role": "llm",
      "mark": "step_1_trail_1",
      "request": {
        "tool": "{cap_id}_execute",
        "arguments": {"agent_codes": {"{agent_name}": "<code>"}}
      }
    }
  }
}
```

Each message must include:

- `role`: `llm` for planner/tool-caller events, or the agent response key (`agent.mark`) for agent-side responses.
- `mark`: the exact execution mark for that event.
- exactly one of `request` or `response`.

History marks still use this exact spelling:

```text
step_{exec_cnt}_trail_{trial_cnt}
```

Use the spelling `trail`, not `trial`, in history marks.

- Before the first execution, use `step_0_trail_0` for reset, documentation, initial observations, and planning.
- The first `execute` attempt uses `step_1_trail_1`.
- A new subtask increments `step`; its first attempt uses `trail_1`.
- A retry keeps the same `step` and increments `trail`.
- After receiving an execution result, use its `exec_cnt` and `trial_cnt` as the authoritative counters for subsequent events.

Before calling a CAP tool, append its request:

```json
{
  "name": "{cap_id}_update_history",
  "arguments": {
    "agent_messages": {
      "{agent_name}": {
        "role": "llm",
        "mark": "step_1_trail_1",
        "request": {
          "tool": "{cap_id}_execute",
          "arguments": {"agent_codes": {"{agent_name}": "<code>"}}
        }
      }
    }
  }
}
```

After the tool returns, append its response to the same mark:

```json
{
  "name": "{cap_id}_update_history",
  "arguments": {
    "agent_messages": {
      "{agent_name}": {
        "role": "<agent_mark>",
        "mark": "step_1_trail_1",
        "response": {
          "tool": "{cap_id}_execute",
          "data": "<complete returned response>"
        }
      }
    }
  }
}
```

Do not combine the request and response into one history update. Call `update_history` once before the request and once after the response. Preserve the complete arguments and response; do not replace them with summaries.

For batched tools, append the relevant request and response event to every participating agent's history under that agent's current mark.

## User Notifications

Use `message` immediately after planning and after every verification result. Use `media` to display verification images and final videos. These UI tools are not CAP agent tools and do not need history entries.

## Workflow

### 1. Reset And Inspect

Using `step_0_trail_0`, log request, call tool, then log response for each operation:

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

Build ordered atomic subtasks. Append the plan as a history event under `step_0_trail_0`:

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

Generate code using only functions from `function_doc`. For LIBERO:

```python
success = libero_vla_episode(task="put the bowl on the stove", max_steps=300)
RESULT = {"success": success, "task": "put the bowl on the stove"}
```

For each agent, log the request under the next `step_N_trail_1`, call execute, then log the complete response:

```json
{"name": "{cap_id}_execute", "arguments": {"agent_codes": {"{agent_name}": "<python code>"}}}
```

Independent agents may be sent in one `agent_codes` mapping. Dependent subtasks must run in order.

### 4. Monitor And Verify

Log the monitor request, call monitor, then log its response under the same mark:

```json
{"name": "{cap_id}_monitor", "arguments": {"agents": ["{agent_name}"], "wait_ms": -1}}
```

`wait_ms=-1` waits for completion, `0` polls immediately, and a positive value waits that many milliseconds before returning current status.

Inspect each result's `ok`, `result`, `stdout`, `stderr`, `exec_cnt`, `trial_cnt`, and `obs.main_image`. Display `main_image` with `media` when present. Append visual verification as another response event under the same mark:

```json
{
  "role": "<agent_mark>",
  "mark": "step_1_trail_1",
  "response": {
    "tool": "verification",
    "data": {
      "subtask": "<subtask>",
      "success": true,
      "image": "<main_image>",
      "notes": "<assessment>"
    }
  }
}
```

Send the SUCCESS/FAIL result with `message`.

### 5. Retry

On failure, increment the trail counter, then log request and response around both retry and monitor:

```json
{"name": "{cap_id}_retry", "arguments": {"agents": ["{agent_name}"]}}
```

Retry no more than `max_retry`. Stop if the result contains `error: "max_retry_exceeded"`.

### 6. Final Record

After all subtasks, log the record request under the latest mark, call record once, then log the returned response:

```json
{"name": "{cap_id}_record", "arguments": {"agents": ["{agent_name}"]}}
```

The complete record contains `info.history`, `info.executes`, `code`, `main_video`, and other videos. Send the executed code with `message`, then display each useful final video with `media`.

## Conceptual Pseudo-Code

```python
def call_cap_with_history(agent, agent_mark, mark, tool_name, arguments):
    # update_history calls are deliberately not logged.
    update_history(agent_messages={
        agent: {"role": "llm", "mark": mark, "request": {"tool": tool_name, "arguments": arguments}}
    })
    response = call_cap_tool(tool_name, arguments)
    update_history(agent_messages={
        agent: {"role": agent_mark, "mark": mark, "response": {"tool": tool_name, "data": response}}
    })
    return response

setup_mark = "step_0_trail_0"
reset_result = call_cap_with_history(
    agent, agent_mark, setup_mark, "{cap_id}_reset", {"agent_options": {agent: {}}}
)
doc_result = call_cap_with_history(
    agent, agent_mark, setup_mark, "{cap_id}_agent_doc", {"agents": [agent]}
)
obs_result = call_cap_with_history(
    agent, agent_mark, setup_mark, "{cap_id}_get_obs", {"agents": [agent]}
)

for step_cnt, subtask in enumerate(subtasks, start=1):
    trail_cnt = 1
    mark = f"step_{step_cnt}_trail_{trail_cnt}"
    call_cap_with_history(
        agent, agent_mark, mark, "{cap_id}_execute", {"agent_codes": {agent: make_code(subtask)}}
    )
    status = call_cap_with_history(
        agent, agent_mark, mark, "{cap_id}_monitor", {"agents": [agent], "wait_ms": -1}
    )
    while not verify(status) and trail_cnt <= max_retry:
        trail_cnt += 1
        mark = f"step_{step_cnt}_trail_{trail_cnt}"
        call_cap_with_history(agent, agent_mark, mark, "{cap_id}_retry", {"agents": [agent]})
        status = call_cap_with_history(
            agent, agent_mark, mark, "{cap_id}_monitor", {"agents": [agent], "wait_ms": -1}
        )

record_result = call_cap_with_history(
    agent, agent_mark, mark, "{cap_id}_record", {"agents": [agent]}
)
```

## Important Rules

1. Record every non-`update_history` CAP request before sending it.
2. Record every complete CAP response immediately after receiving it.
3. Use `step_{cnt}_trail_{cnt}` marks exactly, including the spelling `trail`.
4. Never log `update_history` calls themselves.
5. Never overwrite or summarize earlier events; history updates append.
6. Read `agent_doc` before generating execution code.
7. Use exact LIBERO task strings from `execute_rules`.
8. Use `monitor(..., wait_ms=-1)` for final execution results.
9. Call `record` once after all subtasks.
10. Do not modify YAML configuration files while executing this skill.
