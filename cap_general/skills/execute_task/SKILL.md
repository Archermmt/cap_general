---
name: {agent_id}_execute_task
description: Decompose and execute a robot manipulation task through the {agent_name} CAP agent MCP tools, using agent_doc, execute, retry, get_obs, update_plan, and record.
metadata: {"nanobot":{"emoji":"🤖"}}
---

# Execute Task Skill

Execute a robot manipulation task by decomposing it into visually verifiable subtasks, running each subtask through the `{agent_name}` CAP agent, retrying failures, and producing a final run record.

This skill uses the current CAP agent MCP interface:

- `{agent_id}_reset`
- `{agent_id}_agent_doc`
- `{agent_id}_execute`
- `{agent_id}_retry`
- `{agent_id}_get_obs`
- `{agent_id}_update_plan`
- `{agent_id}_record`

Do not use older tool names such as `step`, `get_record`, or `update_report`.

## Workflow

### 1. Reset The Agent

Always reset before starting a new episode:

```json
{"name": "{agent_id}_reset", "arguments": {"options": {"episode_idx": 0}}}
```

If the environment does not use `episode_idx`, an empty options object is acceptable:

```json
{"name": "{agent_id}_reset", "arguments": {"options": {}}}
```

### 2. Read Agent Docs And Initial State

Call `agent_doc` before writing code:

```json
{"name": "{agent_id}_agent_doc", "arguments": {}}
```

Use the returned fields as authoritative:

- `function_doc`: callable functions available inside `execute` code.
- `execute_rules`: task-specific rules. For LIBERO, this lists valid task descriptions; use them verbatim.
- `policy_doc`: configured policy capabilities.
- `max_retry`: number of retries allowed after the first execute attempt.

Then get the current observation:

```json
{"name": "{agent_id}_get_obs", "arguments": {}}
```

Use `main_image` for visual inspection when present.

### 3. Decompose The Task

Break the user task into ordered atomic subtasks.

Rules:

- Each subtask should be a single manipulation objective that can be visually checked.
- If `execute_rules` lists valid task descriptions, each subtask must use one of those strings exactly.
- Do not paraphrase LIBERO task strings. For example, use `put the bowl on the stove` exactly if that is listed.
- Prefer fine-grained subtasks over broad combined actions.

Store the plan with `update_plan`:

```json
{
  "name": "{agent_id}_update_plan",
  "arguments": {
    "plan": {
      "main_task": "<original user task>",
      "sub_tasks": ["<subtask 1>", "<subtask 2>"]
    }
  }
}
```

### 4. Execute Each Subtask

Generate Python code using only functions from `function_doc`.

For LIBERO, follow the unit test pattern in `tests/libero/test_libero_agent.py`:

```python
success = libero_vla_episode(task="put the bowl on the stove", max_steps=300)
print(f"Episode result: {'SUCCESS' if success else 'FAIL'}")
RESULT = {"success": success, "task": "put the bowl on the stove"}
```

Execute the code:

```json
{"name": "{agent_id}_execute", "arguments": {"code": "<python code string>"}}
```

Read the response:

- `ok`: whether the code executed without exception.
- `stderr` / `stdout`: execution logs.
- `result`: value of `RESULT` from the executed code.
- `exec_cnt`: execute index.
- `trial_cnt`: current trial index. The first execute returns trial 1; retries increment it.
- `step_start` / `step_end`: environment step range.
- `obs`: latest observation dict, including `main_image` when image observations are available.

Use `result["success"]` when available, and use visual inspection of `obs.main_image` as the final check.

### 5. Verify And Retry

After each `execute` or `retry`, inspect `response["obs"]["main_image"]` if present.

If the subtask failed, call `retry`:

```json
{"name": "{agent_id}_retry", "arguments": {}}
```

Retry at most `max_retry` times after the first `execute` attempt. If `retry` returns `ok: false` with `error: "max_retry_exceeded"`, stop retrying that subtask.

For each verification result, update the plan:

```json
{
  "name": "{agent_id}_update_plan",
  "arguments": {
    "plan": {
      "verify_exec_<exec_cnt>_trial_<trial_cnt>": {
        "subtask": "<subtask>",
        "success": true,
        "image": "<obs.main_image path>",
        "notes": "<brief visual or execution assessment>"
      }
    }
  }
}
```

### 6. Final Record

After all subtasks are completed or the task is stopped, call `record` exactly once for the full run:

```json
{"name": "{agent_id}_record", "arguments": {"step_idx": -1}}
```

The result contains:

- `main_video`: primary-camera video path when video recording is enabled.
- `videos`: all recorded camera videos.
- `info`: run plan and per-execute metadata.
- `code`: concatenated code from all executed subtasks.

Present the final summary with the task, subtasks, success/failure status, `main_video` path, and executed code.

## Complete Pseudo-Code

```python
reset(options={"episode_idx": 0})

doc = agent_doc()
function_doc = doc["function_doc"]
execute_rules = doc["execute_rules"]
max_retry = doc["max_retry"]
obs = get_obs()

subtasks = decompose_task_using_execute_rules(user_task, execute_rules)
update_plan(plan={"main_task": user_task, "sub_tasks": subtasks})

for subtask in subtasks:
    code = make_code_from_function_doc(subtask, function_doc)
    response = execute(code=code)

    attempts = 0
    while True:
        success = bool(response.get("result", {}).get("success")) and visually_verify(response.get("obs", {}))
        update_plan(plan={
            f"verify_exec_{response['exec_cnt']}_trial_{response['trial_cnt']}": {
                "subtask": subtask,
                "success": success,
                "image": response.get("obs", {}).get("main_image"),
            }
        })
        if success:
            break
        if attempts >= max_retry:
            break
        attempts += 1
        response = retry()
        if not response.get("ok", False):
            break

record(step_idx=-1)
```

## Important Rules

1. Always call `{agent_id}_agent_doc` before writing `execute` code.
2. Use `execute`, not `step`.
3. Use `record(step_idx=-1)`, not `get_record(record_all=true)`.
4. Use `update_plan(plan=...)`, not `update_report(info=...)`.
5. For LIBERO tasks, map the user request to exact task strings from `execute_rules`.
6. Do not modify yaml configuration files as part of executing this skill.
