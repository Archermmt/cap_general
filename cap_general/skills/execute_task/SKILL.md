---
name: {cap_id}_execute_task
description: Decompose and execute a robot manipulation task through the {agent_name} CAP agent MCP tools, using agent_doc, execute, monitor, retry, get_obs, update_plan, and record. agent_name can be any one of {available_names}.
metadata: {"nanobot":{"emoji":"🤖"}}
---

# Execute Task Skill

Execute a robot manipulation task by decomposing it into visually verifiable subtasks, running each subtask through the `{agent_name}` CAP agent, retrying failures, and producing a final run record.
The placeholder `{agent_name}` can be replaced by any available agent name or alias from this scene: `{available_names}`.

This skill uses the current CAP agent MCP interface:

- `{cap_id}_reset`
- `{cap_id}_agent_doc`
- `{cap_id}_execute`
- `{cap_id}_monitor`
- `{cap_id}_retry`
- `{cap_id}_get_obs`
- `{cap_id}_update_plan`
- `{cap_id}_record`

Do not use older tool names such as `step`, `get_record`, or `update_report`.
Every tool call in this workflow must include `agent="{agent_name}"` so the scene can route the call to the correct agent.
The `agent` value must be one of `{available_names}`.

## CRITICAL: Use `message` For Mid-Task Notifications

This task can execute many subtasks before the final response. The user sees nothing during that time unless you proactively send updates. The `message` tool is for immediate mid-turn notifications; it is not a normal final reply.

You MUST call `message` at these checkpoints:

1. **After planning**: immediately after `{cap_id}_update_plan`, send the main task and numbered subtask list.
2. **After each verification update**: immediately after `{cap_id}_update_plan`, send the SUCCESS/FAIL result and brief notes.
3. **After final record**: send the returned `record.code` in a fenced Python block before displaying the final video.

These `message` calls are proactive status pushes. When `message` is used, the final normal LLM text response may be suppressed, so treat `message` as the communication channel for progress and final text summary during this skill.

Do not use `message` to display images or videos. Use the `media` tool for `obs.main_image` and `main_video`.

## Workflow

### 1. Reset The Agent

Always reset before starting a new episode. Follow the `{cap_id}_reset` MCP tool schema and documentation for supported options:

```json
{"name": "{cap_id}_reset", "arguments": {"agent": "{agent_name}", "options": {}}}
```

### 2. Read Agent Docs And Initial State

Call `agent_doc` before writing code:

```json
{"name": "{cap_id}_agent_doc", "arguments": {"agent": "{agent_name}"}}
```

Use the returned fields as authoritative:

- `function_doc`: callable functions available inside `execute` code.
- `execute_rules`: task-specific rules. For LIBERO, this lists valid task descriptions; use them verbatim.
- `policy_doc`: configured policy capabilities.
- `max_retry`: number of retries allowed after the first execute attempt.

Then get the current observation:

```json
{"name": "{cap_id}_get_obs", "arguments": {"agent": "{agent_name}"}}
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
  "name": "{cap_id}_update_plan",
  "arguments": {
    "agent": "{agent_name}",
    "plan": {
      "main_task": "<original user task>",
      "sub_tasks": ["<subtask 1>", "<subtask 2>"]
    }
  }
}
```

Immediately after `update_plan`, send the plan to the user with the `message` tool. This is a proactive mid-task notification so the user can see the plan before execution starts:

```json
{
  "name": "message",
  "arguments": {
    "content": "Task: <original user task>\n\nSubtasks:\n1. <subtask 1>\n2. <subtask 2>"
  }
}
```

Build the `Subtasks` section as a numbered list from the same subtask list stored in `update_plan`.

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
{"name": "{cap_id}_execute", "arguments": {"agent": "{agent_name}", "code": "<python code string>"}}
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

After each `execute` or `retry`, call `monitor(agent="{agent_name}", wait_finish=true)` and inspect the returned `response["result"]["obs"]["main_image"]` if present.

If the subtask failed, call `retry`:

```json
{"name": "{cap_id}_retry", "arguments": {"agent": "{agent_name}"}}
```

Retry at most `max_retry` times after the first `execute` attempt. After `retry`, call `monitor(wait_finish=true)`; if the monitored result returns `ok: false` with `error: "max_retry_exceeded"`, stop retrying that subtask.

Display the returned `obs.main_image` so the user can see the verification frame:

```json
{"name": "media", "arguments": {"media_type": "image", "mode": "display", "media_path": "<absolute path from obs.main_image>", "prompt": "Verification image: <subtask>"}}
```

If visual judgment is needed, use `media` in `analyze` mode with the same image:

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

For each verification result, update the plan:

```json
{
  "name": "{cap_id}_update_plan",
  "arguments": {
    "agent": "{agent_name}",
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

Immediately after `update_plan`, send the verification result to the user with `message`:

```json
{
  "name": "message",
  "arguments": {
    "content": "Verification result (Exec <exec_cnt>/<total_subtasks> Trial <trial_cnt>/<max_retry>): <subtask>\nResult: <SUCCESS or FAIL>\nNotes: <brief visual or execution assessment>"
  }
}
```

### 6. Final Record

After all subtasks are completed or the task is stopped, call `record` exactly once for the full run:

```json
{"name": "{cap_id}_record", "arguments": {"agent": "{agent_name}", "step_idx": -1}}
```

The result contains:

- `main_video`: primary-camera video path when video recording is enabled.
- `videos`: all recorded camera videos.
- `info`: run plan and per-execute metadata.
- `code`: concatenated code from all executed subtasks.

Present the final summary with the task, subtasks, success/failure status, `main_video` path, and executed code.

Send the returned `code` to the user with `message`:

```json
{
  "name": "message",
  "arguments": {
    "content": "Executed code:\n\n```python\n<record.code>\n```"
  }
}
```

Then display `main_video` with the `media` tool:

```json
{"name": "media", "arguments": {"media_type": "video", "mode": "display", "media_path": "<absolute path from main_video>", "prompt": "Full task execution video"}}
```

If `main_video` is missing but `videos` contains camera videos, display the first useful video path from `videos`.

## Complete Pseudo-Code

```python
reset(agent="{agent_name}", options={})

doc = agent_doc(agent="{agent_name}")
function_doc = doc["function_doc"]
execute_rules = doc["execute_rules"]
max_retry = doc["max_retry"]
obs = get_obs(agent="{agent_name}")

subtasks = decompose_task_using_execute_rules(user_task, execute_rules)
update_plan(agent="{agent_name}", plan={"main_task": user_task, "sub_tasks": subtasks})
subtask_list = "\n".join(f"{idx + 1}. {subtask}" for idx, subtask in enumerate(subtasks))
message(content=f"Task: {user_task}\n\nSubtasks:\n{subtask_list}")

for subtask in subtasks:
    code = make_code_from_function_doc(subtask, function_doc)
    execute(agent="{agent_name}", code=code)
    status = monitor(agent="{agent_name}", wait_finish=True)
    response = status["result"]

    attempts = 0
    while True:
        image = response.get("obs", {}).get("main_image")
        if image:
            media_display(
                media_type="image",
                mode="display",
                media_path=absolute_path(image),
                prompt=f"Verification image: {subtask}",
            )
        success = bool(response.get("result", {}).get("success")) and visually_verify(response.get("obs", {}))
        update_plan(plan={
            f"verify_exec_{response['exec_cnt']}_trial_{response['trial_cnt']}": {
                "subtask": subtask,
                "success": success,
                "image": image,
            }
        })
        message(content=f"Verification result (Exec {response['exec_cnt']} Trial {response['trial_cnt']}): {subtask}\nResult: {'SUCCESS' if success else 'FAIL'}")
        if success:
            break
        if attempts >= max_retry:
            break
        attempts += 1
        retry(agent="{agent_name}")
        status = monitor(agent="{agent_name}", wait_finish=True)
        response = status["result"]
        if not response.get("ok", False):
            break

record_result = record(agent="{agent_name}", step_idx=-1)
message(content=f"Executed code:\n\n```python\n{record_result.get('code', '')}\n```")
if record_result.get("main_video"):
    media_display(
        media_type="video",
        mode="display",
        media_path=absolute_path(record_result["main_video"]),
        prompt="Full task execution video",
    )
```

## Important Rules

1. Always call `{cap_id}_agent_doc` before writing `execute` code.
2. Use `execute`, then `monitor(wait_finish=true)` to retrieve the execution result. Do not use `step`.
3. Use `record(agent="{agent_name}", step_idx=-1)`, not `get_record(record_all=true)`.
4. Use `update_plan(plan=...)`, not `update_report(info=...)`.
5. After storing the plan with `update_plan`, call `message` immediately to send the main task and numbered subtask plan to the user.
6. For each verification attempt, display `obs.main_image` with `media_type="image"` and `mode="display"` when an image is available.
7. After storing a verification result with `update_plan`, call `message` immediately to send the SUCCESS/FAIL result and notes to the user.
8. For LIBERO tasks, map the user request to exact task strings from `execute_rules`.
9. After `record(agent="{agent_name}", step_idx=-1)`, send `record.code` to the user with `message`.
10. After `record(agent="{agent_name}", step_idx=-1)`, display `main_video` with `media_type="video"` and `mode="display"`.
11. Use an absolute path for `main_video` when calling `media`.
12. Do not modify yaml configuration files as part of executing this skill.
