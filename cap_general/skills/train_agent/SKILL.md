---
name: {cap_id}_train_agent
description: Train one or more CAP robot agents for a user-requested number of epochs. Use whenever the user asks a robot or agent to train, learn, fine-tune, or continue training for a specific number of rounds or epochs. Available names are {available_names}. Read this skill before execute, only then call {cap_id}-prefixed tools.
metadata: {"nanobot":{"emoji":"🏋️"}}
---

# Train Agent Skill

Train selected robot agents with `{cap_id}_run_pipe`, poll their status every five seconds with `{cap_id}_monitor`, and send every returned result to the user with `message`.

Available agent names and aliases: `{available_names}`.

## CAP Tools

- `{cap_id}_agent_doc`
- `{cap_id}_run_pipe`
- `{cap_id}_monitor`

Selector tools use an `agents` list. Training uses an `agent_options` mapping keyed by agent name or alias. Response keys use each agent's scene-visible mark, usually `alias(agent_name)` when an alias exists.

## Training Request

Each training request requires:

- `job_options`: A list of job entries. Each entry has a `job` key (job group name, e.g. `"train"`) and an `options` key with job-specific options. The agent configuration determines the training stage (RL or BC).
- `policy_name`: Configured policy name returned by `agent_doc`.

Job options for training include `epoch` (positive integer), `seed`, `train_cfg`, `record_epoch`, and other agent-specific options documented by `agent_doc`.

Example:

```json
{
  "agent_options": {
    "{agent_name}": {
      "policy_name": "runner",
      "job_options": [
        {"job": "train", "options": {"epoch": 100}}
      ]
    }
  }
}
```

Do not send `method`, `stage`, or other unsupported top-level fields to `{cap_id}_run_pipe`.

## Required Workflow

1. Identify the requested agent from `{available_names}` and parse the requested epoch count as a positive integer.
2. Call `{cap_id}_agent_doc` before training to inspect `policy_doc`, the `run_pipe` function signature, and agent-specific options. Read `result["scene"]["async_task"]` and store it as `async_task`.
3. Select the policy from the user's request and `agent_doc`. Ask the user only when multiple valid policies remain genuinely ambiguous.
4. Call `{cap_id}_run_pipe` once with all independent target agents in one `agent_options` mapping.
5. Send a `message` immediately confirming that training started and include the initial run_pipe response.
6. If `async_task=true`: repeatedly call `{cap_id}_monitor` with `wait_ms=5000` for agents that are still running. After every monitor call, immediately send the complete returned status and `result` to the user with `message`. Stop polling an agent when its status has `running=false`.
   If `async_task=false`: the `run_pipe` response already contains the final result — skip monitor polling and send the result directly with `message`.
7. Send a final `message` containing the complete `result`, including any error details when training fails.

## Monitor Loop

Only applicable when `async_task=true`. Use a five-second wait on every poll:

```json
{
  "name": "{cap_id}_monitor",
  "arguments": {
    "agents": ["{agent_name}"],
    "wait_ms": 5000
  }
}
```

Interpret each keyed status independently:

- `running=true`: Send the current status/result with `message`, then poll again after another five-second monitor wait.
- `running=false`: Inspect `result`. A successful training result is flat, for example `{"ok": true, "policy_name": "runner", "stage": "rl", ...}`. Send the complete result with `message`, then stop polling that agent.
- A failed task reports `ok=false` and error details such as `error` or `err_msg` inside `result`; do not expect a separate top-level `error` field.

Do not use `wait_ms=-1`; the user must receive progress at five-second intervals. Do not busy-poll with `wait_ms=0`.

## Example Sequence

```json
{"name": "{cap_id}_agent_doc", "arguments": {"agents": ["{agent_name}"]}}
```

```json
{
  "name": "{cap_id}_run_pipe",
  "arguments": {
    "agent_options": {
      "{agent_name}": {
        "policy_name": "runner",
        "job_options": [
          {"job": "train", "options": {"epoch": 100}}
        ]
      }
    }
  }
}
```

Send the initial response with `message`, then repeat:

```json
{"name": "{cap_id}_monitor", "arguments": {"agents": ["{agent_name}"], "wait_ms": 5000}}
```

Send each monitor response with `message` before the next poll.

## Important Rules

1. Trigger this skill whenever the user asks a robot or agent to train for a number of rounds or epochs.
2. Require `epoch > 0` and preserve the exact number requested by the user.
3. Read `{cap_id}_agent_doc` before selecting a policy or options; check `result["scene"]["async_task"]`.
4. Call `{cap_id}_run_pipe` only once per requested training run.
5. If `async_task=true`: poll only with `{cap_id}_monitor(wait_ms=5000)` while training is running and send every monitor result to the user through `message`.
   If `async_task=false`: skip monitor entirely; the final result is already in the `run_pipe` response.
6. Do not call `{cap_id}_execute` to train a policy.
7. Do not modify YAML configuration files during training.
