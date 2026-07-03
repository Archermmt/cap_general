---
name: {cap_id}_train_agent
description: Train one or more CAP robot agents for a user-requested number of epochs. Use whenever the user asks a robot or agent to train, learn, fine-tune, or continue training for a specific number of rounds or epochs. Available names are {available_names}.
metadata: {"nanobot":{"emoji":"🏋️"}}
---

# Train Agent Skill

Train selected robot agents with `{cap_id}_train`, poll their status every five seconds with `{cap_id}_monitor`, and send every returned result to the user with `message`.

Available agent names and aliases: `{available_names}`.

## CAP Tools

- `{cap_id}_agent_doc`
- `{cap_id}_train`
- `{cap_id}_monitor`

Selector tools use an `agents` list. Training uses an `agent_options` mapping keyed by agent name or alias. Response keys use each agent's scene-visible mark, usually `alias(agent_name)` when an alias exists.

## Training Request

Each training request requires:

- `policy_name`: Configured policy name returned by `agent_doc`.
- `epoch`: Positive integer number of training epochs requested by the user.
- `options`: Optional agent-specific training options documented by `agent_doc`, such as `seed`, `train_cfg`, and `record_epoch`.

The agent configuration determines the training stage, such as RL or BC. Do not
send `method`, `stage`, or other unsupported top-level fields to `{cap_id}_train`.

Example:

```json
{
  "agent_options": {
    "{agent_name}": {
      "policy_name": "runner",
      "epoch": 100,
      "options": {}
    }
  }
}
```

Never place `epoch` inside `options`. It is a required standard parameter of `train`.

## Required Workflow

1. Identify the requested agent from `{available_names}` and parse the requested epoch count as a positive integer.
2. Call `{cap_id}_agent_doc` before training to inspect `policy_doc`, the `train` function signature, and agent-specific options.
3. Select the policy from the user's request and `agent_doc`. Ask the user only when multiple valid policies remain genuinely ambiguous.
4. Call `{cap_id}_train` once with all independent target agents in one `agent_options` mapping.
5. Send a `message` immediately confirming that training started and include the initial train response.
6. Repeatedly call `{cap_id}_monitor` with `wait_ms=5000` for agents that are still running.
7. After every monitor call, immediately send the complete returned status and `result` to the user with `message`.
8. Stop polling an agent when its status has `running=false`.
9. Send a final `message` containing the complete `result`, including any error details when training fails.

## Monitor Loop

Use a five-second wait on every poll:

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
  "name": "{cap_id}_train",
  "arguments": {
    "agent_options": {
      "{agent_name}": {
        "policy_name": "runner",
        "epoch": 100,
        "options": {}
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
3. Read `{cap_id}_agent_doc` before selecting a policy or options.
4. Call `{cap_id}_train` only once per requested training run.
5. Poll only with `{cap_id}_monitor(wait_ms=5000)` while training is running.
6. Send every monitor result to the user through `message`; do not silently wait for completion.
7. Do not call `{cap_id}_execute` to train a policy.
8. Do not modify YAML configuration files during training.
