# History

Multi-turn test cases require prior conversation context to be injected into the agent session before the current turn runs. The `history:` field in a test docstring references a transcript stored in a companion `.meta.yaml` file, which is loaded during hydration and resolved to a list of messages prepended to the conversation.

## `history:` vs `user_context`

`user_context` generates follow-up messages live using a user simulator LLM. Use it when testing whether the agent correctly handles clarification and back-and-forth interaction. The conversation is produced fresh on every run.

`history:` injects a pre-recorded transcript. Use it when you have a reference transcript and want to evaluate a specific later turn in a known, controlled conversation state.

Both can be combined: a turn 2 test can use `history:` to inject turn 1 and `user_context` to simulate user follow-ups within turn 2.

## The `.meta.yaml` File

Place a companion file next to the test file. The name is formed by replacing the `.py` extension with `.meta.yaml`:

```
dataset/test_case/
  my_tests.py
  my_tests.meta.yaml   ← loaded automatically alongside my_tests.py
```

Exactly one `.meta.yaml` is loaded per test file. The framework looks only for `{stem}.meta.yaml` — any other files such as `my_tests_history.meta.yaml` are not loaded and are silently ignored.

The file may contain arbitrary per-test entries (keyed by function name) alongside `$history:`. **The framework only reads `$history:`** — all other entries are ignored at runtime and are available for documentation or future tooling.

Note: `metadata:` is not a supported field in test docstrings. Per-test metadata belongs in the `.meta.yaml` file as a top-level entry alongside `$history:`, not in the Python docstring.

### `$history:` section

Stores named groups of reference transcript messages. Each group is a flat list of messages in conversation order.

```yaml
$history:
  refund_query:
    - T: Text
      message_id: t1
      role: user
      content: What is your return policy?
    - T: ParallelToolCall
      metadata: {}
      tool_calls:
        - T: ToolCall
          name: search
          arguments:
            query: return policy
          id: call_abc123
    - T: ToolResponse
      name: search
      content: '{"status": "ok", "result": [...]}'
      id: call_abc123
    - T: Text
      metadata:
        tag: text
        is_done: true
      role: assistant
      content: Our return policy allows returns within 30 days of purchase.

    - T: Text
      message_id: t2
      role: user
      content: What about items purchased on sale?
    - T: Text
      role: assistant
      content: Sale items are final sale and cannot be returned.
```

Each message uses a `T:` discriminator matching the type names in `chat_types.py`: `Text`, `ToolCall`, `ParallelToolCall`, `ToolResponse`.

`message_id:` marks turn boundaries. It is only needed on messages used as start or end anchors in `history:` range references.

## The `history:` Field

```
history: "key:start:end"
```

| Part | Type | Meaning |
|------|------|---------|
| `key` | str | Group name in `$history:` |
| `start` | int or message_id | First message to include |
| `end` | message_id or `""` | Exclusive end; empty string means to end of list |

The value must be quoted in YAML when `end` is empty, because a trailing `:` is a YAML value indicator.

### Range examples

```yaml
history: "refund_query:0:"      # all messages
history: "refund_query:0:t2"    # from index 0 up to (not including) t2
history: "refund_query:t2:"     # from t2 to end
history: "refund_query:t1:t3"   # from t1 up to (not including) t3
```

### Keys with colons

The parser splits from the right, so the last two colons are always the separators:

```yaml
history: "Benefits & Leave:0:t2"   # key='Benefits & Leave', start=0, end='t2'
```

## How History Injection Works

1. **Resolution** — the `.meta.yaml` is loaded and the range reference resolved to a list of messages.
2. **Message injection** — messages are added to the agent conversation before the first user query.
3. **Server state replay** — every `ToolCall` and `ParallelToolCall` in the history is re-executed against the MCP server in order, restoring server state to match the conversation. Replay errors are logged as warnings and recorded in `DecodeResult.metadata["replay_warnings"]`.
4. **Test turn** — the agent processes the query against the pre-loaded conversation and restored state.

## Example

See `dataset/test_case/cloud_drive.py` and `dataset/test_case/cloud_drive.meta.yaml` for a working example. `test_create_config_T1` creates a config file. `test_update_config_T2` injects that turn as history and asks the agent to append a new setting to "the configuration" — the agent needs conversation context to resolve which file and what its current content is, and needs server state replay to ensure the file exists when the turn runs.

## Message Types Reference

| Type | Required fields | Notes |
|------|----------------|-------|
| `Text` | `role`, `content` | `role` is `user`, `assistant`, or `system` |
| `ToolCall` | `name`, `id`, `arguments` | `arguments` defaults to `{}` if absent |
| `ParallelToolCall` | `tool_calls` | A list of `ToolCall` entries |
| `ToolResponse` | `name`, `content`, `id` | all three are required |

All types accept an optional `metadata:` sub-dict for framework flags such as `is_done: true` and `tag: text`.
