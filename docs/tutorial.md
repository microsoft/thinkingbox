# End-to-end tutorial

How to create a server, a scenario, and a test case.

> **Companion doc:** [`adding_tools.md`](adding_tools.md) covers the *production-grade* tool pattern (custom exception class, `success_response`/`error_response` helpers, the unit-test fixture pattern). This tutorial uses a deliberately minimal server so the focus stays on the framework's dev loop — assertions, state, the LLM judge, the simulated user, and debugging.

## Prerequisites

Install thinkingbox in a virtual env, following the instructions in the README. All commands below assume:

- Both repos are cloned side-by-side and you are running commands from the **`thinkingbox/`** directory:

  ```
  parent/
  ├── thinkingbox/        # framework (← cwd for all commands below)
  └── thinkingbox-data/   # tools, datasets, servers.yaml
  ```

- The MCP Session Proxy is running with `thinkingbox-data`'s `servers.yaml`. Start it in a separate terminal and leave it running:

  ```bash
  THINKINGBOX_DATA="../thinkingbox-data" \
      uv run tb mcp-start --servers ../thinkingbox-data/servers/servers.yaml
  ```

  All `tb` commands below will fail if the proxy is not running. MCP servers are spawned per-conversation, so editing a server file does *not* require restarting `tb mcp-start` — the next conversation will pick up the change.


## Create a server

Create a new Python file `mcp_<servername>.py` in `thinkingbox-data/servers/thinkingbox_tools/thinkingbox_tools/`.

For example, this is a simple "greet" tool that just responds "Hi".

`thinkingbox-data/servers/thinkingbox_tools/thinkingbox_tools/mcp_greet.py`

```python
from fastmcp import FastMCP

mcp = FastMCP("greet", log_level="WARNING")

@mcp.tool(name="say_hi", description="Say Hi")
async def say_hi() -> dict:
    return {"message": "Hi!"}

if __name__ == "__main__":
    mcp.run(transport="stdio", show_banner=False)
```

The server is ready to be used. See [`adding_tools.md`](adding_tools.md) for the full production-style server pattern (custom exceptions, success/error response helpers).

### Register the server in `servers.yaml`

Add an entry for the new server in `thinkingbox-data/servers/servers.yaml` so the Session Proxy can spawn it:

```yaml
servers:
  # ... existing entries ...
  greet:
    type: mcp-process
    command: ["{python}", "-m", "thinkingbox_tools.mcp_greet"]
```

The next `tb infer` / `tb tui` invocation will pick it up automatically (no need to restart `tb mcp-start`, but if you started the proxy *before* editing `servers.yaml` you do need to restart it once for the new entry to register).


## Create a scenario

Create a `<scenarioname>.yaml` file in `thinkingbox-data/dataset/scenario/`. This describes which MCP servers must be running for the scenario, the list of enabled tools, and any system-prompt instructions.

`thinkingbox-data/dataset/scenario/greet.yaml`

```yaml
world_state:
    # "greet" is the server name we just created.
    # Initialization is an empty dictionary because the server has no state yet.
    greet: {}

    # Additional servers can be listed here.

tools:
# The agent will be able to use the "say_hi" tool from whichever server provides it (greet).
- name: say_hi

# Optional additional instruction for the system prompt.
bot_instructions: "It is important that you use the say_hi function when instructed to 'Say Hi'"
```


## Test the scenario

Now use `tb tui` to test the tool and scenario interactively:

```bash
uv run tb tui -c config/config_o4mini.yaml -d ../thinkingbox-data/dataset -a think --scenario greet
```

When prompted for a user message, type `Say Hi`, then press **ESC** then **ENTER** to submit. The model should call `say_hi()`.


## Create a test case

This section shows how to create a simple test case.

Create a new file `<testcases>.py` in `thinkingbox-data/dataset/test_case/`. Files can also be organized into subdirectories — see [File resolution by name](test_case_format.md#file-resolution-by-name).

`thinkingbox-data/dataset/test_case/greet_simple.py`

```python
from thinkingbox.common import Judge, TestContext

def test_say_hi(x: TestContext, judge: Judge):
    """!
    scenario: greet
    query: Please call the Say Hi function, and tell me what is its response.
    """
```

Test the query in this test case using `tb tui`:

```bash
uv run tb tui -c config/config_o4mini.yaml -d ../thinkingbox-data/dataset -a think --name greet_simple.py:test_say_hi
```

Or use the non-interactive `tb infer` and view the result with `tb pp`:

```bash
uv run tb infer -c config/config_o4mini.yaml -d ../thinkingbox-data/dataset -a think --name greet_simple.py:test_say_hi --no-test -o output.yaml
uv run tb pp --less output.yaml
```

Now add a simple assertion to the test function:

`thinkingbox-data/dataset/test_case/greet_simple.py`

```python
from thinkingbox.common import Judge, TestContext

def test_say_hi(x: TestContext, judge: Judge):
    """!
    scenario: greet
    query: Please call the Say Hi function, and tell me what is its response.
    """
    assert "Hi!" in x.response  # check the assistant's final response to the user
```

Re-run with `tb infer`, this time letting it run the test. Use 4 repetitions and write to a JSONL file:

```bash
uv run tb infer -c config/config_o4mini.yaml -d ../thinkingbox-data/dataset -a think --name greet_simple.py:test_say_hi --repeat 4 --batch-size 4 -o output.jsonl

# inspect the first conversation
head -n1 output.jsonl | uv run tb pp --less
```

Show a summary table:

```bash
uv run tb agg --concise output.jsonl
```

Expected output:

```
Test Case ID                | Runs | Pass | Fail | Error | Success% |
greet_simple.py:test_say_hi |    4 |    4 |    0 |     0 |   100.00 |
```


## Server initialization and state

When a server is started, if a special initialization function exists, it is invoked before the agent loop.

Since one server process is started per conversation, state related to the current conversation — and isolated from all others — can be kept in the global scope.

A "get effects" function can also be defined, to retrieve information needed by the test.

A teardown function can be defined as well; it is called before terminating the server (or before terminating the MCP session on remote servers).

Special functions:
- `__reserved__init`
- `__reserved__geteffects`
- `__reserved__teardown`

`thinkingbox-data/servers/thinkingbox_tools/thinkingbox_tools/mcp_greet.py`

```python
import json
from fastmcp import FastMCP

mcp = FastMCP("greet", log_level="WARNING")
global_state = {
    "greeting": "Hi!",
    "count": 0,
}

@mcp.tool(name="__reserved__init")
async def initialize(config: dict):
    # configure the greeting
    global_state["greeting"] = config["greeting"]
    return {"status": "ok"}  # success response

@mcp.tool(name="__reserved__geteffects")
async def geteffects():
    # get the counter value
    return {"count": global_state["count"]}

@mcp.tool(name="say_hi", description="Say Hi")
async def say_hi() -> dict:
    global_state["count"] += 1
    return {"message": global_state["greeting"]}

if __name__ == "__main__":
    mcp.run(transport="stdio", show_banner=False)
```

Now initialization is required, and the configuration dictionary must contain the `greeting` key. Update the scenario:

`thinkingbox-data/dataset/scenario/greet.yaml`

```yaml
world_state:
    greet:
        greeting: "Hello!"
tools:
- name: say_hi
bot_instructions: "It is important that you use the say_hi function when instructed to 'Say Hi'"
```

Test the new server with `tb tui`:

```bash
uv run tb tui -c config/config_o4mini.yaml -d ../thinkingbox-data/dataset -a think --name greet_simple.py:test_say_hi
```

The tool should now respond `"Hello!"`.

Inspect the effects by typing `/effects` as the user prompt in TUI; it should print:

```yaml
greet:
  count: 1
```

This is exactly the `x.effects` dictionary inside the test function. Update the test:

`thinkingbox-data/dataset/test_case/greet_simple.py`

```python
from thinkingbox.common import Judge, TestContext

def test_say_hi(x: TestContext, judge: Judge):
    """!
    scenario: greet
    query: Please call the Say Hi function, and tell me what is its response.
    """
    assert x.effects["greet"]["count"] == 1
```

Run the agent loop with the test, also dumping the effects dictionary into the output:

```bash
uv run tb infer -c config/config_o4mini.yaml -d ../thinkingbox-data/dataset -a think --name greet_simple.py:test_say_hi --dump testcontext -o output.yaml
uv run tb pp --less output.yaml
```


## LLM judge

The `judge` available in test-case functions allows interaction with an LLM judge — useful for asking simple natural-language questions about a piece of text.

`thinkingbox-data/dataset/test_case/greet_simple.py`

```python
from thinkingbox.common import Judge, TestContext

def test_say_hi(x: TestContext, judge: Judge):
    """!
    scenario: greet
    query: Please call the Say Hi function, and tell me what is its response.
    """
    assert x.effects["greet"]["count"] == 1
    # ask a question about the content of `x.response`
    assert judge.text_yesno(
        x.response, "Does the message contain a greeting?"
    )
```

Run and test:

```bash
uv run tb infer -c config/config_o4mini.yaml -d ../thinkingbox-data/dataset -a think --name greet_simple.py:test_say_hi --dump testcontext -o output.yaml
uv run tb pp --less output.yaml
```


## Simulated user

A test case can include additional context, in natural language, for a simulated user. If this context exists, the simulated user will respond to the agent until the agent declares it has finished.

`thinkingbox-data/dataset/test_case/greet_simple.py`

```python
# [...]

def test_say_hi_user(x: TestContext, judge: Judge):
    """!
    scenario: greet
    query: I have a task for you, can you do it?
    user_context: You'd like to call the Say Hi function, and you'd like to know what this function responds.
    """
    assert x.effects["greet"]["count"] == 1
    # ask a question about the content of `x.response`
    assert judge.text_yesno(
        x.response, "Does the message contain a greeting?"
    )
```

Note: this may not always pass — the agent could sometimes decide to terminate the conversation.

Run and test:

```bash
uv run tb infer -c config/config_o4mini.yaml -d ../thinkingbox-data/dataset -a think --name greet_simple.py:test_say_hi_user --dump testcontext -o output.yaml
uv run tb pp --less output.yaml
```

The conversation should contain something like:

```
[assistant::text]
Sure—could you please describe the task you have in mind and any specific requirements or goals?

[user::text]
I want to call the Say Hi function and see what it responds.
```

For a description of all possible fields in a test case, see [Test Case Format](test_case_format.md).


## Debug a test case

Use `tb run-test` to re-run the test only, on the output of `tb infer`, provided that the test context is in the output file (`--dump testcontext`):

```bash
uv run tb infer -c config/config_o4mini.yaml -d ../thinkingbox-data/dataset -a think --name greet_simple.py:test_say_hi --dump testcontext -o output.yaml
# test and print the result without saving it
uv run tb run-test -c config/config_o4mini.yaml --name greet_simple.py:test_say_hi --resultfile output.yaml -o /dev/null --verbose
```

To use a debugger, see [Debugging Tests](debugging_tests.md).
