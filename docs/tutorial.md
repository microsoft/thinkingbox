# End-to-end tutorial

How to create a server, a scenario, and a test case.

## Prerequisites

Install thinkingbox in a virtual env, following the instructions in the README file, and start `tb mcp-start`.

Important: all the commands below will fail if `tb mcp-start` is not running.

MCP servers are started when a conversation starts, it is not necessary to restart `tb mcp-start` after a MCP server is modified the next conversation will use the new server.


## Create a server

Create a new Python file named `mcp_<servername>.py` in `thinkingbox/tools/`.

For example, this is a simple "greet" tool that will just respond "Hi".

`thinkingbox/tools/mcp_greet.py`

```python
from fastmcp import FastMCP

mcp = FastMCP("greet", log_level="WARNING")

@mcp.tool(name="say_hi", description="Say Hi")
async def say_hi() -> dict:
    return {"message": "Hi!"}

if __name__ == "__main__":
    mcp.run(transport="stdio", show_banner=False)
```

The server is ready to be used. Check [Adding Tools](adding_tools.md) for a more complete
example.

## Create a scenario

Create a `<scenarioname>.yaml` file in `dataset/scenario/`. This will describe one or
more MCP servers that need to be running for a particular scenario, and the list of enabled
tools.

`dataset/scenario/greet.yaml`

```yaml
world_state:
    # "greet" is the name of the server ("<servername>") we just created.
    # the initialization is an empty dictionary, because we don't have any.
    greet: {}

    # more servers are allowed too

tools:
# the agent will be able to use the "say_hi" tool from the server that has it (greet in this case)
- name: say_hi

# optional additional instruction for the system prompt
bot_instructions: "It is important that you use the say_hi function when instructed to 'Say Hi'"
```

## Test the scenario

Now you can use `tb tui` to test your tool and scenario interactively

```bash
tb tui -c config/config_o4mini.yaml -d ./dataset -a think --scenario greet
```

When prompted for a user message, type "Say Hi", press ESC and then ENTER to submit.
The model should call the say_hi() function.


## Create a test case

This section shows hot to create a simple test case.

Create a new file `<testcases>.py` in `dataset/test_case/`. Files can also be organized into
subdirectories, see [File resolution by name](test_case_format.md#file-resolution-by-name).

`dataset/test_case/greet_simple.py`

```python
from thinkingbox.common import Judge, TestContext

def test_say_hi(x: TestContext, judge: Judge):
    """!
    scenario: greet
    query: Please call the Say Hi function, and tell me what is its response.
    """
```

You can test running the query in this test case by using `tb tui`

```bash
tb tui -c config/config_o4mini.yaml -d ./dataset -a think --name greet_simple.py:test_say_hi
```

Similarly, you can use the non-interactive tool, `tb infer`, and then see the
result with `tb pp`.

```bash
tb infer -c config/config_o4mini.yaml -d ./dataset -a think --name greet_simple.py:test_say_hi --no-test -o output.yaml
tb pp --less output.yaml
```

Now add a simple assertion to the test function

`dataset/test_case/greet_simple.py`

```python
from thinkingbox.common import Judge, TestContext

def test_say_hi(x: TestContext, judge: Judge):
    """!
    scenario: greet
    query: Please call the Say Hi function, and tell me what is its response.
    """
    assert "Hi!" in x.response  # check the assistant's final response to the user
```

And re-run with `tb infer`, including the test. This time we can run it with 4
repetitions and write the output to a JSONL file.

```bash
tb infer -c config/config_o4mini.yaml -d ./dataset -a think --name greet_simple.py:test_say_hi --repeat 4 --batch-size 4 -o output.jsonl

# check the first conversation
head -n1 output.jsonl | tb pp --less
```

To show a table with some statistics about the results

```bash
tb agg --concise output.jsonl
```

Which should produce the following table

```
Test Case ID                | Runs | Pass | Fail | Error | Success% |
greet_simple.py:test_say_hi |    4 |    4 |    0 |     0 |   100.00 |
```

## Server initialization and state

When a server is started, if a special initialization function exists,
it is invoked before the agent loop.

Since one server process is started per conversation, a state related to the current conversation,
and isolated from all others, can be kept in the global scope.

A special "get effects" function can also be defined, to retrieve information that is needed
in the test.

A teardown function can be defined, it will be called before terminating the server
(or before terminating the MCP session on remote servers).


Special functions:
- `__reserved__init`
- `__reserved__geteffects`
- `__reserved__teardown`

`thinkingbox/tools/mcp_greet.py`

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

Now initialization is required, and the configuration dictionary must contain the "greeting"
key. Also, update the scenario.

`dataset/scenario/greet.yaml`

```yaml
world_state:
    greet:
        greeting: "Hello!"
tools:
- name: say_hi
bot_instructions: "It is important that you use the say_hi function when instructed to 'Say Hi'"
```

We can test the new server with `tb tui`

```bash
tb tui -c config/config_o4mini.yaml -d ./dataset -a think --name greet_simple.py:test_say_hi
```

The tool should have responded "Hello!" now.

Check the "effects" by typing "/effects" as user prompt in TUI, it should print the following:

```yaml
greet:
  count: 1
```

This is exactly the `x.effects` dictionary inside the test function. Now change the testcase
function.

`dataset/test_case/greet_simple.py`

```python
from thinkingbox.common import Judge, TestContext

def test_say_hi(x: TestContext, judge: Judge):
    """!
    scenario: greet
    query: Please call the Say Hi function, and tell me what is its response.
    """
    assert x.effects["greet"]["count"] == 1
```

Run the agent loop and test with `tb infer`, and also save the "effects" dictionary to
the ouput file.

```bash
tb infer -c config/config_o4mini.yaml -d ./dataset -a think --name greet_simple.py:test_say_hi --dump testcontext -o output.yaml
tb pp --less output.yaml
```

## LLM judge

The `judge` available in testcase functions allows to interact with a LLM judge, which can be
used, for example, to "ask" simple natural language questions about a piece of text.

`dataset/test_case/greet_simple.py`

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

Run and test

```bash
tb infer -c config/config_o4mini.yaml -d ./dataset -a think --name greet_simple.py:test_say_hi --dump testcontext -o output.yaml
tb pp --less output.yaml
```

## Simulated user

A test case can contain additional context, in natural language, for a simulated user.
If this context exists, the user will respond to the agent until the agent
declares that it has finished.


`dataset/test_case/greet_simple.py`

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

Note: this may not always pass, as the agent could sometimes decide to terminate the conversation.

Run and test

```bash
tb infer -c config/config_o4mini.yaml -d ./dataset -a think --name greet_simple.py:test_say_hi_user --dump testcontext -o output.yaml
tb pp --less output.yaml
```

Check the conversation, it should contain something similar to the following

```
[assistant::text]
Sure—could you please describe the task you have in mind and any specific requirements or goals?

[user::text]
I want to call the Say Hi function and see what it responds.
```

For a description of possible fields in a test case, see [Test Case Format](test_case_format.md)

# Debug a test case

You can use `tb run-test` to run the test only, on the output of `tb infer`, provided that the test context is in the output file (`--dump testcontext`)

```bash
tb infer -c config/config_o4mini.yaml -d ./dataset -a think --name greet_simple.py:test_say_hi --dump testcontext -o output.yaml
# test and print the result without saving it
tb run-test -c config/config_o4mini.yaml --name greet_simple.py:test_say_hi --resultfile output.yaml -o /dev/null --verbose
```

To use a debugger, see [Debugging Tests](debugging_tests.md)
