# ThinkingBox

## Introduction

ThinkingBox is a framework designed to:
- Define tool mocks as MCP servers.
- Create scenarios and test cases.
- Run an LLM agent and enable interaction with tools.
- Evaluate the outcomes of agent execution.

It can be used to:
- Generate conversations for "offline" LLM training or evaluation.
- Train LLMs with reinforcement learning, using the entire system in the training loop.

It supports:
- Spawning and initializing multiple isolated tool execution environments.
- LLM Agent loop with tool use.
- Interaction with a simulated (LLM) User that responds based on a prompt with added context.

## Setup

*"Just give me the sequence of commands"* -> [TL;DR](docs/tldr.md)

ThinkingBox is only tested on Linux. Most of it might work on other systems but we only target Linux (including WSL) at the moment.

### Repository

Clone this repository

```bash
git clone https://github.com/microsoft/thinkingbox.git
git clone https://github.com/microsoft/thinkingbox-data.git
# OR using GitHub CLI
gh repo clone microsoft/thinkingbox
gh repo clone microsoft/thinkingbox-data
```

### Development Setup

We recommend using `uv` for python management, and python version `3.12`.

```bash
# Install thinkingbox in editable mode with dev dependencies
uv venv --python 3.12
uv sync --group dev

# (for contributors) Install pre-commit hooks
uv run pre-commit install
```

Note that this creates a virtual environment in `.venv` which `uv` uses by default when running from this directory (e.g. `uv run`). Just manually activate this virtual environment in order to use it elsewhere.

```bash
source .venv/bin/activate
```

If use of `uv` is restricted in your environment, you can use pip as an alternative.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools
pip install --config-settings editable-mode=compat -e '.[dev]'
pre-commit install
```

### Pre-commit hooks

**Pre-commit hooks** are configured to automatically format code on commit (check previous section).

This project uses [Black](https://black.readthedocs.io/) for code formatting. A pre-commit hook automatically re-formats changed files on commit. A PR pipeline also enforces that code is correctly formatted before merging into main.

To use pre-commit manually (including the black re-formatter):

```bash
# run pre-commit hooks once on modified files
uv run pre-commit run

# run pre-commit hooks once on all files
uv run pre-commit run --all-files
```

### Running ThinkingBox

All actions in ThinkingBox can be accessed by one command: `tb`.

```bash
> uv run tb --help
Usage: tb [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  agg         Aggregate metrics from a JSONL file.
  dump-tests  Dump test cases for a given agent and dataset.
  infer       Execute inference for a single test or set of tests.
  mcp-start   Start the MCP server.
  pp          Pretty-print a decoded result from a file or stdin.
  run-test    Process decoder results and optionally update or write new...
  sbs         Compare candidate vs baseline JSONL results and report lift...
  tui         Launch the ThinkingBox TUI for a single test case or a...
```

### ThinkingBox unit tests

```bash
# run the session proxy with the test MCP servers configuration
uv run tb mcp-start --servers tests/servers.yaml

# run the tests
uv run pytest -v tests
```

## Run

### MCP Session Proxy

ThinkingBox uses the MCP Session Proxy to interact with tools.

It works as follows:
- TB sends a "scenario" initialization to the Session Proxy, which spawns and initializes MCP servers as needed, creating a new isolated "session" for the current conversation.
- TB requests tool schemas from the Session Proxy.
- TB interacts with tools by sending requests to the Session Proxy.
- TB retrieves side effects from the Session Proxy for judging. This could be any change occurring within the session resulting from tool execution.
- TB sends a destroy request to the Session Proxy, which terminates the related MCP servers and releases the memory.

The Session Proxy must be running while using `uv run tb mcp-start`.

```bash
# Run the session server on 127.0.0.1:7111 (default)
uv run tb mcp-start
```

Note: some tools require additional setup:
- additional running services
- setting environment variable `THINKINGBOX_DATA` for additional data files.

Check [Tools with additional setup](docs/tools_with_additional_setup.md)

Tools that appear in the examples in this repository (in `./dataset`) do not require any additional setup.


### LLM Configuration

See [LLM Endpoint Configuration](docs/llm_endpoint_config.md) for all the options.

If using Azure OpenAI endpoint, log in with azure-cli (`az login`) and configure some endpoints you have access to in the main configuration file. Check the example in `config/config_o4mini.yaml`.

If using OpenAI-Compatible deployments, check the example in `config/config_vllm.yaml`.


### Decode and Test

Single Test

```bash
# Decode one specific case, do not test, dump tools, test context (incl side effects), User-LLM queries
uv run tb infer -c config/config_o4mini.yaml --dataset ./dataset --agent think --name cloud_drive.py:test_append_some_more_text --no-test --dump tools,testcontext,userllm --output output.yaml

# Run test for cloud_drive.py:test_append_some_more_text on test context output.yaml["test_context"]
uv run tb run-test -c config/config_o4mini.yaml --dataset ./dataset --resultfile output.yaml --name cloud_drive.py:test_append_some_more_text --output test_result.yaml

# run test output.yaml["uid"] on test context output.yaml["test_context"] and update output.yaml["test_result"]
uv run tb run-test -c config/config_o4mini.yaml --dataset ./dataset --resultfile output.yaml --update
```

Multiple Tests

```bash
# Decode multiple cases from a file or directory, run tests as well, dump raw messages
uv run tb infer -c config/config_o4mini.yaml --dataset ./dataset --agent think --inputs dataset/test_case/cloud_drive.py --repeat 4 --dump raw --output output.jsonl
```


### Interactive TUI

Start an interactive session to chat with a scenario or a test case.

**IMPORTANT: Use ESC then ENTER to submit a message, or just ENTER for newline. This is necessary for multiline input.**

*Note: check the prompt_toolkit documentation for more information, our instructions are Linux-specific and other platforms have different key bindings.*


```bash
# chat with a scenario
uv run tb tui -c config/config_o4mini.yaml --dataset ./dataset --agent think --scenario cloud_drive --query "Please list all the files"

# chat with a test case but don't send the query
uv run tb tui -c config/config_o4mini.yaml --dataset ./dataset --agent think --name cloud_drive.py:test_append_some_more_text --query ""
```

When prompted with `[user::text]`, provide a user response, or one of the special commands starting with `/`.

Commands:

```
# run a test from file
/test dataset/test_case/cloud_drive.py:test_append_some_more_text

# or if chatting with a test case (--name), to execute its associated test, just run
/test

# show tool definition
/tool get_text_content

# show conversation in raw format
/conversation

# get effects/state from the server
/effects

# exit
/quit
```

### Decoding result visualization and testset aggregation

Print individual conversations with nice formatting for visual inspection, from the JSONL or YAML output of `tb infer`, by running:

```bash
uv run tb pp input_file.yaml

# or (first example in a JSONL)
head -n1 input_file.jsonl | uv run tb pp
```

Aggregate results and statistics into a table summary, from the JSONL output of `tb infer`, by running:

```bash
uv run tb agg input_file.jsonl

# or (for a subset of results)
cat input_file.jsonl | grep "<SOME FILTER>" | uv run tb agg
```


# Configuration and dataset

## Configuration

The configuration file (`--config`, `config_types.py:ConfigFile`) contains:
- MCP session proxy address
- LLM service configurations

See examples in `config/config_o4mini.yaml`, `config/config_vllm.yaml`.

## Dataset

There are 3 types of objects in the dataset: Agent, Scenario, Test Case.


### Agent

Schema: `config_types.py:AgentConfig`

Location: `<dataset>/agent/<agent>.yaml`

`--dataset ./dataset --agent think` selects file `./dataset/agent/think.yaml`

Contains the agent prompts and configuration.


### Scenario

Schema: `config_types.py:ScenarioConfig`

Location: `<dataset>/scenario/<scenario>.yaml`

`--dataset ./dataset` + `scenario: cloud_drive` selects file `./dataset/scenario/cloud_drive.yaml`

Contains the scenario configuration:
- initial state for each server
- list of available tools
- any additional tool configuration


### Test Case

Schema: `config_types.py:TestCase`

Location: `<dataset>/test_case/<test_cases_file>`

`--dataset ./dataset --name cloud_drive.py:test_delete` selects `test_delete` in `./dataset/test_case/cloud_drive.py` or `./dataset/test_case/cloud/cloud_drive.py`

A test case file contains multiple test cases. There are 2 possible equivalent formats:
- python format: described in `python_test_file.py`
- YAML format: schema `config_types.py:TestCaseFile`

A test case contains:
- uid: unique identifier `<filename>:<testname>`
- User query and User-LLM context
- test code

#### Create YAML of test cases
You can get a full list of the testcases in a certain file or directory, or of the useful testcases from a benchmarking run jsonl, by using the scripts in `scripts/dataset_utils/`.

## Third-party code

This repository does not vendor third-party source code. All runtime and development dependencies are declared in `pyproject.toml` / `uv.lock` and installed from public package indexes (PyPI). Each dependency retains its own license.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies.
