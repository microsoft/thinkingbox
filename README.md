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

## Repositories

ThinkingBox is split across two repositories:

- **[thinkingbox](https://github.com/microsoft/thinkingbox)** (this repo) — the
  framework: the `tb` CLI, the MCP Session Proxy, the agent/user/judge loop,
  and the evaluation harness. Ships a single bundled scenario (`cloud_drive`)
  and its matching MCP server (`mcp_cloud_drive.py`) only as an offline
  smoke-test for an install — see [Verify install](#verify-install).
- **[thinkingbox-data](https://github.com/microsoft/thinkingbox-data)** — the
  curated datasets, the MCP tool server packages (under `servers/`, e.g.
  `thinkingbox_tools`, `ms_toloka_servers`), and supporting data files
  (embeddings, knowledge bases, etc.) under `support/`. This is where real
  scenarios, test cases, and tools live; clone it for any non-trivial work.

For tutorial-style worked examples (running scenarios, batch evaluation,
interactive chat against real datasets), see the
[thinkingbox-data README](https://github.com/microsoft/thinkingbox-data#readme).
This README focuses on the framework itself — install, architecture, and
dataset format reference.

## Setup

*"Just give me the sequence of commands"* -> [TL;DR](docs/tldr.md)

ThinkingBox is only tested on Linux. Most of it might work on other systems but we only target Linux (including WSL) at the moment.

### Clone

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

### CLI overview

All actions in ThinkingBox are accessed through one command: `tb`.

```bash
> uv run tb --help
Usage: tb [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  agg         Aggregate metrics from a JSONL file.
  dump-tests  Dump test cases for a given agent and dataset.
  infer       Execute inference for a single test or set of tests.
  mcp-start   Start the MCP Session Proxy.
  pp          Pretty-print a decoded result from a file or stdin.
  run-test    Process decoder results and optionally update or write new...
  sbs         Compare candidate vs baseline JSONL results and report lift...
  tui         Launch the ThinkingBox TUI for a single test case or a...
```

### Verify install

The framework ships a single offline scenario, `cloud_drive`, so you can
sanity-check the install without cloning `thinkingbox-data` first.

In one terminal, start the Session Proxy with no `--servers` flag —
auto-discovery picks up the bundled `mcp_cloud_drive` server:

```bash
uv run tb mcp-start
```

In another terminal, run a single bundled test:

```bash
uv run tb infer -c config/config_o4mini.yaml --dataset ./dataset --agent think \
    --name cloud_drive.py:test_append_some_more_text --output output.yaml
uv run tb pp output.yaml
```

If `tb pp` shows a conversation and the assertions pass, the framework and
your LLM endpoint are wired up. For real scenarios, datasets, and tool
servers, see the
[thinkingbox-data README](https://github.com/microsoft/thinkingbox-data#readme).

### ThinkingBox unit tests

```bash
# run the session proxy with the test MCP servers configuration
uv run tb mcp-start --servers tests/servers.yaml

# run the tests
uv run pytest -v tests
```

## Architecture

This section describes the framework's runtime pieces and their CLI-level
controls. For tutorial-style worked examples, see the
[thinkingbox-data README](https://github.com/microsoft/thinkingbox-data#readme).

### MCP Session Proxy

ThinkingBox interacts with tools through the **MCP Session Proxy** — a
long-running HTTP server that fronts a fleet of MCP tool processes.

It works as follows:
- TB sends a "scenario" initialization to the Session Proxy, which spawns and
  initializes MCP servers as needed, creating a new isolated "session" for
  the current conversation.
- TB requests tool schemas from the Session Proxy.
- TB interacts with tools by sending requests to the Session Proxy.
- TB retrieves side effects from the Session Proxy for judging. This could be
  any change occurring within the session resulting from tool execution.
- TB sends a destroy request to the Session Proxy, which terminates the
  related MCP servers and releases the memory.

Start it with `tb mcp-start`. The choice of `--servers` controls which tool
servers are loaded:

```bash
# Auto-discover the bundled servers under thinkingbox/tools/mcp_*.py
# (only mcp_cloud_drive — useful for the smoke test, nothing else)
uv run tb mcp-start

# Real workloads: point at thinkingbox-data's master servers config
uv run tb mcp-start --servers ../thinkingbox-data/servers/servers.yaml
```

Some tools require additional setup (running services, the
`THINKINGBOX_DATA` environment variable for support files). See
[Tools with additional setup](docs/tools_with_additional_setup.md).

### LLM Configuration

See [LLM Endpoint Configuration](docs/llm_endpoint_config.md) for all the options.

If using Azure OpenAI endpoint, log in with azure-cli (`az login`) and
configure some endpoints you have access to in the main configuration file.
Check the example in `config/config_o4mini.yaml`.

If using OpenAI-Compatible deployments, check the example in
`config/config_vllm.yaml`.

### Interactive TUI

`tb tui` launches an interactive session to chat with a scenario or a test
case. See the
[thinkingbox-data README](https://github.com/microsoft/thinkingbox-data#readme)
for invocation examples; this section covers the TUI's UX details.

**IMPORTANT: Use ESC then ENTER to submit a message, or just ENTER for
newline. This is necessary for multiline input.**

*Note: check the prompt_toolkit documentation for more information, our
instructions are Linux-specific and other platforms have different key
bindings.*

When prompted with `[user::text]`, provide a user response, or one of the
special commands starting with `/`:

```
# run a test from file
/test dataset/test_case/<file>.py:<testname>

# or if chatting with a test case (--name), execute its associated test
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

### Inspecting and aggregating results

Pretty-print individual conversations from the JSONL or YAML output of
`tb infer`:

```bash
uv run tb pp input_file.yaml

# or (first example in a JSONL)
head -n1 input_file.jsonl | uv run tb pp
```

Aggregate results and statistics into a table summary from the JSONL output
of `tb infer`:

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

Contains the agent prompts and configuration.


### Scenario

Schema: `config_types.py:ScenarioConfig`

Location: `<dataset>/scenario/<scenario>.yaml`

Contains the scenario configuration:
- initial state for each server
- list of available tools
- any additional tool configuration


### Test Case

Schema: `config_types.py:TestCase`

Location: `<dataset>/test_case/<test_cases_file>`

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
