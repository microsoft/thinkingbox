# ThinkingBox Onboarding

Agentic testing framework for LLM tool-use. Tests **actions** (tool calls) and **effects** (state changes), not just text output.

*Recommendations:*
* Use WSL
* Use [uv](https://docs.astral.sh/uv/guides/projects/)

---

## Two Repositories

```
AI.ThinkingBox           AI.ThinkingBox.Data
─────────────────────    ─────────────────────────────
Framework code           Your scenarios & test cases
• CLI (tb command)       • dataset/scenario/*.yaml
• session_proxy          • dataset/test_case/*.py
• MCP tool servers       • dataset/agent/*.yaml
• Core libraries         • support/ (embeddings, KBs)
```

**You write tests in AI.ThinkingBox.Data**, the framework lives in AI.ThinkingBox.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  tb infer (CLI Process)                                                     │
│  ─────────────────────                                                      │
│  1. Load config, test case, scenario                                        │
│  2. Create LLM sessions (agent, user, judge)                                │
│  3. Connect to session_proxy, create session                                │
│  4. Run agent loop (decode_turn_iter)                                       │
│  5. Retrieve effects, run test assertions                                   │
└──────┬─────────────────────────────────────┬────────────────────────────────┘
       │                                     │
       │ LLM API calls                       │ HTTP to session_proxy
       │ (agent reasoning,                   │ (tool calls, effects)
       │  user simulation,                   │
       │  judge evaluation)                  │
       ▼                                     ▼
┌──────────────────┐              ┌─────────────────────────────────────────┐
│  Azure OpenAI    │              │  session_proxy (:7111)                  │
│  or Anthropic    │              │  ─────────────────────                  │
│                  │              │  POST /session_create  → spawn servers  │
│  - Agent LLM     │              │  POST /list_tools      → get schemas    │
│  - User LLM      │              │  POST /call_tool       → execute tool   │
│  - Judge LLM     │              │  POST /get_effects     → retrieve state │
└──────────────────┘              │  POST /session_destroy → cleanup        │
                                  └──────────────────┬──────────────────────┘
                                                     │
                                                     │ stdio (JSON-RPC)
                                                     │ one process per server
                                                     ▼
                                  ┌─────────────────────────────────────────┐
                                  │  MCP Server Processes                   │
                                  │  ───────────────────                    │
                                  │  mcp_online_banking.py → account state  │
                                  │  mcp_email.py          → sent emails    │
                                  │  mcp_cloud_drive.py    → file storage   │
                                  │  mcp_slack.py          → messages       │
                                  │  ... (25+ tool servers)                 │
                                  │                                         │
                                  │  Each server has:                       │
                                  │  - __reserved__init (setup state)       │
                                  │  - tool functions (get_accounts, etc)   │
                                  │  - __reserved__geteffects (for testing) │
                                  └─────────────────────────────────────────┘
```

### Data Flow: Single Tool Call

```
Agent LLM returns: ToolCall(name="get_accounts", args={})
        │
        ▼
decode_turn_iter() calls mcp_proxy.call_tool("get_accounts", {})
        │
        ▼
MCPProxyClient POST /call_tool ──► session_proxy
        │                                 │
        │                                 ▼
        │                         ToolDispatcher routes to server
        │                                 │
        │                                 ▼
        │                         mcp_online_banking (JSON-RPC)
        │                                 │
        │                                 ▼
        │                         get_accounts() executes
        │                                 │
        ◄─────────────────────────────────┘
        │                         result: '{"accounts": [...]}'
        ▼
ToolResponse added to conversation, yielded
        │
        ▼
Agent LLM sees tool result, continues reasoning
```

## Key Concepts

| Concept | What it is | Location |
|---------|------------|----------|
| **Agent** | LLM + system prompt + model config | `dataset/agent/*.yaml` |
| **Scenario** | Tools available + initial world state | `dataset/scenario/*.yaml` |
| **Test Case** | User message + assertions on effects | `dataset/test_case/*.py` |
| **Effects** | State changes from tool calls | Captured at runtime |

---

## Golden Path: Get Running

### 1. Clone both repos (side-by-side)

```bash
cd ~/src  # or your preferred location
git clone https://github.com/microsoft/AI.ThinkingBox
git clone https://github.com/microsoft/AI.ThinkingBox.Data
```

### 2. Install dependencies

```bash
cd AI.ThinkingBox
uv venv --python 3.12
uv sync --group dev
# This step installs additional MCP tools so they are accessible to the session proxy.
uv pip install --config-settings editable-mode=compat -e ../AI.ThinkingBox.Data/servers/thinkingbox_tools
```

**Check:** `uv run tb --help` shows command list

### 3. Configure LLM access

```bash
az login  # for Azure OpenAI
```

**Check:** `az account show` returns your subscription

### 4. Set environment variable

```bash
export THINKINGBOX_DATA="path/to/AI.ThinkingBox.Data"
```

**Check:** `echo $THINKINGBOX_DATA` shows the path

### 5. Start the MCP proxy (keep running in Terminal 1)

```bash
uv run tb mcp-start --servers $THINKINGBOX_DATA/servers/servers.yaml
```

**Check:** Output shows `Listening on http://127.0.0.1:7111`

### 6. Run a test (Terminal 2)

```bash
cd ~/src/AI.ThinkingBox
uv run tb infer \
  -c config/config_o4mini.yaml \
  -a think \
  -d ../AI.ThinkingBox.Data/dataset \
  --name banking.py:test_get_balance_savings \
  -o /tmp/output.yaml
```

**Check:** Output file created, no errors

### 7. View the result

```bash
uv run tb pp /tmp/output.yaml
```

**Check:** Conversation shows tool calls and responses

---

## First Task: Modify a Test Case

**Objective:** Change an assertion in `AI.ThinkingBox.Data`, run it, verify it passes.

**Acceptance criteria:**
- [ ] Test passes with modified assertion
- [ ] You understand what `x.effects` contains

### Steps

1. Open `AI.ThinkingBox.Data/dataset/test_case/banking.py`
2. Find `test_get_balance_savings`
3. Note the assertion on `x.effects`
4. Run the test with TUI to see effects interactively:

```bash
uv run tb tui \
  -c config/config_o4mini.yaml \
  -a think \
  -d ../AI.ThinkingBox.Data/dataset \
  --name banking.py:test_get_balance_savings
```

> **TUI tip:** Press **ESC** then **Enter** to submit messages (not just Enter).

5. Type `/effects` to see the effects dictionary
6. Type `/test` to run the test assertions
7. Type `/quit` to exit

---

## See the Simulated User (User LLM)

Tests with `user_context` simulate a user who responds to the agent. The agent asks clarifying questions, and an LLM generates realistic user responses based on the context.

**Example:** `test_transfer_and_balance` asks the agent to transfer money but doesn't specify the destination account. The simulated user provides it when asked.

```bash
# Run with --dump userllm to capture the User LLM prompts
uv run tb infer \
  -c config/config_o4mini.yaml \
  -a think \
  -d ../AI.ThinkingBox.Data/dataset \
  --name banking.py:test_transfer_and_balance \
  --dump userllm \
  -o /tmp/userllm_demo.yaml
```

```bash
# View the conversation (includes simulated user turns)
uv run tb pp /tmp/userllm_demo.yaml
```

**What to look for:**
- `[user::text]` messages after the first query are generated by the User LLM
- The agent asks for missing info (e.g., "which account?")
- The simulated user responds based on `user_context` (e.g., "savings account")

**In a test file:**
```python
def test_transfer_and_balance(x: TestContext, judge: Judge):
    """!
    query: I want to transfer $100 from my checking account.
    user_context: |
        You want to transfer to your savings account.
        You don't know your account numbers.
    """
    assert x.effects["banking"]["transfer_count"] == 1
```

---

## Command Index

### AI.ThinkingBox (Framework)

| Intent | Command |
|--------|---------|
| Start MCP proxy | `uv run tb mcp-start --servers <path_to_servers.yaml>` |
| Run single test | `uv run tb infer -c <config> -a <agent> -d <dataset> --name <file.py:test_name> -o <out>` |
| Run all tests in file | `uv run tb infer -c <config> -a <agent> -d <dataset> --inputs <file.py> -o <out.jsonl>` |
| Interactive debug | `uv run tb tui -c <config> -a <agent> -d <dataset> --name <file.py:test_name>` |
| Chat with scenario | `uv run tb tui -c <config> -a <agent> -d <dataset> --scenario <name>` |
| View result | `uv run tb pp <file.yaml>` |
| Aggregate stats | `uv run tb agg <file.jsonl>` |
| Run framework tests | `uv run pytest -v tests` |
| Install pre-commit | `uv run pre-commit install` |

### AI.ThinkingBox.Data (Tests)

| Intent | Command |
|--------|---------|
| Validate tags | `uv run python scripts/validate_tags.py` |
| Start embeddings server | `uv run python -m thinkingbox.services.embeddings_hf_simple --model ./support/models/intfloat/e5-base-v2` |
| Download HF models | `./scripts/download_hf_models.sh` |

### Key Flags

| Flag | Purpose |
|------|---------|
| `-c, --config` | Config file with LLM endpoints |
| `-d, --dataset` | Path to dataset (scenarios, tests, agents) |
| `-a, --agent` | Agent config name (default: `base`) |
| `-n, --name` | Single test: `file.py:test_function` |
| `-i, --inputs` | Test file or directory for batch |
| `-o, --output` | Output file (`.yaml` single, `.jsonl` batch) |
| `-r, --repeat` | Repeat test N times |
| `--dump` | Extra output: `tools`, `testcontext`, `userllm`, `raw` |
| `--no-test` | Skip running assertions |

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Port 7111 already in use` | Stale proxy process | `lsof -ti:7111 \| xargs kill` |
| `ModuleNotFoundError: thinkingbox` | Venv not activated | `uv sync` or `source .venv/bin/activate` |
| `Scenario not found` | Wrong dataset path | Check `-d` points to `AI.ThinkingBox.Data/dataset` |
| `401 Unauthorized` / timeout | Azure auth expired | Run `az login` |
| `FileNotFoundError: support/...` | Missing data files | Set `THINKINGBOX_DATA` env var |
| `Connection refused localhost:7111` | Proxy not running | Start `uv run tb mcp-start` in another terminal |
| `test_case not found` | Typo in test name | Format is `filename.py:function_name` |
| TUI: can't submit message | Wrong key combo | Press **ESC** then **Enter** (not just Enter) |
| Pre-commit fails | Formatting issues | Run `uv run pre-commit run --all-files` |

---

## Where to Get Help

- **Docs:** `AI.ThinkingBox/docs/` - tutorial, test format, debugging
- **Examples:** `AI.ThinkingBox/dataset/test_case/` - working test cases
- **Issues:** File bugs in the GitHub repo

---

## Next Steps

| Topic | Doc |
|-------|-----|
| Create MCP server from scratch | [tutorial.md](tutorial.md) |
| Write effective tests | [writing_effective_tests.md](writing_effective_tests.md) |
| Configure LLM endpoints | [llm_endpoint_config.md](llm_endpoint_config.md) |
| Tools needing extra setup | [tools_with_additional_setup.md](tools_with_additional_setup.md) |
| Debug tests in VSCode | [debugging_tests.md](debugging_tests.md) |
| Organize tests with tags | [AI.ThinkingBox.Data/docs/Adding-tags.md](../../AI.ThinkingBox.Data/docs/Adding-tags.md) |
