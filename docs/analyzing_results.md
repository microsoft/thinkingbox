# Analyzing results: from `tb infer` output to paper-style metrics

This guide documents the complete, end-to-end workflow for turning `tb infer`
output into the metrics reported in the ThinkingBox paper (pass@1, pass@20,
pass^20), plus how to drill into individual failures. Every command, flag,
and field named below is read directly from the current implementation:

- `thinkingbox/cli/infer.py` — batch/single-test decoding, writes JSONL/YAML
- `thinkingbox/cli/agg_main.py` — aggregation, metrics, tables
- `thinkingbox/cli/show_main.py` (`tb pp`) — pretty-printing one result
- `thinkingbox/cli/runtest_main.py` (`tb run-test`) — re-run assertions against a stored result
- `thinkingbox/cli/sbs_main.py` (`tb sbs`) — baseline vs. candidate comparison
- `thinkingbox/common/chat_types.py` — the `DecodeResult` / `TestResult` / `TestContext` schemas
- `thinkingbox/common/eval_utils.py` — the Beta/credible-interval statistics used by `tb agg` and `tb sbs`

Nothing below is invented: where the code and the paper's terminology
diverge (notably `pass^k`), this guide says so explicitly instead of
papering over it.

## 1. The `tb infer` JSONL artifact

### 1.1 Row granularity

Batch `tb infer` runs (`--inputs <dir_or_file>`, `--test-list <file>`, or
`--name` with `--repeat > 1`) write **one JSON object per line** to the file
given by `--output`/`-o`. Each line is one *attempt*: one `(test case, repetition)`
pair. If you run 30 test cases with `--repeat 20`, the output file has 600
lines. Rows are written in the same order the test cases are read, and results
from `--repeat` are interleaved as `repeat_test_cases()` emits `(tc, i)` for
`i in range(repeat)` per test case before moving to the next
(`thinkingbox/cli/agg_main.py:306`).

If `--output` ends in `.yaml`, a single result is written as YAML instead of
JSONL (`WriterYAML`, only valid for a single test case — it raises if you try
to dump more than one object). JSONL is the format used for batch/benchmark
runs and is what `tb agg` and `tb sbs` consume.

A companion file `<output_stem>_run_metadata.yaml` is always written next to
the output, capturing the config used (`setup_config`, `agent_config`,
`user_config`), the UTC start time, and the total run duration in seconds
(`async_main()` in `infer.py:399-419`). This is not part of the row-per-line
JSONL and is not read by `tb agg`.

### 1.2 Core schema (`DecodeResult`)

Each JSONL line deserializes as `chat_types.py:DecodeResult`:

| Field | Type | Meaning |
|---|---|---|
| `uid` | `str` | `<test_file>:<test_name>`, e.g. `cloud_drive.py:test_append_some_more_text` |
| `messages` | list of `Text`/`ToolCall`/`ParallelToolCall`/`ToolResponse` | the full agent/user/tool conversation |
| `test_result` | `TestResult \| None` | assertion outcome; `None` if `--no-test` or `--skip-agent` without a test |
| `test_context` | `TestContext \| None` | effects/tool-call log; **only populated when `--dump testcontext` is passed** (see §1.4) |
| `test_tags` | `TestCaseTags \| None` | `domain`, `eval_type`, `category`, `labels`, `skip` — always present, taken from the test case's tags |
| `tools` | list of `ToolDef` \| `None` | tool schemas exposed to the agent; only populated with `--dump tools` |
| `raw_messages` | list of `dict` \| `None` | provider-native message log; only populated with `--dump raw` |
| `user_llm_history` | list of message lists \| `None` | the simulated user's own LLM turns; only populated with `--dump userllm` |
| `usage` | list of `Usage` \| `None` | one entry per LLM call (see §1.5) |
| `metadata` | `dict` | repetition index, timers, execution time, errors (see §1.5/§1.6) |
| `is_system_error` | `bool` | `True` if decoding itself raised (agent/tool/proxy crash) — the row is expected to be incomplete |
| `finish_reason` | one of `done`, `end_turn_tool`, `agent_error`, `agent_limit`, `user_limit`, `user_done`, `no_user_llm`, `skipped` | why the conversation loop stopped |

### 1.3 Conversation / tool-trace messages

`messages` is a list of a tagged union (`chat_types.py:115`):

- `Text{role: system|user|assistant, content, metadata.tag: text|think|direct}` —
  a chat turn. `tag == "think"` marks a hidden reasoning message
  (`is_visible == False`); `tag == "direct"` marks text injected from a
  formatted tool response.
- `ToolCall{name, arguments, id}` — a single tool invocation requested by the
  agent.
- `ParallelToolCall{tool_calls: list[ToolCall]}` — one or more `ToolCall`s the
  agent issued in the same turn.
- `ToolResponse{name, content, id, metadata.direct_response}` — the tool's
  result. `metadata.direct_response`, if present, is a canned assistant
  message the framework injected in response to that tool call.

`--dump tools` adds the tool schemas the agent saw (`ToolDef{name,
description, input_schema, direct_response, is_end_turn}`) to `tools`.
`--dump raw` adds the raw, provider-native request/response dicts to
`raw_messages` (useful for debugging exact prompts sent to the LLM API).
`--dump userllm` adds the simulated user's own internal LLM turns to
`user_llm_history`.

### 1.4 Effects / final state (`test_context`)

`test_context` (`chat_types.py:200`, `TestContext`) is the object test code
runs assertions against. It is **always built internally** whenever a test is
about to run (or when `--skip-agent` is combined with a test), but it is only
**written to the output row** when you pass `--dump testcontext`
(`infer.py:230`: `if not self.dump_testcontext: result.test_context = None`).
Fields:

- `effects: dict[str, Any]` — per-MCP-server side effects, i.e. the return
  value of each server's reserved `__reserved__geteffects` function. This is
  the final observable state of the world (files created, emails sent,
  account balances, etc.).
- `tool_calls: list[ToolCallResponse]` — every `(tool_call, tool_response)`
  pair, independent of the free-form conversation transcript.
- `response`, `tool_direct_responses`, `messages`, `init_result`,
  `session_id`, `metadata` — the last assistant message, any direct tool
  responses, the raw conversation, the per-server init state, the session
  proxy session id, and metadata written by fixtures/test code during
  evaluation.

If you need `effects` in your benchmark JSONL for post-hoc inspection
(e.g. to manually classify "Wrong State Update" failures — see §7), you must
run with `--dump testcontext`; it is not there by default.

### 1.5 Test results / assertions

`test_result` (`chat_types.py:247`, `TestResult`) is populated whenever a test
ran (`tc.test_code` is set and `--no-test` was not passed):

- `result: bool` — `True` if all assertions passed.
- `reward: float` (0.0–1.0) — reward value (relevant for rubric/judge-scored
  tests; binary asserts give 0.0/1.0).
- `is_system_error: bool` — `True` if the *test code itself* raised something
  other than an `AssertionError` (a bug in the test, not a real failure) —
  `result` is meaningless in that case.
- `tb`, `lineno`, `line_content` — traceback and the failing line, when
  `result is False`.
- `prints` — anything the test printed.
- `metadata.judge_motivation` — when an LLM judge (`Judge`) was used, a list
  of `{question, answer, motivation}` records explaining each rubric verdict.

### 1.6 Usage / latency metadata

- `usage: list[Usage]` — one entry per LLM call made by the agent during that
  attempt (`usage_types.py:Usage`): `input_tokens`,
  `input_tokens_details.cached_tokens`, `output_tokens`,
  `output_tokens_details.reasoning_tokens`, `total_tokens`. Cached and
  reasoning token counts are populated only when the backend reports them;
  otherwise they default to `0`.
- `metadata.execution_time` — wall-clock seconds for the whole attempt,
  measured by `TBWorker.work()` (`thinkingbox/cli/infer.py:126-128`) around
  decode + test.
- `metadata.time_agent`, `metadata.time_user`, `metadata.time_tool`,
  `metadata.time_test`, `metadata.time_agent_user_loop` — sub-timers
  accumulated with `Timers.measure()` (`thinkingbox/common/utils.py:96`):
  time inside the agent LLM call, the user-simulator LLM call, tool
  invocations, running the test, and the whole agent/user loop respectively.
- On a system error, `metadata.error = {"type", "message", "tb"}"`
  (`ErrorInfo`).

### 1.7 Repetitions

`--repeat N` (`-r`) runs each selected test case `N` times. Each repeated copy
gets `metadata["repetition"] = i` for `i in range(N)`
(`repeat_test_cases()`, `agg_main.py:306-312`). This is the field `tb agg`
and the pass@k/pass^k math rely on to know how many independent attempts
exist per test case (see §3). All rows for a given `uid` in one output file
must share the same repetition count for the pass@k/pass^k columns to be
computed (see §3.3).

### 1.8 Interrupted / resumed runs

There is no checkpoint/replay flag baked into `tb infer` beyond
`--previous-results-file`, and the run does not pause and resume on its own.
What actually happens:

- **Stalled batches**: a watchdog (`--timeout`, default 900s, `0` disables
  it) tracks how long it has been since the last attempt *finished*. If
  nothing completes for longer than `--timeout`, every attempt still
  in-flight is cancelled and written out as a system-error row
  (`metadata.error = {"type": "asyncio.CancelledError", "message": "Task
  cancelled"}`, `TBWorker.get_error_result()`), and the executor moves on to
  the remaining queued items (`thinkingbox/common/ordered_parallel_executor.py`).
  No user action is needed for this case — the run finishes normally and the
  stalled attempts simply show up as system-error rows you can retry (e.g.
  with `--previous-results-file`).
- **Whole-process interruption** (Ctrl+C, crash, node preemption): the output
  file is opened in text-write mode and flushed after every line
  (`WriterJSONL.write()`), so everything decoded before the interruption is
  safely on disk — but the file is **not** closed/finalized cleanly, and
  re-running the same command truncates it (`open(output, "w", ...)`).
  To resume:
  1. Copy the partial output aside, e.g. `cp output.jsonl output.partial.jsonl`.
  2. Re-run the exact same `tb infer` invocation (same `--name`/`--inputs`/
     `--test-list`, same `--repeat`), adding
     `--previous-results-file output.partial.jsonl -o output.jsonl`.
  3. `iter_with_previous_result()` (`thinkingbox/cli/common.py:86`) walks the
     newly-generated test case stream and the previous JSONL together,
     matching on `(metadata["repetition"], uid)`. Any row whose previous
     result exists **and is not a system error** is reused verbatim (no LLM
     calls); any row missing from the previous file, or previously a system
     error, is decoded fresh. Rows from the previous file that don't match a
     currently-generated test case are skipped with a warning (they're
     assumed stale).

This behavior is covered by
`tests/test_tb_cli.py::test_previous_results_file_reruns_system_errors`.

## 2. `tb agg`: aggregating a JSONL file

### 2.1 Exact syntax

```
tb agg [OPTIONS] [INPUT]

  Aggregate metrics from a JSONL file.
  INPUT should be a file in JSONL (multiline) format, or '-' to read from
  standard input.

Options:
  --concise                    Only print Pass/Fail stats
  -f, --output-format [table|json]
                                Output format  [default: table]
```

(`thinkingbox/cli/agg_main.py:476-489`)

`INPUT` defaults to `-` (stdin), so both of these work:

```bash
uv run tb agg output.jsonl
cat output.jsonl | uv run tb agg
```

There is **no CSV or Markdown output option** — only `table` (a Rich console
table, human-readable) and `json` (machine-readable). There is no built-in
`--domain`/`--filter` flag on `tb agg` itself — filtering happens upstream on
the JSONL (see §5).

### 2.2 Concise vs. full table output

Without `--concise`, `tb agg` prints one row per test-case `uid` with these
columns (`PerTestCaseTableRow`, `agg_main.py:109-174`):

`Test Case ID`, `Runs`, `Pass`, `Fail`, `Error`,
`Avg-Ast(min,max)` (assistant turns), `Avg-Usr(min,max)` (user turns),
`Avg-TC(min,max)` (tool-call turns), `Avg-OutTkns(min,max)` (output tokens),
`Avg-ReaTkns(min,max)` (reasoning tokens), `Avg-CharLength(min,max)`,
`Avg-ExecTime(min,max)` (seconds, 2 decimals), `Success%`, `GZ@95%`,
`P(GZ|Data)`.

With `--concise`, only the columns flagged `concise=True` are shown:
`Test Case ID`, `Runs`, `Pass`, `Fail`, `Error`, `Success%`, `GZ@95%`,
`P(GZ|Data)`.

`GZ@95%`/`P(GZ|Data)` are the "Goldilocks Zone" columns: `P(GZ|Data)` is
`P(0.0625 < true_pass_rate < 0.9375 | data)` under a Beta(0.5, 0.5) prior
(`prob_in_zone()`, `eval_utils.py:23`), and `GZ@95%` is `YES` when that
probability is ≥ 0.95. This is a benchmark-design signal for whether a test
case's difficulty is well-calibrated (neither trivial nor unwinnable given
the observed data) — it is **not** one of the paper's reported metrics; it's
independent tooling that happens to live in the same table.

After the per-test table, `tb agg` (table format) prints the aggregate block
(`print_metrics()`, `agg_main.py:456-473`):

```
Number of tests: <N>
Runs per test: <k or "(multiple)">
Total runs: <total>
Mean per-sample accuracy: <mean_pass:.2f> (95% CI: <low:.2f>-<high:.2f>)
Pass@k:
  pass@1: <value>
  pass@5: <value>
  pass@10: <value>
  pass@20: <value>
  ...
Pass^k:
  pass^1: <value>
  pass^5: <value>
  ...
```

The `k` values shown are the subset of `PASS_METRICS_K = [1, 5, 10, 20, 50,
100]` (`agg_main.py:177`) that are `<= runs_per_test`. `Pass@k`/`Pass^k`
blocks are omitted entirely if `runs_per_test` differs across test cases
(see §3.3), or if no `k` in that list is achievable.

### 2.3 JSON output

`tb agg -f json` prints a single JSON object to stdout
(`agg_main.py:509-515`):

```json
{
  "per_test": [ /* one object per PerTestCaseTableRow, all fields (not just concise ones) */ ],
  "metrics": {
    "num_tests": 30,
    "runs_per_test": 20,
    "total_runs": 600,
    "mean_pass": 0.55,
    "mean_pass_ci_low": 0.51,
    "mean_pass_ci_high": 0.59,
    "unbiased_pass_at_k": [[1, 0.55], [5, 0.78], [10, 0.86], [20, 0.92]],
    "unbiased_pass_power_k": [[1, 0.55], [5, 0.14], [10, 0.03], [20, 0.004]]
  }
}
```

`unbiased_pass_at_k` / `unbiased_pass_power_k` are lists of `[k, value]`
pairs (Pydantic serializes the `list[tuple[int, float]]` field as nested
arrays). `runs_per_test` is `-1` if it varies across test cases (see below).

## 3. Metrics, aligned with the ThinkingBox paper's terminology

### 3.1 pass@1

pass@1 is simply the mean success rate over independent attempts. In `tb
agg` output this is the `k=1` entry of the `Pass@k` block (`pass@1: ...`),
equivalently `unbiased_pass_at_k[0]` in JSON. It also numerically matches
`Mean per-sample accuracy` whenever every test case has the same number of
runs (the normal case with `--repeat N`): `mean_pass = sum(c_i) / sum(n_i)`,
and when all `n_i == n`, that equals `mean_i(c_i / n) = pass@1`.
`Mean per-sample accuracy` additionally reports a 95% credible interval
(`cred_int()`, Beta(0.5, 0.5) prior) that the `pass@k`/`pass^k` figures do
not.

### 3.2 pass@k (unbiased estimator) — pass@20

`pass_at_k_unbiased(n, c, k)` (`agg_main.py:191-222`) implements exactly the
unbiased pass@k estimator from the paper (and from the Codex/HumanEval
literature):

```
pass@k = 1 - C(n - c, k) / C(n, k)
```

where `n` is the number of independent attempts for a test case, `c` is the
number of those attempts that passed, and `C(a, b)` is the binomial
coefficient. This is the probability that **at least one** of `k` attempts
drawn (without replacement) from the `n` observed attempts is a pass. It
requires `k <= n`; if `n - c < k` (more failures than there is room for
misses) the code short-circuits to `1.0`. `tb agg` computes this **per test
case**, then reports the plain mean across test cases for each `k` in
`Pass@k:`.

**pass@20** is exactly this formula with `k=20`. To get it, you need
`runs_per_test >= 20` for every test case — i.e. run `tb infer` with
`--repeat 20` (or more) uniformly. This matches the paper's experimental
setup: 20 attempts per task.

### 3.3 pass^k ("all-attempts") — pass^20, and a naming caveat

`pass_power_k_unbiased(n, c, k)` (`agg_main.py:225-249`) computes:

```
pass^k = (c / n) ** k
```

This is the probability that **all** `k` independently-sampled attempts pass,
*if* each attempt is resampled independently with pass probability equal to
the observed rate `c/n` (a plug-in/naive Bernoulli model). **Despite the
function and field being named `..._unbiased` / `unbiased_pass_power_k`,
`(c/n)**k` is the biased plug-in (maximum-likelihood) estimator, not the
combinatorial unbiased estimator for "all k of the n observed attempts
without replacement pass"**, which would be `C(c, k) / C(n, k)` (the
`pass@k`-style hypergeometric analogue for the "all pass" event instead of
"at least one passes"). The paper is aware of this: it reports the biased
plug-in `(c/n)^k` as `pass^k` and explicitly does not report the unbiased
`C(c,k)/C(n,k)` form, because that estimator is degenerate (exactly `0`)
whenever `c < k`, which loses resolution for most tasks at `k=20`. `tb agg`'s
implementation matches what the paper reports numerically, but if you are
reading the code, do not trust the `unbiased` naming for this particular
metric — it is not the same "unbiased" as `pass@k`'s.

**pass^20** is `(c/n) ** 20` per test case, averaged across test cases, shown
in the `Pass^k:` block once `runs_per_test >= 20`.

### 3.4 pass@k vs. pass^k — the distinction

| | Formula | Question it answers | Behavior as `k` grows |
|---|---|---|---|
| `pass@k` | `1 - C(n-c, k) / C(n, k)` | "If I only get to submit `k` of these attempts, what's the chance at least one succeeds?" | Monotonically increases toward 1 |
| `pass^k` | `(c / n) ** k` | "If I need every one of `k` independent attempts to succeed (e.g. reliability/consistency), what's that probability?" | Monotonically decreases toward 0 |

They are not different names for the same quantity: `pass@20` measures
best-of-20 reliability (how good is the model if it gets 20 tries and only
one needs to work), while `pass^20` measures worst-case/all-attempt
reliability (how consistently would the model succeed if it had to pass 20
times in a row). A model can have high pass@20 and very low pass^20 if it is
capable but inconsistent.

## 4. Reproducible command sequence: a 20-repeat benchmark

```bash
# 1. Start the MCP session proxy in one terminal (point --servers at your
#    dataset's servers.yaml for real datasets; omit for the bundled smoke test)
uv run tb mcp-start --servers ../thinkingbox-data/servers/servers.yaml

# 2. Run 20 repeats of every test case in a directory/file, dumping raw
#    provider messages for later debugging, batching 20 attempts concurrently
uv run tb infer -c config/config_o4mini.yaml -d ../thinkingbox-data/dataset \
    --agent think -i ../thinkingbox-data/dataset/test_case \
    --repeat 20 --batch-size 20 --dump raw -o results_20x.jsonl

# 3. Aggregate: human-readable table with per-test pass@k/pass^k rollups
uv run tb agg results_20x.jsonl

# 4. Or get machine-readable numbers (e.g. to script pass@20 / pass^20 out)
uv run tb agg -f json results_20x.jsonl > results_20x.metrics.json
python -c "
import json
m = json.load(open('results_20x.metrics.json'))['metrics']
by_k = dict(m['unbiased_pass_at_k'])
by_k_pow = dict(m['unbiased_pass_power_k'])
print('pass@1 :', by_k[1])
print('pass@20:', by_k[20])
print('pass^20:', by_k_pow[20])
"
```

For a single test case repeated 20 times instead of a whole directory, use
`--name <file>:<test>` in place of `-i`. `--test-list <file.yaml>` runs a
YAML list of `filename:testname` uids instead (useful with the uid lists
produced by `scripts/dataset_utils/create_tc_list.py`).

## 5. Domain / split filtering

`tb infer`/`tb agg` have no built-in `--domain` flag. Every output row does
carry `test_tags.domain` (and `test_tags.eval_type`, `test_tags.category`,
`test_tags.labels`) though, taken straight from the test case's tags, so you
filter the JSONL before piping it into `tb agg` with any standard JSONL tool.

Using `jq`:

```bash
# Keep only rows tagged domain: banking, then aggregate
jq -c 'select(.test_tags.domain == "banking")' results_20x.jsonl \
    | uv run tb agg -

# Split by scenario (the part of uid before ':')
jq -c 'select(.uid | startswith("cloud_drive.py:"))' results_20x.jsonl \
    | uv run tb agg -
```

Using Python (no extra dependency beyond the stdlib):

```bash
python -c "
import json, sys
for line in open('results_20x.jsonl'):
    row = json.loads(line)
    if (row.get('test_tags') or {}).get('domain') == 'banking':
        print(line, end='')
" | uv run tb agg -
```

Both approaches work because `tb agg` accepts `-` (stdin) as `INPUT`
(`agg_main.py:477-479`). This is the same technique the README's
"Inspecting and aggregating results" section shows for arbitrary filters
(`cat input_file.jsonl | grep "<SOME FILTER>" | uv run tb agg`) — `jq` is
just a more precise, JSON-aware filter than `grep`.

## 6. Inspecting individual failures

### 6.1 `tb pp` — pretty-print one result

```
tb pp [OPTIONS] [INPUT]

  Pretty-print a decoded result from a file or stdin.
  INPUT is the path to a file to decode, or '-' to read from stdin.

Options:
  --less           Pipe output through `less -R` for paging.
  --line INTEGER   Line number to read (1-based).
```

(`thinkingbox/cli/show_main.py:100-112`)

`tb pp` accepts a single-result YAML file, a single-line JSONL file, or a
multi-line JSONL file combined with `--line N` to pick one row:

```bash
# Whole-file YAML result (e.g. from --name with a .yaml -o path)
uv run tb pp output.yaml

# First line of a batch JSONL
head -n1 results_20x.jsonl | uv run tb pp

# A specific attempt out of a big batch file, without loading it all
uv run tb pp --line 137 results_20x.jsonl

# Page long output
uv run tb pp --less --line 137 results_20x.jsonl
```

It prints, in order: the `uid` and pass/fail/reward summary, the full
conversation (`Text`/`ToolCall`/`ToolResponse` messages, color-tagged by
role — `think` reasoning is shown distinctly from visible `assistant` text),
`test_context` if present in the row (only if the run used `--dump
testcontext`), and the `TestResult` (including judge motivations and the
failing assertion line, if any). This is the primary way to read the raw
evidence behind a failure — messages, tool calls/responses, and (when
dumped) effects.

### 6.2 `tb run-test` — re-run assertions against a stored result

```
tb run-test -c CONFIG -r RESULTFILE (--update | -o OUTPUT) [OPTIONS]

Options:
  -d, --dataset PATH     Dataset root directory  [default: ./dataset]
  -n, --name TEXT        Test case name (defaults to the result's uid)
  --update               Update RESULTFILE in place
  -v, --verbose          Print the test result
  -o, --output PATH      Write only the new test_result to a separate file
  --debug-test           Import the test file directly (for debugging)
```

(`thinkingbox/cli/runtest_main.py:137-198`)

This re-executes the test code (optionally a *different* test, via `--name`)
against a previously-recorded `test_context` from a YAML result file
(`test_context` must be non-null — i.e. that result was produced with
`--dump testcontext`, or came from `tb infer -o result.yaml` on a single test
which always stores it). Useful for iterating on assertion/rubric logic
without re-running the agent:

```bash
uv run tb run-test -c config/config_o4mini.yaml -d ./dataset \
    -r output.yaml --update --verbose
```

Note `tb run-test` currently takes a single YAML result file
(`-r/--resultfile`), not a JSONL batch — to re-check one attempt out of a
JSONL batch, first extract that line into its own YAML/JSON file (e.g. with
`tb pp --line N` for a human view, or `sed -n 'Np'`/`jq` to extract the raw
line) before pointing `-r` at it.

### 6.3 `tb sbs` — side-by-side comparison

```
tb sbs -b BASELINE -c CANDIDATE [--pct-lift FLOAT]
```

(`thinkingbox/cli/sbs_main.py:111-135`)

`-b/--baseline` and `-c/--candidate` are each a JSONL file of `DecodeResult`s
(same shape as `tb infer` batch output). `tb sbs` aggregates both per test
case (reusing `aggregate_results_per_test` from `tb agg`) and prints one row
per shared `uid` with:

- `Baseline`/`Candidate` run/pass/fail/error counts.
- `Baseline Success%`/`Candidate Success%`.
- `P(pCand > pBase)` — exact posterior probability the candidate's true pass
  rate exceeds the baseline's, via Gauss-Legendre quadrature over Beta
  posteriors (`prob_A_gt_B()`, `eval_utils.py:66`).
- `P(pCand - pBase) > pct_lift` — Monte Carlo estimate (200k samples) of the
  probability the candidate beats the baseline by more than `--pct-lift`
  (default `0.03`, i.e. 3 percentage points) (`prob_lift_gt_eps()`,
  `eval_utils.py:87`).

Test cases where both baseline and candidate are 100% system errors are
skipped. There is no `--output-format json` for `tb sbs` — table only.

## 7. Trajectory-based failure-category analysis: what's built-in vs. manual

The paper's four failure categories — **Tool Usage**, **No State-Changing
Action**, **Incomplete User Resolution**, **Wrong State Update** — are
assigned by inspecting observable trajectory evidence (messages, tool
calls/responses, the final assistant answer, and termination markers) and
picking the single dominant category per failed attempt. **No command in
this repository performs that classification.** Concretely:

- `tb agg`/`tb sbs` only ever report pass/fail/error counts, timing/token
  aggregates, and the pass@k/pass^k statistics above. They do not label
  *why* an attempt failed beyond the executable check's binary verdict
  (`test_result.result`) and, for genuine test bugs, the failing line
  (`test_result.lineno`/`line_content`) or system-error message
  (`metadata.error`). None of that is a natural-language failure category.
- There is no CLI flag, fixture, or script anywhere in this repo (including
  `scripts/dataset_utils/`) that outputs "Tool Usage" / "No State-Changing
  Action" / "Incomplete User Resolution" / "Wrong State Update" labels.

What the CLI gives you is the **raw evidence** you need to apply the paper's
methodology by hand (or with your own external classifier/LLM-judge script):

1. Use `tb agg` (or `jq`/`pandas` over the JSONL, see §5) to find failing
   `uid`s: rows where `test_result.result == False` and
   `test_result.is_system_error == False` are genuine assertion failures;
   `is_system_error == True` at the `DecodeResult` level are decode/agent
   errors; `test_result.is_system_error == True` are bugs in the test itself.
2. Re-run with `--dump testcontext` (and ideally `--dump raw`) so each
   failing row carries `test_context.effects` (final world state) and the
   full tool-call log, not just the chat transcript.
3. For each failing attempt, use `tb pp --line N` to read the conversation,
   tool calls, and the failing assertion side by side, then decide the
   dominant category by hand using the paper's Appendix D criteria (e.g.
   the agent never issued a state-changing tool call → "No State-Changing
   Action"; it called the wrong tool or with wrong arguments → "Tool Usage";
   it ended the conversation without fully addressing the user's request →
   "Incomplete User Resolution"; the tool calls succeeded but left the wrong
   final state per `effects` → "Wrong State Update").
4. `tb run-test` lets you iterate on the assertion itself against a captured
   `test_context` if you find the executable check under- or
   over-classifies "pass"/"fail" relative to what you're trying to measure.

If you build a classifier or scripted heuristic on top of this evidence, it
lives outside `tb agg`/`tb infer` — do not describe such tooling as a
built-in feature of the CLI it is not part of.

## 8. Limitations and caveats (summary)

- `test_context` (and thus `effects`) is stripped from `tb infer` batch
  output unless `--dump testcontext` is passed, even though it's used
  internally to run the test.
- `tb agg`'s `unbiased_pass_power_k` / "pass^k" is a biased plug-in estimator
  `(c/n)**k`, not truly unbiased in the same sense as `pass@k` — see §3.3.
- `pass@k`/`pass^k` blocks are silently omitted if `runs_per_test` is not
  uniform across test cases in the input JSONL, or if no configured `k` in
  `[1, 5, 10, 20, 50, 100]` is `<= runs_per_test`.
- `tb agg`/`tb sbs` support `table` output (and `json` for `tb agg` only) —
  no CSV/Markdown output format exists.
- `tb run-test` operates on a single YAML result file, not a JSONL batch.
- There is no built-in domain/split filter flag; filtering is done on the
  JSONL with standard tools (`jq`, `python`, `pandas`) using the
  `test_tags`/`uid` fields that are always present in the output.
- No CLI command produces the paper's four qualitative failure categories;
  that classification is manual/external, using `tb pp`/`--dump testcontext`
  output as raw evidence.
