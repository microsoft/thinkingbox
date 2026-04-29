# Test Fixtures

## Overview

Test fixtures provide reusable components that can be injected into test functions via dependency injection, similar to pytest fixtures. This allows to share common testing utilities across multiple test cases and avoid code duplication.

## Defining Fixtures

Fixtures are defined in `conftest.yaml` files placed in `dataset/test_case/` directories.
`conftest.yaml` is analogous to pytest's `conftest.py` — it provides directory-scoped shared
configuration that is inherited by all test cases in the same directory and subdirectories (but note that it is explicitly *not* compatible with pytest's `conftest.py`).

Each fixture specifies a type (Python class or function) and optional constructor parameters.

```yaml
# dataset/test_case/my_corpus/conftest.yaml
fixtures:
  # A simple class-based fixture
  filejudge_fn:
    type: my_fixtures.FileContentJudge
    case_sensitive: true

  # Built-in session proxy client to access the MCP session
  session_client:
    type: thinkingbox.common.SessionClientFixture
    endpoint: "http://127.0.0.1:7111"
```

Fixtures can also be declared (or overridden) in a scenario YAML file:

```yaml
# dataset/scenario/my_scenario.yaml
fixtures:
  answer_evaluator:
    type: thinkingbox.fixtures.GeneratedAnswerEvaluator
    fail_on_mismatch: true
```

## Implementation Location

We recommend you implement fixtures in:

* Fixtures likely to be shared across many MCP servers: `AI.ThinkingBox/thinkingbox/fixtures`
* Fixtures specific to a particular MCP server: `AI.ThinkingBox.Data/servers/<server-group>/fixtures` (e.g., `.../servers/ms_toloka_servers/fixtures`)

In either case, include an `__init__.py` that imports the fixture classes/functions and adds them to `__all__`, so they can be referenced by their fully qualified name in the fixture configuration.

### Resolution Order

Fixtures are resolved in this order (later sources override earlier ones):

1. `dataset/test_case/conftest.yaml` — global defaults for the entire dataset
2. `dataset/test_case/<subdir>/conftest.yaml` — per-directory defaults (overrides parent)
3. Scenario YAML `fixtures:` — per-scenario overrides (highest priority)

A fixture with the same name in a later source replaces the one from an earlier source entirely.

## Fixture Types

### Class Constructors

Reference a Python class by its fully qualified name. The fixture system instantiates it with the provided parameters:

```python
class FileContentJudge:
    def __init__(self, case_sensitive: bool = False):
        self.case_sensitive = case_sensitive

    def check_file(self, effects: dict, filename: str, expected_text: str):
        # Implementation
        pass
```

Configuration:
```yaml
fixtures:
  filejudge_fn:
    type: my_fixtures.FileContentJudge
    case_sensitive: true
```

### Context Managers

Use a python context manager (implements `__enter__` and `__exit__`):

```python
import contextlib

@contextlib.contextmanager
def file_content_judge(case_sensitive: bool = False):
    obj = FileContentJudge(case_sensitive=case_sensitive)
    # any setup
    try:
        yield obj
    finally:
        # any teardown
        ...
```

Configuration:
```yaml
fixtures:
  filejudge_ctx:
    type: my_fixtures.file_content_judge
    case_sensitive: true
```

## Using Fixtures in Tests

Declare fixtures as parameters in your test function signature. The test runner automatically injects them:

```python
from thinkingbox.common import Judge, TestContext

def test_append_some_more_text(
    x: TestContext,
    judge: Judge,
    filejudge_ctx,
    filejudge_fn,
    session_client
):
    """!
    query: |
        Find a file named file.txt, then append the following to its contents on a new line: 'some more text'
    """
    # Use the context manager fixture
    filejudge_ctx.check_file(
        x.effects["cloud_drive"],
        "Documents/file.txt",
        "some text\nsome more text"
    )

    # Use the class-based fixture
    filejudge_fn.check_file(
        x.effects["cloud_drive"],
        "Documents/file.txt",
        "some text\nsome more text"
    )

    # Use the session client fixture
    session_client.set_session(x.session_id)
    out_dict = session_client.call_json_tool(
        "cloud_drive",
        "get_text_content",
        {"path": "Documents/file.txt"},
    )
    assert out_dict["text_content"] == "some text\nsome more text"
```

## Dependency Injection

Fixtures can declare other registered fixtures or the `Judge` instance as constructor
parameters. The system resolves the full dependency graph automatically and instantiates
in dependency order.

Two runtime values are available for injection by name: `judge` (the active `Judge` instance)
and `x` (the `TestContext` for the current test run). Both are injected by declaring them as
constructor parameters — no configuration is needed.

### Injecting `judge`

Declare `judge` as a constructor parameter. The runner injects the active `Judge` instance:

```python
from thinkingbox.common import Judge

class VerdictJudge:
    def __init__(self, judge: Judge, threshold: float = 0.5):
        self.judge = judge
        self.threshold = threshold

    def check(self, response: str, question: str) -> bool:
        return self.judge.text_yesno(response, question)
```

```yaml
fixtures:
  verdict_judge:
    type: my_fixtures.VerdictJudge
    threshold: 0.8
    # 'judge' is not listed here — it is injected automatically
```

```python
def test_something(x: TestContext, verdict_judge):
    assert verdict_judge.check(x.response, "Did the agent complete the task?")
```

### Injecting another fixture

Declare a fixture as a constructor parameter using its registered name:

```python
class FileChecker:
    def __init__(self, base_path: str):
        self.base_path = base_path

class ContentVerifier:
    def __init__(self, file_checker: FileChecker, strict: bool = False):
        self.file_checker = file_checker
        self.strict = strict
```

```yaml
fixtures:
  file_checker:
    type: my_fixtures.FileChecker
    base_path: "Documents/"
  content_verifier:
    type: my_fixtures.ContentVerifier
    strict: true
    # 'file_checker' is injected automatically
```

`file_checker` is instantiated first and passed into `content_verifier`. Only
`content_verifier` needs to be declared in the test function signature — transitive
dependencies are resolved internally and not exposed to the test.

### Injecting `x` (TestContext)

Declare `x` as a constructor parameter to receive the live `TestContext`. Because the
fixture holds a reference to the same object the test function receives, any writes to
`x.metadata` inside the fixture are visible in the final `TestResult`:

```python
from thinkingbox.common import TestContext

class RubricJudge:
    def __init__(self, judge: Judge, x: TestContext):
        self._judge = judge
        self._x = x

    def evaluate(self, rubrics: list[RubricConfig]) -> EvaluationResult:
        result = ...  # run evaluation using self._judge and self._x.response
        self._x.metadata["rubric_evaluation"] = result.model_dump()
        return result
```

```python
def test_something(x: TestContext, rubric_judge):
    return rubric_judge.evaluate(rubrics=[...]).reward
    # x.metadata["rubric_evaluation"] is populated automatically
```

### Parameters with defaults

The DI system only injects **required** parameters (those without a default value). Parameters
with a default value are never injected — they use their declared default unless the fixture
config provides an explicit value.

```python
class MyFixture:
    def __init__(self, judge: Judge, x: TestContext, reward_fn=None, max_workers: int = 4):
        # judge, x    — required; injected from runtime (no config needed)
        # reward_fn   — has default; uses None unless provided in config
        # max_workers — has default; overridden by config below
        ...
```

```yaml
fixtures:
  my_fixture:
    type: my_fixtures.MyFixture
    max_workers: 10        # overrides the default of 4
    # reward_fn is absent → uses default None
    # judge and x are injected automatically from the runtime context
```

### Errors

A **cycle** (A depends on B, B depends on A) or an **unresolvable required parameter** (no
default, not in config, not a registered fixture, and not `judge` or `x`) raises a
`RuntimeError` at test execution time with a descriptive message, and the test is recorded
as a system error.

## MCP session persistence

The session remains active during test execution, within the session proxy, allowing fixtures like `SessionClientFixture` to interact with the live MCP session.

By default, the session is destroyed when `infer` finishes processing the test case.
In order to persist the session for debugging with `run-test`, launch `infer` with the `--linger-sessions` flag, then clean up the sessions manually within the session proxy, via a HTTP request or by restarting it.
