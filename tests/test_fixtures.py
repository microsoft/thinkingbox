# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest

from tests.mock_session import MockSession
from thinkingbox.common.chat_types import TestContext as _TestContext
from thinkingbox.common.fixtures import (
    Fixtures,
    _get_injectable_params,
    _resolve_fixture_order,
    fixtures_context,
)
from thinkingbox.common.hydrator import load_conftest_fixtures
from thinkingbox.common.judge import Judge


# Module-level fixture classes (must be at module scope so importlib can find them)
class FixtureA:
    value = "from_A"


class FixtureB:
    def __init__(self, fixture_a: FixtureA):
        self.a = fixture_a


class FixtureC:
    def __init__(self, fixture_b: FixtureB):
        self.b = fixture_b


class _Shared:
    pass


class _Left:
    def __init__(self, shared):
        pass


class _Right:
    def __init__(self, shared):
        pass


class CyclicA:
    def __init__(self, cyclic_b):
        pass


class CyclicB:
    def __init__(self, cyclic_a):
        pass


class SimpleFixture:
    def __init__(self, greeting: str = "hello"):
        self.greeting = greeting


class JudgeDepFixture:
    def __init__(self, judge: Judge):
        self.judge = judge

    def check(self, response: str, question: str) -> bool:
        return self.judge.text_yesno(response, question)


class CtxDepFixture:
    def __init__(self, x):
        self.x = x

    def record(self, key: str, value) -> None:
        self.x.metadata[key] = value


class ContextManagerFixture:
    def __init__(self):
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *args):
        self.exited = True
        return False


class ContextManagerWithDepFixture:
    """Context manager fixture that also requires another fixture injected into its constructor."""

    def __init__(self, fixture_a: FixtureA):
        self.a = fixture_a
        self.entered = False
        self.exited = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *args):
        self.exited = True
        return False


class BrokenFixture:
    def __init__(self):
        raise RuntimeError("broken on purpose")


class FixtureWithOptionalDep:
    """Fixture with a required dep (judge) and an optional dep (extra=None)."""

    def __init__(self, judge: Judge, extra=None):
        self.judge = judge
        self.extra = extra


def _reg(fixtures: Fixtures, name: str, factory, params: dict | None = None) -> None:
    """Register a factory importable via its __module__/__name__."""
    fixtures.add(name, factory.__module__, factory.__name__, params or {})


def test_injectable_skips_config_params():
    """Parameters already supplied by config are not returned as injectable."""

    class F:
        def __init__(self, a, b, c):
            pass

    assert _get_injectable_params(F, {"a": 1}) == ["b", "c"]


def test_injectable_all_in_config():
    """When all parameters are in config, nothing is injectable."""

    class F:
        def __init__(self, a, b):
            pass

    assert _get_injectable_params(F, {"a": 1, "b": 2}) == []


def test_injectable_skips_self():
    """'self' is excluded when inspecting an unbound __init__ that includes it in its signature."""

    class F:
        def __init__(self, value):
            pass

    result = _get_injectable_params(F.__init__, {})
    assert "self" not in result
    assert result == ["value"]


def test_injectable_skips_var_args():
    """*args and **kwargs are not injectable — they have no fixed name."""

    def f(*args, **kwargs):
        pass

    assert _get_injectable_params(f, {}) == []


def test_injectable_skips_keyword_only():
    """Keyword-only parameters (declared after a bare *) are not injectable, consistent with extract_test_fn_param_names."""

    def f(a, *, b):
        pass

    assert _get_injectable_params(f, {}) == ["a"]


def test_injectable_plain_function():
    """Works for plain functions, not just classes."""

    def f(x, y):
        pass

    assert _get_injectable_params(f, {"x": 1}) == ["y"]


def test_injectable_skips_params_with_defaults():
    """Parameters with default values are not injectable — they use their default."""

    class F:
        def __init__(self, required, optional=None):
            pass

    assert _get_injectable_params(F, {}) == ["required"]


def test_resolve_single_no_deps():
    """A single fixture with no dependencies returns just itself."""
    fixtures = Fixtures()
    _reg(fixtures, "a", FixtureA)
    assert _resolve_fixture_order(["a"], fixtures, frozenset()) == ["a"]


def test_resolve_linear_chain():
    """A dependency is resolved before the fixture that needs it."""
    fixtures = Fixtures()
    _reg(fixtures, "fixture_a", FixtureA)
    _reg(fixtures, "fixture_b", FixtureB)
    assert _resolve_fixture_order(["fixture_b"], fixtures, frozenset()) == [
        "fixture_a",
        "fixture_b",
    ]


def test_resolve_transitive_chain():
    """Transitive dependencies are resolved in the correct order."""
    fixtures = Fixtures()
    _reg(fixtures, "fixture_a", FixtureA)
    _reg(fixtures, "fixture_b", FixtureB)
    _reg(fixtures, "fixture_c", FixtureC)
    assert _resolve_fixture_order(["fixture_c"], fixtures, frozenset()) == [
        "fixture_a",
        "fixture_b",
        "fixture_c",
    ]


def test_resolve_shared_dependency_appears_once():
    """A dependency shared by two root fixtures is included only once in the order."""
    fixtures = Fixtures()
    _reg(fixtures, "shared", _Shared)
    _reg(fixtures, "left", _Left)
    _reg(fixtures, "right", _Right)
    order = _resolve_fixture_order(["left", "right"], fixtures, frozenset())
    assert order.count("shared") == 1
    assert order.index("shared") < order.index("left")
    assert order.index("shared") < order.index("right")


def test_resolve_runtime_injectable_not_in_order():
    """Parameters satisfied by runtime_context are not added to the resolved order."""
    fixtures = Fixtures()
    _reg(fixtures, "judge_dependency", JudgeDepFixture)
    order = _resolve_fixture_order(["judge_dependency"], fixtures, frozenset({"judge"}))
    assert order == ["judge_dependency"]
    assert "judge" not in order


def test_resolve_multiple_independent_roots():
    """Multiple independent root fixtures are returned in declaration order."""
    fixtures = Fixtures()
    _reg(fixtures, "fixture_a", FixtureA)
    _reg(fixtures, "simple", SimpleFixture, {"greeting": "hi"})
    assert _resolve_fixture_order(["fixture_a", "simple"], fixtures, frozenset()) == [
        "fixture_a",
        "simple",
    ]


def test_resolve_cycle_raises():
    """A cycle in the dependency graph raises ValueError."""
    fixtures = Fixtures()
    _reg(fixtures, "cyclic_a", CyclicA)
    _reg(fixtures, "cyclic_b", CyclicB)
    with pytest.raises(ValueError):
        _resolve_fixture_order(["cyclic_a"], fixtures, frozenset())


def test_resolve_missing_dependency_raises():
    """A fixture whose dependency is not registered raises ValueError."""
    fixtures = Fixtures()
    _reg(fixtures, "fixture_b", FixtureB)
    with pytest.raises(ValueError):
        _resolve_fixture_order(["fixture_b"], fixtures, frozenset())


def test_resolve_unknown_fixture_raises():
    """Requesting a fixture not in the registry raises KeyError."""
    with pytest.raises(KeyError):
        _resolve_fixture_order(["nonexistent"], Fixtures(), frozenset())


def test_resolve_params_with_defaults_not_injected():
    """Params with defaults are never injected — only required params (no default) are."""
    fixtures = Fixtures()
    _reg(fixtures, "f", FixtureWithOptionalDep)
    # 'judge' is required (no default) → injected; 'extra' has a default → skipped
    order = _resolve_fixture_order(["f"], fixtures, frozenset({"judge"}))
    assert order == ["f"]


def test_fixtures_context_params_with_defaults_use_default():
    """Params with defaults are not injected; the factory receives its declared default."""
    fixtures = Fixtures()
    _reg(fixtures, "f", FixtureWithOptionalDep)
    judge_llm = MockSession(completions=[])
    with fixtures_context(
        fixtures, ["f"], runtime_context={"judge": Judge(judge_llm)}
    ) as out:
        instance = out["f"]
        assert instance.extra is None  # default used, not injected


def test_fixtures_context_basic():
    """A simple fixture with no dependencies is instantiated and yielded."""
    fixtures = Fixtures()
    _reg(fixtures, "fixture_a", FixtureA)
    with fixtures_context(fixtures, ["fixture_a"]) as out:
        assert "fixture_a" in out
        assert isinstance(out["fixture_a"], FixtureA)


def test_fixtures_context_injects_other_fixture():
    """A fixture dependency is automatically injected into the dependent's constructor."""
    fixtures = Fixtures()
    _reg(fixtures, "fixture_a", FixtureA)
    _reg(fixtures, "fixture_b", FixtureB)
    with fixtures_context(fixtures, ["fixture_b"]) as out:
        assert isinstance(out["fixture_b"].a, FixtureA)


def test_fixtures_context_transitive_dependency_not_in_output():
    """Transitive dependencies are instantiated internally but not yielded to the caller."""
    fixtures = Fixtures()
    _reg(fixtures, "fixture_a", FixtureA)
    _reg(fixtures, "fixture_b", FixtureB)
    _reg(fixtures, "fixture_c", FixtureC)
    with fixtures_context(fixtures, ["fixture_c"]) as out:
        assert set(out.keys()) == {"fixture_c"}


def test_fixtures_context_injects_runtime_value():
    """A runtime injectable (e.g. judge) is passed to the fixture constructor."""
    fixtures = Fixtures()
    _reg(fixtures, "judge_dependency", JudgeDepFixture)
    mock_judge = Judge(MockSession(completions=[]))
    with fixtures_context(
        fixtures, ["judge_dependency"], runtime_context={"judge": mock_judge}
    ) as out:
        assert out["judge_dependency"].judge is mock_judge


def test_fixtures_context_injects_test_context():
    """TestContext (x) is passed to the fixture constructor and mutations are shared."""
    fixtures = Fixtures()
    _reg(fixtures, "ctx_fixture", CtxDepFixture)
    x = _TestContext()
    with fixtures_context(fixtures, ["ctx_fixture"], runtime_context={"x": x}) as out:
        out["ctx_fixture"].record("key", "value")
    assert x.metadata["key"] == "value"


def test_fixtures_context_context_manager_entered():
    """A context manager fixture is entered on creation and exited on scope exit."""
    fixtures = Fixtures()
    _reg(fixtures, "ctx", ContextManagerFixture)
    with fixtures_context(fixtures, ["ctx"]) as out:
        assert out["ctx"].entered is True
    assert out["ctx"].exited is True


def test_fixtures_context_context_manager_with_injected_dependency():
    """A context manager fixture that also has a constructor dependency is entered and exited correctly."""
    fixtures = Fixtures()
    _reg(fixtures, "fixture_a", FixtureA)
    _reg(fixtures, "ctx_with_dep", ContextManagerWithDepFixture)
    with fixtures_context(fixtures, ["ctx_with_dep"]) as out:
        assert out["ctx_with_dep"].entered is True
        assert isinstance(out["ctx_with_dep"].a, FixtureA)
    assert out["ctx_with_dep"].exited is True


def test_fixtures_context_config_params_passed():
    """Config-supplied kwargs are forwarded to the factory as-is."""
    fixtures = Fixtures()
    _reg(fixtures, "simple", SimpleFixture, params={"greeting": "hi"})
    with fixtures_context(fixtures, ["simple"]) as out:
        assert out["simple"].greeting == "hi"


def test_fixtures_context_default_used_when_no_config():
    """When a param has a default and no config value is provided, the default is used."""
    fixtures = Fixtures()
    _reg(fixtures, "simple", SimpleFixture)  # no config → greeting uses default "hello"
    with fixtures_context(fixtures, ["simple"]) as out:
        assert out["simple"].greeting == "hello"


def test_fixtures_context_cycle_raises_runtime_error():
    """A dependency cycle raises RuntimeError."""
    fixtures = Fixtures()
    _reg(fixtures, "cyclic_a", CyclicA)
    _reg(fixtures, "cyclic_b", CyclicB)
    with pytest.raises(RuntimeError):
        with fixtures_context(fixtures, ["cyclic_a"]) as _:
            pass


def test_fixtures_context_missing_dependency_raises_runtime_error():
    """An unregistered dependency raises RuntimeError."""
    fixtures = Fixtures()
    _reg(fixtures, "fixture_b", FixtureB)
    with pytest.raises(RuntimeError):
        with fixtures_context(fixtures, ["fixture_b"]) as _:
            pass


def test_fixtures_context_instantiation_error_raises_runtime_error():
    """An exception raised during fixture construction raises RuntimeError."""
    fixtures = Fixtures()
    _reg(fixtures, "broken", BrokenFixture)
    with pytest.raises(RuntimeError):
        with fixtures_context(fixtures, ["broken"]) as _:
            pass


# ── load_conftest_fixtures ──────────────────────────────────────────────────────

_CORPUS_CFG = """\
fixtures:
  my_fixture:
    type: thinkingbox.common.SessionClientFixture
"""

_CORPUS_CFG_OVERRIDE = """\
fixtures:
  my_fixture:
    type: thinkingbox.common.SessionClientFixture
    endpoint: "http://override:9999"
  extra_fixture:
    type: thinkingbox.common.SessionClientFixture
"""


def test_load_conftest_fixtures_no_files(tmp_path):
    """Returns empty dict when no conftest.yaml files exist."""
    test_case_root = tmp_path / "test_case"
    test_case_root.mkdir()
    sub = test_case_root / "my_corpus"
    sub.mkdir()
    test_file = sub / "test_foo.py"
    test_file.touch()
    result = load_conftest_fixtures(test_file, test_case_root)
    assert result == {}


def test_load_conftest_fixtures_single_root_file(tmp_path):
    """A conftest.yaml at the root is loaded."""
    test_case_root = tmp_path / "test_case"
    test_case_root.mkdir()
    (test_case_root / "conftest.yaml").write_text(_CORPUS_CFG)
    test_file = test_case_root / "test_foo.py"
    test_file.touch()
    result = load_conftest_fixtures(test_file, test_case_root)
    assert "my_fixture" in result


def test_load_conftest_fixtures_subdir_overrides_root(tmp_path):
    """A subdir conftest.yaml overrides the root for the same fixture name."""
    test_case_root = tmp_path / "test_case"
    test_case_root.mkdir()
    (test_case_root / "conftest.yaml").write_text(_CORPUS_CFG)
    sub = test_case_root / "my_corpus"
    sub.mkdir()
    (sub / "conftest.yaml").write_text(_CORPUS_CFG_OVERRIDE)
    test_file = sub / "test_foo.py"
    test_file.touch()
    result = load_conftest_fixtures(test_file, test_case_root)
    # subdir value wins for my_fixture
    assert result["my_fixture"].model_extra.get("endpoint") == "http://override:9999"
    # extra_fixture is also present (only in subdir)
    assert "extra_fixture" in result


def test_load_conftest_fixtures_root_only_when_file_at_root(tmp_path):
    """When the test file is directly in test_case_root, only root config is loaded."""
    test_case_root = tmp_path / "test_case"
    test_case_root.mkdir()
    (test_case_root / "conftest.yaml").write_text(_CORPUS_CFG)
    sub = test_case_root / "my_corpus"
    sub.mkdir()
    (sub / "conftest.yaml").write_text(_CORPUS_CFG_OVERRIDE)
    test_file = test_case_root / "test_foo.py"
    test_file.touch()
    result = load_conftest_fixtures(test_file, test_case_root)
    # Root config only — subdir config not applied
    assert "my_fixture" in result
    assert "extra_fixture" not in result


def test_load_conftest_fixtures_outside_root_returns_empty(tmp_path):
    """A test file outside test_case_root returns an empty dict (no crash)."""
    test_case_root = tmp_path / "test_case"
    test_case_root.mkdir()
    (test_case_root / "conftest.yaml").write_text(_CORPUS_CFG)
    test_file = tmp_path / "other" / "test_foo.py"
    test_file.parent.mkdir()
    test_file.touch()
    result = load_conftest_fixtures(test_file, test_case_root)
    assert result == {}


def test_load_conftest_fixtures_empty_yaml(tmp_path):
    """An empty conftest.yaml does not crash and contributes no fixtures."""
    test_case_root = tmp_path / "test_case"
    test_case_root.mkdir()
    (test_case_root / "conftest.yaml").write_text("")
    test_file = test_case_root / "test_foo.py"
    test_file.touch()
    result = load_conftest_fixtures(test_file, test_case_root)
    assert result == {}
