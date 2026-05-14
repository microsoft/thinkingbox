# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import ast
import copy

import pytest

from tests.mock_session import ErrorOnCompletionMockSession, MockSession
from tests.test_fixtures import (
    CtxDepFixture,
    CyclicA,
    CyclicB,
    FixtureA,
    FixtureB,
    JudgeDepFixture,
    SimpleFixture,
)
from thinkingbox.common.chat_types import TestContext as _TestContext
from thinkingbox.common.chat_types import (
    Text,
)
from thinkingbox.common.fixtures import Fixtures
from thinkingbox.common.judge import Judge
from thinkingbox.common.python_test_file import iter_test_functions
from thinkingbox.common.rubrics_judge import RubricJudge
from thinkingbox.common.testrunner import TestScript as _TestScript
from thinkingbox.common.testrunner import TestScriptDebug as _TestScriptDebug

CLOUD_DRIVE_TESTS = r"""
import itertools

def check_file_content(x, filename, expected_text):
    # use some itertools function just to test that imports work
    for f in itertools.dropwhile(
        lambda item: item["path"] != filename,
        x.effects["cloud_drive"]["files"],
    ):
        text = f["text_content"]
        # "assert text == expected_text" must be in the following line, do not change
        assert text == expected_text, (
            f"Expected {expected_text!r}, found {text!r}"
        )
        return
    assert False, f"File {filename} not found"


def check_file_content_with_reward(x, filenames, expected_texts):
    # Returns the number of matching files instead of asserting.

    if isinstance(filenames, str):
        filenames = [filenames]
        expected_texts = [expected_texts]
    elif isinstance(expected_texts, str):
        expected_texts = [expected_texts] * len(filenames)

    matches = 0

    for filename, expected_text in zip(filenames, expected_texts):
        file_found = False
        # use some itertools function just to test that imports work
        for f in itertools.dropwhile(
            lambda item: item["path"] != filename,
            x.effects["cloud_drive"]["files"],
        ):
            file_found = True
            text = f["text_content"]
            if text == expected_text:
                print(f"Expected {expected_text!r} found in {filename}")
                matches += 1
            else:
                print(f"Expected {expected_text!r}, found {text!r} in {filename}")
            break

        if not file_found:
            print(f"File {filename} not found")

    assert matches > 0, f"No files matched their expected content (0/{len(filenames)})"

    return matches/len(filenames)

def test_one(x, judge):
    '''!
    query: none
    '''
    assert judge.text_yesno(
        x.response, "Does the message confirm that file.txt was modified?"
    )
    check_file_content(x, "Documents/file.txt", "some text")

def test_two(x, judge):
    '''!
    query: none
    '''
    # "judge.text_yesno(" must be in the following line, do not change
    assert judge.text_yesno(
        x.response, "Does the message confirm that file.txt was modified?"
    )
    check_file_content(x, "Documents/file.txt", "some other text")

def test_three(x, judge):
    '''!
    query: none
    '''
    assert judge.text_yesno(
        x.response, "Does the message confirm that important.txt was executed?"
    )

    reward = check_file_content_with_reward(x, ["Documents/file.txt", "Documents/important.txt"], ["some text", "IMPORTANT!"])

    return reward

def test_four(x, judge):
    '''!
    query: none
    '''
    assert judge.text_yesno(
        x.response, "Does the message confirm that important.txt was executed?"
    )

    reward = check_file_content_with_reward(x, ["Documents/file.txt", "Documents/important.txt"], ["some text", "NOT IMPORTANT!"])

    return reward

""".lstrip()


CLOUD_DRIVE_EFFECTS = {
    "cloud_drive": {
        "files": [
            {"path": "Documents/file.txt", "text_content": "some text"},
            {"path": "Documents/important.txt", "text_content": "IMPORTANT!"},
        ]
    }
}


def _get_tests_code(test_fn_code: str):
    return {
        test_fn.name: test_fn.test_code
        for test_fn in iter_test_functions(test_fn_code, check_config=True)
    }


def _msg_text(content):
    return Text(role="assistant", content=content, metadata={"tag": "text"})


def test_load_test_with_globals():
    tests = _get_tests_code(CLOUD_DRIVE_TESTS)
    expected_tests = ["test_one", "test_two", "test_three", "test_four"]
    assert set(tests.keys()) == set(expected_tests)

    # but not other test functions
    for name, code in tests.items():
        # check that the import and extra function are in there
        assert "check_file_content" in code
        assert "import itertools" in code

        # check that the code can be parsed
        _ = ast.parse(code)

        # check that other test cases are not there
        for other_name in expected_tests:
            if other_name != name:
                assert other_name not in code


@pytest.mark.asyncio
async def test_testscript_exec_pass():
    """Test that the 'exec' variant (TestScript) works when a test passes"""
    tests = _get_tests_code(CLOUD_DRIVE_TESTS)

    test_context = _TestContext(
        response="something",
        effects=copy.deepcopy(CLOUD_DRIVE_EFFECTS),
    )
    judge_llm = MockSession(completions=[[_msg_text("yes")]])
    test = _TestScript(
        tests["test_one"],
        judge=Judge(judge_llm),
    )
    test_result = await test.evaluate(test_context)
    assert test_result.result, f"test result is False, traceback: {test_result.tb}"
    assert not judge_llm.completions, "Not all judge messages have been emitted"


@pytest.mark.asyncio
async def test_testscript_exec_testfail():
    """Test that the 'exec' variant (TestScript) works when a test fails"""
    tests = _get_tests_code(CLOUD_DRIVE_TESTS)

    test_context = _TestContext(
        response="something",
        effects=copy.deepcopy(CLOUD_DRIVE_EFFECTS),
    )
    judge_llm = MockSession(completions=[[_msg_text("yes")]])
    test = _TestScript(
        tests["test_two"],
        judge=Judge(judge_llm),
    )
    test_result = await test.evaluate(test_context)
    assert not test_result.result, "test result is True"
    assert not test_result.is_system_error, "Expected test failure, got system error"
    assert not judge_llm.completions, "Not all judge messages have been emitted"
    # should be the line from the topmost frame still belonging to the test file
    # in this case it's the top frame where the exception actually originated
    assert "assert text == expected_text" in test_result.line_content


@pytest.mark.asyncio
async def test_testscript_exec_reward_pass():
    """Test that the 'exec' variant (TestScript) works when a reward test passes"""
    tests = _get_tests_code(CLOUD_DRIVE_TESTS)

    test_context = _TestContext(
        response="something",
        effects=copy.deepcopy(CLOUD_DRIVE_EFFECTS),
    )
    judge_llm = MockSession(completions=[[_msg_text("yes")]])
    test = _TestScript(
        tests["test_three"],
        judge=Judge(judge_llm),
    )
    test_result = await test.evaluate(test_context)
    assert test_result.result, f"test result is False, traceback: {test_result.tb}"
    assert test_result.reward == 1.0, f"Expected reward 1.0, got {test_result.reward}"
    assert not judge_llm.completions, "Not all judge messages have been emitted"


@pytest.mark.asyncio
async def test_testscript_exec_reward_partial():
    """Test that the 'exec' variant (TestScript) works when a reward test passes with partial reward"""
    tests = _get_tests_code(CLOUD_DRIVE_TESTS)

    test_context = _TestContext(
        response="something",
        effects=copy.deepcopy(CLOUD_DRIVE_EFFECTS),
    )
    judge_llm = MockSession(completions=[[_msg_text("yes")]])
    test = _TestScript(
        tests["test_four"],
        judge=Judge(judge_llm),
    )
    test_result = await test.evaluate(test_context)
    assert test_result.result, f"test result is False, traceback: {test_result.tb}"
    assert test_result.reward == 0.5, f"Expected reward 0.5, got {test_result.reward}"
    assert not test_result.is_system_error, "Expected test success, got system error"
    assert not judge_llm.completions, "Not all judge messages have been emitted"


@pytest.mark.asyncio
async def test_testscript_exec_sysfail():
    """Test that the 'exec' variant (TestScript) works when a test produces an error"""
    tests = _get_tests_code(CLOUD_DRIVE_TESTS)

    test_context = _TestContext(
        response="something",
        effects=copy.deepcopy(CLOUD_DRIVE_EFFECTS),
    )
    judge_llm = ErrorOnCompletionMockSession(completions=[])
    test = _TestScript(
        tests["test_two"],
        judge=Judge(judge_llm),
    )
    test_result = await test.evaluate(test_context)
    assert not test_result.result, "test result is True"
    assert test_result.is_system_error, "Expected system error, got test failure"
    # should be the line from the topmost frame still belonging to the test file
    assert "judge.text_yesno(" in test_result.line_content


def _make_test_script_debug(tmpdir, code, test_name, judge):
    path = tmpdir / "cloud_drive_tests.py"
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    return _TestScriptDebug(
        test_name=test_name,
        source_file=path,
        judge=judge,
    )


@pytest.mark.asyncio
async def test_testscript_import_pass(tmp_path):
    """Test that the 'import' variant (TestScriptDebug) works"""
    test_context = _TestContext(
        response="something",
        effects=copy.deepcopy(CLOUD_DRIVE_EFFECTS),
    )
    judge_llm = MockSession(completions=[[_msg_text("yes")]])
    test = _make_test_script_debug(
        tmp_path,
        CLOUD_DRIVE_TESTS,
        "test_one",
        judge=Judge(judge_llm),
    )
    test_result = await test.evaluate(test_context)
    assert test_result.result, f"test result is False, traceback: {test_result.tb}"
    assert not judge_llm.completions, "Not all judge messages have been emitted"


@pytest.mark.asyncio
async def test_testscript_import_testfail(tmp_path):
    """Test that the 'import' variant (TestScriptDebug) works when a test fails"""
    test_context = _TestContext(
        response="something",
        effects=copy.deepcopy(CLOUD_DRIVE_EFFECTS),
    )
    judge_llm = MockSession(completions=[[_msg_text("yes")]])
    test = _make_test_script_debug(
        tmp_path,
        CLOUD_DRIVE_TESTS,
        "test_two",
        judge=Judge(judge_llm),
    )
    test_result = await test.evaluate(test_context)
    assert not test_result.result, "test result is True"
    assert not test_result.is_system_error, "Expected test failure, got system error"
    assert not judge_llm.completions, "Not all judge messages have been emitted"
    # should be the line from the topmost frame still belonging to the test file
    # in this case it's the top frame where the exception actually originated
    assert "assert text == expected_text" in test_result.line_content


@pytest.mark.asyncio
async def test_testscript_import_reward_pass(tmp_path):
    """Test that the 'import' variant (TestScriptDebug) works when a reward test passes"""
    test_context = _TestContext(
        response="something",
        effects=copy.deepcopy(CLOUD_DRIVE_EFFECTS),
    )
    judge_llm = MockSession(completions=[[_msg_text("yes")]])
    test = _make_test_script_debug(
        tmp_path,
        CLOUD_DRIVE_TESTS,
        "test_three",
        judge=Judge(judge_llm),
    )
    test_result = await test.evaluate(test_context)
    assert test_result.result, f"test result is False, traceback: {test_result.tb}"
    assert test_result.reward == 1.0, f"Expected reward 1.0, got {test_result.reward}"
    assert not judge_llm.completions, "Not all judge messages have been emitted"


@pytest.mark.asyncio
async def test_testscript_import_reward_partial(tmp_path):
    """Test that the 'import' variant (TestScriptDebug) works when a reward test passes with partial reward"""
    test_context = _TestContext(
        response="something",
        effects=copy.deepcopy(CLOUD_DRIVE_EFFECTS),
    )
    judge_llm = MockSession(completions=[[_msg_text("yes")]])
    test = _make_test_script_debug(
        tmp_path,
        CLOUD_DRIVE_TESTS,
        "test_four",
        judge=Judge(judge_llm),
    )
    test_result = await test.evaluate(test_context)
    assert test_result.result, f"test result is False, traceback: {test_result.tb}"
    assert test_result.reward == 0.5, f"Expected reward 0.5, got {test_result.reward}"
    assert not test_result.is_system_error, "Expected test success, got system error"
    assert not judge_llm.completions, "Not all judge messages have been emitted"


@pytest.mark.asyncio
async def test_testscript_import_sysfail(tmp_path):
    """Test that the 'import' variant (TestScriptDebug) works when a test produces an error"""
    test_context = _TestContext(
        response="something",
        effects=copy.deepcopy(CLOUD_DRIVE_EFFECTS),
    )
    judge_llm = ErrorOnCompletionMockSession(completions=[])
    test = _make_test_script_debug(
        tmp_path,
        CLOUD_DRIVE_TESTS,
        "test_two",
        judge=Judge(judge_llm),
    )
    test_result = await test.evaluate(test_context)
    assert not test_result.result, "test result is True"
    assert test_result.is_system_error, "Expected system error, got test failure"
    # should be the line in the topmost frame still belonging to the test file
    assert "judge.text_yesno(" in test_result.line_content


@pytest.mark.asyncio
async def test_testcontext_metadata_propagated():
    """metadata written to x.metadata during the test should appear in TestResult.metadata"""
    code = '''
def __tb_test_fn(x, judge):
    """!
    query: What is an LLM?
    """
    x.metadata["word_count"] = len(x.response.split())
    x.metadata["response_has_content"] = len(x.response) > 0
'''
    test = _TestScript(code, judge=Judge(MockSession(completions=[])))
    ctx = _TestContext(
        response="An LLM is a large language model trained on text data."
    )
    result = await test.evaluate(ctx)
    assert result.metadata["word_count"] == len(ctx.response.split())
    assert result.metadata["response_has_content"] is True


@pytest.mark.asyncio
async def test_testcontext_metadata_empty_by_default():
    """TestResult.metadata should only contain judge_motivation when x.metadata is not written to"""
    code = '''
def __tb_test_fn(x, judge):
    """!
    query: What is an LLM?
    """
    assert judge.text_yesno(x.response, "Does the response define what an LLM is?")
'''
    test = _TestScript(code, judge=Judge(MockSession(completions=[[_msg_text("yes")]])))
    ctx = _TestContext(
        response="An LLM is a large language model trained on text data."
    )
    result = await test.evaluate(ctx)
    assert set(result.metadata.keys()) == {"judge_motivation"}


def _fixtures(*pairs) -> Fixtures:
    """Build a Fixtures registry from (name, factory) pairs."""
    f = Fixtures()
    for name, factory in pairs:
        f.add(name, factory.__module__, factory.__name__, {})
    return f


def _fixtures_with_params(name, factory, params) -> Fixtures:
    """Build a Fixtures registry for a single factory with config kwargs."""
    f = Fixtures()
    f.add(name, factory.__module__, factory.__name__, params)
    return f


@pytest.mark.asyncio
async def test_fixture_judge_injected_exec():
    """TestScript injects judge into a fixture constructor."""
    code = """
def __tb_test_fn(x, judge, judge_dep):
    '''!
    query: test
    '''
    assert judge_dep.check(x.response, "Is the response non-empty?")
"""
    judge_llm = MockSession(completions=[[_msg_text("yes")]])
    fixtures = _fixtures(("judge_dep", JudgeDepFixture))
    test = _TestScript(code, judge=Judge(judge_llm), fixtures=fixtures)
    result = await test.evaluate(_TestContext(response="hello"))
    assert result.result, result.tb
    assert not judge_llm.completions


@pytest.mark.asyncio
async def test_fixture_judge_injected_debug(tmp_path):
    """TestScriptDebug injects judge into a fixture constructor."""
    test_file = tmp_path / "test_judge_injection.py"
    test_file.write_text(
        """
def test_judge_injected(x, judge, judge_dep):
    '''!
    query: test
    '''
    assert judge_dep.check(x.response, "Is the response non-empty?")
"""
    )
    judge_llm = MockSession(completions=[[_msg_text("yes")]])
    fixtures = _fixtures(("judge_dep", JudgeDepFixture))
    test = _TestScriptDebug(
        test_name="test_judge_injected",
        source_file=test_file,
        judge=Judge(judge_llm),
        fixtures=fixtures,
    )
    result = await test.evaluate(_TestContext(response="hello"))
    assert result.result, result.tb
    assert not judge_llm.completions


@pytest.mark.asyncio
async def test_fixture_to_fixture_injection_exec():
    """TestScript injects fixture_a into fixture_b automatically."""
    code = """
def __tb_test_fn(x, judge, fixture_b):
    '''!
    query: test
    '''
    assert fixture_b.a.value == "from_A"
"""
    fixtures = _fixtures(("fixture_a", FixtureA), ("fixture_b", FixtureB))
    test = _TestScript(
        code, judge=Judge(MockSession(completions=[])), fixtures=fixtures
    )
    result = await test.evaluate(_TestContext(response="ok"))
    assert result.result, result.tb


@pytest.mark.asyncio
async def test_fixture_cycle_is_system_error():
    """A cycle in fixtures causes is_system_error=True without raising."""
    code = """
def __tb_test_fn(x, judge, cyclic_a):
    '''!
    query: test
    '''
    pass
"""
    fixtures = _fixtures(("cyclic_a", CyclicA), ("cyclic_b", CyclicB))
    test = _TestScript(
        code, judge=Judge(MockSession(completions=[])), fixtures=fixtures
    )
    result = await test.evaluate(_TestContext(response="ok"))
    assert not result.result
    assert result.is_system_error
    assert "Cycle" in result.tb


@pytest.mark.asyncio
async def test_fixture_config_only_unchanged():
    """Fixtures with only config params continue to work after injection changes."""
    code = """
def __tb_test_fn(x, judge, simple):
    '''!
    query: test
    '''
    assert simple.greeting == "hi"
"""
    fixtures = _fixtures_with_params("simple", SimpleFixture, {"greeting": "hi"})
    test = _TestScript(
        code, judge=Judge(MockSession(completions=[])), fixtures=fixtures
    )
    result = await test.evaluate(_TestContext(response="ok"))
    assert result.result, result.tb


@pytest.mark.asyncio
async def test_x_and_judge_not_looked_up_as_fixtures():
    """x and judge declared in a test function signature are not looked up in the fixtures registry."""
    code = """
def __tb_test_fn(x, judge):
    '''!
    query: test
    '''
    pass
"""
    test = _TestScript(
        code, judge=Judge(MockSession(completions=[])), fixtures=Fixtures()
    )
    result = await test.evaluate(_TestContext(response="ok"))
    assert result.result, result.tb


@pytest.mark.asyncio
async def test_none_fixtures_does_not_raise():
    """TestScript with fixtures=None runs successfully when the test uses no fixtures."""
    code = """
def __tb_test_fn(x, judge):
    '''!
    query: test
    '''
    pass
"""
    test = _TestScript(code, judge=Judge(MockSession(completions=[])), fixtures=None)
    result = await test.evaluate(_TestContext(response="ok"))
    assert result.result, result.tb


@pytest.mark.asyncio
async def test_missing_fixture_dependency_is_system_error():
    """A fixture with an unregistered dependency is recorded as a system error in the result."""
    code = """
def __tb_test_fn(x, judge, fixture_b):
    '''!
    query: test
    '''
    pass
"""
    fixtures = _fixtures(("fixture_b", FixtureB))  # fixture_a is not registered
    test = _TestScript(
        code, judge=Judge(MockSession(completions=[])), fixtures=fixtures
    )
    result = await test.evaluate(_TestContext(response="ok"))
    assert not result.result
    assert result.is_system_error


@pytest.mark.asyncio
async def test_fixture_receives_test_context_and_mutations_appear_in_result():
    """A fixture injected with TestContext can write to x.metadata and those values appear in the test result."""
    code = """
def __tb_test_fn(x, judge, ctx_fixture):
    '''!
    query: test
    '''
    ctx_fixture.record("fixture_key", "fixture_value")
"""
    fixtures = _fixtures(("ctx_fixture", CtxDepFixture))
    test = _TestScript(
        code, judge=Judge(MockSession(completions=[])), fixtures=fixtures
    )
    result = await test.evaluate(_TestContext(response="ok"))
    assert result.result, result.tb
    assert result.metadata["fixture_key"] == "fixture_value"


@pytest.mark.asyncio
async def test_fixture_cycle_is_system_error_debug(tmp_path):
    """A cycle in fixtures causes is_system_error=True in the TestScriptDebug import path."""
    test_file = tmp_path / "test_cycle.py"
    test_file.write_text(
        """
def test_cycle(x, judge, cyclic_a):
    '''!
    query: test
    '''
    pass
"""
    )
    fixtures = _fixtures(("cyclic_a", CyclicA), ("cyclic_b", CyclicB))
    test = _TestScriptDebug(
        test_name="test_cycle",
        source_file=test_file,
        judge=Judge(MockSession(completions=[])),
        fixtures=fixtures,
    )
    result = await test.evaluate(_TestContext(response="ok"))
    assert not result.result
    assert result.is_system_error
    assert "Cycle" in result.tb


@pytest.mark.asyncio
async def test_missing_fixture_dependency_is_system_error_debug(tmp_path):
    """A fixture with an unregistered dependency is recorded as a system error in the import path."""
    test_file = tmp_path / "test_missing_dep.py"
    test_file.write_text(
        """
def test_missing_dep(x, judge, fixture_b):
    '''!
    query: test
    '''
    pass
"""
    )
    fixtures = _fixtures(("fixture_b", FixtureB))  # fixture_a is not registered
    test = _TestScriptDebug(
        test_name="test_missing_dep",
        source_file=test_file,
        judge=Judge(MockSession(completions=[])),
        fixtures=fixtures,
    )
    result = await test.evaluate(_TestContext(response="ok"))
    assert not result.result
    assert result.is_system_error


@pytest.mark.asyncio
async def test_rubric_judge_fixture_injects_judge_and_writes_metadata():
    """RubricJudge fixture receives judge and x via DI; evaluate() writes metadata."""
    code = """
def __tb_test_fn(x, judge, rubric_judge):
    '''!
    query: test
    '''
    result = rubric_judge.evaluate(
        response="The capital of France is Paris.",
        rubrics=[rubric_judge.Config(criterion="Is Paris mentioned?", weight=1.0)],
    )
    assert result.passed
"""
    fixtures = _fixtures(("rubric_judge", RubricJudge))
    judge_llm = MockSession(completions=[[_msg_text("yes")]])
    test = _TestScript(code, judge=Judge(judge_llm), fixtures=fixtures)
    ctx = _TestContext(response="The capital of France is Paris.")
    result = await test.evaluate(ctx)
    assert result.result, result.tb
    assert result.metadata["rubric_reward"] == 1.0
    assert "rubric_scores" in result.metadata
    assert len(result.metadata["rubric_scores"]) == 1
