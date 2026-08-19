# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import asyncio
import importlib.util
import inspect
import io
import json
import string
import sys
import traceback
from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import TypeAdapter
from pydantic_core import to_jsonable_python

from thinkingbox.common.chat_types import TestContext, TestResult
from thinkingbox.common.config_types import TEST_CASE_FN_NAME, FixtureConfig
from thinkingbox.common.fixtures import (
    Fixtures,
    build_fixtures,
    fixtures_context_for_test_fn,
)
from thinkingbox.common.http_client import get_dns_cache, initialize_dns_cache
from thinkingbox.common.judge import Judge
from thinkingbox.common.llm_session_factory import LLMSessionConfigT, create_llm_session
from thinkingbox.common.utils import syncify


class MockPrint:
    def __init__(self):
        self.buf = io.StringIO()

    def __call__(self, *args, **kwargs):
        if "file" in kwargs:
            del kwargs["file"]
        print(*args, file=self.buf, **kwargs)

    def getvalue(self):
        return self.buf.getvalue()


def get_exc_line(tb, code: str) -> tuple[int, str]:
    lineno = -1
    line_content = ""
    tb_stack = traceback.extract_tb(tb)
    if tb_stack:
        last_entry = tb_stack[-1]
        lineno = last_entry.lineno
        code_lines = code.splitlines()
        if lineno <= len(code_lines):
            line_content = code_lines[lineno - 1]
    return (lineno, line_content)


class TestScriptBase(ABC):
    @abstractmethod
    def evaluate_sync(self, ctx: TestContext) -> TestResult: ...

    @abstractmethod
    async def evaluate(self, ctx: TestContext) -> TestResult: ...


def _get_safe_module_name(path: str | Path):
    allowed_characters = string.ascii_letters + string.digits + "_"
    base_name = Path(path).stem
    base_name = "".join((c if c in allowed_characters else "_") for c in base_name)
    base_name = f"_tb_dbg_{base_name}"
    out = base_name
    i = 0
    while out in sys.modules:
        i += 1
        out = f"{base_name}_{i}"
    return out


def _iter_traceback_frames(exc_tb):
    frame = exc_tb
    while frame:
        yield frame
        frame = frame.tb_next


def _exc_info_to_test_result_inplace(
    is_system_error: bool,
    exc_info: tuple,
    sources: dict[str, str],
    res: TestResult,
):
    """
    Set test result to false and store exception information.
    exc_info is the tuple returned by sys.exc_info()
    """
    res.result = False
    res.reward = 0.0
    res.is_system_error = is_system_error
    exc_type, exc_value, exc_tb = exc_info

    # Get the last traceback frame in our test module, or the last one if not found
    testscript_module = ""
    testscript_frame = None
    tb_last = None
    for tb_last in _iter_traceback_frames(exc_tb):
        name = tb_last.tb_frame.f_globals.get("__name__", None)
        if name and name in sources:
            testscript_module = name
            testscript_frame = tb_last
    if testscript_frame is None:
        testscript_frame = tb_last

    res.lineno = testscript_frame.tb_lineno
    # Format the full traceback as a string
    res.tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))

    # try to get the error line
    if testscript_module in sources:
        code = sources[testscript_module]
        try:
            res.line_content = code.splitlines()[res.lineno - 1]
        except IndexError:
            res.line_content = f"<line {res.lineno} not found in {testscript_module}>"
    else:
        res.line_content = f"<source for {testscript_module} is not available>"


class TestScript(TestScriptBase):
    """
    Test script implementation that uses exec to execute
    a test without importing it as a module

    It appends a call to the test case function at the end of the test code string

    ```
    # test code...

    # line added by the test loader
    __tb_test_fn = my_actual_test_case_defined_above

    # line added by TestScript, assuming __tb_test_fn is the test function
    __tb_test_fn(**__tb_test_kwargs)
    ```

    `__tb_test_kwargs` is passed in the in the globals+locals dict when executing
    the code above.

    As the code block above is executed, `__tb_test_fn`, which is the user-written test
    case function, is executed.

    It is expected to either complete without exceptions (test pass), or raise an
    assertion error (test failed). This information constitutes the test result.
    Other exceptions are also captured and stored in the result object for
    debugging, but are not expected outcomes.
    """

    def __init__(
        self,
        code: str,
        judge: Judge,
        fixtures: Fixtures | None = None,
        test_uid: str | None = None,
    ):
        # add call to the test function, which is always __tb_test_fn
        # __tb_test_kwargs is passed as globals+locals to exec
        self.code = (
            f"{code}\n__tb_test_reward = {TEST_CASE_FN_NAME}(**__tb_test_kwargs)"
        )
        self.judge = judge
        self.fixtures = fixtures
        self.test_uid = test_uid

    def evaluate_sync(self, ctx: TestContext) -> TestResult:
        """
        Execute the test on a given test context. This function does not propagate exceptions.
        It catches any exceptions and returns a TestResult object containing exception information.
        If an exception is raised that is not related to a test failure (not an assertion),
        then TestResult.is_system_error is True, and TestResult.result should be ignored.
        """

        # NO ISOLATION HERE!
        print_fn = MockPrint()

        # note: this is not real, it doesn't get imported in the global modules namespace
        module_name = _get_safe_module_name("string")

        fn_kwargs = {
            "x": ctx,
            "judge": self.judge,
        }
        ctx_globals = {
            "print": print_fn,
            "__tb_test_kwargs": fn_kwargs,
            "__name__": module_name,
            "__tb_test_reward": None,  # Placeholder, will be set later if supported by the test
        }
        res = TestResult(result=True, reward=1.0, is_system_error=False)
        try:
            # pass the same dictionary to both globals and locals. This is required to
            # allow calling a function in another one, when both defined in the exec
            # code block.

            # Explanation
            #
            # def fn(): ...  # <-- this goes in locals
            #
            # def fn2():
            #   fn()  # <-- this only works if fn is in globals, because locals changed here
            #
            # overall, this works only if globals = locals in the global context where
            # functions are declared.

            with fixtures_context_for_test_fn(
                self.fixtures,
                self.code,
                function_name=TEST_CASE_FN_NAME,
                test_name=self.test_uid,
                runtime_context={"judge": self.judge, "x": ctx},
            ) as fixture_instances:
                ctx_globals["__tb_test_kwargs"].update(fixture_instances)
                exec(self.code, ctx_globals, ctx_globals)

            # Capture the return value after successful execution
            reward_value = ctx_globals.get("__tb_test_reward")

            if reward_value is not None:
                res.reward = reward_value

        except Exception as e:
            _exc_info_to_test_result_inplace(
                is_system_error=not isinstance(e, AssertionError),
                exc_info=sys.exc_info(),
                sources={module_name: self.code},
                res=res,
            )

        res.metadata.update(ctx.metadata)
        motivations = self.judge.drain_decisions()
        res.metadata["judge_motivation"] = motivations or None
        res.prints = print_fn.getvalue()
        return res

    async def evaluate(self, ctx: TestContext) -> TestResult:
        return await asyncio.to_thread(
            self.evaluate_sync,
            ctx,
        )


class TestScriptDebug(TestScriptBase):
    """
    Test script implementation that imports a test case file
    and executes a given test function, allowing to set breakpoints
    in a IDE and debug the test.

    This requires the file to exist on the filesystem. It will be
    loaded as a module with a unique name
    The `print` function is not captured.
    """

    def __init__(
        self,
        test_name: str,
        source_file: str | Path,
        judge: Judge,
        fixtures: Fixtures | None = None,
        test_uid: str | None = None,
    ):
        self.test_name = test_name
        self.module_name = _get_safe_module_name(source_file)
        spec = importlib.util.spec_from_file_location(self.module_name, source_file)
        self.module = importlib.util.module_from_spec(spec)
        sys.modules[self.module_name] = self.module
        spec.loader.exec_module(self.module)
        self.code = inspect.getsource(self.module)
        self.judge = judge
        self.fixtures = fixtures
        self.test_uid = test_uid

    def evaluate_sync(self, ctx: TestContext) -> TestResult:
        # NO ISOLATION HERE!
        res = TestResult(result=True, reward=1.0, is_system_error=False)
        try:
            with fixtures_context_for_test_fn(
                self.fixtures,
                self.code,
                function_name=self.test_name,
                test_name=self.test_uid,
                runtime_context={"judge": self.judge, "x": ctx},
            ) as fixture_instances:
                test_fn = getattr(self.module, self.test_name)
                reward_value = test_fn(
                    x=ctx,
                    judge=self.judge,
                    **fixture_instances,
                )

            if reward_value is not None:
                res.reward = reward_value

        except Exception as e:
            _exc_info_to_test_result_inplace(
                is_system_error=not isinstance(e, AssertionError),
                exc_info=sys.exc_info(),
                sources={self.module_name: self.code},
                res=res,
            )
        res.metadata.update(ctx.metadata)
        motivations = self.judge.drain_decisions()
        res.metadata["judge_motivation"] = motivations or None
        res.prints = ""
        return res

    async def evaluate(self, ctx: TestContext) -> TestResult:
        return await asyncio.to_thread(
            self.evaluate_sync,
            ctx,
        )


class TestScriptSubprocess(TestScriptBase):
    """
    Test script implementation that uses TestScript to execute
    the test in a separate process
    """

    def __init__(
        self,
        code: str,
        judge_config: LLMSessionConfigT,
        judge_type: str = "legacy",
        fixtures_config: dict[str, FixtureConfig] | None = None,
        test_uid: str | None = None,
    ):
        self.code = code
        self.judge_config = judge_config
        self.judge_type = judge_type
        self.fixtures_config: dict[str, FixtureConfig] = fixtures_config or {}
        self.test_uid = test_uid

    @staticmethod
    def test_process_entrypoint(input_stream, output_stream):
        # read request json from an input stream (e.g. stdin)
        request = json.load(input_stream)
        if not isinstance(request, dict):
            raise ValueError("Request payload must be a dictionary")

        initialize_dns_cache(request.get("dns_cache"))
        # deserialize request
        code = request["code"]
        judge_config = TypeAdapter(LLMSessionConfigT).validate_python(
            request["judge_config"]
        )
        judge_type = request.get("judge_type", "legacy")

        fixtures_config = TypeAdapter(dict[str, FixtureConfig]).validate_python(
            request["fixtures_config"],
        )
        test_context = TestContext(**request["test_context"])
        test_uid = request.get("test_uid")

        # execute test
        judge = Judge(create_llm_session(judge_config), judge_type=judge_type)
        fixtures = build_fixtures(fixtures_config)
        testscript = TestScript(
            code,
            judge,
            fixtures=fixtures,
            test_uid=test_uid,
        )
        result = testscript.evaluate_sync(test_context)

        # serialize response
        result_json = result.model_dump_json()

        # write response to the output stream after a "\0" separator, to distinguish
        # the response from anything else that might have been written before.
        # assuming no other threads write after this point, this is reliable, because
        # "\0" is not valid in a JSON string
        output_stream.flush()
        output_stream.write("\0")
        output_stream.write(result_json)
        output_stream.flush()

    def evaluate_sync(self, ctx: TestContext) -> TestResult:
        return syncify(self.evaluate(ctx))

    async def evaluate(self, ctx: TestContext) -> TestResult:
        res = TestResult(result=True, reward=1.0, is_system_error=True)

        dns_cache = get_dns_cache()
        dns_cache_copy: dict | None = None
        if dns_cache is not None:
            dns_cache_copy = await dns_cache.copy()

        try:
            # judge_config is of type LLMSessionConfigT
            judge_config = to_jsonable_python(self.judge_config)

            # serialize code, judge_config, test_context
            request = json.dumps(
                {
                    "code": self.code,
                    "judge_config": judge_config,
                    "judge_type": self.judge_type,
                    "test_context": to_jsonable_python(ctx),
                    "dns_cache": dns_cache_copy,
                    "fixtures_config": to_jsonable_python(self.fixtures_config),
                    "test_uid": self.test_uid,
                }
            )
            request = request.encode("utf-8")

            # run test in a separate process
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "thinkingbox.cli.testscript_worker",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate(input=request)

            # look for the last occurrence of the response separator.
            # This is reliable, because "\0" is not allowed in a JSON string
            pos = stdout.rfind(b"\0")
            if pos == -1:
                raise ValueError(
                    "The test process did not return any result."
                    + f"\nstdout: {stdout.decode('utf-8')}\nstderr: {stderr.decode('utf-8')}"
                )

            # replace result with the one from the json
            res = TestResult.model_validate_json(stdout[pos + 1 :].decode("utf-8"))
        except Exception:
            # keep the promise to not raise exception in evaluate
            _exc_info_to_test_result_inplace(
                is_system_error=True,
                exc_info=sys.exc_info(),
                sources={},
                res=res,
            )
        return res
