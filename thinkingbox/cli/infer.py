#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import asyncio
import logging
import sys
import time
import traceback
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import click
import yaml
from pydantic import BaseModel, TypeAdapter
from pydantic_core import to_jsonable_python

from thinkingbox.cli.common import iter_with_previous_result, load_yaml
from thinkingbox.cli.show_main import pprint_result
from thinkingbox.common.agent_session_factory import get_agent_session_factory
from thinkingbox.common.agent_user_loop import run_agent_user_loop
from thinkingbox.common.chat_types import (
    DecodeResult,
    TestContext,
    TestResult,
)
from thinkingbox.common.config_types import (
    ConfigFile,
    HydratedTestCase,
    merge_init_config,
)
from thinkingbox.common.fixtures import build_fixtures
from thinkingbox.common.http_client import initialize_dns_cache
from thinkingbox.common.hydrator import (
    _get_agent_config,
    get_dataset_case_by_name,
    iter_cases_by_names,
    iter_cases_from_file_or_folder,
)
from thinkingbox.common.judge import Judge
from thinkingbox.common.llm_session_factory import create_llm_session
from thinkingbox.common.mcp_proxy_client import MCPProxyClient
from thinkingbox.common.ordered_parallel_executor import (
    Worker,
    WorkResult,
    iter_map_parallel_ordered,
    iter_map_sequential,
)
from thinkingbox.common.testrunner import TestScriptDebug, TestScriptSubprocess
from thinkingbox.common.user_simulated_answer import SYSTEM_PROMPT, USER_PROMPT
from thinkingbox.common.utils import (
    ErrorInfo,
    Timers,
    enter_async_context_if_not_none,
)

logger = logging.getLogger(__name__)


class TBWorker(Worker[HydratedTestCase, DecodeResult]):
    def __init__(
        self,
        config: ConfigFile,
        skip_test: bool = False,
        skip_agent: bool = False,
        dump_tools: bool = False,
        dump_testcontext: bool = False,
        dump_userllm: bool = False,
        dump_raw: bool = False,
        debug_test: bool = False,
        linger_sessions: bool = False,
        print_lock: asyncio.Lock | None = None,
    ):
        self.config = config
        self.skip_test = skip_test
        self.skip_agent = skip_agent
        self.dump_tools = dump_tools
        self.dump_testcontext = dump_testcontext
        self.dump_userllm = dump_userllm
        self.dump_raw = dump_raw
        self.debug_test = debug_test
        self.linger_sessions = linger_sessions
        self.print_lock = print_lock

    async def work(self, work: HydratedTestCase) -> WorkResult[DecodeResult]:
        previous_result = work.metadata.pop("previous_result", None)
        if isinstance(previous_result, DecodeResult):
            if not previous_result.is_system_error:
                is_correct = (
                    previous_result.test_result.result
                    if previous_result.test_result is not None
                    else False
                )
                return WorkResult(result=previous_result, is_correct=is_correct)

        result, error = None, None
        start_time = time.time()
        timers = Timers()
        timers.ensure("time_agent")
        timers.ensure("time_user")
        timers.ensure("time_tool")
        timers.ensure("time_test")
        timers.ensure("time_agent_user_loop")
        try:
            result = await self._decode(work, timers)
        except Exception as e:
            # If an exception happened, we don't have a result.
            # Create an error result and populate it with exception information

            metadata = work.metadata.copy()
            error_metadata = ErrorInfo(
                type=type(e).__name__,
                message=str(e),
                tb=traceback.format_exc(),
            )
            metadata["error"] = error_metadata.model_dump()
            result = self.get_error_result(work)
            result.metadata = metadata

        # we have a result here
        #   either result.is_system_error is False
        #   or result.is_system_error is True and result.metadata["error"]
        #   contains error information
        end_time = time.time()
        execution_time = end_time - start_time
        result.metadata["execution_time"] = execution_time

        if result.is_system_error:
            error = ErrorInfo(**result.metadata.get("error", {}))
            async with enter_async_context_if_not_none(self.print_lock):
                logger.error("Error decoding test case %s: %s", work.uid, error.message)
                if error.tb:
                    print("Traceback:", file=sys.stderr)
                    print(error.tb, file=sys.stderr)

        is_correct = (
            result.test_result.result if result.test_result is not None else False
        )
        return WorkResult(
            result=result, is_system_error=result.is_system_error, is_correct=is_correct
        )

    async def _decode(self, tc: HydratedTestCase, timers: Timers) -> DecodeResult:
        # create a session within the MCP session proxy
        server_config = merge_init_config(
            tc.scenario.world_state,
            tc.init,
        )

        async with MCPProxyClient.session_context_from_config(
            config=self.config.mcp_proxy,
            server_config=server_config,
            available_tools=[t.name for t in tc.scenario.tools],
        ) as mcp_proxy:
            if self.skip_agent:
                # Skip agent loop — only validate init and optionally run test
                result = DecodeResult(
                    uid=tc.uid,
                    messages=[],
                    test_result=None,
                    test_context=None,
                    test_tags=tc.tags,
                    tools=None,
                    raw_messages=None,
                    user_llm_history=None,
                    usage=[],
                    metadata=tc.metadata.copy(),
                    finish_reason="skipped",
                )
                should_test = bool(tc.test_code) and not self.skip_test
                should_store_testcontext = should_test or self.dump_testcontext
                test_context: TestContext | None = None
                if should_store_testcontext:
                    effects = await mcp_proxy.get_effects()
                    test_context = TestContext(
                        response="",
                        tool_calls=[],
                        tool_direct_responses=[],
                        effects=effects,
                        messages=[],
                        session_id=mcp_proxy.session_id,
                        init_result=mcp_proxy.init_result,
                    )
                if should_test:
                    assert test_context is not None
                    with timers.measure("time_test"):
                        test_result = await self._run_test(tc, test_context)
                    result.test_result = test_result
                if self.dump_testcontext:
                    result.test_context = test_context
                timers.merge_into(result.metadata)
                if self.linger_sessions:
                    mcp_proxy.linger()
                return result

            user_llm = (
                create_llm_session(self.config.user_model)
                if self.config.user_model
                else None
            )

            should_test = bool(tc.test_code) and not self.skip_test

            agent_session_factory = get_agent_session_factory(
                config=self.config.orchestrator,
            )

            with timers.measure("time_agent_user_loop"):
                result = await run_agent_user_loop(
                    tc,
                    agent_session_factory=agent_session_factory,
                    mcp_proxy=mcp_proxy,
                    user_model=user_llm,
                    store_user_sim_history=self.dump_userllm,
                    store_tool_definitions=self.dump_tools,
                    store_test_context=should_test or self.dump_testcontext,
                    user_can_end_conversation=self.config.user_can_end_conversation,
                    store_raw_messages=self.dump_raw,
                )

            if self.linger_sessions:
                mcp_proxy.linger()

            if should_test and not result.is_system_error:
                with timers.measure("time_test"):
                    test_result = await self._run_test(tc, result.test_context)
                result.test_result = test_result
            if not self.dump_testcontext:
                result.test_context = None
            timers.merge_into(result.metadata)
            return result

    async def _run_test(
        self,
        tc: HydratedTestCase,
        test_ctx: TestContext,
    ) -> TestResult:
        if not self.debug_test:
            test = TestScriptSubprocess(
                code=tc.test_code,
                judge_config=self.config.judge_model,
                judge_type=self.config.judge_type,
                fixtures_config=tc.fixtures,
                test_uid=tc.uid,
            )
        else:
            # with --debug-test, --name is required, so we expect that the test comes
            # from the dataset directory and the related information in metadata is
            # correct
            judge_llm = create_llm_session(self.config.judge_model)
            test = TestScriptDebug(
                tc.metadata["test_case_name"],
                tc.metadata["test_case_file"],
                judge=Judge(judge_llm, judge_type=self.config.judge_type),
                fixtures=build_fixtures(tc.fixtures),
                test_uid=tc.uid,
            )
        res = await test.evaluate(test_ctx)
        return res

    def get_error_result(self, work: HydratedTestCase) -> DecodeResult:
        return DecodeResult(
            uid=work.uid,
            messages=[],
            test_result=None,
            test_context=None,
            test_tags=work.tags,
            tools=None,
            raw_messages=None,
            user_llm_history=None,
            metadata={
                "error": ErrorInfo(
                    message="Task cancelled",
                    type="asyncio.CancelledError",
                ).model_dump(),
            },
            is_system_error=True,
        )


class WriterJSONL:
    def __init__(self, f):
        self.f = f

    def write(self, obj: BaseModel):
        self.f.write(obj.model_dump_json())
        self.f.write("\n")
        self.f.flush()


class WriterYAML:
    def __init__(self, f):
        self.f = f
        self.count = 0

    def write(self, obj: BaseModel):
        if self.count > 0:
            raise ValueError("Can only dump one output to YAML")
        yaml.dump(to_jsonable_python(obj), self.f, sort_keys=False)
        self.f.flush()
        self.count += 1


def repeat_test_cases(test_case_iterator: Iterator[HydratedTestCase], repeat: int = 1):
    for tc in test_case_iterator:
        for i in range(repeat):
            tc_copy = tc.model_copy()
            tc_copy.metadata = tc.metadata.copy()
            tc_copy.metadata["repetition"] = i
            yield tc_copy


async def async_main(
    decoder_args: dict,
    dataset: str,
    agent: str,
    input_name: str | None,
    input_file_or_folder: str | None,
    test_list_file: str | None,
    output: str | Path,
    repeat: int = 1,
    batch_size: int = 5,  # Adjust based on API rate limits
    max_output_queue: int = -1,
    watchdog_timeout: float | None = None,
    previous_results_file: str | Path | None = None,
    verbose: bool = False,
):
    if input_name is not None:
        tc = get_dataset_case_by_name(
            input_name,
            base_dir=dataset,
            agent=agent,
        )
        test_case_gen = iter([tc])
    elif test_list_file is not None:
        # Load test cases from a YAML file containing a list of test case names
        with open(test_list_file, "r", encoding="utf-8") as f:
            test_names = yaml.safe_load(f)
        if not isinstance(test_names, list):
            raise ValueError(
                f"Test list file {test_list_file} must contain a YAML list of test case names"
            )
        test_case_gen = iter_cases_by_names(
            test_names,
            base_dir=dataset,
            agent=agent,
            strict=True,
        )
    else:
        test_case_gen = iter_cases_from_file_or_folder(
            input_file_or_folder,
            base_dir=dataset,
            agent=agent,
        )

    output = Path(output).expanduser()
    output_cfg = output.parent / f"{output.stem}_run_metadata.yaml"
    is_yaml = output.suffix == ".yaml"
    if max_output_queue < 0:
        max_output_queue = batch_size * 5

    tests = repeat_test_cases(test_case_gen, repeat)
    if previous_results_file is not None:
        tests = iter_with_previous_result(tests, previous_results_file)

    print_lock = asyncio.Lock()
    decoder_args["print_lock"] = print_lock
    worker = TBWorker(**decoder_args)

    t_beg = time.monotonic()
    dt_beg = datetime.now(timezone.utc)
    with open(output, "w", encoding="utf-8") as f:
        writer = (WriterYAML if is_yaml else WriterJSONL)(f)
        if batch_size == 1:
            # Use sequential variant for easier debugger when not batching
            results_iterator = iter_map_sequential(
                tests,
                worker=worker,
            )
        else:
            results_iterator = iter_map_parallel_ordered(
                tests,
                max_parallelism=batch_size,
                max_input_queue=batch_size,
                max_output_queue=max_output_queue,
                worker=worker,
                watchdog_timeout=watchdog_timeout,
            )
        async for result in results_iterator:
            if verbose:
                async with print_lock:
                    pprint_result(result)
            writer.write(result)

    t_elapsed = time.monotonic() - t_beg

    # Capture Run Configurations:
    with open(output_cfg, "w", encoding="utf-8") as f:
        writer = WriterYAML(f)

        agent_config = {}
        if dataset and agent:
            agent_path = Path(dataset).expanduser().resolve()
            agent_config = _get_agent_config(agent_path, agent)

        writer.write(
            {
                "setup_config": decoder_args["config"],
                "agent_config": agent_config,
                "user_config": {
                    "system_instructions": SYSTEM_PROMPT,
                    "user_prompt": USER_PROMPT,
                },
                "run_start_dt": dt_beg,
                "run_duration (sec)": t_elapsed,
            }
        )

    logger.info("Time elapsed (seconds): %.2f", t_elapsed)
    logger.info("Test results written to: %s", output)
    logger.info("Test config written to: %s", output_cfg)


def dump_arg(ctx, param, value):
    adapter = TypeAdapter(list[Literal["tools", "testcontext", "userllm", "raw"]])
    if not value:
        return []
    return adapter.validate_python([x.strip() for x in value.split(",")])


@click.command()
@click.option("-c", "--config", required=True, help="Configuration file path")
@click.option("-d", "--dataset", default=None, help="Dataset path")
@click.option("-a", "--agent", default=None, help="Agent name")
@click.option(
    "-n",
    "--name",
    default=None,
    help="Test case name (single test). Example: --name banking.py:test_transfer_and_balance",
)
@click.option(
    "-i",
    "--inputs",
    default=None,
    help="Test case directory or file (multiple tests), or JSONL file with hydrated test cases",
)
@click.option(
    "--test-list",
    default=None,
    help="YAML file containing a list of test case names in 'filename:testname' format",
)
@click.option("--no-test", is_flag=True, help="Do not run test")
@click.option(
    "--skip-agent",
    is_flag=True,
    help="Skip agent loop, only validate init and optionally run test (if tests exist and --no-test is not set)",
)
@click.option(
    "--dump",
    callback=dump_arg,
    help="Comma-separated list of: tools, testcontext, userllm, raw",
    default="",
)
@click.option("-r", "--repeat", type=int, default=1, help="Repeat decoding N times")
@click.option(
    "-b",
    "--batch-size",
    type=int,
    default=5,
    help="Batch size for decoding (default: 5)",
)
@click.option(
    "--output-queue-size",
    type=int,
    default=-1,
    help="How many out-of-order items to keep in memory before blocking (default: 10 * batch-size)",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
@click.option("-o", "--output", required=True, help="Output file path")
@click.option(
    "--debug-test",
    is_flag=True,
    help="Import the test code file directly, to allow debugging",
)
@click.option(
    "--linger-sessions",
    is_flag=True,
    help="Do not destroy sessions in the session proxy after finishing decoding",
)
@click.option(
    "--timeout",
    type=float,
    default=900.0,
    help="Timeout (in seconds) for batch processing queue inactivity; zero or negative values disable timeout",
)
@click.option(
    "--previous-results-file",
    default=None,
    help="Previous results file with errors: non-error rows are copied, error rows are re-tried",
)
def infer(
    config,
    dataset,
    agent,
    name,
    inputs,
    test_list,
    no_test,
    skip_agent,
    dump,
    repeat,
    batch_size,
    output_queue_size,
    verbose,
    output,
    debug_test,
    linger_sessions,
    timeout,
    previous_results_file,
):
    """Execute inference for a single test or set of tests."""

    if sum([inputs is None, name is None, test_list is None]) != 2:
        raise click.ClickException(
            "Exactly one of --name, --inputs, --test-list must be set"
        )

    if debug_test and ((not name) or (repeat != 1)):
        raise click.ClickException(
            "--debug-test is only supported for a single test run (using --name) and repeat=1"
        )

    config_obj = load_yaml(config, ConfigFile)

    decoder_kwargs = {
        "config": config_obj,
        "skip_test": no_test,
        "skip_agent": skip_agent,
        "dump_tools": "tools" in dump,
        "dump_testcontext": "testcontext" in dump,
        "dump_userllm": "userllm" in dump,
        "dump_raw": "raw" in dump,
        "debug_test": debug_test,
        "linger_sessions": linger_sessions,
    }

    initialize_dns_cache()

    timeout = timeout if timeout > 0 else None

    asyncio.run(
        async_main(
            decoder_args=decoder_kwargs,
            dataset=dataset,
            agent=agent,
            input_name=name,
            input_file_or_folder=inputs,
            test_list_file=test_list,
            output=output,
            repeat=repeat,
            batch_size=batch_size,
            max_output_queue=output_queue_size,
            watchdog_timeout=timeout,
            previous_results_file=previous_results_file,
            verbose=verbose,
        )
    )
