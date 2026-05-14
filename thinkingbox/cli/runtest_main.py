#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import asyncio
from pathlib import Path

import click
import yaml
from prompt_toolkit import HTML
from pydantic import BaseModel
from pydantic_core import to_jsonable_python

from thinkingbox.cli.common import load_yaml, pprint
from thinkingbox.common.chat_types import DecodeResult, TestResult
from thinkingbox.common.config_types import ConfigFile, TestCase
from thinkingbox.common.fixtures import build_fixtures
from thinkingbox.common.hydrator import Dataset, load_conftest_fixtures, load_test_file
from thinkingbox.common.judge import Judge
from thinkingbox.common.llm_session_factory import create_llm_session
from thinkingbox.common.testrunner import TestScriptDebug, TestScriptSubprocess


class TestResultOnlyFile(BaseModel):
    test_result: TestResult | None = None


def get_test_location(base_dir: str | Path, name: str) -> tuple[Path, str]:
    dataset = Dataset(base_dir)
    filename, testname = name.split(":")
    testfile_path = dataset.get_testfile_path(filename)
    return testfile_path, testname


def get_test_by_name(base_dir: str | Path, name: str) -> TestCase:
    testfile_path, testname = get_test_location(base_dir, name)
    test_file = load_test_file(
        testfile_path,
        check_config=False,
    )
    for tc in test_file.test_cases:
        if tc.uid == testname:
            return tc
    raise ValueError(f"Test not found: {name}")


async def async_main(
    config: Path,
    dataset: Path,
    resultfile: Path,
    name: str | None,
    update: bool,
    verbose: bool,
    output: Path | None,
    debug_test: bool,
):
    if sum([update, output is not None]) != 1:
        raise ValueError(
            "Use --update to update the result file in-place, "
            "or --output <file> to write the test result only to a new file"
        )

    config_obj = load_yaml(config, ConfigFile)
    decode_result = load_yaml(resultfile, DecodeResult)

    if update:
        output_file = resultfile
        output_obj = decode_result
    else:
        output_file = output
        output_obj = TestResultOnlyFile()

    if decode_result.test_context is None:
        raise ValueError("test_context is required, found null")

    # find and load test from test file by id
    test_ref = name
    if test_ref is None:
        test_ref = decode_result.uid

    if verbose:
        pprint(f"Loading test case {test_ref}")

    # Load corpus and scenario fixtures (shared by both debug and non-debug paths)
    ds = Dataset(dataset)
    filename = test_ref.split(":")[0]
    test_case_file = ds.get_testfile_path(filename)
    conftest_fixtures = load_conftest_fixtures(
        test_case_file, Path(dataset) / "test_case"
    )

    # run the test
    if not debug_test:
        tc = get_test_by_name(
            base_dir=dataset,
            name=test_ref,
        )
        scenario = ds.get_scenario(tc.scenario)
        resolved_fixtures = {**conftest_fixtures, **scenario.fixtures}
        test = TestScriptSubprocess(
            code=tc.test_code,
            judge_config=config_obj.judge_model,
            judge_type=config_obj.judge_type,
            fixtures_config=resolved_fixtures,
            test_uid=tc.uid,
        )
    else:
        testfile_path, testname = get_test_location(
            base_dir=dataset,
            name=test_ref,
        )
        tc = get_test_by_name(base_dir=dataset, name=test_ref)
        scenario = ds.get_scenario(tc.scenario)
        resolved_fixtures = {**conftest_fixtures, **scenario.fixtures}
        judge_llm = create_llm_session(config_obj.judge_model)

        test = TestScriptDebug(
            testname,
            testfile_path,
            judge=Judge(judge_llm, judge_type=config_obj.judge_type),
            fixtures=build_fixtures(resolved_fixtures),
            test_uid=test_ref,
        )
    res = await test.evaluate(decode_result.test_context)

    if verbose:
        for pp_text in res.pp(html=True):
            pprint(HTML(pp_text))

    # write output
    output_obj.test_result = res

    with open(output_file, "w", encoding="utf-8") as f:
        yaml.dump(to_jsonable_python(output_obj), f, sort_keys=False)


@click.command()
@click.option(
    "-c",
    "--config",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="Path to YAML config file.",
)
@click.option(
    "-d",
    "--dataset",
    default="./dataset",
    type=click.Path(path_type=Path, exists=True),
    show_default=True,
    help="Dataset root directory.",
)
@click.option(
    "-r",
    "--resultfile",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="Decoder result YAML file.",
)
@click.option(
    "-n",
    "--name",
    default=None,
    help="Test case name (single test).",
)
@click.option(
    "--update",
    is_flag=True,
    help="Update the input YAML file instead of writing to a new file.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Enable verbose output.",
)
@click.option(
    "-o",
    "--output",
    default=None,
    type=click.Path(path_type=Path, dir_okay=False, writable=True),
    help="Optional output file path.",
)
@click.option(
    "--debug-test",
    is_flag=True,
    help="Import the test code file directly, to allow debugging.",
)
def run_test(
    config: Path,
    dataset: Path,
    resultfile: Path,
    name: str | None,
    update: bool,
    verbose: bool,
    output: Path | None,
    debug_test: bool,
) -> None:
    """
    Process decoder results and optionally update or write new test files.
    """
    asyncio.run(
        async_main(
            config=config,
            dataset=dataset,
            resultfile=resultfile,
            name=name,
            update=update,
            verbose=verbose,
            output=output,
            debug_test=debug_test,
        )
    )
