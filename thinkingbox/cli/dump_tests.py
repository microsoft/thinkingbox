# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
from pathlib import Path

import click
import yaml

from thinkingbox.common.hydrator import (
    iter_cases_by_names,
    iter_cases_from_file_or_folder,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def dump_test_cases(
    dataset: str,
    agent: str,
    input_file_or_folder: str | Path | None,
    test_list_file: str | Path | None,
    output: str | Path,
):
    if test_list_file is not None:
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

    cnt = 0
    with open(output, "w") as f:
        for tc in test_case_gen:
            logger.debug(
                f"Dumping test case: {tc.metadata['test_case_file']}{tc.metadata['test_case_name']}"
            )
            f.write(tc.model_dump_json() + "\n")
            cnt += 1
    logger.info(f"Dumped {cnt} test cases to {output}")


@click.command()
@click.option("-d", "--dataset", required=True, help="Dataset name (required).")
@click.option("-a", "--agent", default="base", show_default=True, help="Agent name.")
@click.option(
    "-i",
    "--inputs",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Test case directory or file (multiple tests).",
)
@click.option(
    "--test-list",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="YAML file containing a list of test case names in 'filename:testname' format",
)
@click.option(
    "-o",
    "--output",
    required=True,
    type=click.Path(path_type=Path),
    help="Output file (required).",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging.")
def dump_tests(
    dataset: str,
    agent: str,
    inputs: Path | None,
    test_list: Path | None,
    output: Path,
    verbose: bool,
):
    """
    Dump test cases for a given agent and dataset.

    Examples:
      uv run tb dump-tests -d mydataset -a agentX -i tests/ -o output.json
      uv run tb dump-tests -d mydataset -i case.json -o results.json
      uv run tb dump-tests -d mydataset --test-list list.yaml -o output.json
    """
    if verbose:
        logger.setLevel(logging.DEBUG)

    if sum([inputs is None, test_list is None]) != 1:
        raise click.UsageError("Exactly one of --inputs, --test-list must be set")

    logger.debug(f"Dumping test cases for agent {agent} to {output}")

    dump_test_cases(
        dataset=dataset,
        agent=agent,
        input_file_or_folder=inputs,
        test_list_file=test_list,
        output=output,
    )
