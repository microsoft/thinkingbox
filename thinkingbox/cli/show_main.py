# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import functools
import io
import json
import os
import subprocess
import sys

import click
import yaml
from prompt_toolkit import HTML
from prompt_toolkit.output.color_depth import ColorDepth
from prompt_toolkit.output.vt100 import Vt100_Output
from pydantic_core import to_jsonable_python

from thinkingbox.cli.common import pprint, pprint_yaml
from thinkingbox.common.chat_types import DecodeResult


def pprint_result(r: DecodeResult, out_stream=sys.stdout):
    output = Vt100_Output(
        out_stream,
        default_color_depth=ColorDepth.TRUE_COLOR,
        enable_bell=False,
        get_size=lambda: tuple(os.get_terminal_size()),
    )

    pprint_kw = dict(
        output=output,
        color_depth=ColorDepth.TRUE_COLOR,
    )
    pprint_with_output = functools.partial(pprint, **pprint_kw)
    pprint_yaml_with_output = functools.partial(pprint_yaml, **pprint_kw)

    result_text: str = ""
    if r.is_system_error:
        result_text = "Decode Error"
    elif r.test_result:
        if r.test_result.is_system_error:
            result_text = "Test Error"
        else:
            result_text = f"{r.test_result.result!s} ({r.test_result.reward:.2f})"

    pprint_yaml_with_output(
        HTML("<i>[info]</i>"),
        {
            "uid": r.uid,
            "result": result_text,
        },
    )

    for msg in r.messages:
        for pp in msg.pp(html=True):
            pprint_with_output(HTML(pp))

    if r.test_context:
        pprint_yaml_with_output(
            HTML("<i>[test_context]</i>"),
            to_jsonable_python(r.test_context),
        )

    if r.test_result:
        for pp in r.test_result.pp(html=True):
            pprint_with_output(HTML(pp))


def _read_line_from_file(file: io.TextIOWrapper, line_number: int) -> str:
    """Read a specific line (1-based) from a file without loading the entire file."""
    for current_line, content in enumerate(file, start=1):
        if current_line == line_number:
            return content.strip()
    raise ValueError(f"Line {line_number} not found in file")


def _is_jsonl_format(first_line: str) -> bool:
    """Try to parse the first line as JSON to determine if this is JSONL format."""
    try:
        json.loads(first_line)
        return True
    except json.JSONDecodeError:
        return False


def _read_jsonl_single_entry(file: io.TextIOWrapper, first_line: str) -> str:
    """
    Read a JSONL file and validate it contains only one non-empty line.
    Returns the data if valid, raises ValueError otherwise.
    """
    data = first_line
    for line in file:
        if line.strip():
            raise ValueError(
                "More than one line of input. Use --line to filter to one line"
            )
    return data


@click.command()
@click.argument(
    "input_filename", required=False, default="-", metavar="INPUT", type=click.File("r")
)
@click.option(
    "--less",
    is_flag=True,
    help="Pipe output through `less -R` for paging.",
)
@click.option(
    "--line", type=click.IntRange(min=1), help="Line number to read (1-based)."
)
def pp(input_filename: io.TextIOWrapper, less: bool, line: int | None) -> None:
    """
    Pretty-print a decoded result from a file or stdin.

    INPUT is the path to a file to decode, or '-' to read from stdin.
    """
    # Choose output stream
    less_proc = None
    if less:
        less_proc = subprocess.Popen(
            ["less", "-R"], stdin=subprocess.PIPE, encoding="utf-8"
        )
        out_stream = less_proc.stdin
    else:
        out_stream = sys.stdout

    # Note: despite the name, input_filename is not a path,
    # it's already a stream (click auto-opens it)
    try:
        # Read input data
        if line is not None:
            # Read line by line until we reach the desired line (1-based)
            data = _read_line_from_file(input_filename, line)
            # When --line is used, we assume JSONL format and parse as JSON
            r: DecodeResult = DecodeResult.model_validate_json(data)
        else:
            # Read the first line to determine format
            first_line = input_filename.readline().strip()

            # Empty first line or non-JSON line indicates YAML format
            # (JSONL doesn't allow empty lines)
            if first_line and _is_jsonl_format(first_line):
                # JSONL format: validate only one non-empty line exists
                data = _read_jsonl_single_entry(input_filename, first_line)
                r = DecodeResult.model_validate_json(data)
            else:
                # YAML format (also covers multi-line JSON): read entire file
                full_content = first_line + "\n" + input_filename.read()
                r = DecodeResult(**yaml.safe_load(full_content))

        # Pretty print
        pprint_result(r, out_stream)
    finally:
        # Cleanup if paging
        if less_proc is not None:
            out_stream.close()
            less_proc.wait()
