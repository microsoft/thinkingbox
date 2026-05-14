# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import html
import io
import logging
import shlex
from typing import Any, Iterator

import yaml
from prompt_toolkit import HTML, PromptSession, print_formatted_text
from prompt_toolkit.styles import Style

from thinkingbox.common.chat_types import DecodeResult
from thinkingbox.common.config_types import HydratedTestCase
from thinkingbox.common.utils import ThinkingBoxError, iter_validate_jsonl

logger = logging.getLogger(__name__)

pp_style = Style.from_dict(
    {
        "error": "fg:ansired",
        "good": "fg:ansibrightgreen",
        "system": "fg:ansiwhite",
        "user": "fg:ansicyan",
        "assistant": "fg:ansibrightgreen",
        "think": "fg:ansiwhite",
        "toolcall": "fg:ansiyellow",
        "toolresp": "fg:ansiwhite",
    }
)


def pprint(*args, **kwargs) -> None:
    print_formatted_text(*args, style=pp_style, **kwargs)


def load_yaml(path, cls=None):
    with open(path, "r") as f:
        out = yaml.safe_load(f)
    if cls is not None:
        return cls(**out)
    return out


def tui_parse_command(s: str) -> tuple[str, list[str]]:
    cmd = shlex.split(s)
    if not cmd:
        return "", []
    return cmd[0], cmd[1:]


async def tui_input(prompt) -> str:
    session = PromptSession(multiline=True)
    return await session.prompt_async(
        prompt,
        multiline=True,
        wrap_lines=True,
        style=pp_style,
    )


def pprint_yaml(title, obj: Any, **kwargs):
    pprint(title, **kwargs)
    buf = io.StringIO()
    yaml.dump(obj, buf, sort_keys=False)
    pprint(buf.getvalue().rstrip("\n"), **kwargs)
    pprint("", **kwargs)


def wrap_pprint_tb_error_async(func):
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except ThinkingBoxError as e:
            exception_message = html.escape(str(e))
            pprint(HTML(f"<error>Failure: {exception_message}</error>"))
            # If the exception was raised from another one, re-raise the original
            if e.__cause__ is not None:
                raise e.__cause__
            raise

    return wrapper


def iter_with_previous_result(tests: Iterator[HydratedTestCase], previous_results_file):
    with open(previous_results_file, "r", encoding="utf-8") as prev_f:
        prev_iter = iter_validate_jsonl(prev_f, model=DecodeResult)
        prev = next(prev_iter, None)

        for test in tests:
            test_key = (test.metadata.get("repetition"), test.uid)

            # Advance previous_results until we find a match or pass the current test
            while prev is not None:
                prev_key = (prev.metadata.get("repetition"), prev.uid)
                if prev_key == test_key:
                    test.metadata["previous_result"] = prev
                    prev = next(prev_iter, None)
                    break
                # If prev_key < test_key, this previous result has no matching test; skip it
                logger.warning(
                    f"Skipping unmatched previous result: rep={prev_key[0]} uid={prev_key[1]}"
                )
                prev = next(prev_iter, None)

            yield test
