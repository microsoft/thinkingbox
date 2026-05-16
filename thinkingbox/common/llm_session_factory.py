# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import importlib
from typing import Callable, NamedTuple

from thinkingbox.common.anthropic_messages_session import AnthropicMessagesSession
from thinkingbox.common.aoai_responses_session import AOAIResponsesSession
from thinkingbox.common.aoai_session import AOAISession
from thinkingbox.common.config_types import (
    AnthropicMessagesSessionConfig,
    AOAIResponsesSessionConfig,
    AOAISessionConfig,
    CustomSessionConfig,
    LLMSessionConfigT,
)
from thinkingbox.common.llm_session_base import LLMSessionBase


class _LLMSessionFactory(NamedTuple):
    factory: Callable[[dict], LLMSessionBase]


_g_factory_cache: dict[str, _LLMSessionFactory] = {}


def _load_factory(factory_path: str) -> _LLMSessionFactory:
    if "." not in factory_path:
        raise ValueError(
            f"Invalid factory path '{factory_path}': must be in format"
            + " 'module.path.ClassName'"
        )
    module_path, attr_name = factory_path.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"Cannot import module part '{module_path}'" + f" for '{factory_path}'"
        ) from e
    factory_func = getattr(module, attr_name, None)
    if factory_func is None:
        raise ImportError(
            f"Cannot import '{attr_name}' from '{module_path}'"
            + f" for '{factory_path}'"
        )
    if not callable(factory_func):
        raise TypeError(f"'{factory_path}' is not callable")
    return _LLMSessionFactory(factory=factory_func)


def create_llm_session(config: LLMSessionConfigT) -> LLMSessionBase:
    if isinstance(config, AOAISessionConfig):
        return AOAISession.from_config(config)

    if isinstance(config, AOAIResponsesSessionConfig):
        return AOAIResponsesSession.from_config(config)

    if isinstance(config, AnthropicMessagesSessionConfig):
        return AnthropicMessagesSession.from_config(config)

    if isinstance(config, CustomSessionConfig):
        config_dict = config.model_dump()
        config_dict.pop("type")
        factory_path: str = config_dict.pop("factory")
        factory: _LLMSessionFactory | None = _g_factory_cache.get(factory_path)
        if factory is None:
            factory = _load_factory(factory_path)
            _g_factory_cache[factory_path] = factory

        return factory.factory(**config_dict)

    raise ValueError(f"Cannot create LLM session for config: {config!r}")
