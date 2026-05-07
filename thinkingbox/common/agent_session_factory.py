# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Callable

from thinkingbox.common.agent_session import AgentSession
from thinkingbox.common.agent_session_base import AgentSessionBase
from thinkingbox.common.config_types import OrchestratorConfigT
from thinkingbox.common.llm_session_factory import create_llm_session


def get_agent_session_factory(
    config: OrchestratorConfigT,
) -> Callable[..., AgentSessionBase]:
    session_config = config.agent_model

    def factory(**kwargs) -> AgentSessionBase:
        agent_model = create_llm_session(session_config)
        return AgentSession.from_config(agent_model=agent_model, **kwargs)

    return factory
