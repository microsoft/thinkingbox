# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Callable

from thinkingbox.botdesigner.bd_agent import BotDesignerAgentSession
from thinkingbox.botdesigner.bd_client import BotDesignerClient
from thinkingbox.botdesigner.bot_override import load_template_file
from thinkingbox.common.agent_session import AgentSession
from thinkingbox.common.agent_session_base import AgentSessionBase
from thinkingbox.common.config_types import (
    BotDesignerOrchestratorConfig,
    OrchestratorConfigT,
    ThinkingBoxOrchestratorConfig,
)
from thinkingbox.common.credential_factory import create_credential
from thinkingbox.common.llm_session_factory import create_llm_session


def get_agent_session_factory(
    config: OrchestratorConfigT,
) -> Callable[..., AgentSessionBase]:
    if isinstance(config, ThinkingBoxOrchestratorConfig):
        session_config = config.agent_model

        def factory(**kwargs) -> AgentSessionBase:
            agent_model = create_llm_session(session_config)
            return AgentSession.from_config(agent_model=agent_model, **kwargs)

        return factory

    if isinstance(config, BotDesignerOrchestratorConfig):

        def factory(**kwargs) -> AgentSessionBase:
            credential = (
                create_credential(config.credential) if config.credential else None
            )
            botdesigner_client = BotDesignerClient(
                endpoint=config.endpoint_url,
                environment_id=config.environment_id,
                base_bot_id=config.base_bot_id,
                feature_overrides=config.feature_overrides,
                timeout=config.timeout,
                use_dns_cache=config.use_dns_cache,
                credential=credential,
                client_certificate=config.client_certificate,
                trust_ca_path=config.trust_ca_path,
                headers=config.headers,
                max_retries_server_error=config.max_retries_server_error,
                retryable_server_errors=config.retryable_server_errors,
                use_sse_protocol=config.use_sse_protocol,
            )
            return BotDesignerAgentSession.from_config(
                botdesigner_client=botdesigner_client,
                bot_template=load_template_file(config.bot_template_file),
                bot_variables=config.bot_variables,
                locale=config.locale,
                connector_endpoint_override=config.connector_endpoint_override,
                tool_translation_mode=config.tool_translation_mode,
                recognizer_kind=config.recognizer_kind,
                **kwargs,
            )

        return factory

    raise ValueError(f"Cannot create agent session factory for config: {config!r}")
