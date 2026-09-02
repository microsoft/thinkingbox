# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
import sys
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from thinkingbox.common.chat_types import MessageT, ToolDef
from thinkingbox.common.history_loader import HistoryRef
from thinkingbox.common.recursive_merge import recursive_merge
from thinkingbox.common.tag_types import TestCaseTags

logger = logging.getLogger(__name__)

ReasoningEffort = str | None

# config.yaml


class CredentialConfig(BaseModel):
    type: Literal[""] = ""


class ApiKeyCredentialConfig(BaseModel):
    type: Literal["api-key"] = "api-key"
    api_key: str


class AzureCliCredentialConfig(BaseModel):
    type: Literal["az-cli"] = "az-cli"
    resource: str = "https://cognitiveservices.azure.com/"
    duration: float = 600.0


class AzureManagedIdentityCredentialConfig(BaseModel):
    type: Literal["managed-identity"] = "managed-identity"
    resource: str = "https://cognitiveservices.azure.com/"
    client_id: str | None = None
    duration: float = 600.0


class AzureIdentityCredentialConfig(BaseModel):
    type: Literal["azure-identity"] = "azure-identity"
    name: str
    scope: str

    model_config = {"extra": "allow"}


CredentialConfigT = (
    ApiKeyCredentialConfig
    | AzureCliCredentialConfig
    | AzureManagedIdentityCredentialConfig
    | AzureIdentityCredentialConfig
)


class LLMSessionConfig(BaseModel):
    type: Literal[""] = ""


class HTTPLLMSessionConfig(LLMSessionConfig):
    type: Literal[""] = ""
    credential: CredentialConfigT | None = None
    client_certificate: str | None = None
    trust_ca_path: str | None = None
    headers: dict[str, str] | None = None
    timeout: float = 60.0
    use_dns_cache: bool = False
    max_retries_server_error: int = 5
    retryable_server_errors: tuple[int | str, ...] = (502, 503, 504)

    @field_validator("credential", mode="before")
    @classmethod
    def _coerce_credential(cls, value) -> CredentialConfigT | None:
        if value == "az-cli":
            return AzureCliCredentialConfig()
        return value


class AOAISessionConfig(HTTPLLMSessionConfig):
    type: Literal["aoai"] = "aoai"
    deployment: str
    account_name: str | None = None
    endpoint_url: str | None = None
    seed: int | None = None
    temperature: float = 1.0
    max_completion_tokens: int = 4096
    is_reasoning: bool = False
    reasoning_effort: ReasoningEffort = None
    api_version: str = "2024-10-21"
    disabled_params: list[str] = Field(default_factory=list)
    parallel_tool_calls: bool = False


class AOAIResponsesSessionConfig(HTTPLLMSessionConfig):
    type: Literal["aoai_responses"] = "aoai_responses"
    deployment: str
    account_name: str | None = None
    endpoint_url: str | None = None
    seed: int | None = None
    temperature: float = 1.0
    max_completion_tokens: int = 4096
    is_reasoning: bool = False
    reasoning_source: Literal["none", "summary", "content"] = "summary"
    reasoning_effort: ReasoningEffort = None
    use_stateful_protocol: bool = False
    api_version: str = "2024-10-21"
    parallel_tool_calls: bool = False


class AnthropicMessagesSessionConfig(HTTPLLMSessionConfig):
    type: Literal["anthropic"] = "anthropic"
    deployment: str
    endpoint_url: str
    max_completion_tokens: int = 4096
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    stop_sequences: list[str] | None = None
    thinking: dict | None = None
    output_config: dict | None = None
    parallel_tool_calls: bool = False

    @field_validator("top_k", mode="before")
    @classmethod
    def validate_top_k(cls, v):
        if v is not None and v < 0:
            raise ValueError("top_k must be >= 0")
        return v

    @field_validator("top_p", mode="before")
    @classmethod
    def validate_top_p(cls, v):
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("top_p must be between 0.0 and 1.0")
        return v

    @model_validator(mode="after")
    def check_temperature_and_top_p(self):
        if self.temperature is not None and self.top_p is not None:
            raise ValueError(
                "temperature and top_p cannot both be set at the same time"
            )
        return self


class CustomSessionConfig(BaseModel):
    """
    Configuration for custom LLM session implementations.

    The `factory` should be a fully qualified path to a callable (class or function)
    (e.g. `mymodule.MySession`) that accepts keyword arguments matching the extra
    fields in this config and returns an instance of LLMSessionBase.
    """

    type: Literal["custom"] = "custom"
    factory: str
    model_config = ConfigDict(extra="allow")


LLMSessionConfigT = (
    AOAISessionConfig
    | AOAIResponsesSessionConfig
    | AnthropicMessagesSessionConfig
    | CustomSessionConfig
)


class AgentConfig(BaseModel):
    system_instructions: str
    builtin_tools: list[ToolDef]


class FixtureConfig(BaseModel):
    """
    A fixture specification where:
      - type: dotted python path "package.module.attr"
      - all remaining (extra) fields are treated as keyword arguments for the callable/context manager.
    """

    type: str

    model_config = ConfigDict(extra="allow")


class SessionProxyConfig(BaseModel):
    endpoint_url: str
    timeout: float = 120.0
    credential: CredentialConfigT | None = None
    use_dns_cache: bool = False
    client_certificate: str | None = None
    trust_ca_path: str | None = None
    headers: dict[str, str] | None = None
    max_retries_server_error: int = 5
    retryable_server_errors: tuple[int | str, ...] = (502, 503, 504)
    always_json_output: bool = False
    geteffects_proxy_info: bool = False


class ThinkingBoxOrchestratorConfig(BaseModel):
    type: Literal["thinkingbox"] = "thinkingbox"
    agent_model: LLMSessionConfigT


OrchestratorConfigT = ThinkingBoxOrchestratorConfig


class ConfigFile(BaseModel):
    mcp_proxy: SessionProxyConfig
    judge_model: LLMSessionConfigT
    judge_type: str = "legacy"  # "motivation" opt-in
    user_model: LLMSessionConfigT | None = None
    orchestrator: OrchestratorConfigT
    user_can_end_conversation: bool = False

    mcp_proxy_timeout: float | None = Field(default=None, exclude=True)  # deprecated
    mcp_proxy_use_dns_cache: bool | None = Field(
        default=None, exclude=True
    )  # deprecated
    agent_model: LLMSessionConfigT | None = Field(
        default=None, exclude=True
    )  # deprecated: use orchestrator.agent_model

    @model_validator(mode="before")
    @classmethod
    def _migrate_deprecated_agent_model(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        agent_model = data.get("agent_model")
        orchestrator = data.get("orchestrator")
        if agent_model is not None:
            deprecation_msg = (
                "ConfigFile.agent_model is deprecated; "
                "set orchestrator.agent_model instead."
            )
            if orchestrator is not None:
                raise ValueError(deprecation_msg)
            logger.warning(deprecation_msg)

            # convert previously valid configurations only
            data = data.copy()
            del data["agent_model"]
            data["orchestrator"] = {
                "type": "thinkingbox",
                "agent_model": agent_model,
            }
            return data

        return data

    @field_validator("mcp_proxy", mode="before")
    @classmethod
    def _coerce_mcp_proxy(cls, value) -> SessionProxyConfig | dict:
        if isinstance(value, str):
            return SessionProxyConfig(endpoint_url=value)
        return value

    @model_validator(mode="after")
    def _migrate_deprecated_mcp_proxy_fields(self):
        if self.mcp_proxy_timeout is not None:
            self.mcp_proxy.timeout = self.mcp_proxy_timeout
            self.mcp_proxy_timeout = None
        if self.mcp_proxy_use_dns_cache is not None:
            self.mcp_proxy.use_dns_cache = self.mcp_proxy_use_dns_cache
            self.mcp_proxy_use_dns_cache = None
        return self


# scenario.yaml


class ToolDefOverride(BaseModel):
    name: str
    override_description: str | None = None
    override_arg_description: dict[str, str] | None = None
    direct_response: str | None = None
    is_end_turn: bool = False


class ConftestConfig(BaseModel):
    fixtures: dict[str, FixtureConfig] = Field(default_factory=dict)


class ScenarioConfig(BaseModel):
    # For each server, a configuration that is passed to its initialization function
    world_state: dict[str, Any]

    # list of tools that are visible to the agent, possibly including
    # additional scenario-specific configuration
    tools: list[ToolDefOverride]

    # scenario-specific bot instructions
    bot_instructions: str | None = None

    # tags for all test cases in the scenario
    tags: TestCaseTags = Field(default_factory=TestCaseTags)

    # fixtures usable by all tests in this scenario
    fixtures: dict[str, FixtureConfig] = Field(default_factory=dict)

    # additional scenario metadata, it may contain additional scenario-specific
    # configuration for specific orchestrators
    metadata: dict[str, Any] = Field(default_factory=dict)


# testcase.yaml


class TestCase(BaseModel):
    __test__ = False  # prevent pytest from collecting this as a test class

    uid: str

    # pointer to scenario
    scenario: str

    # user query
    query: str

    # python test code
    test_code: str

    # History reference resolved by the hydrator via .meta.yaml
    history: HistoryRef | None = None

    # additional system instructions
    bot_instructions: str | None = None

    # User-LLM context prompt
    user_context: str | None = None

    # maximum number of simulated user turns
    max_user_sim_turns: int = 10

    # maximum number of agent turns (decode iterations); default is unlimited
    max_agent_sim_turns: int = Field(default=sys.maxsize, ge=0)

    # additional initialization config to be merged with the scenario's world_state
    init: dict[str, dict] = Field(default_factory=dict)

    # whether query, user_context, bot_instructions should be formatted
    # with the dictionary returned by init
    format_query: bool = False

    metadata: dict[str, Any] = Field(default_factory=dict)

    # taxonomy of test cases
    tags: TestCaseTags = Field(default_factory=TestCaseTags)


# Name that identifies the test function in a test code string
TEST_CASE_FN_NAME = "__tb_test_fn"


class TestCaseFile(BaseModel):
    test_cases: list[TestCase]


# Hydrated test case
# = TestCase augmented with AgentConfig, InitConfig


class HydratedTestCase(BaseModel):
    # unique identifier
    uid: str

    # prompts and agent behavior
    agent: AgentConfig

    # scenario
    scenario: ScenarioConfig

    # Resolved prior conversation history injected before the agent loop
    history: list[MessageT] | None = None

    # user query
    query: str

    # python test code
    test_code: str

    # additional system instructions
    bot_instructions: str | None = None

    # User-LLM context prompt
    user_context: str | None = None

    # maximum number of simulated user turns
    max_user_sim_turns: int = 10

    # maximum number of agent turns (decode iterations); default is unlimited
    max_agent_sim_turns: int = Field(default=sys.maxsize, ge=0)

    # additional initialization config to be merged with the scenario's
    init: dict[str, dict] = Field(default_factory=dict)

    # whether query, user_context, bot_instructions should be formatted
    # with the dictionary returned by init
    format_query: bool = False

    metadata: dict[str, Any] = Field(default_factory=dict)

    # taxonomy of test cases
    tags: TestCaseTags = Field(default_factory=TestCaseTags)

    # resolved fixtures (conftest.yaml + scenario overrides)
    fixtures: dict[str, FixtureConfig] = Field(default_factory=dict)


def update_tools_with_client_config(
    tools: list[ToolDef], config_tools: list[ToolDefOverride]
):
    config_tools_map = {tool.name: tool for tool in config_tools}
    for tool in tools:
        override = config_tools_map.get(tool.name)
        if override is None:
            continue
        tool.direct_response = override.direct_response
        tool.is_end_turn = override.is_end_turn
        if override.override_description is not None:
            tool.description = override.override_description
        if override.override_arg_description is not None:
            properties = tool.input_schema.get("properties", {})
            for param_name, desc in override.override_arg_description.items():
                if param_name in properties:
                    properties[param_name]["description"] = desc
                else:
                    logger.warning(
                        f"Tool {tool.name}: override_arg_description key {param_name} does not match "
                        "any param in input_schema. Ignoring param desc override"
                    )


def format_string_or_none(value: str | None, format_values: dict):
    if not value:
        return value
    return value.format(init=format_values)


def merge_bot_instructions(
    scenario_instructions: str | None = None,
    testcase_instructions: str | None = None,
) -> str | None:
    instructions = [scenario_instructions, testcase_instructions]
    instructions = [s for s in instructions if s is not None]
    if not instructions:
        return None
    return "\n".join(instructions)


def merge_init_config(
    scenario_world_state: dict[str, dict],
    test_case_init: dict[str, dict] | None = None,
):
    """Merge the initialization dictionaries from the scenario's world_state
    with additional ones from a test case (if any)
    """
    if not test_case_init:
        return scenario_world_state
    out = scenario_world_state.copy()
    for k, v in test_case_init.items():
        scenario_v = out.get(k)
        if scenario_v:
            out[k] = recursive_merge(scenario_v, v)
        else:
            out[k] = v
    return out


# Rubric configuration types
class RubricConfig(BaseModel):
    """Configuration for a single rubric criterion.

    Attributes:
        criterion: The evaluation criterion/question to be assessed
        weight: The importance weight of this rubric
            - For positive rubrics: maximum points earnable (must be >= 0)
            - For deduction penalties: maximum points deductible (must be >= 0)
            - For multiplicative penalties: maximum scale-down factor (must be in [0.0, 1.0])
        is_penalty: Whether this rubric is a penalty (reduces score) rather than positive
        penalty_type: How the penalty is applied ("deduction" or "multiplicative")
    """

    criterion: str
    weight: float = Field(ge=0)
    is_penalty: bool = False
    penalty_type: Literal["deduction", "multiplicative"] = "deduction"

    @model_validator(mode="after")
    def multiplicative_penalty_weight_must_be_in_range(self) -> "RubricConfig":
        if self.is_penalty and self.penalty_type == "multiplicative":
            if self.weight > 1:
                raise ValueError("multiplicative penalty weight must be in [0.0, 1.0]")
        return self
