# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import io
from typing import Any, AsyncIterator

import pytest
import yaml

from tests.mock_session import MockSession, MockSessionLog
from thinkingbox.botdesigner.bd_agent import BotDesignerAgentSession
from thinkingbox.botdesigner.bd_client import (
    BotDesignerClient,
    BotDesignerRequest,
    ConversationContext,
    StepResult,
)
from thinkingbox.botdesigner.bot_override import load_template_file
from thinkingbox.common.agent_user_loop import run_agent_user_loop
from thinkingbox.common.chat_types import (
    DecodeResult,
    ParallelToolCall,
    Text,
    ToolResponse,
)
from thinkingbox.common.config_types import (
    AgentConfig,
    HydratedTestCase,
    ScenarioConfig,
    merge_init_config,
)
from thinkingbox.common.judge import Judge
from thinkingbox.common.mcp_proxy_client import MCPProxyClient
from thinkingbox.common.python_test_file import iter_test_functions
from thinkingbox.common.testrunner import TestScript as _TestScript

SESSION_PROXY_TIMEOUT = 60.0
SESSION_PROXY_ENDPOINT = "http://127.0.0.1:7111"


AGENT_CONFIG = AgentConfig(
    system_instructions="System Instructions",
    builtin_tools=[],
)


# Scenario tools don't matter here, we add some just to test
# connector translation


CLOUD_DRIVE_SCENARIO = ScenarioConfig(
    world_state={
        "cloud_drive": {
            "files": [],
        }
    },
    tools=[
        {"name": "upload_file"},
        {"name": "get_text_content"},
    ],
    bot_instructions="",
)


TEST_WEATHER_60F = r"""
def test_weather_60f(x, judge):
    '''!
    query: none  # ignored
    '''
    assert "60" in x.response
""".lstrip()


def _get_test_code(test_fn_code: str):
    test_fns = list(iter_test_functions(test_fn_code, check_config=True))
    assert (
        len(test_fns) == 1
    ), "this is a problem with the unit test, fix the test code string"
    return test_fns[0].test_code


def _get_bot_override(result: DecodeResult) -> dict:
    for msg in result.raw_messages:
        if msg.get("source") == "bd_agent":
            bot_override = msg.get("data", {}).get("bot_override")
            if bot_override is not None:
                return yaml.safe_load(io.StringIO(bot_override))
    raise ValueError(f"Bot override not found in raw_messages {result.raw_messages!r}")


def _get_connector_dialogs(bot_override: dict[str, Any], connector_id: str):
    conn_refs = {}
    operations = {}
    dialogs = {}
    for conn_ref in bot_override.get("connectionReferences", []):
        if conn_ref.get("connectorId") != connector_id:
            continue
        conn_refs[conn_ref["connectionReferenceLogicalName"]] = conn_ref
    for conn_def in bot_override.get("connectorDefinitions", []):
        if conn_def.get("connectorId") != connector_id:
            continue
        for operation in conn_def.get("operations", []):
            operations[operation["operationId"]] = operation
    for component in bot_override.get("components", []):
        if component.get("kind") != "DialogComponent":
            continue
        if component.get("dialog", {}).get("kind") != "TaskDialog":
            continue
        dialog = component["dialog"]
        if dialog.get("action", {}).get("kind") != "InvokeConnectorTaskAction":
            continue
        dialogs[dialog["modelDisplayName"]] = {
            "dialog": dialog,
            "ref": conn_refs[dialog["action"]["connectionReference"]],
            "operation": operations[dialog["action"]["operationId"]],
        }
    return dialogs


_CONVERSATION_ID = "00000000-0000-0000-0000-00000000002a"


class BotDesignerClientMock(BotDesignerClient):
    def __init__(self, activities: list[dict], log: MockSessionLog | None = None):
        super().__init__(
            endpoint="http://localhost:5000",
            environment_id="00000000-0000-0000-0000-000000000001",
            base_bot_id="00000000-0000-0000-0000-000000000002",
        )
        self.conversation_id = _CONVERSATION_ID
        self.activities = activities.copy()
        self.log = log
        if log is not None:
            if log.instance_created:
                raise RuntimeError(
                    "MockSessionLog.instance_created already set; "
                    "a second session was created with the same log"
                )
            log.instance_created = True
            log.remaining_completions = len(self.activities)

    async def _post(self, *args, **kwargs):
        raise RuntimeError("unreachable")

    async def conversation_start(self, *args, **kwargs) -> ConversationContext:
        resp = self.activities.pop(0)
        if self.log is not None:
            self.log.remaining_completions = len(self.activities)
        result = StepResult(
            activities=resp["activities"],
            action=resp["action"],
            num_steps=1,
        )
        return ConversationContext(
            conversation_id=self.conversation_id,
            result=result,
            req=BotDesignerRequest(bot_id="id"),
            raw_response={},
        )

    async def conversation_continue_iter(
        self, conversation: ConversationContext, limit: int = -1
    ) -> AsyncIterator[StepResult]:
        while True:
            resp = self.activities.pop(0)
            if self.log is not None:
                self.log.remaining_completions = len(self.activities)
            result = StepResult(
                activities=resp["activities"],
                action=resp["action"],
                num_steps=1,
            )
            yield result
            if not result.should_continue():
                break

    async def conversation_send_message(
        self, conversation: ConversationContext, activity: dict
    ) -> StepResult:
        return StepResult(activities=[], action="continue", num_steps=1)


def _agent_session_factory(
    activities: list[dict],
    log: MockSessionLog | None = None,
    **kwargs,
):
    outer_kwargs = kwargs

    def factory(**kwargs):
        botdesigner_client = BotDesignerClientMock(activities, log=log)
        kwargs = {
            **outer_kwargs,
            **kwargs,
        }
        return BotDesignerAgentSession.from_config(
            botdesigner_client=botdesigner_client,
            bot_template=load_template_file(),
            bot_variables={},
            **kwargs,
        )

    return factory


BD_CONVERSATION_RECOG_GENAI = [
    {
        "activities": [
            {
                "type": "message",
                "id": "49c6ca4d-d31b-4115-a342-0b505f3fb328",
                "timestamp": "2026-03-30T15:53:25.6314556+00:00",
                "channelId": "pva-studio",
                "from": {
                    "id": "11111111-1111-1111-1111-111111111111/22222222-2222-2222-2222-222222222222",
                    "name": "aibtest",
                    "role": "bot",
                },
                "conversation": {"id": "c62d0715-73dd-4749-b971-7d927a1c0b6d"},
                "recipient": {
                    "id": "00000000-0000-0000-0000-000000000001",
                    "aadObjectId": "00000000-0000-0000-0000-000000000001",
                    "role": "user",
                },
                "textFormat": "markdown",
                "membersAdded": [],
                "membersRemoved": [],
                "reactionsAdded": [],
                "reactionsRemoved": [],
                "locale": "en-US",
                "text": "Hello, I'm Assistant. How can I help?",
                "inputHint": "acceptingInput",
                "attachments": [],
                "entities": [],
                "channelData": {"feedbackLoop": {"type": "default"}},
                "replyToId": "f9256e49-a7f7-43b6-bf81-0b09ab2d0227",
                "listenFor": [],
                "textHighlights": [],
            },
            {
                "type": "event",
                "id": "013d3adc-69a1-45ba-8420-7cea4d07e0d4",
                "timestamp": "2026-03-30T15:53:25.6314621+00:00",
                "channelId": "pva-studio",
                "from": {
                    "id": "11111111-1111-1111-1111-111111111111/22222222-2222-2222-2222-222222222222",
                    "name": "aibtest",
                    "role": "bot",
                },
                "conversation": {"id": "c62d0715-73dd-4749-b971-7d927a1c0b6d"},
                "recipient": {
                    "id": "00000000-0000-0000-0000-000000000001",
                    "aadObjectId": "00000000-0000-0000-0000-000000000001",
                    "role": "user",
                },
                "membersAdded": [],
                "membersRemoved": [],
                "reactionsAdded": [],
                "reactionsRemoved": [],
                "locale": "en-US",
                "attachments": [],
                "entities": [],
                "replyToId": "f9256e49-a7f7-43b6-bf81-0b09ab2d0227",
                "valueType": "DialogTracingInfo",
                "value": {
                    "actions": [
                        {
                            "actionId": "sendMessage_greeting",
                            "topicId": "crd37_Hi.topic.ConversationStart",
                            "triggerId": "main",
                            "dialogComponentId": "0470d35b-0dae-4481-986e-67b6fceab684",
                            "actionType": "SendActivity",
                            "conditionItemExit": [],
                            "variableState": {"dialogState": {}, "globalState": {}},
                            "exception": "",
                            "resultTrace": {},
                        }
                    ]
                },
                "name": "DialogTracing",
                "listenFor": [],
                "textHighlights": [],
            },
        ],
        "action": "waiting",
    },
    {
        "activities": [
            {
                "type": "event",
                "id": "7861680a-e429-42a4-b304-18a28120b98d",
                "timestamp": "2026-03-30T15:55:11.1433046+00:00",
                "channelId": "pva-studio",
                "from": {
                    "id": "11111111-1111-1111-1111-111111111111/22222222-2222-2222-2222-222222222222",
                    "name": "aibtest",
                    "role": "bot",
                },
                "conversation": {"id": "c62d0715-73dd-4749-b971-7d927a1c0b6d"},
                "recipient": {
                    "id": "00000000-0000-0000-0000-000000000001",
                    "aadObjectId": "00000000-0000-0000-0000-000000000001",
                    "role": "user",
                },
                "membersAdded": [],
                "membersRemoved": [],
                "reactionsAdded": [],
                "reactionsRemoved": [],
                "attachments": [],
                "entities": [],
                "replyToId": "01d6bec3-9c90-4b15-a820-18f9db2f2489",
                "valueType": "DynamicPlanReceived",
                "value": {
                    "steps": ["thinkingbox_poc.action.tb001-get_current_weather"],
                    "isFinalPlan": False,
                    "planIdentifier": "7db2050b-2ed0-4459-ba2d-1949d6a109c6",
                    "toolDefinitions": [
                        {
                            "$kind": "ToolDefinition",
                            "displayName": "get_current_weather",
                            "description": "Get the current weather for a specified location.",
                            "iconUri": "https://defaulticons.powerapps.com/defaulticons/api-dedicated.png",
                            "inputs": [
                                {
                                    "$kind": "ToolInput",
                                    "name": "location",
                                    "description": "Location to get the weather for",
                                    "type": {"$kind": "String"},
                                    "propertyName": "location",
                                    "isRequired": True,
                                    "isAutomatic": True,
                                    "shouldPromptUser": True,
                                    "order": 0,
                                }
                            ],
                            "schemaName": "thinkingbox_poc.action.tb001-get_current_weather",
                            "toolKind": "InvokeConnectorTaskAction",
                        }
                    ],
                    "toolKinds": {
                        "thinkingbox_poc.action.tb001-get_current_weather": "InvokeConnectorTaskAction"
                    },
                },
                "name": "DynamicPlanReceived",
                "listenFor": [],
                "textHighlights": [],
            },
            {
                "type": "event",
                "id": "80628346-33b8-488f-a860-92f324597c31",
                "timestamp": "2026-03-30T15:55:11.1433079+00:00",
                "channelId": "pva-studio",
                "from": {
                    "id": "11111111-1111-1111-1111-111111111111/22222222-2222-2222-2222-222222222222",
                    "name": "aibtest",
                    "role": "bot",
                },
                "conversation": {"id": "c62d0715-73dd-4749-b971-7d927a1c0b6d"},
                "recipient": {
                    "id": "00000000-0000-0000-0000-000000000001",
                    "aadObjectId": "00000000-0000-0000-0000-000000000001",
                    "role": "user",
                },
                "membersAdded": [],
                "membersRemoved": [],
                "reactionsAdded": [],
                "reactionsRemoved": [],
                "attachments": [],
                "entities": [],
                "replyToId": "01d6bec3-9c90-4b15-a820-18f9db2f2489",
                "valueType": "DynamicPlanReceivedDebug",
                "value": {
                    "summary": "",
                    "ask": "Get the current weather in 'Seattle', Fahrenheit",
                    "planIdentifier": "7db2050b-2ed0-4459-ba2d-1949d6a109c6",
                    "isFinalPlan": False,
                },
                "name": "DynamicPlanReceivedDebug",
                "listenFor": [],
                "textHighlights": [],
            },
            {
                "type": "message",
                "id": "550591d9-3335-440d-ba77-6c4da74a15f8",
                "timestamp": "2026-03-30T15:55:11.1433131+00:00",
                "channelId": "pva-studio",
                "from": {
                    "id": "11111111-1111-1111-1111-111111111111/22222222-2222-2222-2222-222222222222",
                    "name": "aibtest",
                    "role": "bot",
                },
                "conversation": {"id": "c62d0715-73dd-4749-b971-7d927a1c0b6d"},
                "recipient": {
                    "id": "00000000-0000-0000-0000-000000000001",
                    "aadObjectId": "00000000-0000-0000-0000-000000000001",
                    "role": "user",
                },
                "textFormat": "markdown",
                "membersAdded": [],
                "membersRemoved": [],
                "reactionsAdded": [],
                "reactionsRemoved": [],
                "text": '{"location":"Seattle","explanation_of_tool_call":"This action needs to be done to provide the user with accurate and up-to-date weather conditions for Seattle in Fahrenheit."}',
                "inputHint": "acceptingInput",
                "attachments": [],
                "entities": [],
                "channelData": {"feedbackLoop": {"type": "default"}},
                "replyToId": "01d6bec3-9c90-4b15-a820-18f9db2f2489",
                "listenFor": [],
                "textHighlights": [],
            },
            {
                "type": "event",
                "id": "51815d6a-d7b1-41f2-9f5b-1503ad3f4c43",
                "timestamp": "2026-03-30T15:55:11.1458117+00:00",
                "channelId": "pva-studio",
                "from": {
                    "id": "11111111-1111-1111-1111-111111111111/22222222-2222-2222-2222-222222222222",
                    "name": "aibtest",
                    "role": "bot",
                },
                "conversation": {"id": "c62d0715-73dd-4749-b971-7d927a1c0b6d"},
                "recipient": {
                    "id": "00000000-0000-0000-0000-000000000001",
                    "aadObjectId": "00000000-0000-0000-0000-000000000001",
                    "role": "user",
                },
                "membersAdded": [],
                "membersRemoved": [],
                "reactionsAdded": [],
                "reactionsRemoved": [],
                "attachments": [],
                "entities": [],
                "replyToId": "01d6bec3-9c90-4b15-a820-18f9db2f2489",
                "valueType": "DynamicPlanStepTriggered",
                "value": {
                    "planIdentifier": "7db2050b-2ed0-4459-ba2d-1949d6a109c6",
                    "stepId": "53992371-7922-41e1-99f7-b9c39b1ff3ae",
                    "taskDialogId": "thinkingbox_poc.action.tb001-get_current_weather",
                    "thought": "This action needs to be done to provide the user with accurate and up-to-date weather conditions for Seattle in Fahrenheit.",
                    "state": "inProgress",
                    "hasRecommendations": False,
                    "type": "Action",
                },
                "name": "DynamicPlanStepTriggered",
                "listenFor": [],
                "textHighlights": [],
            },
            {
                "type": "event",
                "id": "1b17e70c-1c80-47d1-9a88-b7588f3eaae9",
                "timestamp": "2026-03-30T15:55:11.1517765+00:00",
                "channelId": "pva-studio",
                "from": {
                    "id": "11111111-1111-1111-1111-111111111111/22222222-2222-2222-2222-222222222222",
                    "name": "aibtest",
                    "role": "bot",
                },
                "conversation": {"id": "c62d0715-73dd-4749-b971-7d927a1c0b6d"},
                "recipient": {
                    "id": "00000000-0000-0000-0000-000000000001",
                    "aadObjectId": "00000000-0000-0000-0000-000000000001",
                    "role": "user",
                },
                "membersAdded": [],
                "membersRemoved": [],
                "reactionsAdded": [],
                "reactionsRemoved": [],
                "attachments": [],
                "entities": [],
                "replyToId": "01d6bec3-9c90-4b15-a820-18f9db2f2489",
                "valueType": "DynamicPlanStepBindUpdate",
                "value": {
                    "taskDialogId": "thinkingbox_poc.action.tb001-get_current_weather",
                    "stepId": "53992371-7922-41e1-99f7-b9c39b1ff3ae",
                    "arguments": {"location": "Seattle"},
                    "planIdentifier": "7db2050b-2ed0-4459-ba2d-1949d6a109c6",
                    "autoFilledArguments": ["location"],
                },
                "name": "DynamicPlanStepBindUpdate",
                "listenFor": [],
                "textHighlights": [],
            },
        ],
        "action": "continue",
    },
    {
        "activities": [
            {
                "type": "event",
                "id": "2610077c-8af2-47a5-ad93-980a17816c32",
                "timestamp": "2026-03-30T15:57:09.8918221+00:00",
                "channelId": "pva-studio",
                "from": {
                    "id": "11111111-1111-1111-1111-111111111111/22222222-2222-2222-2222-222222222222",
                    "name": "aibtest",
                    "role": "bot",
                },
                "conversation": {"id": "c62d0715-73dd-4749-b971-7d927a1c0b6d"},
                "recipient": {
                    "id": "00000000-0000-0000-0000-000000000001",
                    "aadObjectId": "00000000-0000-0000-0000-000000000001",
                    "role": "user",
                },
                "membersAdded": [],
                "membersRemoved": [],
                "reactionsAdded": [],
                "reactionsRemoved": [],
                "attachments": [],
                "entities": [],
                "replyToId": "01d6bec3-9c90-4b15-a820-18f9db2f2489",
                "valueType": "DynamicPlanStepFinished",
                "value": {
                    "taskDialogId": "thinkingbox_poc.action.tb001-get_current_weather",
                    "stepId": "53992371-7922-41e1-99f7-b9c39b1ff3ae",
                    "observation": {
                        "Response": '{"status":"ok","obj":{"location":"Seattle","temperature":60,"unit":"F"}}'
                    },
                    "planUsedOutputs": {},
                    "planIdentifier": "7db2050b-2ed0-4459-ba2d-1949d6a109c6",
                    "state": "completed",
                    "hasRecommendations": False,
                    "executionTime": "00:01:58.7468155",
                },
                "name": "DynamicPlanStepFinished",
                "listenFor": [],
                "textHighlights": [],
            },
        ],
        "action": "continue",
    },
    {
        "activities": [
            {
                "type": "message",
                "id": "ad754376-2fb3-4d8a-bd75-f68dbbfbf6d4",
                "timestamp": "2026-03-30T15:57:12.4957564+00:00",
                "channelId": "pva-studio",
                "from": {
                    "id": "11111111-1111-1111-1111-111111111111/22222222-2222-2222-2222-222222222222",
                    "name": "aibtest",
                    "role": "bot",
                },
                "conversation": {"id": "c62d0715-73dd-4749-b971-7d927a1c0b6d"},
                "recipient": {
                    "id": "00000000-0000-0000-0000-000000000001",
                    "aadObjectId": "00000000-0000-0000-0000-000000000001",
                    "role": "user",
                },
                "textFormat": "markdown",
                "membersAdded": [],
                "membersRemoved": [],
                "reactionsAdded": [],
                "reactionsRemoved": [],
                "text": "Arr, matey! The current temperature in Seattle be **60\u00b0F**.",
                "inputHint": "acceptingInput",
                "attachments": [],
                "entities": [],
                "channelData": {"feedbackLoop": {"type": "default"}},
                "replyToId": "01d6bec3-9c90-4b15-a820-18f9db2f2489",
                "listenFor": [],
                "textHighlights": [],
            },
            {
                "type": "event",
                "id": "35c0e24c-abd5-49c6-a38e-9139a1763914",
                "timestamp": "2026-03-30T15:57:12.4986216+00:00",
                "channelId": "pva-studio",
                "from": {
                    "id": "11111111-1111-1111-1111-111111111111/22222222-2222-2222-2222-222222222222",
                    "name": "aibtest",
                    "role": "bot",
                },
                "conversation": {"id": "c62d0715-73dd-4749-b971-7d927a1c0b6d"},
                "recipient": {
                    "id": "00000000-0000-0000-0000-000000000001",
                    "aadObjectId": "00000000-0000-0000-0000-000000000001",
                    "role": "user",
                },
                "membersAdded": [],
                "membersRemoved": [],
                "reactionsAdded": [],
                "reactionsRemoved": [],
                "attachments": [],
                "entities": [],
                "replyToId": "01d6bec3-9c90-4b15-a820-18f9db2f2489",
                "valueType": "DynamicPlanFinished",
                "value": {
                    "planId": "7db2050b-2ed0-4459-ba2d-1949d6a109c6",
                    "wasCancelled": False,
                },
                "name": "DynamicPlanFinished",
                "listenFor": [],
                "textHighlights": [],
            },
        ],
        "action": "waiting",
    },
]


@pytest.mark.asyncio
async def test_bd_conversation_recog_genai():
    scenario = CLOUD_DRIVE_SCENARIO
    agent_log = MockSessionLog()
    agent_factory = _agent_session_factory(BD_CONVERSATION_RECOG_GENAI, log=agent_log)
    judge_llm = MockSession(completions=[])
    tc = HydratedTestCase(
        uid="0",
        agent=AGENT_CONFIG,
        scenario=scenario,
        query="weather?",
        test_code=_get_test_code(TEST_WEATHER_60F),
    )
    async with MCPProxyClient.session_context(
        endpoint=SESSION_PROXY_ENDPOINT,
        timeout=SESSION_PROXY_TIMEOUT,
        server_config=merge_init_config(scenario.world_state, tc.init),
        available_tools=[t.name for t in scenario.tools],
    ) as mcp_proxy:
        result = await run_agent_user_loop(
            tc,
            agent_session_factory=agent_factory,
            mcp_proxy=mcp_proxy,
            user_model=None,
            store_test_context=True,
            store_raw_messages=True,
        )
    assert not result.is_system_error, repr(result)
    assert result.test_context is not None

    assert agent_log.remaining_completions == 0, "Not all activities have been emitted"
    test = _TestScript(
        tc.test_code,
        Judge(judge_llm),
    )
    test_result = await test.evaluate(result.test_context)
    assert test_result.result, f"test result is False, traceback: {test_result.tb}"
    assert not judge_llm.completions, "Not all judge messages have been emitted"

    tool_calls = [msg for msg in result.messages if isinstance(msg, ParallelToolCall)]
    assert len(tool_calls) == 1
    assert "get_current_weather" in tool_calls[0].tool_calls[0].name
    assert tool_calls[0].tool_calls[0].arguments == {"location": "Seattle"}
    tool_responses = [msg for msg in result.messages if isinstance(msg, ToolResponse)]
    assert "get_current_weather" in tool_responses[0].name
    assert "60" in tool_responses[0].content
    assert len(tool_responses) == 1

    bot_override = _get_bot_override(result)
    assert (
        bot_override["entity"]["configuration"]["recognizer"]["kind"]
        == "GenerativeAIRecognizer"
    )

    # tools translation: connector
    dialogs = _get_connector_dialogs(
        bot_override, "/providers/Microsoft.PowerApps/apis/shared_thinkingbox_connector"
    )
    assert len(dialogs) == 2
    assert list(dialogs) == ["upload_file", "get_text_content"]
    assert list(dialogs["upload_file"]["operation"]["inputType"]["properties"]) == [
        "path",
        "text_content",
        "overwrite",
    ]
    assert list(
        dialogs["get_text_content"]["operation"]["inputType"]["properties"]
    ) == ["path"]


BD_CONVERSATION_RECOG_CLIAGENT = [
    {
        "activities": [
            {
                "id": "typing-1",
                "type": "typing",
                "from": {
                    "id": "11111111-1111-1111-1111-111111111111/22222222-2222-2222-2222-222222222222"
                },
            },
            {
                "id": "d138d73f-e39c-4ba1-8f03-4149fd918162",
                "type": "typing",
                "text": "Hello! I\u0027m Assistant. How can I help you today?",
                "from": {
                    "id": "11111111-1111-1111-1111-111111111111/22222222-2222-2222-2222-222222222222"
                },
                "conversation": {"id": "6f071fd5-81b0-443a-b646-49a374917acd"},
                "channelData": {"streamType": "streaming", "streamSequence": 1},
                "entities": [
                    {
                        "streamType": "streaming",
                        "streamSequence": 1,
                        "type": "streaminfo",
                        "properties": {},
                    }
                ],
            },
            {
                "type": "message",
                "id": "7cb2ec26-4e5f-4ecc-b07f-9035e90e93bc",
                "channelId": "pva-studio",
                "from": {
                    "id": "11111111-1111-1111-1111-111111111111/22222222-2222-2222-2222-222222222222",
                    "name": "aibtest",
                    "role": "bot",
                },
                "conversation": {"id": "6f071fd5-81b0-443a-b646-49a374917acd"},
                "recipient": {
                    "id": "00000000-0000-0000-0000-000000000001",
                    "aadObjectId": "00000000-0000-0000-0000-000000000001",
                    "role": "user",
                },
                "membersAdded": [],
                "membersRemoved": [],
                "reactionsAdded": [],
                "reactionsRemoved": [],
                "locale": "en-US",
                "text": "Hello! I\u0027m Assistant. How can I help you today?",
                "attachments": [],
                "entities": [
                    {
                        "streamId": "d138d73f-e39c-4ba1-8f03-4149fd918162",
                        "streamType": "final",
                        "type": "streaminfo",
                    }
                ],
                "channelData": {
                    "streamType": "final",
                    "streamId": "d138d73f-e39c-4ba1-8f03-4149fd918162",
                },
                "replyToId": "56f79b59-230b-49a5-a2c8-172ef5c7f53e",
                "listenFor": [],
                "textHighlights": [],
            },
        ],
        "action": "waiting",
    },
    {
        "activities": [
            {
                "id": "typing-1",
                "type": "typing",
                "from": {
                    "id": "11111111-1111-1111-1111-111111111111/22222222-2222-2222-2222-222222222222"
                },
            },
            {
                "id": "f14be332-d81e-44fc-a48e-d09a2bae19f7",
                "type": "typing",
                "text": "Provisioning sandbox...",
                "from": {
                    "id": "11111111-1111-1111-1111-111111111111/22222222-2222-2222-2222-222222222222"
                },
                "conversation": {"id": "6f071fd5-81b0-443a-b646-49a374917acd"},
                "channelData": {"streamType": "informative", "streamSequence": 1},
                "entities": [
                    {
                        "streamType": "informative",
                        "streamSequence": 1,
                        "type": "streaminfo",
                        "properties": {},
                    }
                ],
            },
            {
                "id": "f14be332-d81e-44fc-a48e-d09a2bae19f7-1",
                "type": "typing",
                "text": "Initializing sandbox...",
                "from": {
                    "id": "11111111-1111-1111-1111-111111111111/22222222-2222-2222-2222-222222222222"
                },
                "conversation": {"id": "6f071fd5-81b0-443a-b646-49a374917acd"},
                "channelData": {
                    "streamType": "informative",
                    "streamId": "f14be332-d81e-44fc-a48e-d09a2bae19f7",
                    "streamSequence": 2,
                },
                "entities": [
                    {
                        "streamId": "f14be332-d81e-44fc-a48e-d09a2bae19f7",
                        "streamType": "informative",
                        "streamSequence": 2,
                        "type": "streaminfo",
                        "properties": {},
                    }
                ],
            },
            {
                "type": "message",
                "id": "713c1c88-d922-47b6-99b6-74f94beac019",
                "channelId": "pva-studio",
                "from": {
                    "id": "11111111-1111-1111-1111-111111111111/22222222-2222-2222-2222-222222222222",
                    "name": "aibtest",
                    "role": "bot",
                },
                "conversation": {"id": "6f071fd5-81b0-443a-b646-49a374917acd"},
                "recipient": {
                    "id": "00000000-0000-0000-0000-000000000001",
                    "aadObjectId": "00000000-0000-0000-0000-000000000001",
                    "role": "user",
                },
                "membersAdded": [],
                "membersRemoved": [],
                "reactionsAdded": [],
                "reactionsRemoved": [],
                "text": "Arrr matey, I be needin’ to know where ye be sailin’! Give me the name of the port or the city so I can fetch ye the current weather. \ud83c\udf26️ Where be it?",
                "attachments": [],
                "entities": [
                    {
                        "streamId": "f14be332-d81e-44fc-a48e-d09a2bae19f7",
                        "streamType": "final",
                        "type": "streaminfo",
                    }
                ],
                "channelData": {
                    "streamType": "final",
                    "streamId": "f14be332-d81e-44fc-a48e-d09a2bae19f7",
                },
                "replyToId": "5e0aa7ce-44ef-461c-aed3-e420f7952f91",
                "listenFor": [],
                "textHighlights": [],
            },
            {
                "type": "event",
                "id": "6d66f91e-61a4-42a0-a7e7-8bb40e2c55dd",
                "channelId": "pva-studio",
                "from": {
                    "id": "11111111-1111-1111-1111-111111111111/22222222-2222-2222-2222-222222222222",
                    "name": "aibtest",
                    "role": "bot",
                },
                "conversation": {"id": "6f071fd5-81b0-443a-b646-49a374917acd"},
                "recipient": {
                    "id": "00000000-0000-0000-0000-000000000001",
                    "aadObjectId": "00000000-0000-0000-0000-000000000001",
                    "role": "user",
                },
                "membersAdded": [],
                "membersRemoved": [],
                "reactionsAdded": [],
                "reactionsRemoved": [],
                "attachments": [],
                "entities": [],
                "replyToId": "5e0aa7ce-44ef-461c-aed3-e420f7952f91",
                "name": "turn.complete",
                "listenFor": [],
                "textHighlights": [],
            },
        ],
        "action": "waiting",
    },
    {
        "activities": [
            {
                "id": "typing-1",
                "type": "typing",
                "from": {
                    "id": "11111111-1111-1111-1111-111111111111/22222222-2222-2222-2222-222222222222"
                },
            },
            {
                "id": "14843874-3d60-4dee-992a-d731194c02a1",
                "type": "typing",
                "text": "Calling get_current_weather...",
                "from": {
                    "id": "11111111-1111-1111-1111-111111111111/22222222-2222-2222-2222-222222222222"
                },
                "conversation": {"id": "6f071fd5-81b0-443a-b646-49a374917acd"},
                "channelData": {"streamType": "informative", "streamSequence": 1},
                "entities": [
                    {
                        "type": "toolCall",
                        "toolCallId": "08182194-0869-4605-b9d0-bbea0d372cfe",
                        "toolName": "get_current_weather",
                        "toolDisplayName": "get_current_weather",
                        "status": "started",
                        "filledParameters": {"location": "\u0022Seattle\u0022"},
                        "unfilledParameters": [],
                        "hiddenFilledParameters": {},
                        "hiddenUnfilledParameters": [],
                    },
                    {
                        "streamType": "informative",
                        "streamSequence": 1,
                        "type": "streaminfo",
                        "properties": {},
                    },
                ],
            },
            {
                "id": "14843874-3d60-4dee-992a-d731194c02a1-1",
                "type": "typing",
                "text": "get_current_weather completed",
                "from": {
                    "id": "11111111-1111-1111-1111-111111111111/22222222-2222-2222-2222-222222222222"
                },
                "conversation": {"id": "6f071fd5-81b0-443a-b646-49a374917acd"},
                "channelData": {
                    "streamType": "informative",
                    "streamId": "14843874-3d60-4dee-992a-d731194c02a1",
                    "streamSequence": 2,
                },
                "entities": [
                    {
                        "type": "toolCall",
                        "toolCallId": "08182194-0869-4605-b9d0-bbea0d372cfe",
                        "toolName": "get_current_weather",
                        "toolDisplayName": "get_current_weather",
                        "status": "completed",
                        "durationMs": 16,
                        "result": "{\u0022status\u0022:\u0022ok\u0022,\u0022obj\u0022:{\u0022location\u0022:\u0022Seattle\u0022,\u0022temperature\u0022:60,\u0022unit\u0022:\u0022F\u0022}}",
                    },
                    {
                        "streamId": "14843874-3d60-4dee-992a-d731194c02a1",
                        "streamType": "informative",
                        "streamSequence": 2,
                        "type": "streaminfo",
                        "properties": {},
                    },
                ],
            },
            {
                "type": "message",
                "id": "47f53ff0-cac2-4af3-894c-a68486b22a0c",
                "channelId": "pva-studio",
                "from": {
                    "id": "11111111-1111-1111-1111-111111111111/22222222-2222-2222-2222-222222222222",
                    "name": "aibtest",
                    "role": "bot",
                },
                "conversation": {"id": "6f071fd5-81b0-443a-b646-49a374917acd"},
                "recipient": {
                    "id": "00000000-0000-0000-0000-000000000001",
                    "aadObjectId": "00000000-0000-0000-0000-000000000001",
                    "role": "user",
                },
                "membersAdded": [],
                "membersRemoved": [],
                "reactionsAdded": [],
                "reactionsRemoved": [],
                "text": "Arrr, in the port o’ Seattle it be a fair 60°F, matey! A fine breeze for a stroll or a short sail. \ud83c\udf24️",
                "attachments": [],
                "entities": [
                    {
                        "streamId": "14843874-3d60-4dee-992a-d731194c02a1",
                        "streamType": "final",
                        "type": "streaminfo",
                    }
                ],
                "channelData": {
                    "streamType": "final",
                    "streamId": "14843874-3d60-4dee-992a-d731194c02a1",
                },
                "replyToId": "d22d9d6c-0de8-4713-b62d-8dcc0a2621cc",
                "listenFor": [],
                "textHighlights": [],
            },
            {
                "type": "event",
                "id": "1d286c1b-a235-4302-95cb-da562801774b",
                "channelId": "pva-studio",
                "from": {
                    "id": "11111111-1111-1111-1111-111111111111/22222222-2222-2222-2222-222222222222",
                    "name": "aibtest",
                    "role": "bot",
                },
                "conversation": {"id": "6f071fd5-81b0-443a-b646-49a374917acd"},
                "recipient": {
                    "id": "00000000-0000-0000-0000-000000000001",
                    "aadObjectId": "00000000-0000-0000-0000-000000000001",
                    "role": "user",
                },
                "membersAdded": [],
                "membersRemoved": [],
                "reactionsAdded": [],
                "reactionsRemoved": [],
                "attachments": [],
                "entities": [],
                "replyToId": "d22d9d6c-0de8-4713-b62d-8dcc0a2621cc",
                "name": "turn.complete",
                "listenFor": [],
                "textHighlights": [],
            },
        ],
        "action": "waiting",
    },
]


@pytest.mark.asyncio
async def test_bd_conversation_recog_cliagent_userllm():
    scenario = CLOUD_DRIVE_SCENARIO
    agent_log = MockSessionLog()
    agent_factory = _agent_session_factory(
        BD_CONVERSATION_RECOG_CLIAGENT,
        log=agent_log,
        recognizer_kind="CLIAgentRecognizer",
        tool_translation_mode="none",
    )
    judge_llm = MockSession(completions=[])
    user_llm = MockSession(
        completions=[
            [Text(role="user", content="here")],
            [Text(role="user", content="bye <DONE>")],
        ],
    )
    tc = HydratedTestCase(
        uid="0",
        agent=AGENT_CONFIG,
        scenario=scenario,
        query="weather?",
        user_context="some",
        test_code=_get_test_code(TEST_WEATHER_60F),
    )
    async with MCPProxyClient.session_context(
        endpoint=SESSION_PROXY_ENDPOINT,
        timeout=SESSION_PROXY_TIMEOUT,
        server_config=merge_init_config(scenario.world_state, tc.init),
        available_tools=[t.name for t in scenario.tools],
    ) as mcp_proxy:
        result = await run_agent_user_loop(
            tc,
            agent_session_factory=agent_factory,
            mcp_proxy=mcp_proxy,
            user_model=user_llm,
            store_test_context=True,
            user_can_end_conversation=True,
            store_raw_messages=True,
        )
    assert not result.is_system_error, repr(result)
    assert result.test_context is not None

    assert agent_log.remaining_completions == 0, "Not all activities have been emitted"
    test = _TestScript(
        tc.test_code,
        Judge(judge_llm),
    )
    test_result = await test.evaluate(result.test_context)
    assert test_result.result, f"test result is False, traceback: {test_result.tb}"
    assert not judge_llm.completions, "Not all judge messages have been emitted"
    assert not user_llm.completions, "Not all user messages have been emitted"
    tool_calls = [msg for msg in result.messages if isinstance(msg, ParallelToolCall)]
    assert len(tool_calls) == 1
    assert "get_current_weather" in tool_calls[0].tool_calls[0].name
    assert tool_calls[0].tool_calls[0].arguments == {"location": "Seattle"}
    tool_responses = [msg for msg in result.messages if isinstance(msg, ToolResponse)]
    assert "get_current_weather" in tool_responses[0].name
    assert "60" in tool_responses[0].content
    assert len(tool_responses) == 1

    bot_override = _get_bot_override(result)
    assert (
        bot_override["entity"]["configuration"]["recognizer"]["kind"]
        == "CLIAgentRecognizer"
    )

    # tools translation: none
    dialogs = _get_connector_dialogs(
        bot_override, "/providers/Microsoft.PowerApps/apis/shared_thinkingbox_connector"
    )
    assert not dialogs
    assert not bot_override.get("connectorDefinitions")
    assert not bot_override.get("connectionReferences")
