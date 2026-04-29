# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import contextlib
import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from thinkingbox.common.chat_types import (
    Message,
    ParallelToolCall,
    Text,
    ToolCall,
    ToolResponse,
)

logger = logging.getLogger(__name__)


class LLMCall(BaseModel):
    """Captured LLM call from BotDesigner."""

    timestamp: str
    url: str
    request: dict[str, Any]  # {messages, tools, parameters}
    response: dict[str, Any]


class BotDesignerActivityErrorInfo(BaseModel):
    activity_type: str
    error_type: str
    content: str
    path: str = ""

    def __str__(self):
        return f"activity_type={self.activity_type} error_type={self.error_type} content={self.content}"


class BotDesignerActivityParserError(Exception):
    pass


class ParsedActivities(BaseModel):
    """Parsed events from BotDesigner."""

    messages: list[Message] = Field(default_factory=list)
    available_tools: list[str] = Field(default_factory=list)
    plan_finished: bool = False
    plan_cancelled: bool = False
    errors: list[BotDesignerActivityErrorInfo] = Field(default_factory=list)
    warnings: list[BotDesignerActivityErrorInfo] = Field(default_factory=list)


class ParserContext(BaseModel):
    result: ParsedActivities = Field(default_factory=ParsedActivities)
    path: list[str | int] = Field(default_factory=lambda: ["root"])

    def get_path_string(self):
        return ".".join([str(item) for item in self.path])

    @contextlib.contextmanager
    def push_path(self, item: str | int):
        self.path.append(item)
        try:
            yield
        except BotDesignerActivityParserError:
            raise
        except Exception as e:
            raise BotDesignerActivityParserError(
                f"{e} (at {self.get_path_string()})"
            ) from e
        finally:
            self.path.pop(-1)


def _get_tool_response_from_observation(observation: dict[str, Any]):
    # TODO need documentation ...
    structured = observation.get("structuredContent")
    if structured:
        return json.dumps(structured)

    content_list = observation.get("content", [])
    if content_list:
        return content_list[0].get("text", "{}")

    response = observation.get("Response", None)
    if response is not None:
        return response

    return "{}"


def _get_activity_value_obj(activity: dict[str, Any]) -> dict[str, Any]:
    value = activity.get("value")
    if not value:
        return {}
    if not isinstance(value, dict):
        raise ValueError("Expected dictionary")
    return value


# Non-comprehensive list of errors that can appear in bot text responses.
# BotDesigner's test endpoint may not report them as error events,
# therefore we need to make a best effort to match error strings...

BD_ERROR_STRINGS = (
    "I encountered an error while processing your request",
    "EXCEPTION_BEGIN:",
    "GenAIToolPlannerRateLimitReached",
)


def _is_bd_error_text(
    content: str,
    conversation_id: str | None = None,
):
    # if conversation ID is in content, something must have gone wrong!
    # The LLM does not know the random conversation ID, it will never emit it
    if conversation_id is not None and conversation_id in content:
        return True

    # match common error strings
    for err in BD_ERROR_STRINGS:
        if err in content:
            return True

    return False


class ActivityParser:
    """Parse BotDesigner activities into ThinkingBox message types."""

    def __init__(self, conversation_id: str | None = None):
        self.conversation_id: str | None = conversation_id

        # pending_tool_calls persists through multiple parser calls
        self.pending_tool_calls: dict[str, ToolCall] = {}

    def set_conversation_id(self, conversation_id: str | None):
        self.conversation_id = conversation_id

    def parse_activities(
        self,
        activities: list[dict[str, Any]],
    ) -> ParsedActivities:
        """Parse events into tool call responses, messages, and chat history.

        Args:
            events: List of BotDesigner activity events
        """
        ctx = ParserContext()
        # Track pending tool calls by stepId (BotDesigner) or tool_name (legacy)

        for i, activity in enumerate(activities):
            with ctx.push_path(i):
                self._parse_activity(ctx, activity=activity)

        return ctx.result

    def _parse_activity(
        self,
        ctx: ParserContext,
        activity: dict[str, Any],
    ):
        activity_type = activity.get("type")
        with ctx.push_path(f"<activity:{activity_type}>"):
            match activity_type:
                case "message":
                    self._parse_activity_message(
                        ctx,
                        activity=activity,
                    )
                case "event":
                    self._parse_activity_event(ctx, activity=activity)
                case "typing":
                    self._parse_activity_typing(ctx, activity=activity)
                case "trace":
                    self._parse_activity_trace(ctx, activity=activity)

    def _parse_activity_message(
        self,
        ctx: ParserContext,
        activity: dict[str, Any],
    ) -> None:
        from_role: str = activity.get("from", {}).get("role", "")
        text: str = activity.get("text", "")
        is_error: bool = False
        role_map = {
            "user": "user",
            "bot": "assistant",
        }

        if _is_bd_error_text(text, conversation_id=self.conversation_id):
            ctx.result.errors.append(
                BotDesignerActivityErrorInfo(
                    activity_type="message",
                    error_type="text",
                    content=text,
                    path=ctx.get_path_string(),
                )
            )
            is_error = True

        if from_role in role_map:
            msg = Text(role=role_map[from_role], content=text)
            if is_error:
                msg.metadata["bd_is_error"] = True
            ctx.result.messages.append(msg)

    def _parse_activity_typing(
        self,
        ctx: ParserContext,
        activity: dict[str, Any],
    ):
        """Parse typing activities for tool call information.

        In Dracarys + SSE mode, tool calls arrive as typing activities
        """
        entities = activity.get("entities", [])
        for entity in entities:
            if entity.get("type") != "toolCall":
                continue

            tool_call_id = entity.get("toolCallId")
            tool_name = entity.get("toolName", "UnknownTool")
            status = entity.get("status")

            if not tool_call_id:
                continue

            if status == "started":
                filled_params = entity.get("filledParameters", {})
                # filledParameters values may be JSON-encoded strings
                # TODO are they always JSON-encoded?
                arguments = {}
                if isinstance(filled_params, dict):
                    for key, value in filled_params.items():
                        if isinstance(value, str):
                            try:
                                arguments[key] = json.loads(value)
                            except (json.JSONDecodeError, ValueError):
                                arguments[key] = value
                        else:
                            arguments[key] = value

                tool_call = ToolCall(
                    name=tool_name,
                    arguments=arguments,
                    metadata={"bd_tool_call_id": tool_call_id},
                )
                self.pending_tool_calls[tool_call_id] = tool_call

            elif status == "completed":
                if tool_call_id in self.pending_tool_calls:
                    tool_call = self.pending_tool_calls.pop(tool_call_id)
                    result = entity.get("result", "{}")
                    ctx.result.messages.append(ParallelToolCall(tool_calls=[tool_call]))
                    tool_response = ToolResponse(
                        name=tool_call.name,
                        content=result,
                        id=tool_call.id,
                        metadata={"bd_tool_call_id": tool_call_id},
                    )
                    ctx.result.messages.append(tool_response)
                else:
                    ctx.result.warnings.append(
                        BotDesignerActivityErrorInfo(
                            activity_type="typing",
                            error_type="toolCall_completed",
                            content=f"unknown tool_call_id: {tool_call_id!r}",
                            path=ctx.get_path_string(),
                        )
                    )

    def _parse_activity_trace(
        self,
        ctx: ParserContext,
        activity: dict[str, Any],
    ):
        value_type = activity.get("valueType")
        with ctx.push_path(f"<type:{value_type}>"):
            match value_type:
                case "ErrorCode":
                    self._parse_activity_trace_ErrorCode(ctx, activity)

    def _parse_activity_trace_ErrorCode(
        self,
        ctx: ParserContext,
        activity: dict[str, Any],
    ):
        value = activity.get("value")
        ctx.result.errors.append(
            BotDesignerActivityErrorInfo(
                activity_type="trace",
                error_type="ErrorCode",
                content=json.dumps(value),
                path=ctx.get_path_string(),
            )
        )

    def _parse_activity_event(
        self,
        ctx: ParserContext,
        activity: dict[str, Any],
    ) -> None:
        """Parse an event activity."""
        value_type = activity.get("valueType")

        with ctx.push_path(f"<type:{value_type}>"):
            match value_type:
                case "DynamicPlanStepTriggered":
                    self._parse_activity_event_DynamicPlanStepTriggered(
                        ctx,
                        activity=activity,
                    )
                case "DynamicPlanStepBindUpdate":
                    self._parse_activity_event_DynamicPlanStepBindUpdate(
                        ctx,
                        activity=activity,
                    )
                case "DynamicPlanStepFinished":
                    self._parse_activity_event_DynamicPlanStepFinished(
                        ctx,
                        activity=activity,
                    )
                case "DynamicServerToolsList":
                    self._parse_activity_event_DynamicServerToolsList(
                        ctx,
                        activity=activity,
                    )
                case "DynamicPlanFinished":
                    self._parse_activity_event_DynamicPlanFinished(
                        ctx,
                        activity=activity,
                    )
                case "DynamicPlanStepBlocked":
                    self._parse_activity_event_DynamicPlanStepBlocked(
                        ctx,
                        activity=activity,
                    )

    def _parse_activity_event_DynamicPlanStepTriggered(
        self,
        ctx: ParserContext,
        activity: dict[str, Any],
    ):
        value = _get_activity_value_obj(activity)
        # Tool call triggered - extract tool name from taskDialogId
        tool_name = value.get("taskDialogId", "UnknownTool")
        step_id = value.get("stepId")
        if not step_id:
            return

        # Create tool call with empty arguments (will be updated by BindUpdate)
        tool_call = ToolCall(
            name=tool_name,
            arguments={},
            metadata={"bd_step_id": step_id},
        )
        self.pending_tool_calls[step_id] = tool_call

    def _parse_activity_event_DynamicPlanStepBindUpdate(
        self,
        ctx: ParserContext,
        activity: dict[str, Any],
    ):
        value = _get_activity_value_obj(activity)
        step_id = value.get("stepId")
        # Update pending tool call with arguments
        if step_id in self.pending_tool_calls:
            arguments = value.get("arguments", {})
            tool_call = self.pending_tool_calls[step_id]
            tool_call.arguments = arguments
        else:
            ctx.result.warnings.append(
                BotDesignerActivityErrorInfo(
                    activity_type="event",
                    error_type="DynamicPlanStepBindUpdate",
                    content=f"unknown step_id: {step_id!r}",
                    path=ctx.get_path_string(),
                )
            )

    def _parse_activity_event_DynamicPlanStepFinished(
        self,
        ctx: ParserContext,
        activity: dict[str, Any],
    ):
        value = _get_activity_value_obj(activity)
        # Tool call finished - extract result
        step_id = value.get("stepId")

        if step_id in self.pending_tool_calls:
            observation = value.get("observation", {})
            # Get result from structuredContent or content text
            content = _get_tool_response_from_observation(observation)
            tool_call = self.pending_tool_calls.pop(step_id)
            ctx.result.messages.append(ParallelToolCall(tool_calls=[tool_call]))
            tool_response = ToolResponse(
                name=tool_call.name,
                content=content,
                id=tool_call.id,
                metadata={"bd_step_id": step_id},
            )
            ctx.result.messages.append(tool_response)
        else:
            ctx.result.warnings.append(
                BotDesignerActivityErrorInfo(
                    activity_type="event",
                    error_type="DynamicPlanStepFinished",
                    content=f"unknown step_id: {step_id!r}",
                    path=ctx.get_path_string(),
                )
            )

    def _parse_activity_event_DynamicServerToolsList(
        self,
        ctx: ParserContext,
        activity: dict[str, Any],
    ):
        value = _get_activity_value_obj(activity)
        tools = value.get("tools", [])
        if not tools:
            tools = value.get("toolsList", [])
        for tool in tools:
            tool_name = tool.get("name") or tool.get("displayName")
            if tool_name:
                ctx.result.available_tools.append(tool_name)

    def _parse_activity_event_DynamicPlanFinished(
        self,
        ctx: ParserContext,
        activity: dict[str, Any],
    ):
        value = _get_activity_value_obj(activity)
        ctx.result.plan_finished = True
        ctx.result.plan_cancelled = value.get("wasCancelled", False)

    def _parse_activity_event_DynamicPlanStepBlocked(
        self,
        ctx: ParserContext,
        activity: dict[str, Any],
    ):
        value = activity.get("value")
        ctx.result.errors.append(
            BotDesignerActivityErrorInfo(
                activity_type="event",
                error_type="DynamicPlanStepBlocked",
                content=json.dumps(value),
                path=ctx.get_path_string(),
            )
        )


def message_to_activity(msg: Message) -> dict:
    if not (isinstance(msg, Text) and msg.role == "user"):
        raise ValueError("Can only convert user messages to activities")
    sender_id = msg.metadata.get("bd_sender_id", "user-1")
    return {
        "type": "message",
        "text": msg.content,
        "from": {"id": sender_id, "role": "user"},
    }
