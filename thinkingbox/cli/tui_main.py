#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import asyncio
import html
import itertools
import traceback
from pathlib import Path

import click
from prompt_toolkit import HTML
from pydantic_core import to_jsonable_python

from thinkingbox.cli.common import (
    load_yaml,
    pprint,
    pprint_yaml,
    tui_input,
    tui_parse_command,
    wrap_pprint_tb_error_async,
)
from thinkingbox.common.agent_session_base import AgentSessionBase
from thinkingbox.common.agent_session_factory import get_agent_session_factory
from thinkingbox.common.chat_types import Message, Text
from thinkingbox.common.config_types import (
    ConfigFile,
    format_string_or_none,
    merge_bot_instructions,
    merge_init_config,
    update_tools_with_client_config,
)
from thinkingbox.common.http_client import initialize_dns_cache
from thinkingbox.common.hydrator import (
    Dataset,
    HydratedTestCase,
    get_dataset_case_by_name,
    load_test_file,
)
from thinkingbox.common.judge import Judge
from thinkingbox.common.llm_session_factory import create_llm_session
from thinkingbox.common.mcp_proxy_client import MCPProxyClient
from thinkingbox.common.testrunner import TestScript
from thinkingbox.common.user_simulated_answer import UserSimulator


class TUI:
    def __init__(self, config: ConfigFile, tc: HydratedTestCase):
        self.config = config
        self.tc = tc
        self.judge_llm = create_llm_session(self.config.judge_model)
        # User LLM session (optional)
        self.user_model = (
            create_llm_session(self.config.user_model)
            if self.config.user_model
            else None
        )
        self.user_context = tc.user_context or ""

    @wrap_pprint_tb_error_async
    async def main(self):
        # whether the first query is from command line
        tc = self.tc

        query: str | None = None
        should_inject_query = False
        if bool(tc.query):
            query = tc.query
            should_inject_query = True

        test_specific_bot_instructions = tc.bot_instructions

        # create a session within the MCP session proxy
        server_config = merge_init_config(
            tc.scenario.world_state,
            tc.init,
        )

        async with MCPProxyClient.session_context_from_config(
            config=self.config.mcp_proxy,
            server_config=server_config,
            available_tools=[t.name for t in tc.scenario.tools],
        ) as mcp_proxy:
            mcp_tools = await mcp_proxy.list_tools()
            update_tools_with_client_config(
                mcp_tools,
                tc.scenario.tools,
            )

            if tc.format_query:
                test_specific_bot_instructions = format_string_or_none(
                    test_specific_bot_instructions,
                    mcp_proxy.init_result,
                )
                query = format_string_or_none(query, mcp_proxy.init_result)

            bot_instructions = merge_bot_instructions(
                scenario_instructions=tc.scenario.bot_instructions,
                testcase_instructions=test_specific_bot_instructions,
            )

            agent_session_factory = get_agent_session_factory(
                config=self.config.orchestrator,
            )

            # create agent session
            agent = agent_session_factory(
                config=tc.agent,
                mcp_proxy=mcp_proxy,
                mcp_tools=mcp_tools,
                bot_instructions=bot_instructions,
                scenario_metadata=tc.scenario.metadata,
            )

            # print tools
            pprint(HTML("<system>[tools]</system>"))
            tools_txt = ""
            for tool in itertools.chain(self.tc.agent.builtin_tools, mcp_tools):
                tool_args_str = ", ".join(list(tool.input_schema["properties"]))
                tools_txt += f"{tool.name}({tool_args_str})\n"
            # tools_txt = yaml.safe_dump([tool.model_dump() for tool in mcp_tools.values()], sort_keys=False)
            pprint(tools_txt)
            pprint("")

            while True:
                if should_inject_query:
                    should_inject_query = False
                    msg_user = Text(role="user", type="text", content=query)
                    for pp_text in msg_user.pp(html=True):
                        pprint(HTML(pp_text))
                else:
                    new_query = await tui_input(HTML("<user>[user::text]</user>\n"))
                    pprint("")
                    msg_user = Text(role="user", type="text", content=new_query)

                if msg_user.content.startswith("/"):
                    # command
                    cmd_name, cmd_args = tui_parse_command(msg_user.content[1:])
                    should_exit = False
                    generated_msg: Text | None = None
                    try:
                        should_exit, generated_msg = await self.handle_command(
                            agent=agent,
                            cmd_name=cmd_name,
                            cmd_args=cmd_args,
                        )
                    except KeyboardInterrupt:
                        raise
                    except Exception:
                        traceback.print_exc()
                    if should_exit:
                        break
                    if generated_msg is None:
                        # Command handled, no message to send
                        continue
                    # Command produced a message - use it
                    msg_user = generated_msg

                # user message (either typed or generated by /user command)
                if msg_user.content == "":
                    msg_user = None

                # Print the user message if it was generated
                if msg_user is not None and msg_user.metadata.get("is_user_llm"):
                    for pp_text in msg_user.pp(html=True):
                        pprint(HTML(pp_text))
                    pprint("")

                async for msg in agent.decode_turn_iter(msg_user):
                    for pp_text in msg.pp(html=True):
                        pprint(HTML(pp_text))
                    pprint("")
                pprint("")

    async def handle_command(
        self,
        agent: AgentSessionBase,
        cmd_name: str,
        cmd_args: list[str],
    ) -> tuple[bool, Text | None]:
        """
        Handle slash commands.

        Returns:
            tuple of (should_exit, message_to_send)
            - should_exit: True if the TUI should exit
            - message_to_send: Optional Text message to send to the agent
        """
        if cmd_name == "help":
            pprint(HTML("<i>[commands]</i>"))
            pprint("  /help                  - Show this help")
            pprint("  /quit                  - Exit the TUI")
            pprint("  /effects               - Show side effects")
            pprint("  /test [file:name]      - Run test assertions")
            pprint("  /conversation          - Show raw LLM conversation")
            pprint("  /tool <name>           - Show tool schema")
            pprint("  /user                  - Generate a simulated user message")
            pprint("  /user context          - Show current user context")
            pprint("  /user context <text>   - Set user context")
            return (False, None)
        if cmd_name == "quit":
            pprint(HTML("<i>[goodbye]</i>"))
            return (True, None)
        if cmd_name == "effects":
            effects = await agent.get_effects()
            pprint_yaml(
                HTML("<i>[effects]</i>"),
                effects,
            )
            return (False, None)
        if cmd_name == "test" and len(cmd_args) <= 1:
            await self.handle_test_command(
                agent, test_ref=cmd_args[0] if cmd_args else None
            )
            return (False, None)
        if cmd_name == "conversation":
            raw_conversation = agent.get_raw_messages()
            if raw_conversation is not None:
                if isinstance(raw_conversation, list):
                    for raw_message in raw_conversation:
                        pprint(raw_message)
                else:
                    pprint(raw_conversation)
            else:
                pprint("Raw conversation is not available for this orchestrator")
            return (False, None)
        elif cmd_name == "tool" and len(cmd_args) == 1:
            tool = [t for t in agent.tools if t.name == cmd_args[0]]
            if not tool:
                pprint("Error: tool not found")
            else:
                pprint_yaml(
                    HTML("<i>[tool]</i>"),
                    to_jsonable_python(tool[0]),
                )
            return (False, None)
        elif cmd_name == "user":
            msg = await self.generate_user_message(agent, cmd_args)
            return (False, msg)

        pprint(HTML("<i>[command not recognized]</i>"))
        return (False, None)

    async def generate_user_message(
        self,
        agent: AgentSessionBase,
        cmd_args: list[str],
    ) -> Text | None:
        """
        Handle /user command variants:
          /user              - Generate a user LLM response (returned for main loop to send)
          /user context      - Show current user context
          /user context <text> - Set user context

        Returns:
            Text message to send to agent, or None if no message to send
        """
        # Subcommand: context
        if cmd_args and cmd_args[0] == "context":
            if len(cmd_args) == 1:
                # Show current context
                pprint(HTML("<i>[user context]</i>"))
                pprint(self.user_context or "(none)")
            else:
                # Set context
                self.user_context = " ".join(cmd_args[1:])
                pprint(HTML("<i>[user context updated]</i>"))
            return None

        # Default: generate user LLM response
        if not self.user_model:
            pprint(HTML("<i>[error: user_model not configured in config]</i>"))
            return None

        if not self.user_context:
            pprint(HTML("<i>[error: no user_context set - use /user context ...]</i>"))
            return None

        # Generate user response
        user_sim = UserSimulator(self.user_model)
        chat_history: list[Message] = agent.conversation.messages
        pprint(HTML("<i>[generating user response...]</i>"))

        try:
            msg_user = await user_sim.generate(
                chat_history=chat_history,
                user_context=self.user_context,
            )
        except Exception as e:
            pprint(
                HTML(
                    f"<i>[error: failed to generate user response: {html.escape(str(e))}]</i>"
                )
            )
            return None

        if not msg_user.content.strip():
            pprint(HTML("<i>[warning: user LLM returned empty response, skipping]</i>"))
            return None

        msg_user = msg_user.model_copy(deep=True)
        msg_user.metadata["is_user_llm"] = True

        return msg_user

    async def handle_test_command(
        self,
        agent: AgentSessionBase,
        test_ref: str | None = None,
    ):
        if test_ref is None:
            # expect a test in the test case
            if not self.tc.test_code:
                raise ValueError("No test code here, use /test file.py:test_name")
            test_code = self.tc.test_code
        else:
            test_code = ""
            filename, testname = test_ref.split(":")
            test_file = load_test_file(filename, check_config=False)
            for tc in test_file.test_cases:
                if tc.uid == testname:
                    test_code = tc.test_code
                    break
            if not test_code:
                raise ValueError("Test code not found")

        test_ctx = await agent.make_test_context()
        test = TestScript(
            code=test_code,
            judge=Judge(self.judge_llm),
        )
        res = await test.evaluate(test_ctx)
        for pp_text in res.pp(html=True):
            pprint(HTML(pp_text))


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
    required=True,
    type=click.Path(path_type=Path, exists=True),
    help="Dataset root directory.",
)
@click.option(
    "-a",
    "--agent",
    default="base",
    show_default=True,
    help="Agent name.",
)
@click.option(
    "-n",
    "--name",
    default=None,
    help="Test case name (mutually exclusive with --scenario).",
)
@click.option(
    "-s",
    "--scenario",
    default=None,
    help="Scenario name (mutually exclusive with --name).",
)
@click.option(
    "-q",
    "--query",
    default="",
    show_default=True,
    help="Optional query to override or seed the test.",
)
def tui(
    config: Path,
    dataset: Path,
    agent: str,
    name: str | None,
    scenario: str | None,
    query: str,
) -> None:
    """
    Launch the ThinkingBox TUI for a single test case or a scenario template.

    Exactly one of --name or --scenario must be provided.
    """
    # Validate mutual exclusivity: exactly one of name/scenario
    if (name is None) == (scenario is None):
        raise click.UsageError("Exactly one of --name or --scenario must be set.")

    cfg = load_yaml(config, ConfigFile)

    if name is not None:
        tc = get_dataset_case_by_name(
            name,
            base_dir=dataset,
            agent=agent,
        )
        if query:
            tc.query = query
    elif scenario is not None:
        ds = Dataset(dataset)
        tc = HydratedTestCase(
            uid="none:none",
            agent=ds.get_agent_config(agent),
            scenario=ds.get_scenario(scenario),  # type: ignore[arg-type]
            query=query,
            test_code="",
            user_context="",
        )
    else:
        raise Exception("Unexpected code path.")

    initialize_dns_cache()

    ui = TUI(cfg, tc)
    asyncio.run(ui.main())
