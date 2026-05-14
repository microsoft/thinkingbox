# How prompts are configured

## System Prompt

A system prompt shared among all scenarios is configured in the Agent configuration file.
See an example in `dataset/agent/think.yaml` and schema in `thinkingbox/common/config_types.py` class `AgentConfig`

An additional scenario-specific system message can be added in the Scenario configuration file.
See schema in `thinkingbox/common/config_types.py` class `ScenarioConfig` (field `bot_instructions`)

Moreover, additional bot instructions can be added per test case; these will be
appended to the scenario bot instructions.

The end result is

```
[first message, role=system or developer]
(implicitly includes tools and other relevant settings)
<agent_config.system_instructions>

[second message, role=system or developer]
<scenario_config.bot_instructions>
<test_case.bot_instructions>
```
