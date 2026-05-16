# Tools configuration in scenario

## Tools list

In `tools` section of the scenario configuration, list all the tools that will be available to the agent. This can be a subset of the tools provided by MCP servers in the `world_state` section.

If multiple MCP servers provide one of those tools, the tool from the last server will be used.


## Client-side tool configuration

Each entry of the `tools` section is a configuration, which at minimum can be just the name of the tool.

See schema in `thinkingbox/common/config_types.py` class `ToolDefOverride`.

Examples

```yaml
tools:
# this tool responds to the agent with the text returned by the MCP server as is
- name: get_weather

# this tool responds directly to the user; this is a client-side configuration
- name: get_weather_direct_resp
  direct_response: "The temperature is {temperature}"
  # the direct_response string formatted as follows:
  # text = direct_response.format(**json.loads(tool_response))
  # e.g. if the tool returns {"temperature": "22"}
  # message "The temperature is 22" would be presented

# this tool is not really called in the MCP server, it just stops the agent loop
- name: end_conversation
  is_end_turn: true

# override the tool description shown to the agent
- name: search_documents
  override_description: "Search internal company documents by keyword"

# override individual parameter descriptions
- name: send_email
  override_arg_description:
    recipient: "The email address of the recipient (must be a valid company email)"
    subject: "A short summary of the email content"

# combine tool and parameter description overrides
- name: create_ticket
  override_description: "Create a support ticket in the internal tracking system"
  override_arg_description:
    priority: "Ticket priority: low, medium, high, or critical"
    assignee: "Username of the team member to assign the ticket to"
```

## Direct responses behavior

When a tool is configured with a direct response, an assistant message containing
the formatted text is added to the conversation, after the tool response message.

Often the Agent will rephrase and repeat the message in its last response. This
behavior is not desired. We should prompt the Agent not to do this, and grade it
as part of the reward function in training.
