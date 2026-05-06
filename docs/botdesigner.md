
# BotDesigner integration

ThinkingBox can interface with the BotDesigner test API exposed by `VirtualAgent.Fabric`.

```
POST /environments/{environment}/bots/{bot}/test/conversations
POST /environments/{environment}/bots/{bot}/test/conversations/{conversation}
POST /environments/{environment}/bots/{bot}/test/conversations/{conversation}/continue
```

The API is currently undocumented, and the ThinkingBox integration is based on BotDesigner behavior as of commit 5fe1d1fa.


## How it works

### Mainline architecture (GenerativeAIRecognizer)

```
                            tb mcp-start
  ┌──────────┐               ▲       ▲          ┌───────────────┐
  │          ├───────────────┘       └──────────┤               │
  │ tb infer │                                  │ VirtualAgent  │
  │          ├─────────────────────────────────►|               │
  └────┬─────┘                                  └───────┬───────┘
       │                                                │
       ▼                                                ▼
  AOAI (User/Judge LLM)                            CAPI (Agent LLM)
```

### Dracarys architecture (CLIAgentRecognizer)

```
                            tb mcp-start
  ┌──────────┐               ▲       ▲       ┌──────────────┐
  │          ├───────────────┘       └───────┤              │
  │ tb infer │                               │ AgenticLoop  ├──► dracarys orchestrator
  │          ├──► VirtualAgent ─────────────►|              │            OR
  └────┬─────┘                               └──────┬───────┘       sandbox pool
       │                                            │
       ▼                                            ▼
  AOAI (User/Judge LLM)                      CAPI (Agent LLM)
```


ThinkingBox creates a bot definition based on a template and the list of MCP tools. The generated bot definition includes:
- Bot Instructions
- Agent name
- Connectors, converted from MCP tool definitions

It submits requests to BotDesigner with the following overrides:
- Bot definition override: completely replaces the existing bot definition
- Connector override URL, pointing to a specific TB session in the session proxy: connector requests are redirected to this server


## Setup

### Overview

Run `VirtualAgent.Fabric` and `tb mcp-start` (session proxy) on the same machine.

For Dracarys, also run `AgenticLoop.Service` and make sure it can connect to the Dracarys sandbox. For local Dracarys sandbox, also run `dracarys-orchestrator` and configure `AgenticLoop.Service` accordingly.

ThinkingBox will need to be able to connect to:

- BotDesigner (`VirtualAgent.Fabric`)
- Session Proxy (`tb mcp-start`)

Use `tb infer` or `tb tui` with a configuration file (`tb infer -c config.yaml ...`) that includes setup for the BotDesigner orchestrator.


### BotDesigner

For mainline testing, start at least `VirtualAgent.Fabric`, or start a configuration that includes it (e.g. `BotDesigner.Web + VirtualAgent.Fabric LocalBotWithCds` in the VS solution)

For Dracarys testing, start at least `VirtualAgent.Fabric` and `AgenticLoop.Service`, or start a configuration that includes both (e.g. `BotDesigner.Web + VirtualAgent.Fabric LocalBotWithCds + AgenticLoop`).

The following environment variables must be set:

- `VirtualAgent.Fabric`: `ASPNETCORE_URLS = "http://0.0.0.0:5000/"`
- `AgenticLoop.Service`: `ASPNETCORE_URLS = "https://localhost:5002;http://localhost:5003"`

VirtualAgent.Fabric needs to listen on `0.0.0.0` in order to be reachable from the ThinkingBox process in WSL.

The following feature flag must be set:

- `EngineSecureHttpHandler.ValidateUrl = false`

This is necessary for BotDesigner to connect to the Session Proxy on same host or local network (see [Known Limitations](#known-limitations)).

Make sure everything is correctly setup according to the documentation in the BotDesigner repo, in particular:

- The test endpoints in VirtualAgent.Fabric will require an account to get an authentication token.
- VirtualAgent.Fabric and AgenticLoop.Service will need some certificates to be installed on the system, in order to authenticate with services like CAPI.
- The test endpoints will require an existing environment and a bot, even if we override the entire bot definition.

Note: by default AgenticLoop will use a remote sandbox service for Dracarys.


### Interoperability Setup

Due to limitations in BotDesigner (see [Known Limitations](#known-limitations)) we need to start the Session Proxy on HTTPS 443 with a valid certificate. We hope BotDesigner can lift these restrictions in the future when running in local debugging mode, to simplify the setup.

Create a host name for the session proxy by adding a mapping to the hosts file.

- In WSL, get the IP address of the WSL/Windows network interface

```bash
ip -4 addr show eth0 | grep -oP 'inet \K[^/]+'
```

- Open `C:\Windows\System32\drivers\etc\hosts` with a text editor as Administrator, and add the address from the previous command

```
<wsl-address-here> tbmcp.local
```

- In WSL, add the following to `/etc/hosts`

```
127.0.0.1 tbmcp.local
```

In WSL, create a CA and a self-signed certificate for the server

```bash
# Generate Certificate Authority
openssl genrsa -out ca.key 4096

openssl req -x509 -new -nodes -key ca.key \
    -sha256 -days 3650 \
    -out ca.crt \
    -subj "/C=US/ST=Local/L=Local/O=ThinkingBox Dev CA/CN=ThinkingBox Dev CA"

# Generate server key and certificate
openssl genrsa -out tbmcp.local.key 4096

openssl req -new -key tbmcp.local.key \
    -out tbmcp.local.csr \
    -subj "/CN=tbmcp.local" \
    -addext "subjectAltName=DNS:tbmcp.local,DNS:*.tbmcp.local,IP:127.0.0.1,IP:::1"

openssl x509 -req -in tbmcp.local.csr \
    -CA ca.crt -CAkey ca.key -CAcreateserial \
    -out tbmcp.local.crt \
    -days 365 -sha256 \
    -copy_extensions copy

# Merge certificate and key into a single file
cat tbmcp.local.crt tbmcp.local.key > tbmcp.local.pem

# Copy ca.crt somewhere on Windows
cp ca.crt /mnt/c/src/ca.crt
```

- `tbmcp.local.pem` will be needed to start the Session Proxy
- `ca.crt` will be needed to connect to it

**IMPORTANT: DO NOT SHARE ca.key**

Trust the new CA at system level on Windows, so that BotDesigner can connect to Session Proxy.

Open PowerShell as Administrator

```powershell
Import-Certificate -FilePath "C:\src\ca.crt" -CertStoreLocation Cert:\LocalMachine\Root
```

**IMPORTANT: Uninstall the certificate when done**


### ThinkingBox

Follow instructions in README, or [docs/onboarding.md](docs/onboarding.md), verify that testing with the ThinkingBox orchestrator works.

Restart the Session Proxy on `https://0.0.0.0:443` with the BotDesigner connector override endpoint enabled.

```bash
# allow listening on 443
sudo sysctl net.ipv4.ip_unprivileged_port_start=443

THINKINGBOX_DATA="/path/to/thinkingbox-data" \
    tb mcp-start \
    --servers "/path/to/thinkingbox-data/servers/servers.yaml" \
    --ssl-pem /path/to/tbmcp.local.pem \
    --host 0.0.0.0 \
    --port 443 \
    --bd-connectors
```

Edit the example configuration (`config/config_botdesigner.yaml`) as needed, (see [ThinkingBox Configuration](#thinkingbox-configuration)).

Use `tb infer` or `tb tui` with the new configuration file.


### ThinkingBox Configuration

```yaml

mcp_proxy:
    # If using HTTPS, a hostname is needed. Create an alias in /etc/hosts if necessary
    endpoint_url: "https://tbmcp.local"

    # If using HTTPS with self-signed cert, and CA is not trusted as system level, configure here the additional CA to trust
    trust_ca_path: /path/to/ca.crt

    # Retrieve tool calls and responses directly from the session proxy,
    # and insert them into TestContext.tool_calls
    geteffects_proxy_info: true

    timeout: 300.0


orchestrator:
    type: botdesigner

    # `host.docker.internal` should map to the Windows address in WSL
    endpoint_url: "http://host.docker.internal:5000"

    credential:
        type: api-key

        # Use a valid API key if BotDesigner requires it,
        # otherwise you can set credential: null
        api_key: BOTDESIGNER_API_KEY

    # Environment must exist (see "Known Limitations")
    environment_id: "<YOUR_ENVIRONMENT_ID>"

    # Bot must exist (see "Known Limitations").
    # Its content is not important, since it will be completely replaced
    # by the override
    base_bot_id: "<YOUR_BASE_BOT_ID>"

    # Endpoint that BotDesigner can reach for connector calls
    # (`tb mcp-start --bd-connectors ...`)

    # Make sure:
    #   - it listens on port 443 with HTTPS
    #   - its certificate is trusted at system level where BotDesigner runs
    #   - DNS resolves it and the address is reachable from where BotDesigner runs
    # If the server is on a local network address, make sure BotDesigner cluster setting
    #   EngineSecureHttpHandler.ValidateUrl is configured to False, or BotDesigner will
    #   refuse to connect. This cannot be set with an override.
    # (see [Known Limitations](#known-limitations))
    connector_endpoint_override: "https://tbmcp.local/connectors"

    # connector: Translate all MCP tools to connectors, creating a bot definition
    #            on-the-fly
    # mcp:       Insert a MCP connector and let it retrieve all tools directly
    #            from the session proxy
    #            (This won't work yet, as BotDesigner cannot redirect it)
    # none:      Do not convert tools, in case a YAML with hardcoded tools is already
    #            provided, which does not expand the tool entries, and possible
    #            conversion failures are not relevant

    # Note that some tools have input schemas that are not compatible with a
    # connector's input schema, therefore conversion can fail
    tool_translation_mode: "connector"

    # Mainline: GenerativeAIRecognizer
    # Dracarys: CLIAgentRecognizer
    recognizer_kind: "GenerativeAIRecognizer"

    # Set to True to use SSE when communicating with BotDesigner.
    # Note that when enabled, SSE chunks are still accumulated on the client,
    # and not streamed in real time to the TB UI or result file.
    # This option is provided to allow logging tool call events with
    # Dracarys, since Dracarys sends them only on SSE.
    use_sse_protocol: false

    # Additional feature overrides to pass in the `x-ms-feature-overrides` header
    # Note: features to enable connector and bot definition override are
    # already included when needed
    feature_overrides: {}

    # Path to the bot template file for all tests
    # If not provided, the default one is used
    # (see: thinkingbox/botdesigner/bot_template.yaml)
    bot_template_file: null

    # Additional variables to expand in the bot YAML template for all tests
    bot_variables:
        agent_name: Assistant  # (default)

    timeout: 300.0

```

Additional scenario-specific configuration can be passed in the `metadata` field of a TB scenario file. These will have no effect for other orchestrator types

```yaml
metadata:
    # set this to a template object like thinkingbox/botdesigner/bot_template.yaml
    # to override the template for this scenario
    bd_bot_template: null

    # Override variables to expand in the template, e.g. different agent name for
    # this scenario
    bd_bot_variables:
        agent_name: Weather Assistant

```


## Known Limitations

There are several known limitations, mainly due to gaps in BotDesigner.


**Agent registration**

BotDesigner requires an environment and a bot to be already registered in order to use the test endpoint, even with the bot override feature.

`VirtualAgent.Fabric` does not have APIs to create an environment or an agent, therefore ThinkingBox cannot create them.


**Session proxy**

- Cannot run on the local network: BotDesigner connector override rejects any domain that resolves a
  non-routable (private/internal) IP address
- Must be served with HTTPS: BotDesigner connector override refuses to connect to HTTP
- Must be served on port 443: BotDesigner connector override ignores port and uses 443
- Must offer a valid certificate: BotDesigner connector override cannot bypass certificate verification

Workarounds:
    - Compile BotDesigner with "EngineSecureHttpHandler.ValidateUrl" set to false by default
    - Generate a CA and self-signed certificate for the server, trust the CA at system level
    - Run Session Proxy on port 443 with HTTPS protocol


**Tools support**

Since the BotDesigner connectors schema does not cover the full jsonschema specification, it is not always possible to convert a tool's input schema to connector's input schema.

There is no workaround for this:
- We can only override connectors providing a "connector input schema", our tools are MCP servers and use "jsonschema"
- We convert MCP tool's "jsonschema" to "connector input schema" where possible
- BotDesigner converts "connector input schema" to "jsonschema" for the LLM tool call
- BotDesigner's LLM tool call "jsonschema" must match the MCP tool's "jsonschema"
- Therefore a tool can be supported only if round-trip conversion between "connector input schema" and "jsonschema" is possible

The special MCP connector retrieves input schemas from the MCP server directly, bypassing "connector input schema", which would solve the issue, however the connector override feature in BotDesigner currently does not support the MCP connector.


**UniversalSearchTool override**

BotDesigner does not allow overriding UniversalSearchTool, however it is always included in the list of tools available to the model.


**Events parsing**

Since there are no documented schemas for request inputs and outputs, it is possible that some events are not parsed correctly, which could cause decoding to fail or messages to not appear in the conversation.


**Error detection**

In some situations BotDesigner does not surface software errors (such as connection problems, LLM call temporary failures, orchestration failures) as error events, and includes these errors in the bot's message text.

We mitigate the issue by detecting some known error strings, but this may not always work.

We also cannot distinguish RAI errors from the LLM endpoint from calls ModifierRequestsInToolExecutionTool, since the latter is reported as a LLM API call error.


**LLM Call throttling**

BotDesigner re-try logic on errors and rate-limit responses is not optimal for testing. Errors due to failing LLM calls will be the most common failure mode on a setup that is otherwise correct.

In some cases, ThinkingBox may not even record these failures as "system error", depending on how BotDesigner handles them.

SSE Responses API client is more susceptible due to the limitation that it cannot handle rate-limit errors that occur in the SSE response body. When such error occurs, the conversation will just look truncated, as BotDesigner sends a "plan finished" event.

Workaround:
- Redirect all BotDesigner CAPI requests to a proxy that implements re-try logic
- Increase BotDesigner LLM request timeout
- Disable BotDesigner's client-side LLM call throttling


**LLM sampling parameters**

Some tests require `temperature > 0` and `seed = null` to make repetitions meaningful, however BotDesigner does not allow configuring LLM sampling parameters.

Workaround: CAPI proxy with overrides (see [LLM Call throttling](#llm-call-throttling) section)


**Testing a non-local, deployed BotDesigner instance**

BotDesigner connector override does not support any authentication method, therefore it is not possible to securely deploy `tb mcp-start` and let BotDesigner instance authenticate with it.

Testing is currently possible only with a local setup.


**End of conversation**

Unlike GenerativeAIRecognizer, CLIAgentRecognizer (Dracarys) does not emit a "plan finished" event, it continues the conversation indefinitely.

In order to run tests with Dracarys, we added a feature to let User-LLM end the conversation. This feature is in an early stage of development and evaluation. The prompt and/or implementation may change in the future based on further evaluations.
