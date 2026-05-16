# Session Proxy configuration

## Configuration file

By default, the session proxy (`tb mcp-start`) looks for server files in the `tools/` directory, according to the following name convention:

```
tools/mcp_<server-name>.py
```

It is possible to change this behavior by passsing a YAML configuration file.

```bash
tb mcp-start --servers servers.yaml
```

The configuration file schema is in `tools/client/common.py` class `ServersConfig`.

```yaml
# Example servers.yaml

# do not discover servers in tools/
# (the default is True)
use_internal_servers: false

# List of servers. These take priority over tools/
servers:

  # a server with stdio transport
  # one process is created for each conversation, no need to handle
  # or isolate sessions.
  test_tool_process:
    type: mcp-process

    # The new process inherits the parent process environment variables

    # (optional) add or update environment variables for the new process
    env:
      VARNAME: "value"

    # (optional) set working directory for the new process
    cwd: "tests"

    # Command
    # each part is formatted with the following:
    # {
    #   "python": sys.executable,
    #   "env": os.environ,
    # }
    # Use double {{ ... }} to escape { ... }
    command: ["{python}", "mcp_test_tool.py", "--transport", "stdio"]

  # Connect to a running instance with streamable-http or sse transport
  # The server is expected to handle and isolate sessions
  test_tool_http:
    type: mcp-remote

    # (optional) additional request headers
    headers:
        CustomHeader: Value

    # HTTP client timeout
    timeout: 60.0

    # endpoint URL
    endpoint: http://127.0.0.1:10611/mcp/

    # streamable-http or sse
    transport: streamable-http

```
