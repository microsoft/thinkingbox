# How to create new tools for the ThinkingBox project

The `thinkingbox/tools` directory contains a server example for the ThinkingBox project. The tool server is implemented as a separate Python module (e.g., `mcp_cloud_drive.py`)
and follows a consistent structure for maintainability and ease of integration.

The tools consist of reusable components to be orchestrated by agentic LLMs using the MCP (Modular Control Protocol) framework, but with added control
and introspection of effects and state that can be later used in conversation simulation test cases.

All new tools will be expected to be added in the `AI.ThinkingBox.Data` repository under the `servers/thinkingbox_tools/thinkingbox_tools` directory. We keep the tools separate from
the main engine code to have better separation of concerns and also facilitate paired versioning of tools and tests. See more information in `AI.ThinkingBox.Data/servers/README.md`.

Follow these steps to add a new tools server:

### 1. **Create a New MCP Server Module**
   - Copy an existing module (e.g., `servers/thinkingbox_tools/thinkingbox_tools/mcp_cloud_drive.py`) as a starting point, or create a new file named `servers/thinkingbox_tools/thinkingbox_tools/mcp_<your_tool_server_name>.py`.
   - Let's see a simple example tool that greets a user by name, named `mcp_greet.py`.
   ```python
   # mcp_greet.py
   import traceback
   from typing import Annotated
   from pydantic import Field
   from pydantic_core import to_jsonable_python
   from fastmcp import FastMCP


   # 1. Declare the MCP server with a unique server name and desired log level.
   mcp = FastMCP("greet", log_level="INFO")

   # 2. Add custom exception
   class SayHiToolException(Exception):
       """Custom exception for your tool."""
       pass

   # 3. Define success and error handling functions.
   def success_response(**kwargs) -> dict:
       """Create a successful response with status ok."""
       obj = {
           "status": "ok",
           **to_jsonable_python(kwargs),
       }
       return obj

   def error_response(exc) -> dict:
       if isinstance(exc, SayHiToolException):
           return {"error": "SayHiToolException: " + str(exc)}
       traceback.print_exc()
       return {"error": "Unexpected error!"}

   # 4. Reserved and required method to initialize the tool's state
   @mcp.tool(name="__reserved__init")
   def initialize(config: dict) -> dict:
       # Set up initial state here
       return {}  # Return an empty JSON object

   # 5. Reserved method to get the effects of the tool's operations
   @mcp.tool(name="__reserved__geteffects")
   def get_effects() -> dict:
       # Here you could fill in effects from the tools operations
       effects = {}
       return effects

   # 6. Define your tool/s function/s with the @mcp.tool decorator
   # Always include a description and the expected input parameters with annotations.
   # Always catch all errors so the method returns a string, even on error.
   @mcp.tool(name="say_hi", description="This tool creates a Hi greeting for a user.")
   def say_hi(
       name: Annotated[str, Field(description="The name of the person to greet.")]
   ) -> dict:
       if not name or name is None:
           return error_response(SayHiToolException("Name cannot be empty or None."))
       try:
           greeting = f"Hi, {name}!"
           return success_response(greeting=greeting)
       except Exception as exc:
           return error_response(exc)

   # 7. Ensure correct server initialization on direct call
   if __name__ == "__main__":
       mcp.run(transport="stdio", show_banner=False)
   ```
   - The definition of a state can be done with custom objects declared at the top level of the file. See `mcp_cloud_drive.py` or `mcp_email_system.py` for examples of how to define a state object.

#### 1.1 Add new server to the servers configuration:
In `AI.ThinkingBox.Data/servers/servers.yaml`, add your new server to the configuration file so it can be made available to the MCP proxy. For example:
```yaml
use_internal_servers: true
servers:
  #[...] <--- existing servers --->
  # Add this:
  greet:
    type: mcp-process
    command: ["{python}", "-m", "thinkingbox_tools.mcp_greet"]
```

### 2. **Agentic Tests**
   - Create a scenario configuration in dataset/scenario/<your_tool_server_name>.yaml to define the initial state and available tools. See `dataset/scenario/cloud_drive.yaml` for an example.
   - For the greeting example, create a file `dataset/scenario/greet.yaml`:
     ```yaml
     world_state:
        greet: {}
     tools:
     - name: say_hi

     bot_instructions: ""
     ```
   - Create a test case file in `dataset/test_case/greet.py` to define the agentic test cases. For example:
     ```python
     from thinkingbox.common import TestContext, Judge

     """!
     scenario: greet
     """

     def test_say_hi(x: TestContext, judge: Judge):
         """!
         query: |
            Greet me!
         user_context: |
            The user's name is Bob.
         """
         assert len(x.tool_calls) == 1, "Expected one tool call"
         assert judge.text_yesno(x.response, "Does the message contain a greeting for a person named Bob?")
     ```
   - Run the background MCP proxy server with `tb mcp-start --servers $THINKINBOX_DATA/servers/servers.yaml` to make the tool available for testing.
   - Run the agentic tests using calls like `tb infer -c config/config.yaml --dataset ./dataset --agent think --name greet.py:test_say_hi --output output.yaml` to ensure the tool works as expected in an agentic context.

### 3. [OPTIONAL] **Unit Tests**

Small, experimental tools may not require unit tests, but for larger or more complex tools, it's good practice to add unit tests to ensure the tool's functionality is correct and stable down the line.

   - Add unittests for the tools you added in step 5 in `AI.ThinkingBox.Data/servers/thinkingbox_tools/tests` directory (e.g., `AI.ThinkingBox.Data/servers/thinkingbox_tools/tests/test_<your_tool_server_name>_server.py`). See `AI.ThinkingBox.Data/servers/thinkingbox_tools/tests/test_slack_server.py` as an example, including how to handle state.
   - For the greeting example:
   ```python
   import pytest
   import pytest_asyncio
   from fastmcp import Client
   from thinkingbox_tools import mcp_greet

   TOOLS = ["say_hi"]
   SERVER_CONFIG = {
       "greet": {}
   }

   @pytest_asyncio.fixture
   async def greet_client():
       async with Client(mcp_greet.mcp) as client:
            yield client

   @pytest.mark.asyncio
   async def test_say_hi_success(greet_client):
       # Call the tool
       response = await greet_client.call_tool("say_hi", name="Bob")
       # Assert successful response
       assert response.structured_content["status"] == "ok"
       assert response.structured_content["greeting"] == "Hi, Bob!"
   ```
   - Run the tests using `pytest -k test_say_hi_success ./test_tools` to ensure everything works as expected.
