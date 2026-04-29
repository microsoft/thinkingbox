# LLM Endpoint configuration

## LLM requirements

ThinkingBox needs access to 3 LLMs:
- Agent: The LLM to evaluate, with tool calling and reasoning
- User: The simulated user LLM, no tool calling and no reasoning
- Judge: A LLM to use as judge in tests, no tool calling and no reasoning

Other configurations may work mechanically (e.g. Agent without reasoning, Judge with reasoning) but they are not tested. We will focus on other configurations only if we target them for training.

## Configuration file

The configuration file format is described in `thinkingbox/common/config_types.py` class `ConfigFile`.

Most tools accept a `-c/--config` argument pointing to the configuration file in YAML format.

## API support

### Chat Completions

**Azure OpenAI with token from azure-cli**

```yaml
# non-reasoning
judge_model:
    type: aoai

    # authentication: use azure-cli for the token
    credential:
        type: az-cli

    # https://<account-name>.openai.azure.com/openai/deployments/<deployment-name>/chat/completions?api-version=<api-version>"
    account_name: "<account-name>"
    deployment: "<deployment-name>"
    api_version: "<api-version>"

    temperature: 0.0
    seed: 42
    max_completion_tokens: 128
    timeout: 60.0
```

**OpenAI-compatible chat completions**

```yaml
# reasoning, deepseek-style "reasoning_content" expected in response
agent_model:
    type: aoai

    # endpoint URL is the full URL, such as "http://127.0.0.1:8000/v1/chat/completions"
    # In request body: {"model": "<model-name>"}
    deployment: "<model-name>"  # served model name
    endpoint_url: "<chat-completions-endpoint>"

    is_reasoning: true
    temperature: 1.0
    max_completion_tokens: 4096
    timeout: 60.0

# non-reasoning
judge_model:
  type: aoai
  deployment: "<model-name>"  # served model name
  endpoint_url: "http://127.0.0.1:8001/v1/chat/completions"
  is_reasoning: false
  temperature: 0.0
  seed: 42
  max_completion_tokens: 128
  timeout: 60.0
```

For API key authentication, use credential type `api_key`.

```yaml
agent_model:
  credential:
    type: api-key
    api_key: "<api-key>"
    # ...
```

For client certificate authentication, provide the path to the certificate in `client_certificate`.


### Responses

**Azure OpenAI Responses with token from azure-cli or managed identity**

*Note: this does not use the stateful protocol*

```yaml
agent_model:
    type: aoai_responses

    # authentication: use azure-cli for the token
    credential:
        type: az-cli

    # https://<account-name>.openai.azure.com/openai/responses?api-version=<api-version>
    # In request body: {"model": "<deployment-name>"}
    account_name: "<account-name>"
    deployment: "<deployment-name>"
    api_version: "<api-version>"

    is_reasoning: true
    reasoning_effort: medium
    temperature: 1.0
    max_completion_tokens: 4096
    timeout: 60.0
```

**OpenAI-compatible Responses**

Provide the URL directory in `endpoint_url`, model name in `deployment` and do not set `account_name`.

For API key authentication, use credential type `api_key`.

For client certificate authentication, provide the path to the certificate in `client_certificate`.


**Reasoning Content**

If the endpoint provides reasoning in content (e.g. gpt-oss on vLLM), use `reasoning_source: content`.

If the endpoint does not provide reasoning, use `reasoning_source: none`.

Note that the default setting (`reasoning_source: summary`) requests reasoning summary, which causes an error if not enabled on the account.


### Custom (external implementation)

Implement the `LLMSessionBase` interface in a separate module, and point a factory function
to create an instance of your class (this could also be the class constructor).

Make sure the module can be imported (install as package).

```yaml
# config.yaml

agent_model:
    type: custom
    factory: my_module.create_session
    my_arg: my_value
    # ...
```

```python
# my_module.py

def create_session(my_arg: str) -> MyLLMClient:
    return MyLLMClient(my_arg=my_arg)

```
