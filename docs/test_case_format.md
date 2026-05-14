# Test case

## Python format

Test cases can be written as python functions containing a YAML string
literal containing the query and other information.
Multiple test cases can be written in each file.

The first string literal starting with `"!"` in the module body or function
body is interpreted as YAML.

Test case functions are required to have a YAML string (starting with `"!"`) in their body (root level).

```python

# Global, applies to all test cases unless overridden
"""!
# scenario name
scenario: cloud_drive
"""

# This function is not a test case, because it does not have a YAML string
def check_file_content(x: TestContext, filename: str, expected_text: str):
    found = False
    for f in x.effects["cloud_drive"]["files"]:
        if f["path"] == filename:
            found = True
            text = f["text_content"]
            assert text == "some text\nsome more text", (
                f"File content, expected: {expected_text!r}, found {text!r}"
            )
            break
    assert found, f"File was not found: {filename}"


# This function is a test case
def test_append_ok(x: TestContext, judge: Judge):
    # The following only applies to this test case
    """!
    # scenario name. This was already defined above,
    # it can be omitted, unless the test needs a different one.
    scenario: cloud_drive

    # User query (str)
    query: |
        Find a file named more.txt, then append the following to its contents on a new line: 'END'

    # User context (str, optional), used to prompt the User simulator
    # model with additional context or instructions
    user_context: "If asked to confirm anything, confirm"

    # Bot instructions (str, optional), if present they're appended to the
    # scenario's bot instructions
    bot_instructions: "Current date: 2025-06-22"

    # Maximum user simulator turns (int, default: 10)
    max_user_sim_turns: 2

    # Maximum agent decode turns per user message (int, default: unlimited)
    # Each turn = one agent decode pass (may produce multiple assistant messages/tool calls)
    max_agent_sim_turns: 5

    # Additional initialization for the MCP servers. If present, this
    # is merged recursively into the scenario's world_state
    init:
        cloud_drive:
            files:
            -   path: Documents/more.txt
                text_content: "more text"
    """
    check_file_content(x, "Documents/file.txt", "more text\nEND")
    assert judge.text_yesno(
        x.response, "Does the message confirm that file.txt was modified?"
    )
```

## File resolution by name

Each test case has a unique identifier in the form `filename.py:test_name`.
This allows test cases to be organized into subdirectories inside `test_case/`
while still being referenced by just their filename, without needing to know
the full path. This is used by `--name` and `--test-list` in `tb infer`,
`tb runtest`, and `tb dump-tests`. When loading by path (`--inputs`), the
path is used directly and this resolution does not apply.

Given a filename like `my_test_case.py`, the lookup tries the following
locations in order:

1. `test_case/my_test_case.py`
2. `test_case/my_test/my_test_case.py`
3. `test_case/my/my_test_case.py`

The filename is split on underscores and each candidate directory is built
from a prefix of those parts (longest first). The first match wins. Only a
single subdirectory level is supported. Nested paths like
`test_case/my/test/my_test_case.py` will not be found.

## Recursive merge of server initialization

The each server config in the scenario's world_state is merged with the one in the test cases's init (if present) according to the following rules:

- Dictionaries are merged by recursive update, keys from test case init take precedence if there is a collision
- Lists are merged by concatenation, elements from the test case init are appended

```yaml
# scenario
world_state:
    my_server:
        my_list: [1, 2, 3]
        my_dict:
            a: 1
            b: 2

# test case
init:
    my_server:
        my_list: [4, 5]
        my_dict:
            b: 11
            c: 12

# result
world_state:
    my_server:
        my_list: [1, 2, 3, 4, 5]
        my_dict:
            a: 1
            b: 11
            c: 12
```


## Test results

The schema for test results is `thinkingbox.common.chat_types.TestResult`.

**Pass result**

`TestResult(result=True, reward=1.0)`

The test function did not raise any exception and did not return anything.

**Pass with numeric reward**

`TestResult(result=True, reward=s)` (`0 <= s <= 1`)

The test function did not raise any exception and returned a number.

**Fail**

`TestResult(result=False, reward=0.0)`

The test function raised an AssertionError.

**Error**

`TestResult(result=False, reward=0.0, is_system_error=True)`

The test function raised an exception other than AssertionError.
Other fields in `TestResult` capture more information about the exception.


## TestCase object

The schema for a test case is `thinkingbox.common.config_types.TestCase`.

When parsing test cases from python files, fields are set as follows:
- `uid`: `<file-name>:<function-name>`
- `test_code`: Code in the python file, excluding code related to other test case functions, which is replaced by newline characters, so that line numbers in `test_code` match with line numbers in the original files.
- other fields: from the testcase's YAML
