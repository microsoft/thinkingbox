# Guidelines

- This project uses Pydantic version 2
- Use lowercase type annotations, compatible with python 3.10+
- This project uses pytest for unit tests; use test functions, not test classes
- Always run python using uv, like `uv run python ...`


# Test cases

A test case includes:
- definition of a task for an agent to complete
- test code to verify that the task was completed successfully

Test cases are part of a ThinkingBox dataset, they are distinct from unit tests. This section only applies to test cases.

## Anatomy of a test case

You'll see tests like this:

```
"""!
scenario: sales_rep_email
"""

def test_generate_outreach_email(x: TestContext, judge: Judge):
    """!
    query: |
       Can you send a personalized email to this prospect?
    user_context: |
       Name: Alex Johnson
    """
    assert len(x.tool_calls) == 1, "Expected one tool call to generate_outreach_email"

    # Must include proper greeting with name
    assert judge.text_yesno(
        x.response,
        "Does the email properly greet Alex Johnson by name?",
    )
```

There are several parts to this test:

* the scenario `sales_rep_email` is a yaml file that has content about this test.
* the query is the test case itself.
* the user_context is used in a user model to simulate a user with more info.

### Good Test Definitions

Good tests should test meaningful side effects from running the test case. They should allow the model to be flexible in *how* it answers a question.

Not all existing tests meet this definition of good.

## Modifying tests

If you are asked to make a test "goldilocks" or between 20% or 80% of success, then here are some tips:

* Use a command like `uv run tb infer -c config/config_o4mini.yaml --dataset ./dataset --agent think --name gdrive.py:test_create_folder_and_organize --repeat 20 --batch-size 20 --dump raw --output <unique file>` to execute the test.
* Use a command like `uv run tb agg <unique file>` to read the file and aggregate success rate.

If you get timeouts on `infer` then rerun infer. If there are not 20 results in the batch then rerun.

The best ways to make an example more complex are:
* More realistic or stricter criteria that you check.
* More ambiguous (but still achievable) goals.
* Data can be added to make search/retrieval tasks harder.
* If the judge is the reason for failures, consider removing it IF you have good coverage of what happened. Judges can be less reliable than deterministic checks.
* If the judge is the reason for failures, you should determine if the deterministic assertions really validate the scenario. Focus on high quality questions and high quality deterministic checks.
* Tests that are underspecified are tests where the query + user_context is not sufficient to correctly accomplish the task. These should be avoided. If a test is underspecified then the right outcome is for the agent to ask the user for more information, not to guess. These tests are rarely goldilocks.
* Add additional tools to the scenario that are not needed as "decoys"

See the docs folder for information about how tests are constructed.

If a test is too easy, make a plan for realistic scenarios that are harder before executing.

## Writing Rubric Criteria

When adding rubric criteria to a test, especially penalty rubrics, describe specific, observable violations that can be detected directly from the response. Avoid double negatives and vague phrasing. See [docs/rubrics_judge.md](docs/rubrics_judge.md) for detailed guidance and examples.
