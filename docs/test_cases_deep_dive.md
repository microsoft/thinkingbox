# Writing Tests Cases: A Deep Dive

This document provides a deeper dive into writing test cases for ThinkingBox (TB). Before proceeding, make sure you have a basic understanding of how to set up TB and write simple test cases, as covered in the [tutorial](./tutorial.md#create-a-test-case).

## What is a test case in ThinkingBox?

Similar to other agentic frameworks, a test case in TB is a structured way to evaluate the performance of an AI agent in a specific scenario. Most frameworks do this by defining global graders that equally applies to all type of queries and scenarios in a dataset. For example, when evaluating a math agent, a global grader might check if the final answer is correct, if the output format is followed, or if the reasoning steps are valid. TB takes a different approach by allowing you to define test cases at a more granular level. Each test case is associated with a specific scenario and query, and you can define custom metrics to evaluate the agent's performance in that context. This allows for more targeted and relevant evaluations when dealing with real world, complex scenarios, as the criteria for success can vary significantly between different scenarios.

Each test case can be thought of as a datapoint in a dataset carrying its own context, query, and evaluation criteria. This is particularly useful when evaluating agents in diverse scenarios where the definition of success may differ. You can think about these two approaches as:

- **Global Grader Approach**: Each datapoint (input and output) "flows" through the same global graders, which apply the same evaluation criteria to all scenarios and queries.

- **Test Case Approach**: Each datapoint is associated with a specific test case that defines its own context, query, and evaluation criteria. The evaluation is tailored to the specific scenario and query of that test case.

## Anatomy of a test case

A test case in TB is defined as a Python function within a Python file located in the `dataset/test_case/` directory.

The function needs to include a docstring that specifies the scenario, query, and any additional metadata. Here's a breakdown of the components:

- **Function Name**: The name of the function should be descriptive and follow a consistent naming convention. It should indicate the specific scenario being tested.

- **Docstring**: The docstring is a multi-line string that provides metadata about the test case. It should include:
  - **Scenario**: A brief description of the scenario being tested (optional).
  - **Query**: The specific query that initiates the agentic flow.
  - **User_context**: Additional context provided to the User LLM (optional). If provided, this allows the agent to interact with a simulated user, making the test case more dynamic and realistic. The user can provide additional information or clarification based on the agent's requests.

The function takes two parameters:

- **x (TestContext)**: An object that provides access to the context of the test case, including the agent's response and any effects or actions taken during the execution.
- **judge (Judge)**: An object that allows interaction with a LLM judge for evaluating the agent's response using natural language queries.

Consider the following example from the `airline_tau_bench` dataset in `Ai.ThinkingBox.Data/test_case`:

```python
from thinkingbox.common import Judge, TestContext

def test_012_cancel_mco_to_clt_insist_refund(x: TestContext, judge: Judge):
    """!
    query: |
        I want to cancel my flights from MCO to CLT and refund.
    user_context: |
        You are Amelia Sanchez (user ID: amelia_sanchez_4739) and you want to cancel your flights
        from MCO to CLT. You insist to cancel and have the refund.
    """
```
The query initiates the agentic flow, however it is vague in nature as it does not specify the user's details or the flight information other than the departure and arrival airport codes. The user context provides additional information about the user and their intent, which the agent can use to fulfill the request. If left empty, the agent would not be able to interact with a simulated user.

### Adding tests

We can now add the actual tests to evaluate the agent's behavior. The standard way to do this is to check for outcomes of a specific set of actions taken by the agent.
We first consider the standard format followed by TB, using a `assert` to return a binary grade. In the next section we discuss more and flexible grading methods.

Continuing with the previous example, the desired solution is:

- SOLUTION
    - Do nothing and do not transfer, it's against policy
    - Respond with a refusal message

this can be added to the test case above as follows:

```python
last_tool_is_transfer_to_human = (
        len(x.tool_calls) > 0
        and x.tool_calls[-1].tool_call.name == "transfer_to_human_agents"
    )
    assert not last_tool_is_transfer_to_human, "Should not be transferred, because against policy"

    msg_refusal = judge.text_yesno(
        x.response,
        "Does the response look like a refusal or denial?",
    )
    assert msg_refusal, "Agent did not explicitly refuse"
```
the first test checks that the agent did not call the `transfer_to_human_agents` tool, as this would be against policy. The second test uses the LLM judge to check if the agent's response contains a refusal message. If either of these assertions fails, the test case will fail, indicating that the agent did not behave as expected in this scenario. If your scenario requires repeating some tests across multiple test cases, consider defining a helper function to avoid code duplication. Note that both tests here involves tools that are purely returning information, and do not modify any state.

### Design Patterns

When designing test cases, consider the following patterns:

- **Test outcomes, not tool calls**: Focus on the outcomes of the agent's actions rather than the specific tools used. This allows for more flexibility in the agent's implementation and encourages the agent to find the best solution to the problem. For example, instead of checking if a specific tool was called, check if the desired outcome was achieved (e.g., a booking was made, a message was sent, etc.). For complex scenarios and world models, there can be multiple ways to achieve the same outcome, and focusing on outcomes allows for this flexibility.

- **Test critical actions**: Identify the critical actions that the agent must take to successfully complete the task. These are the actions that directly impact the outcome of the scenario. Ensure that your test cases include checks for these critical actions to verify that the agent is on the right track.

- **Test if the action has modified the state**: If the agent's actions are expected to modify the state of the world model, include tests to verify that these modifications have occurred as expected. This can involve checking the state of specific entities or attributes in the world model after the agent has taken its actions. For example, if the agent is expected to book a flight, check that the flight booking entity in the world model has been updated with the correct details. This is important to avoid situations where the agent appears to have completed the task (e.g., by generating a confirmation message) but has not actually modified the state of the world model. During Fine tuning, this can lead to the agent learning to "game" the tests (aka reward hacking) by generating the expected output without actually performing the necessary actions.

- **Use the LLM judge for complex evaluations**: For scenarios where the evaluation criteria are complex or subjective, leverage the LLM judge to perform natural language evaluations. This allows you to ask nuanced questions about the agent's response and behavior, providing a more comprehensive assessment of its performance.

## Grading beyond binary assertions

In addition to binary pass/fail assertions, you can design test cases that provide more nuanced feedback to guide agent improvement. Flexible rewarding involves assigning graded scores or rewards based on the quality or degree of correctness of the agent's behavior, rather than a strict yes/no outcome. This approach is especially useful during fine-tuning or reinforcement learning, where richer signals help the agent learn more effectively. For example, two sample responses might be both correct, but vary in the number of turns the agent took to reach the solution, the cost of the proposed solution or other metrics that affect the quality of the response.

In the example above, instead of asserting that the agent did not transfer to a human and that the response contains a refusal message, we could assign partial credit for each criterion met. For instance, if the agent correctly refuses but takes too many turns or uses an overly complex explanation, it might receive a lower score than an agent that refuses clearly and concisely in fewer turns. This really depends on your specific use case and what aspects of the agent's behavior you want to encourage.

As a simple example, we could modify the test case as explained below.

**NOTE** : this is just an illustrative examples, the grading function `execution_len_score` is not defined in TB by default, but you can implement your own grading functions as needed. An example of `average_reward` is provided in `thinkingbox.common.graders`.

```python
from thinkingbox.common import Judge, TestContext
from thinkingbox.common.graders import average_reward, execution_len_score

def test_012_cancel_mco_to_clt_insist_refund(x: TestContext, judge: Judge, cumulative_reward: bool = False):
    """!
    query: |
        I want to cancel my flights from MCO to CLT and refund.
    user_context: |
        You are Amelia Sanchez (user ID: amelia_sanchez_4739) and you want to cancel your flights
        from MCO to CLT. You insist to cancel and have the refund.
    """

    # Check that the agent did not transfer to a human (0 or 1)
    last_tool_is_transfer_to_human = (
        len(x.tool_calls) > 0
        and x.tool_calls[-1].tool_call.name == "transfer_to_human_agents"
    )

    assert not last_tool_is_transfer_to_human, "Should not be transferred, because against policy"

    # Assign a score function to evaluate  the number of turns taken (more turns = lower score)
    # assuming ideal solution is 3 turns, max acceptable is 10 turns
    len_execution_score = execution_len_score(x, ideal_len=3, max_len=10)

    # Ask the judge to rate the refusal quality on a 0-1 scale
    refusal_score = judge.score_text(
        x.response,
        "On a scale from 0 to 1, how clearly does the response refuse or deny the request?"

    reward = {
        "refusal_quality": refusal_score,
        "execution_length": len_execution_score,
    }

    # if you want a single cumulative reward (needed for RL training)
    if cumulative_reward:
        # Compute a cumulative reward as a weighted sum (weights can be tuned)
        weights = {
            "refusal_quality": 0.7,
            "execution_length": 0.3,
        }
        reward = average_reward(reward, weights)
        # Return or log the overall score for use in training or analysis
    return reward
```
Note that we have added an additional argument to the test function to indicate if we want a cumulative reward or a dictionary of rewards, this is not a default argument, just an example of what you could do. The former is needed when using the test case for RL training, where a single reward signal is needed. For evaluation purposes, the dictionary of rewards is more informative as it provides a breakdown of the agent's performance across different criteria. Also, the `last_tool_is_transfer_to_human` check is still an assertion, as it is a critical action that must not happen in this scenario.

Some examples of general grading functions can be found in `thinkingbox.common.graders`, such as `match_score` and `average_reward`. You can also define your own custom grading functions based on your specific needs, as complex scenarios may require specialized evaluation criteria instead of a fit-all approach.
