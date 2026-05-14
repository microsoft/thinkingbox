# Writing Effective Test Cases

This document covers how to write test cases that produce useful signal for evaluation and RL training. For syntax and format, see [test_case_format.md](./test_case_format.md). For detailed examples, see [test_cases_deep_dive.md](./test_cases_deep_dive.md).

---

## The Goal

A good test case answers: **"Can the agent complete this task?"**

A bad test case answers: **"Did the agent do exactly what I expected in exactly the way I expected?"**

---

## The Three Requirements

Every test case must satisfy three requirements:

### 1. The task must be completable

Given the query, user_context, and available tools, a competent agent should be able to succeed.

**Check yourself:** Could YOU complete this task with only the information provided?

Bad:
```yaml
query: Send an email to John.
```
Who is John? What email? About what? The agent may guess instead of asking for clarification, which may not be the behavior you want to test.

Good:
```yaml
query: Send an email to John about the project update.
user_context: |
  John's email is john@company.com.
  The project is on track for the March deadline.
```

### 2. Success must be measurable

Your assertions must reliably distinguish success from failure.

**Check yourself:** If the agent succeeded, will your assertions pass? If the agent failed, will they fail?

Bad:
```python
assert len(x.tool_calls) == 2
```
Why 2? What if the agent solves it in 1 call? What if it solves it in 3?

Good:
```python
sent = x.effects["email"]["sent"]
assert len(sent) == 1, "Expected one email to be sent"
assert sent[0]["to"] == "john@company.com", "Expected email to be sent to john@company.com"
```

### 3. The task must be realistic

The test should reflect a real scenario - either something a user would ask, or something that would trigger autonomously in production.

**Types of realistic tasks:**

| Type | Trigger | Example |
|------|---------|---------|
| User-initiated | User asks directly | "Schedule a meeting with the sales team" |
| Event-triggered | Email arrives, file uploaded, etc. | "New support email received - classify and route it" |
| Scheduled | Time-based automation | "Generate weekly sales report and send to stakeholders" |
| Reactive | System state changes | "Deal stage changed to 'Closed Won' - update CRM and notify finance" |

**Check yourself:** Would this happen in production? Is this a scenario the agent needs to handle?

Bad:
```yaml
query: Use the send_email tool with parameters to="john@company.com" subject="Update" body="Hello"
```
This tests copy/paste, not capability.

Good (user-initiated):
```yaml
query: Let John know the project is on track.
user_context: |
  John's email is john@company.com.
```

Good (event-triggered):
```yaml
query: |
  New email received from customer.
  Subject: "Urgent - API not working"
  Body: "We're getting 500 errors on the /users endpoint since this morning."
bot_instructions: |
  You are an autonomous support agent. Classify the issue, check system status,
  and either resolve or escalate appropriately.
```

---

## Jobs to Be Done (JTBD)

When designing test cases, think about the **job** that needs to be accomplished, not just the immediate action.

### The Framework

A job has three layers:

1. **Functional job** - What task needs to be done?
2. **Circumstance** - What's the context? What constraints exist?
3. **Outcome** - What does success look like?

### Example: "Send an email"

**Simple task:**
```yaml
query: Send an email to the sales team.
user_context: |
  The sales team distribution list is sales-team@company.com.
  You want to remind them about the Friday deadline.
```
This is a valid test - it checks basic email capability.

**Fuller job (with JTBD):**
```yaml
query: |
  I need to update the sales team about the Q3 numbers before the Monday meeting.
user_context: |
  You're the sales director. The Q3 numbers are: revenue $2.1M (up 15%),
  new deals closed: 23, pipeline: $4.5M. The Monday meeting is at 9am.
  The sales team distribution list is sales-team@company.com.
```

The second version adds complexity by capturing:
- **Why** they're sending the email (Monday meeting prep)
- **What success looks like** (team is informed with the right data)
- **More to get right** (synthesizing numbers, right audience, correct details)

Both are valid. Use JTBD when you need to increase complexity or test more realistic scenarios.

### JTBD Helps With Complexity

Instead of artificially adding difficulty, JTBD reveals natural complexity:

| Simple Task | Real Job (with JTBD) |
|--------------|---------------------|
| "Book a meeting" | "Find a time when both the NYC and London teams are available, avoiding their lunch hours" |
| "Look up a contact" | "Find the right person to ask about the API issue - someone technical on the platform team" |
| "Send a reminder" | "Follow up on the proposal we sent last week, reference their concerns about timeline, and ask for a decision by Friday" |

### JTBD Reveals Multi-System Jobs

Real jobs often span multiple systems. JTBD helps you see this:

| Simple (single system) | Real Job (multiple systems) |
|-------------------------|----------------------------|
| "Create a calendar event" | "Schedule a client call, send them an invite, and add the prep notes to the CRM" |
| "Find a file" | "Get the latest contract from Drive, check if legal approved it in email, and update the deal stage in Salesforce" |
| "Send a message" | "Let the team know the deployment is done - post in Slack, update the Jira ticket, and email the stakeholders" |

This is where test complexity should come from - not artificial hurdles, but the natural reality that jobs cross system boundaries.

### Questions to Uncover the Job

When writing a test case, ask:

1. **Why does this need to happen?** (Not just what)
2. **What happens next?** (The job is rarely isolated)
3. **What would make this a failure?** (Even if technically "complete")
4. **What constraints exist?** (Time, audience, policy)
5. **What systems does this job touch?** (Email, calendar, CRM, files, chat)

---

## What to Put Where

| Field | Purpose | What goes here |
|-------|---------|----------------|
| `query` | The task or trigger | User request or event description |
| `user_context` | Info the simulated user knows (NOT the agent) | Details the user would provide if asked |
| `bot_instructions` | Additional instructions for the agent | Date, special constraints, persona, job details |
| `init` | Initial state of the world | Data the agent can discover via tools |

### Common Confusion: query vs user_context

The agent sees the `query` directly as a message.

The agent does NOT see `user_context` - it's for the simulated user to answer follow-up questions.

If the agent needs info to complete the task, you have options:

| Approach | When to use |
|----------|-------------|
| Put it in `query` | Agent should have this info upfront |
| Put it in `init` | Agent should discover it via tools |
| Put it in `user_context` only | Agent should ask for it - tests clarification behavior |

All three are valid. The principle: **always test whether the agent completed the task.**

```python
# Example 1: "Send email to John" (email only in user_context)
# If the email was sent correctly, the agent must have asked for clarification.
# No need to explicitly check - the outcome proves it.
sent = x.effects["email"]["sent"]
assert len(sent) == 1, "Only 1 email should have been sent"
assert sent["to"] == "john@company.com"

# Example 2: "Transfer money, and always confirm before transferring"
# Here confirmation IS part of the task requirement.
# x.transcript(n) returns the tail up to the last n assistant messages
assert judge.text_yesno(x.transcript(2), "Did the agent ask for confirmation before transferring?")
assert x.effects["banking"]["transactions"][0]["amount"] == 100.0
```

---

## Assertions: Test Outcomes, Not Paths

The agent's job is to complete the task. Your job is to verify the task was completed.

**Don't test HOW the agent solved it.** Test THAT it solved it.

### Bad: Testing the path

```python
# Why these specific tools? Why this order?
assert x.tool_calls[0].tool_call.name == "search_contacts"
assert x.tool_calls[1].tool_call.name == "send_email"
assert len(x.tool_calls) == 2
```

### Good: Testing the outcome

```python
# Did the email get sent correctly?
sent = x.effects["email"]["sent"]
assert len(sent) == 1, "Expected at least one email to be sent"
assert sent["to"] == "john@company.com", "Expected email to john@company.com"
# For content, use judge if the wording could vary
assert judge.text_yesno(sent["body"], "Does this mention the project update?")
```

### When to check tool calls

Only when the tool call itself IS the outcome:

```python
# Testing that agent did NOT do something forbidden
assert not any(tc.tool_call.name == "delete_all_files" for tc in x.tool_calls)

# Testing that agent used a specific required tool (policy requirement)
assert any(tc.tool_call.name == "log_audit_event" for tc in x.tool_calls)
```

### Read-only operations

For lookups and queries, there are no side effects to check. Verify `x.response` contains the correct information:

```python
# Task: "What's John's phone number?"
# The answer should be in the response
assert judge.text_yesno(x.response, "Does the response include the phone number 555-1234?")
```

---

## Using the Judge

The judge is an LLM. It's good at **specific, narrow questions** - not complex evaluation.

### What judges are good at

- **Semantic presence** - "Does this mention the deadline?" (when variations like "Monday" or "the 23rd" are acceptable)
- **Simple classification** - "Is this a refusal or an answer?"
- **Native concepts** - "Does this apologize?" "Does this confirm?" (concepts the judge understands inherently)

### What judges are bad at

- **Complex reasoning** - "Is this the best approach?"
- **Vague evaluation** - "Is this good?"
- **Multi-part questions** - "Does it do A, B, and C correctly?"

### Keep questions narrow

The narrower the question, the more reliable the judge.

```python
# BAD: Too vague
assert judge.text_yesno(x.response, "Is this a good response?")

# BAD: Too complex
assert judge.text_yesno(x.response, "Does the response correctly summarize the Q3 results and present them professionally?")

# GOOD: Narrow and specific
assert judge.text_yesno(x.response, "Does the response mention the Friday deadline?")
assert judge.text_yesno(x.response, "Does the response address the customer by name (John Smith)?")
```

Split complex checks into multiple narrow questions.

### Prefer deterministic checks when possible

Check `x.effects` for actions - this is ground truth:

```python
# Did the email get sent to the right person? (use any() for safety)
sent = x.effects["email"]["sent"]
assert any(s["to"] == "john@company.com" for s in sent)

# Did the transfer happen with the right amount?
txns = x.effects["banking"]["transactions"]
assert any(t["amount"] == 100.0 for t in txns)
```

Use judge when the concept can be expressed many ways:

```python
# The judge handles variations naturally - you don't need to list them all
assert judge.text_yesno(x.response, "Does the response mention the deadline?")
assert judge.text_yesno(x.response, "Does the response state the revenue growth percentage?")
```

Use judge for concepts the judge understands natively:

```python
# Apology/regret
assert judge.text_yesno(x.response, "Does the response apologize or express regret?")

# Refusal
assert judge.text_yesno(x.response, "Does the response refuse or decline the request?")

# Confirmation
assert judge.text_yesno(x.response, "Does the response confirm the action was completed?")
```

Avoid vague subjective assessments like "Is the tone professional?" - different judges will interpret this differently.

---

## Difficulty Calibration

Run your test 20 times. This gives enough samples for a reliable estimate while keeping iteration fast. Check the pass rate and Goldilocks probability.

```bash
uv run tb infer -c config/config.yaml -d ./dataset \
    --name mytest.py:test_something --repeat 20 --batch-size 20 -o results.jsonl
uv run tb agg results.jsonl
```

The output includes:
- **Success%** - raw pass rate
- **P(GZ|Data)** - probability the true pass rate is in the Goldilocks zone (approximately 6% to 94% pass rate)

**Target: P(GZ|Data) >= 0.95**

This means we're 95% confident the test has appropriate difficulty. A test that passes 100% or 0% of the time gives no signal for training or evaluation.

### If P(GZ|Data) is low because pass rate is too low

The test is too hard or broken. Check:

1. **Is the task completable?** Try it yourself with the available tools.
2. **Is the query clear enough?** Add more context.
3. **Are the assertions correct?** Maybe the agent is succeeding but your checks are wrong.
4. **Is the data set up correctly?** Check `init` and scenario `world_state`.

### If P(GZ|Data) is low because pass rate is too high

The test is too easy. Consider:

1. **Add complexity** - require multi-step reasoning
2. **Add ambiguity** - make the agent figure something out (resolvable via tools or clarification, not underspecification)
3. **Add decoy tools** - tools that seem relevant but aren't needed
4. **Add more data** - more contacts to search, more files to filter
5. **Stricter assertions** - check more details

---

## Debugging Failing Tests

When a test fails, here's how to diagnose the problem:

### 1. View what happened

```bash
# Run test and save full context
uv run tb infer -c config/config.yaml -d ./dataset \
    --name mytest.py:test_something --dump testcontext -o output.yaml

# View the conversation and results
uv run tb pp --less output.yaml
```

### 2. Re-run just the test (without re-running the agent)

```bash
# Useful when iterating on assertions
uv run tb run-test -c config/config.yaml \
    --name mytest.py:test_something --resultfile output.yaml -o /dev/null --verbose
```

### 3. Common failure causes

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Agent succeeded but test failed | Assertions too strict or checking wrong thing | Review assertions - are you testing outcome or path? |
| Agent failed but should have succeeded | Query unclear or data not set up | Check query, init, and scenario world_state |
| Inconsistent pass/fail across runs | Assertions depend on wording that varies | Use judge for semantic checks, not string matching |
| Judge assertions fail unexpectedly | Question too vague or complex | Make judge questions narrower and more specific |
| Test always fails on same assertion | Assertion is wrong or checking impossible condition | Print the actual value and verify your expectation |

### 4. Check the effects

Look at `x.effects` to see what actually happened. Effects are included in the output file when using `--dump testcontext`, and visible with `tb pp`:

```bash
uv run tb pp --less output.yaml
```

If effects show the right outcome but assertions fail, your assertions are wrong, not the agent.

---

## Checklist Before Submitting

- [ ] I can complete this task myself with query + tools + available data
- [ ] Assertions check outcomes, not specific tool sequences
- [ ] Judge questions are specific and unambiguous
- [ ] I ran the test 20x and P(GZ|Data) >= 0.95
- [ ] The task resembles something that would happen in production
