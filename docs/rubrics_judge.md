# Rubric Judge - Design and Tutorial

## Overview

**Rubric Judge** is a multi-criteria evaluation engine for LLM responses, inspired by academic grading rubrics. It evaluates a response against independent criteria (rubrics) with configurable weights—some contribute positively while others act as penalties (deductions or multipliers). Scores are aggregated into a final reward in `[0.0, 1.0]` and compared against a threshold for pass/fail decisions.

---

## Quick Example

**Scenario:** Evaluate an email response for quality.

**Step 1: Define rubrics and get LLM judge ratings**

Each rubric is rated YES/NO by the LLM judge. The **effect** is the score contribution:
- Positive YES → +weight; NO → +0
- Penalty YES → applies penalty; NO → no effect

| Criterion | Weight | Type | Rating | Effect |
|-----------|--------|------|--------|--------|
| "Does the email address the recipient's question?" | 50 | Positive | YES | +50 pts |
| "Is the tone professional and appropriate?" | 30 | Positive | YES | +30 pts |
| "Does it include all requested information?" | 20 | Positive | NO | +0 pts |
| "Contains factual errors or misinformation" | 15 | Deduction | YES | −15 pts |
| "Contains offensive or inappropriate content" | 0.5 | Multiplicative | YES | ×0.5 |

**Step 2: Compute reward**

- Earned from positive rubrics: 50 + 30 + 0 = 80
- Deduction penalty: −15
- Total positive weight: 50 + 30 + 20 = 100
- Base reward = (80 − 15) / 100 = **0.65**
- Multiplicative penalty scales by (1 − 0.5) = 0.5
- Final reward = 0.65 × 0.5 = **0.325**

**Step 3: Compare to threshold**
- Threshold: 0.70
- Reward 0.325 < 0.70 → **FAIL**

This example illustrates the key concepts: **rubrics** (criteria), **weights** (max points), **rating** (YES/NO from LLM), **reward** (weighted sum), and **threshold** (pass/fail gate).

---

## Definitions

| Term | Description |
|------|-------------|
| **Rubric** | A single evaluation question answered YES/NO by the judge |
| **Weight** | Maximum points for positive rubrics; deduction/scaling amount for penalties |
| **Rating** | YES (1) or NO (0) from the LLM judge |
| **Reward** | Final aggregated score, clamped to [0.0, 1.0] |
| **Threshold** | Pass/fail cutoff (e.g., 0.70 means reward ≥ 0.70 passes) |
| **Penalty** | A rubric that reduces the score when its criterion is met |
| ↳ **Deduction** | Subtracts weight from the score |
| ↳ **Multiplicative** | Scales reward by (1 - weight); weight must be in [0, 1] |

---

## Reward Calculation

**Rubric scores:** Positive rubric $s_i \in \{0, w_i\}$; Deduction penalty $p_j \in \{0, d_j\}$

**Base reward:**
$$
\mathrm{base\_reward} = \frac{\sum_i s_i - \sum_j p_j}{\sum_i w_i} \quad \text{clamped to } [0, 1]
$$

**Final reward:** Each multiplicative penalty $m_k \in [0, 1]$ scales the result:
$$
\mathrm{reward} = \mathrm{base\_reward} \times \prod_k (1 - m_k) \quad \text{clamped to } [0, 1]
$$

---

## Usage

### Setup

Add the fixture to your config:

```yaml
fixtures:
    rubric_judge:
        type: thinkingbox.common.rubrics_judge.RubricJudge
        max_workers: 10
```

The framework automatically injects `judge` and `x` (TestContext) into the constructor from
the runtime context. No extra config is needed.

Or instantiate directly in a test function:
`rubric_judge = RubricJudge(judge=judge, x=x, max_workers=1)`

### Example

```python
from thinkingbox.common.rubrics_judge import RubricJudge

def test_response_quality(x, judge, rubric_judge: RubricJudge):
    '''!
    query: Summarize the document
    '''
    rubrics = [
        rubric_judge.Config(criterion="Is the answer factually correct?", weight=50),
        rubric_judge.Config(criterion="Does it include all required details?", weight=30),
        rubric_judge.Config(criterion="Is the writing clear?", weight=20),
        # Penalties
        rubric_judge.Config(
            criterion="Contains hallucinated information",
            weight=15, is_penalty=True, penalty_type="deduction",
        ),
        rubric_judge.Config(
            criterion="Contains unsafe content",
            weight=0.8, is_penalty=True, penalty_type="multiplicative",
        ),
    ]

    result = rubric_judge.evaluate(
        response=x.response, rubrics=rubrics,
        global_threshold=0.70, throw_on_failure=True,
    )
    return result.reward
```

The `evaluate()` call:
- Evaluates response against all rubrics using `self.judge`
- Aggregates scores into a final reward
- Writes `rubric_reward` and `rubric_scores` to `x.metadata` automatically
- Raises `AssertionError` if below threshold (when `throw_on_failure=True`)

> See [`dataset/test_case/cloud_drive.py`](../dataset/test_case/cloud_drive.py) for a complete example.

---

## Best Practices

### General Guidelines

- **Weight by importance** — higher weights for critical criteria
- **Positive rubrics** for expected behavior; **penalties** for must-not behaviors
- **Deduction penalties** for localized issues; **multiplicative** for trust-breaking failures
- Use `throw_on_failure=True` for CI enforcement, `False` for analysis/logging

### Writing Effective Penalty Rubrics

Describe specific, observable violations. Avoid double negatives.

| ❌ Bad | ✅ Good |
|---|---|
| "Does not mention the correct percentage" | "States an incorrect percentage" |
| "Fails to include required information" | "Missing required information" |
| "The response is not professional" | "Contains unprofessional language" |
| "Doesn't cite sources properly" | "Contains missing or incorrect citations" |

Test each penalty with responses that should/shouldn't trigger it. If inconsistent, rewrite to be more explicit.
