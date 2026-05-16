# GeneratedAnswerEvaluator

`GeneratedAnswerEvaluator` is the standard fixture for evaluating knowledge-QA (RAG) test
cases — scenarios where the agent's job is to answer a question by retrieving information
from a document collection. It evaluates a generated answer on semantic correctness,
citation accuracy, textual quality, etc.

Use `GeneratedAnswerEvaluator` when:
- The agent is expected to produce a **factual answer** grounded in specific source documents.
- You want to assert that the answer is **semantically correct** relative to a reference answer.
- You want to record (and eventually enforce) that the answer **cites the right document and location**.

**See also:**
- [`fixtures.md`](fixtures.md) — how to wire fixtures via `conftest.yaml` and dependency injection
- [`rubrics_judge.md`](rubrics_judge.md) — how rubric scoring and rewards are calculated
- [`thinkingbox-data/docs/adding_search_sources_scenario.md`](https://github.com/microsoft/thinkingbox-data/blob/main/docs/adding_search_sources_scenario.md) — end-to-end guide for creating a knowledge-QA scenario (including use of this evaluator)/

---

## Contents

1. [Quick start](#1-quick-start)
2. [Semantic correctness check](#2-semantic-correctness-check)
3. [Rubric-based evaluation](#3-rubric-based-evaluation)
4. [Evaluation modes (`fail_on_mismatch`)](#4-evaluation-modes-fail_on_mismatch)
5. [Citation locations (`ref_doc` and `ref_loc`)](#5-citation-locations-ref_doc-and-ref_loc)
   - [5.1 Specifying the source document (`ref_doc`)](#51-specifying-the-source-document-ref_doc)
   - [5.2 Location specifiers (`ref_loc`)](#52-location-specifiers-ref_loc)
6. [Configuration reference](#6-configuration-reference)
7. [Notes and limitations](#7-notes-and-limitations)

---

## 1. Quick start

Add the fixture to your scenario YAML or `conftest.yaml`:

```yaml
fixtures:
  answer_evaluator:
    type: thinkingbox.fixtures.GeneratedAnswerEvaluator
    fail_on_mismatch: true
    rubrics_threshold: 0.7
    max_workers: 4
```

Use it in a test function:

```python
from thinkingbox.common import TestContext, Judge
from thinkingbox.fixtures import GeneratedAnswerEvaluator

def test_cbo_deficit_estimate(
    x: TestContext, judge: Judge, answer_evaluator: GeneratedAnswerEvaluator
):
    """!
    query: |
        Did CBO overestimate or underestimate the 2023 deficit, and by how much?
    """
    answer_evaluator.evaluate(
        reference="CBO underestimated the 2023 deficit by $1 trillion, or 3.9% of GDP.",
        ref_doc="59682-Accuracy.pdf",
        ref_loc="p2,p12",
    )
```

`judge` and `x` are injected automatically by the fixture system — no configuration needed.
See [`fixtures.md`](fixtures.md) for details on dependency injection.

For the full process of building a knowledge-QA scenario (scenario YAML, collection setup,
DVC tracking), see
[`adding_search_sources_scenario.md`](https://github.com/microsoft/thinkingbox-data/blob/main/docs/adding_search_sources_scenario.md)
in thinkingbox-data.

---

## 2. Semantic correctness check

The core evaluation is a judge-mediated comparison between the generated answer and a short
reference answer:

> *"Does the candidate answer communicate the full and complete semantics of the reference?"*

The judge is instructed to:
- Treat formatting differences in numeric answers as irrelevant.
- Accept reasonable currency markers in generated answers even when absent from the reference.
- Require a currency marker in the generated answer if the reference includes one.

`evaluate()` runs this check internally. You can also call
`candidate_answer_matches_reference()` directly when you only need the boolean result and
do not want rubrics:

```python
matched = answer_evaluator.candidate_answer_matches_reference(
    query=x.query(),
    reference="$1 trillion",
    candidate=x.response,
)
```

---

## 3. Rubric-based evaluation

In addition to the semantic check, `evaluate()` runs real-valued scoring via
[`RubricJudge`](rubrics_judge.md) internally. As of this writing, it scores
up to 4 equally-weighted rubrics (although the answer generation subsystem
does not currently generate citations, so the first two cannot be scored accurately):

| Rubric | Triggered by | Question asked of the judge |
|---|---|---|
| Document citation | `ref_doc` is provided | Does the response cite or reference `<filename>`? |
| Location citation | `ref_loc` is provided | Does the response indicate the information comes from `<location>`? |
| Readability | always | Is the response written in clear, readable language? |
| Conciseness | always | Does the response avoid unnecessary verbosity? |

The combined rubric reward and individual scores are written to `x.metadata` automatically.
See [`rubrics_judge.md`](rubrics_judge.md) for how the reward is calculated from rubric scores.

---

## 4. Evaluation modes (`fail_on_mismatch`)

`fail_on_mismatch` controls how semantic correctness interacts with the rubric evaluation:

| `fail_on_mismatch` | Semantic check | Rubrics |
|---|---|---|
| `True` *(default)* | Hard `AssertionError` on mismatch; test stops immediately | Run afterward; raise `AssertionError` if reward < `rubrics_threshold` |
| `False` | Folded into rubrics at 50% weight | Combined reward = 50% correctness + 50% rubrics; raise if below threshold |

Recommendation: use `fail_on_mismatch=True` (the default) in most test cases — it stops early on a wrong
answer and gives a clear failure message. Use `fail_on_mismatch=False` when you are running
exploratory or RL training scenarios where a partial score is more useful than a hard pass/fail.

`rubrics_threshold` (default `0.7`): the minimum combined rubric reward required to pass.
Set to `0.0` to record rubric scores without enforcing any threshold.

Both can be set at the fixture level in `conftest.yaml` and overridden per `evaluate()` call:

```python
# Override for a single call — disable threshold, allow mismatch to be scored
answer_evaluator.evaluate(
    reference="...",
    fail_on_mismatch=False,
    rubrics_threshold=0.0,
)
```

---

## 5. Citation locations (`ref_doc` and `ref_loc`)

### 5.1 Specifying the source document (`ref_doc`)

`ref_doc` is the exact filename of the source document the answer should cite (e.g.
`"59682-Accuracy.pdf"` or `"contoso-timesheet.xlsx"`). When provided, a rubric asks the judge
whether the response mentions or references that filename.

### 5.2 Location specifiers (`ref_loc`)

`ref_loc` records where within the document the relevant information can be found. It is used
to construct a rubric that asks the judge whether the response indicates that the information
comes from that location.

A `ref_loc` value is built from one or more **location specifiers** combined with operators.

#### Location specifiers

A single location specifier identifies a specific page, position, or range within a document —
for example `p3` (page 3) or `Justin:r47-62` (rows 47–62 of the sheet named 'Justin').

**Paginated documents**

| Pattern | Meaning | Example |
|---|---|---|
| `p<N>` | Single page | `p3` — page 3 |
| `p<N>-<M>` | Consecutive page range (inclusive) | `p95-100` — pages 95 through 100 |

This includes documents with explicit or consistent pagination:

* PDF
* PPTX and other slide formats
* DOCX (pagination may vary *slightly* across platforms or depending on installed fonts, but we treat DOCX as a paginated format)

This does **not** apply to HTML or Markdown (which are not paginated) or to tabular formats such as CSV, TSV, and XLSX.

**Spreadsheet documents (XLSX, CSV)**

| Pattern | Meaning | Example |
|---|---|---|
| `r<N>` | Single row | `r42` — row 42 |
| `r<N>-<M>` | Row range (inclusive) | `r7-180` — rows 7 through 180 |
| `<Sheet>:r<N>` | Single row on a named sheet | `July2024:r42` |
| `<Sheet>:r<N>-<M>` | Row range on a named sheet | `Justin:r47-62` |
| `Sheet<N>:r<M>` | Row on sheet by 1-based index | `Sheet2:r5` |

When no `<Sheet>:` prefix is given, the first sheet of the workbook is assumed.

Row numbers refer to physical spreadsheet rows (1-based, including header rows), matching
what a spreadsheet application displays.

#### Operators

Location specifiers are combined using two operators. `+` binds more tightly than `,`.

| Operator | Meaning | Example |
|---|---|---|
| `+` | AND — all locations must be cited | `p3+p8` — both page 3 and page 8 required |
| `,` | OR — any one alternative is sufficient | `p3,p8` — page 3 or page 8 is sufficient |

Use `+` when the answer requires information spread across multiple locations (e.g. a table
spanning two pages, or a value requiring two separate row lookups). Use `,` when the same
information appears in more than one place and citing any one is acceptable.

#### Examples

Paginated documents:

```
p3                   # single page
p95-100              # pages 95 through 100 (consecutive range)
p3+p8                # page 3 AND page 8 (both required; non-consecutive)
p3,p8                # page 3 OR page 8 (either is sufficient)
p1-2,p3+p8           # pages 1–2 OR (page 3 AND page 8)
p30+p31,p329         # (pages 30 and 31) OR page 329
```

Spreadsheet:

```
r42                  # row 42 (first sheet by default)
r7-180               # rows 7 through 180 (first sheet by default)
July2024:r42         # row 42 of the sheet named 'July2024'
Justin:r47-62        # rows 47 through 62 of the sheet named 'Justin'
February2024:r7-180  # rows 7 through 180 of the sheet named 'February2024'
```

---

## 6. Configuration reference

All constructor parameters and their defaults:

```yaml
fixtures:
  answer_evaluator:
    type: thinkingbox.fixtures.GeneratedAnswerEvaluator
    fail_on_mismatch: true      # hard assert on semantic mismatch (default: true)
    rubrics_threshold: 0.7      # minimum rubric reward to pass (default: 0.7)
    max_workers: 2              # thread-pool size for parallel rubric evaluation (default: 2)
    # judge and x are injected automatically — do not configure them here
```

`fail_on_mismatch` and `rubrics_threshold` can be overridden per `evaluate()` call (see
[§4](#4-evaluation-modes-fail_on_mismatch)). `max_workers` is fixed at fixture instantiation.

The fixture can also be declared in a scenario YAML `fixtures:` block to override the
directory-level `conftest.yaml` for a specific scenario. See [`fixtures.md`](fixtures.md)
for resolution order and the full fixture wiring reference.

---

## 7. Notes and limitations

- **Citation information is not provided by answer generation subsystem** so `ref_doc` and
  `ref_loc` cannot be evaluated
- **Cross-document citations are not yet supported.** Each `evaluate()` call accepts a single
  `ref_doc`. If the answer requires information from two different documents, this cannot yet
  be expressed in `ref_loc`. This limitation is tracked as a TODO in `answer_evaluator.py`.
- **`ref_pages` is a deprecated alias** for `ref_loc` in `evaluate()`. Existing test cases
  using `ref_pages=` continue to work; new test cases should use `ref_loc=`.
