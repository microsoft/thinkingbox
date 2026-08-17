# Analyzing inference results

## Result file

Run `tb infer` as described in the [ThinkingBox README](../README.md) to
produce a batch result JSONL file. Each line represents one task attempt. To
report pass@20 and pass^20, run 20 attempts per task.

## Compute the reported metrics

```bash
uv run tb agg output_thinkingbox_bench_v1.jsonl
```

For a JSONL containing 20 attempts for every task, the aggregate output
includes the paper's pass@1, pass@20, and pass^20 values. pass@20 measures
whether at least one of 20 attempts succeeds; pass^20 estimates whether all 20
attempts succeed. These metrics are omitted if tasks have unequal attempt
counts.
