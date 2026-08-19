# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from pathlib import Path
from typing import Dict, List, Literal, Union

import pandas as pd
from create_tc_list import _resolve_output_path


def _extract_test_result(result_dict) -> bool:
    """Extract boolean result from test_result dictionary"""
    try:
        return result_dict.get("result", False) if result_dict else False
    except Exception:
        return False


def _write_structured_yaml(
    filtered_uids: List[str],
    output_path: Path,
    filter_params: Dict[str, Union[float, str]],
) -> None:
    """Write filtered UIDs to YAML file with metadata and comments."""
    with open(output_path, "w") as f:
        # Write header comment with filter parameters
        f.write("# Filtered test case UIDs\n")
        f.write("# Filter criteria:\n")
        f.write(f"#   - Bottom threshold: {filter_params['bottom_threshold']}%\n")
        f.write(f"#   - Top threshold: {filter_params['top_threshold']}%\n")
        f.write(f"#   - Filter by: {filter_params['filter_by']}\n")
        f.write(f"#   - Total UIDs: {len(filtered_uids)}\n")
        f.write("\n")

        # Write the UIDs
        for uid in filtered_uids:
            f.write(f"- {uid}\n")


def filter_high_quality_samples(
    benchmark_file: Union[str, Path],
    bottom_threshold: float = 1.0,
    top_threshold: float = 99.0,
    filter_by: Literal["testcase", "scenario"] = "testcase",
    output_filename: str = None,
    save_yaml: bool = True,
) -> List[str]:
    """
    Filter high-quality samples from benchmark results based on configurable thresholds.
    Excludes perfect scores (100%) and includes only those between thresholds.

    Args:
        benchmark_file: Path to the benchmark results JSONL file
        bottom_threshold: Minimum result percentage to include (0-100)
        top_threshold: Maximum result percentage to include (0-100)
        filter_by: Level to filter by - "testcase" for individual test cases or
                  "scenario" to filter by scenario-level performance
        output_filename: Full path to output file (defaults to [benchmark_filename]_filtered.yaml in script directory)
        save_yaml: Whether to save results to YAML file

    Returns:
        List of filtered UIDs
    """

    benchmark_file = Path(benchmark_file).expanduser().resolve()

    if not benchmark_file.exists():
        raise FileNotFoundError(f"Benchmark file not found: {benchmark_file}")

    # Load the benchmark results
    try:
        df = pd.read_json(benchmark_file, lines=True)
    except Exception as e:
        raise ValueError(f"Failed to load benchmark file {benchmark_file}: {e}")

    if df.empty:
        print("Warning: Benchmark file is empty")
        return []

    # Extract the boolean result from test_result field
    df["result"] = df["test_result"].apply(_extract_test_result)

    if filter_by == "testcase":
        # Filter by individual test case performance
        # Group by uid and calculate mean success rate (as percentage)
        results_tc = df[["uid", "result"]].groupby("uid").mean() * 100

        # Filter based on thresholds
        filtered_results = results_tc[
            (results_tc["result"] >= bottom_threshold)
            & (results_tc["result"] <= top_threshold)
        ]

        filtered_uids = filtered_results.index.tolist()

    elif filter_by == "scenario":
        # Filter by scenario-level performance
        # Extract scenario from uid (assumes format "scenario:test_name")
        df["scenario"] = df["uid"].apply(lambda x: x.split(":")[0])

        # First aggregate by uid to get per-test-case results
        results_by_uid = (
            df[["scenario", "uid", "result"]]
            .groupby("uid")
            .agg({"result": "mean", "scenario": "first"})
            .reset_index()
        )
        results_by_uid["result"] = results_by_uid["result"] * 100

        # Filter out scores above top_threshold
        results_by_uid = results_by_uid[results_by_uid["result"] <= top_threshold]

        # Then group by scenario and aggregate
        results_scenario = results_by_uid.groupby("scenario").agg(
            {"result": "mean", "uid": list}
        )

        # Apply bottom threshold
        results_scenario = results_scenario[
            results_scenario["result"] >= bottom_threshold
        ]

        # Explode to get individual UIDs
        results_scenario = results_scenario.explode("uid")
        filtered_uids = results_scenario["uid"].tolist()

    else:
        raise ValueError("filter_by must be either 'testcase' or 'scenario'")

    # Save to YAML if requested
    if save_yaml and filtered_uids:
        output_path = _resolve_output_path(
            benchmark_file, output_filename, suffix="_filtered"
        )
        filter_params = {
            "bottom_threshold": bottom_threshold,
            "top_threshold": top_threshold,
            "filter_by": filter_by,
        }
        _write_structured_yaml(filtered_uids, output_path, filter_params)
        print(f"Filtered {len(filtered_uids)} UIDs and saved to {output_path}")
    elif save_yaml:
        print("No UIDs matched the filter criteria, no file saved")
    else:
        print(f"Filtered {len(filtered_uids)} UIDs")

    return filtered_uids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter high-quality samples from benchmark results and save to YAML"
    )
    parser.add_argument(
        "--benchmark_file", type=str, help="Path to the benchmark results JSONL file"
    )
    parser.add_argument(
        "--bottom_threshold",
        type=float,
        default=1.0,
        help="Minimum result percentage to include (0-100, default: 1.0)",
    )
    parser.add_argument(
        "--top_threshold",
        type=float,
        default=99.0,
        help="Maximum result percentage to include (0-100, default: 99.0)",
    )
    parser.add_argument(
        "--filter_by",
        type=str,
        choices=["testcase", "scenario"],
        default="testcase",
        help="Level to filter by - 'testcase' or 'scenario' (default: testcase)",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default=None,
        help="Full path to output file (default: [benchmark_filename]_filtered.yaml in script directory)",
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Don't save results to YAML file, only print to console",
    )

    args = parser.parse_args()

    filtered_uids = filter_high_quality_samples(
        benchmark_file=args.benchmark_file,
        bottom_threshold=args.bottom_threshold,
        top_threshold=args.top_threshold,
        filter_by=args.filter_by,
        output_filename=args.output_filename,
        save_yaml=not args.no_save,
    )

    if args.no_save:
        print(f"\nFiltered UIDs ({len(filtered_uids)}):")
        for uid in filtered_uids:
            print(f"  - {uid}")
