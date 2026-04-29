# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import os
from collections import defaultdict
from pathlib import Path

from thinkingbox.common.python_test_file import iter_test_functions


def extract_test_case_uids(
    path: Path | str,
    output_filename: str | None = None,
) -> list[str]:
    """
    Extract all test case UIDs from Python test files and dump them to a YAML file.

    Args:
        path: Path to test case file or directory
        output_filename: path to output file

    Returns:
        List of test case UIDs
    """

    path = Path(path).expanduser().resolve()

    if path.is_file():
        uids_by_file = {path.name: _extract_uids_from_file(path)}
    elif path.is_dir():
        uids_by_file = _extract_uids_from_directory(path)
    else:
        raise FileNotFoundError(f"Path not found: {path}")

    # Flatten to get total count
    all_uids = []
    for file_uids in uids_by_file.values():
        all_uids.extend(file_uids)

    # Save to YAML file with comments
    if output_filename:
        output_path = Path(output_filename).expanduser().resolve()
        _write_structured_yaml(uids_by_file, output_path)
        print(f"Extracted {len(all_uids)} test case UIDs to {output_path}")

    return all_uids


def _extract_uids_from_file(file_path: Path) -> list[str]:
    """Extract UIDs from a single Python test file."""
    if not file_path.suffix == ".py":
        return []

    uids = []
    try:
        code = file_path.read_text(encoding="utf-8")
        for test_func in iter_test_functions(code, check_config=False):
            uid = f"{file_path.name}:{test_func.name}"
            uids.append(uid)
    except Exception as e:
        print(f"Warning: Could not extract UIDs from {file_path}: {e}")

    return uids


def _extract_uids_from_directory(dir_path: Path) -> dict[str, list[str]]:
    """Extract UIDs from all Python test files in a directory, grouped by file."""
    uids_by_file = defaultdict(list)

    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".py"):
                file_path = Path(root) / file
                uids = _extract_uids_from_file(file_path)
                if uids:  # Only add files that have test functions
                    # Use stem (filename without extension) as the scenario name
                    scenario_name = file_path.stem
                    uids_by_file[scenario_name] = uids

    return dict(uids_by_file)


def _write_structured_yaml(
    uids_by_file: dict[str, list[str]], output_path: Path
) -> None:
    """Write UIDs to YAML file with comments for each scenario."""
    with open(output_path, "w") as f:
        for scenario_name, uids in uids_by_file.items():
            # Write comment for the scenario
            f.write(f"# Scenario: {scenario_name}\n")

            # Write the UIDs for this scenario
            for uid in uids:
                f.write(f"- {uid}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract test case UIDs to a YAML file"
    )
    parser.add_argument("--path", help="Path to test case file or directory")
    parser.add_argument(
        "--output_filename",
        type=str,
        help="Full path to output file",
    )

    args = parser.parse_args()

    extract_test_case_uids(
        path=args.path,
        output_filename=args.output_filename,
    )
