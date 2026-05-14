# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from pathlib import Path

import yaml
from create_tc_list import extract_test_case_uids


def process(test_dir: Path, train_file: Path, out_file: Path, sample: bool = False):

    train_file = train_file.expanduser().resolve()
    out_file = out_file.expanduser().resolve()

    test_cases = extract_test_case_uids(test_dir)

    with train_file.open("r", encoding="utf-8") as f:
        train_data = yaml.safe_load(f)

    eval_data = [tc for tc in test_cases if tc not in train_data]

    if sample:
        seen = set()
        sample_eval_data = []
        for eval_item in eval_data:
            if eval_item.split(":")[0] not in seen:
                seen.add(eval_item.split(":")[0])
                sample_eval_data.append(eval_item)

        eval_data = sample_eval_data

    with out_file.open("w", encoding="utf-8") as f:
        yaml.safe_dump(eval_data, f)

    print(
        f"Generated evaluation test case list with {len(eval_data)} entries at {out_file}"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate evaluation test case list from directory of YAML files."
    )
    parser.add_argument(
        "--test-dir", type=Path, help="Directory containing test case YAML files."
    )
    parser.add_argument("--train-file", type=Path, help="YAML file with training data.")
    parser.add_argument(
        "--out-file", type=Path, help="Output YAML file for evaluation list."
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="If set, only include a sample of the test cases.",
    )

    args = parser.parse_args()

    process(args.test_dir, args.train_file, args.out_file, sample=args.sample)
