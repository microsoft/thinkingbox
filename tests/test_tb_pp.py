# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import subprocess
from pathlib import Path

import yaml


def _get_sample_decode_result(uid: str = "test123"):
    """Return a sample DecodeResult-like dict for testing."""
    return {
        "uid": uid,
        "messages": [
            {
                "T": "Text",
                "role": "system",
                "content": uid,
            }
        ],
    }


def _run_tb_pp(args: list[str]) -> subprocess.CompletedProcess:
    """Run tb pp command with the given arguments."""
    return subprocess.run(
        ["tb", "pp"] + args,
        capture_output=True,
        text=True,
    )


def test_tb_pp_jsonl_single_line(tmp_path: Path):
    """Test tb pp with a JSONL file containing a single JSON object on one line."""
    file_path = tmp_path / "data.jsonl"
    data = _get_sample_decode_result("test123")
    file_path.write_text(json.dumps(data))

    result = _run_tb_pp([str(file_path)])
    assert result.returncode == 0
    assert "test123" in result.stdout


def test_tb_pp_json_multiline(tmp_path: Path):
    """Test tb pp with a JSON file containing a single JSON object over multiple lines."""
    file_path = tmp_path / "data.json"
    data = _get_sample_decode_result("test123")
    file_path.write_text(json.dumps(data, indent=4))

    result = _run_tb_pp([str(file_path)])
    assert result.returncode == 0
    assert "test123" in result.stdout


def test_tb_pp_line_option(tmp_path: Path):
    """Test tb pp --line with a JSONL file with 3 lines."""
    file_path = tmp_path / "data.jsonl"
    data = [
        _get_sample_decode_result("test123"),
        _get_sample_decode_result("test456"),
        _get_sample_decode_result("test789"),
    ]
    file_path.write_text("\n".join(json.dumps(obj) for obj in data))

    # Test reading line 2 (1-based)
    result = _run_tb_pp(["--line", "2", str(file_path)])
    assert result.returncode == 0
    assert "test123" not in result.stdout
    assert "test456" in result.stdout
    assert "test789" not in result.stdout


def test_tb_pp_yaml(tmp_path: Path):
    """Test tb pp with a YAML file."""
    file_path = tmp_path / "data.yaml"
    data = _get_sample_decode_result("test123")
    file_path.write_text(yaml.dump(data, default_flow_style=False))

    result = _run_tb_pp([str(file_path)])
    assert result.returncode == 0
    assert "test123" in result.stdout


def test_tb_pp_jsonl_multiple_lines_fails(tmp_path: Path):
    """Test that tb pp fails when JSONL file contains multiple lines without --line."""
    file_path = tmp_path / "data.jsonl"
    data = _get_sample_decode_result("test123")
    # Write the same object 2 times
    lines = [json.dumps(data) for _ in range(2)]
    file_path.write_text("\n".join(lines))

    result = _run_tb_pp([str(file_path)])
    assert result.returncode != 0
    assert "More than one line of input" in result.stderr
