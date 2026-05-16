# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


def _check_file(cloud_drive_effects: dict, filename: str, expected_content: str):
    found = False
    for f in cloud_drive_effects["files"]:
        if f["path"] == filename:
            found = True
            assert f["text_content"] == expected_content
            break
    assert found


def check_target_file_fn(filename: str):
    # a function fixture

    def fn(cloud_drive_effects: dict, expected_content: str):
        _check_file(
            cloud_drive_effects=cloud_drive_effects,
            filename=filename,
            expected_content=expected_content,
        )

    return fn


class CheckTargetFileCtx:
    # a context manager fixture

    def __init__(self, filename: str):
        self._filename = filename
        self._state = "init"

    def __call__(self, cloud_drive_effects: dict, expected_content: str):
        assert self._state == "enter", f"State {self._state} != 'enter'"
        _check_file(
            cloud_drive_effects=cloud_drive_effects,
            filename=self._filename,
            expected_content=expected_content,
        )

    def __enter__(self):
        self._state = "enter"
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._state = "exit"
