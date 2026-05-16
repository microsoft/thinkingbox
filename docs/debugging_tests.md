# How to debug a test

## In VSCode

By default ThinkingBox executes the test cases test functions in a way that does not work with a debugger. This is because it does not make the assumptions that these exist as files on the filesystem (e.g. when decoding from JSONL)

To debug the test function execution in `tb infer` or `tb run-test`, use `--debug-test`, which will import the test file directly (for supported configurations) and allow debugging.

In `tb infer` this is only supported when selecting a single test with `--name` and using `--repeat 1` (default).

To run a test with the VSCode python debugger integration:
- Configure the correct virtual env in VSCode
- Create a `.vscode/launch.json` file with the contents below
- Edit `args` as needed (configuration, test name...) but keep `--debug-test`
- Set breakpoints in the test file
- Pick a configuration from the Run and Debug tab and start debugging

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "tb-test",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/thinkingbox/cli/runtest_main.py",
            "console": "integratedTerminal",
            "args": "--debug-test -c config/config.yaml -d ./dataset -o /dev/null --verbose -r result_file.yaml"
        },
        {
            "name": "tb-cli",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/thinkingbox/cli/main.py",
            "console": "integratedTerminal",
            "args": "--debug-test -c config/config.yaml -d ./dataset --agent think --name cloud_drive.py:test_append_some_more_text --dump testcontext -o output.yaml"
        }
    ]
}
```
