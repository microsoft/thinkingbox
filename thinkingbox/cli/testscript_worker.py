# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys

from thinkingbox.common.testrunner import TestScriptSubprocess

if __name__ == "__main__":
    TestScriptSubprocess.test_process_entrypoint(sys.stdin, sys.stdout)
