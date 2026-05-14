# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import click

from thinkingbox.common.utils import setup_logging


class LazyGroup(click.Group):
    """A click Group that lazily imports commands to avoid pulling in heavy dependencies."""

    _lazy_commands: dict[str, str] = {}

    def list_commands(self, ctx):
        return sorted(set(super().list_commands(ctx)) | set(self._lazy_commands))

    def get_command(self, ctx, cmd_name):
        # Try already-registered commands first
        cmd = super().get_command(ctx, cmd_name)
        if cmd is not None:
            return cmd
        # Lazily import if known
        import_path = self._lazy_commands.get(cmd_name)
        if import_path is None:
            return None
        module_path, attr = import_path.rsplit(".", 1)
        from importlib import import_module

        mod = import_module(module_path)
        return getattr(mod, attr)


@click.group(cls=LazyGroup)
def main():
    setup_logging()


main._lazy_commands = {
    "agg": "thinkingbox.cli.agg_main.agg",
    "dump-tests": "thinkingbox.cli.dump_tests.dump_tests",
    "infer": "thinkingbox.cli.infer.infer",
    "mcp-start": "thinkingbox.tools.session_proxy.mcp_start",
    "pp": "thinkingbox.cli.show_main.pp",
    "run-test": "thinkingbox.cli.runtest_main.run_test",
    "sbs": "thinkingbox.cli.sbs_main.sbs",
    "tui": "thinkingbox.cli.tui_main.tui",
}


if __name__ == "__main__":
    main()
