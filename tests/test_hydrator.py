# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import textwrap
from pathlib import Path

import pytest

from thinkingbox.common.chat_types import Text
from thinkingbox.common.config_types import HydratedTestCase
from thinkingbox.common.hydrator import (
    Dataset,
    iter_cases_by_names,
    iter_cases_from_file_or_folder,
)


def get_agent():
    return """
system_instructions: You like apples
builtin_tools:
- name: InjectionAttackInToolResponse
  is_end_turn: true
  description: this is the most powerful tool
  input_schema:
    type: object
    properties:
      reason:
        description: guess it
        type: string
    required:
    - reason
"""


def get_scenario_data(
    bot_instructions: str | None = None, tags: list[str] | None = None
):
    base = """
world_state:
  test_server: {{}}

tools:
- name: test_tool
bot_instructions: {bot_instructions}
""".format(
        bot_instructions=json.dumps(bot_instructions)
    )

    if tags:
        # Convert list to YAML list format
        tags_yaml = "tags: " + json.dumps(tags)
        base += "\n" + tags_yaml

    return base


def get_test_data():
    return '''
from thinkingbox.common import Judge, TestContext

"""!
scenario: myscenario
"""

def test_1(x: TestContext, judge: Judge):
    """!
    query: An apple a day...
    bot_instructions: "There are some bot instructions"
    """
    assert judge.text_yesno(x.response, "say yes")

def test_2(x: TestContext, judge: Judge):
    """!
    query: Two apples a day...
    user_context: "There is some user context"
    """
    assert judge.text_yesno(x.response, "say no")
'''


def _create_file(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def test_hydrate_directory(tmp_path):
    _create_file(
        tmp_path / "agent" / "myagent.yaml",
        get_agent(),
    )
    _create_file(
        tmp_path / "scenario" / "myscenario.yaml",
        get_scenario_data(bot_instructions="Scenario instructions"),
    )
    _create_file(
        tmp_path / "test_case" / "mytest.py",
        get_test_data(),
    )
    cases = list(
        iter_cases_from_file_or_folder(
            path=tmp_path / "test_case",
            base_dir=tmp_path,
            agent="myagent",
        )
    )
    # Do some basic checks to verify that the information is parsed
    # and interpreted correctly
    assert len(cases) == 2
    for tc in cases:
        assert isinstance(tc, HydratedTestCase)
        assert tc.agent.system_instructions == "You like apples"
        assert len(tc.agent.builtin_tools) == 1
        assert len(tc.scenario.tools) == 1
        assert "test_server" in tc.scenario.world_state
        assert tc.scenario.bot_instructions == "Scenario instructions"
    assert cases[0].query == "An apple a day..."
    assert cases[0].user_context is None
    assert cases[0].bot_instructions == "There are some bot instructions"
    assert "def test_1(" in cases[0].test_code
    assert "say yes" in cases[0].test_code
    assert cases[1].query == "Two apples a day..."
    assert cases[1].user_context == "There is some user context"
    assert cases[1].bot_instructions is None
    assert "def test_2(" in cases[1].test_code
    assert "say no" in cases[1].test_code


def test_hydrate_test_case_file(tmp_path):
    _create_file(
        tmp_path / "agent" / "myagent.yaml",
        get_agent(),
    )
    _create_file(
        tmp_path / "scenario" / "myscenario.yaml",
        get_scenario_data(bot_instructions="Scenario instructions"),
    )
    _create_file(
        tmp_path / "test_case" / "mytest.py",
        get_test_data(),
    )
    cases = list(
        iter_cases_from_file_or_folder(
            path=tmp_path / "test_case" / "mytest.py",
            base_dir=tmp_path,
            agent="myagent",
        )
    )
    assert len(cases) == 2
    for tc in cases:
        assert isinstance(tc, HydratedTestCase)


def test_hydrate_with_tags_none(tmp_path):
    """Test that hydration works when tags are None in both scenario and test case."""
    _create_file(
        tmp_path / "agent" / "myagent.yaml",
        get_agent(),
    )
    _create_file(
        tmp_path / "scenario" / "myscenario.yaml",
        get_scenario_data(),
    )
    _create_file(
        tmp_path / "test_case" / "mytest.py",
        get_test_data(),
    )
    cases = list(
        iter_cases_from_file_or_folder(
            path=tmp_path / "test_case" / "mytest.py",
            base_dir=tmp_path,
            agent="myagent",
        )
    )
    assert len(cases) == 2
    # Verify tags are created with default values when None
    for tc in cases:
        assert tc.tags.labels == []
        assert tc.tags.domain is None
        assert tc.tags.category == []
        assert tc.tags.skip is False


def test_hydrate_with_scenario_tags_only(tmp_path):
    """Test that scenario tags are used when test case has no tags."""
    _create_file(
        tmp_path / "agent" / "myagent.yaml",
        get_agent(),
    )
    _create_file(
        tmp_path / "scenario" / "myscenario.yaml",
        get_scenario_data(tags=["domain:customer-service", "scenario-label"]),
    )
    _create_file(
        tmp_path / "test_case" / "mytest.py",
        get_test_data(),
    )
    cases = list(
        iter_cases_from_file_or_folder(
            path=tmp_path / "test_case" / "mytest.py",
            base_dir=tmp_path,
            agent="myagent",
        )
    )
    assert len(cases) == 2
    # Verify scenario tags are inherited
    for tc in cases:
        assert "scenario-label" in tc.tags.labels
        assert tc.tags.domain == "customer-service"
        assert tc.tags.skip is False


def test_hydrate_with_test_case_tags_only(tmp_path):
    """Test that test case tags are used when scenario has no tags."""
    _create_file(
        tmp_path / "agent" / "myagent.yaml",
        get_agent(),
    )
    _create_file(
        tmp_path / "scenario" / "myscenario.yaml",
        get_scenario_data(),
    )
    test_with_tags = '''
from thinkingbox.common import Judge, TestContext

"""!
scenario: myscenario
"""

def test_1(x: TestContext, judge: Judge):
    """!
    query: An apple a day...
    tags: [domain:customer-service, test-label, category:user]
    """
    assert judge.text_yesno(x.response, "say yes")
'''
    _create_file(
        tmp_path / "test_case" / "mytest.py",
        test_with_tags,
    )
    cases = list(
        iter_cases_from_file_or_folder(
            path=tmp_path / "test_case" / "mytest.py",
            base_dir=tmp_path,
            agent="myagent",
        )
    )
    assert len(cases) == 1
    # Verify test case tags are used
    tc = cases[0]
    assert "test-label" in tc.tags.labels
    assert tc.tags.domain == "customer-service"
    from thinkingbox.common.tag_types import Category

    assert Category.USER in tc.tags.category
    assert tc.tags.skip is False


def test_hydrate_with_merged_tags(tmp_path):
    """Test that tags are properly merged when both scenario and test case have tags."""
    _create_file(
        tmp_path / "agent" / "myagent.yaml",
        get_agent(),
    )
    _create_file(
        tmp_path / "scenario" / "myscenario.yaml",
        get_scenario_data(tags=["domain:customer-service", "scenario-label"]),
    )
    test_with_tags = '''
from thinkingbox.common import Judge, TestContext

"""!
scenario: myscenario
"""

def test_1(x: TestContext, judge: Judge):
    """!
    query: An apple a day...
    tags: [test-label, category:xpia, category:user]
    """
    assert judge.text_yesno(x.response, "say yes")
'''
    _create_file(
        tmp_path / "test_case" / "mytest.py",
        test_with_tags,
    )
    cases = list(
        iter_cases_from_file_or_folder(
            path=tmp_path / "test_case" / "mytest.py",
            base_dir=tmp_path,
            agent="myagent",
        )
    )
    assert len(cases) == 1
    tc = cases[0]
    # Labels should be merged (both scenario and test labels)
    assert "scenario-label" in tc.tags.labels
    assert "test-label" in tc.tags.labels
    # Domain should come from scenario (test case doesn't override it)
    assert tc.tags.domain == "customer-service"
    # Categories should come from test case
    from thinkingbox.common.tag_types import Category

    assert Category.XPIA in tc.tags.category
    assert Category.USER in tc.tags.category
    assert tc.tags.skip is False


def test_hydrate_skips_tests_with_skip_true(tmp_path):
    """Test that test cases with skip=True are automatically filtered out."""
    _create_file(
        tmp_path / "agent" / "myagent.yaml",
        get_agent(),
    )
    _create_file(
        tmp_path / "scenario" / "myscenario.yaml",
        get_scenario_data(),
    )

    # Create test file with mix of skipped and non-skipped tests
    test_with_skip = '''
from thinkingbox.common import Judge, TestContext

"""!
scenario: myscenario
"""

def test_should_run_1(x: TestContext, judge: Judge):
    """!
    query: This test should run
    """
    assert judge.text_yesno(x.response, "say yes")

def test_should_be_skipped(x: TestContext, judge: Judge):
    """!
    query: This test should be skipped
    tags: [domain:customer-service, skip, experimental, wip]
    """
    assert judge.text_yesno(x.response, "say yes")

def test_should_run_2(x: TestContext, judge: Judge):
    """!
    query: This test should also run
    tags: [domain:customer-service, baseline]
    """
    assert judge.text_yesno(x.response, "say yes")

def test_another_skipped(x: TestContext, judge: Judge):
    """!
    query: Another skipped test
    tags: [skip, category:xpia]
    """
    assert judge.text_yesno(x.response, "say yes")
'''
    _create_file(
        tmp_path / "test_case" / "mytest.py",
        test_with_skip,
    )

    # Load test cases
    cases = list(
        iter_cases_from_file_or_folder(
            path=tmp_path / "test_case" / "mytest.py",
            base_dir=tmp_path,
            agent="myagent",
        )
    )

    # Should only get 2 test cases (the ones with skip=True are filtered out)
    assert len(cases) == 2

    # Verify that all returned tests have skip=False
    assert all(tc.tags.skip is False for tc in cases)

    # Verify we got the right tests
    test_names = [tc.metadata["test_case_name"] for tc in cases]
    assert "test_should_run_1" in test_names
    assert "test_should_run_2" in test_names
    assert "test_should_be_skipped" not in test_names
    assert "test_another_skipped" not in test_names


def test_hydrate_with_case_insensitive_enums(tmp_path):
    """Test that domain and category parsing is case-insensitive."""
    _create_file(
        tmp_path / "agent" / "myagent.yaml",
        get_agent(),
    )
    _create_file(
        tmp_path / "scenario" / "myscenario.yaml",
        get_scenario_data(tags=["domain:Customer-Service", "scenario-label"]),
    )
    test_with_mixed_case = '''
from thinkingbox.common import Judge, TestContext

"""!
scenario: myscenario
"""

def test_1(x: TestContext, judge: Judge):
    """!
    query: An apple a day...
    tags: [category:User, category:XPIA, category:Direct-Msg]
    """
    assert judge.text_yesno(x.response, "say yes")
'''
    _create_file(
        tmp_path / "test_case" / "mytest.py",
        test_with_mixed_case,
    )
    cases = list(
        iter_cases_from_file_or_folder(
            path=tmp_path / "test_case" / "mytest.py",
            base_dir=tmp_path,
            agent="myagent",
        )
    )

    assert len(cases) == 1
    tc = cases[0]

    # Verify domain is normalized (from scenario)
    from thinkingbox.common.tag_types import Category, Domain

    assert tc.tags.domain == Domain.CUSTOMER_SERVICE

    # Verify categories are normalized
    assert len(tc.tags.category) == 3
    assert Category.USER in tc.tags.category
    assert Category.XPIA in tc.tags.category
    assert Category.DIRECT_MSG in tc.tags.category


def test_hydrate_with_test_case_scenario_overwrite(tmp_path):
    """Test that domain and category parsing is case-insensitive."""
    _create_file(
        tmp_path / "agent" / "myagent.yaml",
        get_agent(),
    )
    _create_file(
        tmp_path / "scenario" / "myscenario.yaml",
        get_scenario_data(tags=["domain:Customer-Service", "scenario-label"]),
    )
    test_with_mixed_case = '''
from thinkingbox.common import Judge, TestContext

"""!
scenario: myscenario
tags: [domain:hr]
"""

def test_1(x: TestContext, judge: Judge):
    """!
    query: An apple a day...
    tags: [category:User, category:XPIA, category:Direct-Msg]
    """
    assert judge.text_yesno(x.response, "say yes")
'''
    _create_file(
        tmp_path / "test_case" / "mytest.py",
        test_with_mixed_case,
    )
    cases = list(
        iter_cases_from_file_or_folder(
            path=tmp_path / "test_case" / "mytest.py",
            base_dir=tmp_path,
            agent="myagent",
        )
    )

    assert len(cases) == 1
    tc = cases[0]

    # Verify domain is normalized (from scenario)
    from thinkingbox.common.tag_types import Category, Domain

    assert tc.tags.domain == Domain.HUMAN_RESOURCES

    # Verify categories are normalized
    assert len(tc.tags.category) == 3
    assert Category.USER in tc.tags.category
    assert Category.XPIA in tc.tags.category
    assert Category.DIRECT_MSG in tc.tags.category


def test_hydrate_with_test_case_overwrite(tmp_path):
    """Test that domain and category parsing is case-insensitive."""
    _create_file(
        tmp_path / "agent" / "myagent.yaml",
        get_agent(),
    )
    _create_file(
        tmp_path / "scenario" / "myscenario.yaml",
        get_scenario_data(tags=["domain:Customer-Service", "scenario-label"]),
    )
    test_with_mixed_case = '''
from thinkingbox.common import Judge, TestContext

"""!
scenario: myscenario
tags: [domain:hr]
"""

def test_1(x: TestContext, judge: Judge):
    """!
    query: An apple a day...
    tags: [category:User, category:XPIA, category:Direct-Msg, domain:misc]
    """
    assert judge.text_yesno(x.response, "say yes")
'''
    _create_file(
        tmp_path / "test_case" / "mytest.py",
        test_with_mixed_case,
    )
    cases = list(
        iter_cases_from_file_or_folder(
            path=tmp_path / "test_case" / "mytest.py",
            base_dir=tmp_path,
            agent="myagent",
        )
    )

    assert len(cases) == 1
    tc = cases[0]

    # Verify domain is normalized (from scenario)
    from thinkingbox.common.tag_types import Category, Domain

    assert tc.tags.domain == Domain.MISC

    # Verify categories are normalized
    assert len(tc.tags.category) == 3
    assert Category.USER in tc.tags.category
    assert Category.XPIA in tc.tags.category
    assert Category.DIRECT_MSG in tc.tags.category


def _make_base(tmp_path):
    """Write minimal agent and scenario files needed by all meta.yaml tests."""
    _create_file(tmp_path / "agent" / "myagent.yaml", get_agent())
    _create_file(tmp_path / "scenario" / "myscenario.yaml", get_scenario_data())


def test_hydrate_history_ref_resolved_to_messages(tmp_path):
    """A history: HistoryRef in the test docstring is resolved to list[MessageT] in HydratedTestCase."""
    _make_base(tmp_path)
    _create_file(
        tmp_path / "test_case" / "mytest.py",
        textwrap.dedent(
            '''\
            from thinkingbox.common import Judge, TestContext

            """!
            scenario: myscenario
            """

            def test_1(x: TestContext, judge: Judge):
                """!
                query: Follow-up
                history: "mygroup:0:"
                """
                pass
        '''
        ),
    )
    _create_file(
        tmp_path / "test_case" / "mytest.meta.yaml",
        textwrap.dedent(
            """\
            $history:
              mygroup:
                - T: Text
                  message_id: t1
                  role: user
                  content: Prior question
                - T: Text
                  role: assistant
                  content: Prior answer
        """
        ),
    )

    cases = list(
        iter_cases_from_file_or_folder(
            path=tmp_path / "test_case" / "mytest.py",
            base_dir=tmp_path,
            agent="myagent",
        )
    )
    assert len(cases) == 1
    tc = cases[0]
    assert tc.history is not None
    assert len(tc.history) == 2
    assert isinstance(tc.history[0], Text)
    assert tc.history[0].content == "Prior question"


def test_hydrate_missing_history_key_raises(tmp_path):
    """Raise ValueError during hydration when the history: key is absent from $history."""

    _make_base(tmp_path)
    _create_file(
        tmp_path / "test_case" / "mytest.py",
        textwrap.dedent(
            '''\
            from thinkingbox.common import Judge, TestContext

            """!
            scenario: myscenario
            """

            def test_1(x: TestContext, judge: Judge):
                """!
                query: Follow-up
                history: "nokey:0:"
                """
                pass
        '''
        ),
    )
    _create_file(
        tmp_path / "test_case" / "mytest.meta.yaml",
        textwrap.dedent(
            """\
            $history:
              othergroup:
                - T: Text
                  role: user
                  content: hi
        """
        ),
    )

    with pytest.raises(ValueError, match="Failed hydrating"):
        list(
            iter_cases_from_file_or_folder(
                path=tmp_path / "test_case" / "mytest.py",
                base_dir=tmp_path,
                agent="myagent",
            )
        )


def test_hydrate_no_meta_file_history_none(tmp_path):
    """Without a .meta.yaml, history is None and no external metadata is added."""
    _make_base(tmp_path)
    _create_file(tmp_path / "test_case" / "mytest.py", get_test_data())

    cases = list(
        iter_cases_from_file_or_folder(
            path=tmp_path / "test_case" / "mytest.py",
            base_dir=tmp_path,
            agent="myagent",
        )
    )
    for tc in cases:
        assert tc.history is None
        assert "scenario" in tc.metadata


def test_hydrate_history_declared_but_no_meta_file_raises(tmp_path):
    """Raise ValueError when a test declares history: but no .meta.yaml file exists."""

    _make_base(tmp_path)
    _create_file(
        tmp_path / "test_case" / "mytest.py",
        textwrap.dedent(
            '''\
            from thinkingbox.common import Judge, TestContext

            """!
            scenario: myscenario
            """

            def test_1(x: TestContext, judge: Judge):
                """!
                query: Follow-up
                history: "mygroup:0:"
                """
                pass
        '''
        ),
    )
    # No .meta.yaml written

    with pytest.raises(ValueError, match="Failed hydrating"):
        list(
            iter_cases_from_file_or_folder(
                path=tmp_path / "test_case" / "mytest.py",
                base_dir=tmp_path,
                agent="myagent",
            )
        )


def test_folder_scan_ignores_meta_yaml_files(tmp_path):
    """Folder discovery must not attempt to load .meta.yaml files as test case files."""
    _create_file(tmp_path / "agent" / "myagent.yaml", get_agent())
    _create_file(tmp_path / "scenario" / "myscenario.yaml", get_scenario_data())
    _create_file(tmp_path / "test_case" / "mytest.py", get_test_data())
    _create_file(
        tmp_path / "test_case" / "mytest.meta.yaml",
        textwrap.dedent(
            """\
            $history:
              mygroup:
                - T: Text
                  role: user
                  content: hi
        """
        ),
    )

    # Should load 2 test cases from mytest.py and not crash on mytest.meta.yaml
    cases = list(
        iter_cases_from_file_or_folder(
            path=tmp_path / "test_case",
            base_dir=tmp_path,
            agent="myagent",
        )
    )
    assert len(cases) == 2


def test_only_names_preserves_test_code_isolation(tmp_path):
    """When loading a subset of test cases via only_names, the selected test's
    test_code must not contain function definitions from unselected tests."""
    _create_file(tmp_path / "agent" / "myagent.yaml", get_agent())
    _create_file(tmp_path / "scenario" / "myscenario.yaml", get_scenario_data())
    _create_file(tmp_path / "test_case" / "mytest.py", get_test_data())

    # Load only test_1 via iter_cases_by_names
    cases = list(
        iter_cases_by_names(
            ["mytest.py:test_1"],
            base_dir=tmp_path,
            agent="myagent",
        )
    )
    assert len(cases) == 1
    tc = cases[0]
    assert tc.metadata["test_case_name"] == "test_1"
    # test_1's own definition must be present
    assert "def test_1(" in tc.test_code
    # test_2's definition must NOT leak into test_1's test_code
    assert "def test_2(" not in tc.test_code
    assert "Two apples a day" not in tc.test_code


def test_get_testfile_path_finds_file_in_prefix_subdirectory(tmp_path):
    _create_file(tmp_path / "test_case" / "my" / "my_test.py", "")
    ds = Dataset(tmp_path)
    assert (
        ds.get_testfile_path("my_test.py")
        == tmp_path / "test_case" / "my" / "my_test.py"
    )


def test_get_testfile_path_direct_file_takes_precedence(tmp_path):
    _create_file(tmp_path / "test_case" / "my_test.py", "direct")
    _create_file(tmp_path / "test_case" / "my" / "my_test.py", "subdir")
    ds = Dataset(tmp_path)
    assert ds.get_testfile_path("my_test.py") == tmp_path / "test_case" / "my_test.py"


def test_get_testfile_path_does_not_find_filename_as_directory(tmp_path):
    _create_file(tmp_path / "test_case" / "my_test.py" / "my_test.py", "")
    ds = Dataset(tmp_path)
    with pytest.raises(FileNotFoundError):
        ds.get_testfile_path("my_test.py")
