# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import itertools
import os
from pathlib import Path
from typing import Any, Iterator, Literal, Sequence

import yaml

from thinkingbox.common.config_types import (
    AgentConfig,
    ConftestConfig,
    FixtureConfig,
    HydratedTestCase,
    ScenarioConfig,
    TestCase,
    TestCaseFile,
)
from thinkingbox.common.history_loader import HistoryLoader, MetaFile, parse_messages
from thinkingbox.common.python_test_file import (
    iter_test_functions as py_file_iter_test_functions,
)
from thinkingbox.common.tag_types import TestCaseTags
from thinkingbox.common.utils import iter_validate_jsonl


def load_conftest_fixtures(
    test_file: Path,
    test_case_root: Path,
) -> dict[str, FixtureConfig]:
    """Load and merge fixtures from conftest.yaml files found along the path
    from test_case_root down to test_file.parent (inclusive). Later files override earlier.
    """
    try:
        rel = test_file.parent.relative_to(test_case_root)
    except ValueError:
        return {}
    dirs = [test_case_root]
    for part in rel.parts:
        dirs.append(dirs[-1] / part)
    merged: dict[str, FixtureConfig] = {}

    # Iterate down the directory structure, reading and applying any "conftest.yaml" files, and reporting
    # any parse or validation errors.
    for d in dirs:
        cfg_file = d / "conftest.yaml"
        if cfg_file.is_file():
            try:
                with open(cfg_file, "r", encoding="utf-8") as f:
                    raw = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(
                    f"Error parsing YAML in conftest file {cfg_file}"
                ) from e
            if raw is None:
                continue
            if not isinstance(raw, dict):
                raise ValueError(
                    f"Top-level YAML in conftest file {cfg_file} must be a mapping, "
                    f"got {type(raw).__name__}"
                )
            try:
                cfg = ConftestConfig(**raw)
            except Exception as e:
                raise ValueError(
                    f"Error loading conftest config from {cfg_file}"
                ) from e
            merged.update(cfg.fixtures)
    return merged


def _get_agent_config(base_dir: Path, name: str) -> AgentConfig:
    path = base_dir / "agent" / f"{name}.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return AgentConfig(**yaml.safe_load(f))


class Dataset:
    def __init__(self, base_dir: str | Path, cache_scenarios=True):
        self.base_dir = Path(base_dir)
        self.should_cache_scenarios = cache_scenarios
        self.scenarios = {}

    def get_agent_config(self, name: str) -> AgentConfig:
        return _get_agent_config(self.base_dir, name)

    def get_scenario(self, name: str) -> ScenarioConfig:
        out = self.scenarios.get(name)
        if out is None:
            path = self.base_dir / "scenario" / f"{name}.yaml"
            with open(path, "r", encoding="utf-8") as f:
                out = ScenarioConfig(**yaml.safe_load(f))
            if self.should_cache_scenarios:
                self.scenarios[name] = out
        return out.model_copy(deep=True)

    def get_testfile_path(self, filename: str) -> Path:
        test_cases_dir = self.base_dir / "test_case"
        path = test_cases_dir / filename
        if path.is_file():
            return path
        prefixes = filename.split("_")
        for k in range(len(prefixes) - 1, 0, -1):
            path = test_cases_dir / "_".join(prefixes[:k]) / filename
            if path.is_file():
                return path
        raise FileNotFoundError(filename)

    def hydrate_test_case(
        self,
        tc: TestCase,
        agent_config: AgentConfig,
        meta_file: MetaFile | None = None,
        conftest_fixtures: dict[str, FixtureConfig] | None = None,
    ) -> HydratedTestCase:
        # read scenario file
        try:
            scenario = self.get_scenario(tc.scenario)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Failed loading scenario '{tc.scenario}': {e}") from e

        # Merge labels (concatenate lists)
        scenario_labels = scenario.tags.labels if scenario.tags else []
        tc_labels = tc.tags.labels if tc.tags else []
        merged_labels = scenario_labels + tc_labels

        # Domain comes from scenario, test case can override
        domain = None
        if tc.tags and tc.tags.domain:
            domain = tc.tags.domain
        elif scenario.tags and scenario.tags.domain:
            domain = scenario.tags.domain

        # Category always comes from test case
        category = tc.tags.category if tc.tags else []

        # Skip if either scenario or test case has skip=True
        skip = bool(
            (scenario.tags and scenario.tags.skip) or (tc.tags and tc.tags.skip)
        )

        tags = TestCaseTags(
            labels=merged_labels,
            domain=domain,
            category=category,
            skip=skip,
        )

        merged_metadata = {
            "scenario": tc.scenario,
            **tc.metadata,
        }

        # Resolve history messages (if any)
        history = None
        if tc.history is not None:
            loader = HistoryLoader(meta_file)
            try:
                raw_msgs = loader.resolve_raw(tc.history)
                history = parse_messages(raw_msgs)
            except (KeyError, ValueError) as e:
                raise ValueError(
                    f"Failed resolving history for test case {tc.uid!r}: {e}"
                ) from e

        merged_fixtures = {**(conftest_fixtures or {}), **scenario.fixtures}

        hydrated_tc = HydratedTestCase(
            uid=tc.uid,
            agent=agent_config,
            scenario=scenario,
            query=tc.query,
            test_code=tc.test_code,
            bot_instructions=tc.bot_instructions,
            user_context=tc.user_context,
            max_user_sim_turns=(
                tc.max_user_sim_turns if tc.max_user_sim_turns is not None else 10
            ),
            max_agent_sim_turns=tc.max_agent_sim_turns,
            init=tc.init,
            metadata=merged_metadata,
            tags=tags,
            history=history,
            fixtures=merged_fixtures,
        )
        return hydrated_tc


def load_test_file(
    path: str | Path,
    file_format: Literal["auto", "py", "yaml"] = "auto",
    check_config: bool = False,
    only_names: set[str] | None = None,
) -> TestCaseFile:
    path = Path(path).expanduser().resolve()
    if file_format == "auto":
        file_format = "py" if path.suffix == ".py" else "yaml"

    if file_format == "py":
        code = path.read_text(encoding="utf-8")
        tc_file = TestCaseFile(test_cases=[])
        for fn in py_file_iter_test_functions(
            code, check_config=check_config, only_names=only_names
        ):
            tc = TestCase(
                uid=fn.name,
                test_code=fn.test_code,
                **fn.config,
            )
            tc_file.test_cases.append(tc)
    else:
        with open(path, "r", encoding="utf-8") as f:
            try:
                tc_file = TestCaseFile(**yaml.safe_load(f))
            except Exception as e:
                raise ValueError(f"Failed to load test case file {path}: {e}") from e
        if only_names is not None:
            tc_file.test_cases = [
                tc for tc in tc_file.test_cases if tc.uid in only_names
            ]
    return tc_file


def _iter_cases_from_file(
    test_file: str | Path,
    dataset: Dataset,
    agent_config: AgentConfig,
    file_format: Literal["auto", "py", "yaml"] = "auto",
    metadata: dict[str, Any] = None,
    skip_filtered: bool = True,
    only_names: set[str] | None = None,
) -> Iterator[HydratedTestCase]:
    test_file = Path(test_file)
    tc_file = load_test_file(
        test_file,
        file_format=file_format,
        check_config=True,
        only_names=only_names,
    )
    # Load .meta.yaml once per test file, shared across all test cases in the file
    meta_file = MetaFile.load_for_test_file(test_file)
    # Load corpus fixtures once per test file, from test_case root down to test_file directory
    test_case_root = dataset.base_dir / "test_case"
    conftest_fixtures = load_conftest_fixtures(test_file, test_case_root)

    for i, tc in enumerate(tc_file.test_cases):

        if skip_filtered and tc.tags and tc.tags.skip:
            continue

        if metadata is not None:
            tc.metadata.update(metadata)
        tc.metadata["test_case_file"] = str(test_file)
        tc.metadata["test_case_name"] = tc.uid
        tc.uid = test_file.name + ":" + tc.uid
        try:
            hydrated_tc = dataset.hydrate_test_case(
                tc,
                agent_config=agent_config,
                meta_file=meta_file,
                conftest_fixtures=conftest_fixtures,
            )
            yield hydrated_tc
        except (FileNotFoundError, ValueError) as e:
            raise ValueError(
                f"Failed hydrating test case {test_file!s}:{tc.uid} (#{i})"
            ) from e


def iter_cases_from_jsonl(
    path: str | Path,
):
    with open(path, "r", encoding="utf-8") as f:
        yield from iter_validate_jsonl(f, model=HydratedTestCase)


def iter_cases_from_test_case_file(
    test_file: str | Path,
    base_dir: str | Path,
    agent: str,
    file_format: Literal["auto", "py", "yaml"] = "auto",
    metadata: dict[str, Any] = None,
    skip_filtered: bool = True,
) -> Iterator[HydratedTestCase]:
    base_dir = Path(base_dir).expanduser().resolve()
    dataset = Dataset(base_dir)
    agent_config = dataset.get_agent_config(agent)
    metadata = metadata.copy() if metadata is not None else {}
    metadata["agent"] = agent
    yield from _iter_cases_from_file(
        test_file=test_file,
        dataset=dataset,
        agent_config=agent_config,
        file_format=file_format,
        metadata=metadata,
        skip_filtered=skip_filtered,
    )


def _parse_test_name(name: str) -> tuple[str, str]:
    try:
        filename, testname = name.split(":")
    except ValueError:
        raise ValueError("Invalid test name, must be 'filename:testname'")
    return filename, testname


def iter_cases_by_names(
    names: Sequence[str],
    base_dir: str | Path,
    agent: str,
    file_format: Literal["auto", "py", "yaml"] = "auto",
    metadata: dict[str, Any] = None,
    strict: bool = True,
) -> Iterator[HydratedTestCase]:
    parsed = [_parse_test_name(n) for n in names]
    base_dir = Path(base_dir).expanduser().resolve()
    dataset = Dataset(base_dir)
    agent_config = dataset.get_agent_config(agent)
    metadata = metadata.copy() if metadata is not None else {}
    metadata["agent"] = agent

    for filename, group in itertools.groupby(parsed, key=lambda p: p[0]):
        testnames = [testname for _, testname in group]
        try:
            test_file = dataset.get_testfile_path(filename)
        except FileNotFoundError:
            if strict:
                raise
            continue
        only_names = set(testnames)
        # index hydrated cases by test_case_name for ordered lookup
        hydrated: dict[str, HydratedTestCase] = {}
        for tc in _iter_cases_from_file(
            test_file=test_file,
            dataset=dataset,
            agent_config=agent_config,
            file_format=file_format,
            metadata=metadata.copy(),
            only_names=only_names,
        ):
            hydrated[tc.metadata["test_case_name"]] = tc
        for testname in testnames:
            tc = hydrated.get(testname)
            if tc is None:
                if strict:
                    raise FileNotFoundError(f"{filename}:{testname}")
                continue
            yield tc.model_copy(deep=True)


def get_dataset_case_by_name(
    name: str,
    base_dir: str | Path,
    agent: str,
    file_format: Literal["auto", "py", "yaml"] = "auto",
    metadata: dict[str, Any] = None,
) -> HydratedTestCase:
    try:
        return next(
            iter_cases_by_names(
                [name],
                base_dir=base_dir,
                agent=agent,
                file_format=file_format,
                metadata=metadata,
            )
        )
    except (FileNotFoundError, StopIteration) as e:
        raise FileNotFoundError(f"Test case {name!r} not found in {base_dir}") from e


def iter_cases_from_folder(
    test_case_dir: str | Path,
    base_dir: str | Path,
    agent: str,
    metadata: dict[str, Any] = None,
    skip_filtered: bool = True,
) -> Iterator[HydratedTestCase]:
    test_case_dir = Path(test_case_dir).expanduser().resolve()
    base_dir = Path(base_dir).expanduser().resolve()
    dataset = Dataset(base_dir)
    agent_config = dataset.get_agent_config(agent)
    metadata = metadata.copy() if metadata is not None else {}
    metadata["agent"] = agent

    for root, _, files in os.walk(test_case_dir):
        for f in files:
            if f.endswith(".py") or (
                f.endswith(".yaml")
                and not f.endswith(".meta.yaml")
                and f != "conftest.yaml"
            ):
                yield from _iter_cases_from_file(
                    test_file=Path(root) / f,
                    dataset=dataset,
                    agent_config=agent_config,
                    metadata=metadata,
                    skip_filtered=skip_filtered,
                )


def iter_cases_from_file_or_folder(
    path: str | Path,
    base_dir: str | Path | None = None,
    agent: str | None = None,
    skip_filtered: bool = True,
) -> Iterator[HydratedTestCase]:
    path = Path(path).expanduser().resolve()
    is_file = path.is_file()

    if not is_file and not path.is_dir():
        raise FileNotFoundError(str(path))

    if is_file and path.name.endswith(".jsonl"):
        if base_dir is not None or agent is not None:
            raise ValueError(
                "base_dir and agent must be null when loading a JSONL file, the information is already in the file."
                + f" found agent={agent}, base_dir={base_dir!s}"
            )
        yield from iter_cases_from_jsonl(path)
        return

    if base_dir is None or agent is None:
        raise ValueError(
            "base_dir and agent cannot be null when loading from test case files."
            + f" found agent={agent}, base_dir={base_dir!s}"
        )

    if is_file:
        yield from iter_cases_from_test_case_file(
            path,
            base_dir=base_dir,
            agent=agent,
            skip_filtered=skip_filtered,
        )
        return

    # it is a directory
    yield from iter_cases_from_folder(
        path,
        base_dir=base_dir,
        agent=agent,
        skip_filtered=skip_filtered,
    )
