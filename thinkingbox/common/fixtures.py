# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import ast
import contextlib
import importlib
import inspect
from dataclasses import dataclass
from typing import Any, Callable

from thinkingbox.common.config_types import FixtureConfig

NON_FIXTURE_PARAMS = [
    "x",
    "judge",
]


# Type alias for a loader returning the actual factory callable on demand
Factory = Callable[..., Any]


@dataclass
class _FixtureDef:
    module_path: str
    attr_name: str
    full_path: str
    params: dict[str, Any]
    _cached_factory: Factory | None = None

    def get_factory(self) -> Factory:
        if self._cached_factory is None:
            try:
                module = importlib.import_module(self.module_path)
            except ImportError as e:
                raise ImportError(
                    f"Cannot import module part '{self.module_path}'"
                    + f" for fixture '{self.full_path}'"
                ) from e
            try:
                self._cached_factory = getattr(module, self.attr_name)
            except AttributeError as e:
                raise AttributeError(
                    f"Module '{self.module_path}' has no attribute '{self.attr_name}'"
                    + f" for fixture '{self.full_path}'"
                ) from e
        return self._cached_factory


class Fixtures:
    """
    Registry for test/session fixtures.

    Usage:
        fixtures = Fixtures()
        fixtures.add("client", "mymodule", "make_client", {"host": "localhost"})

        with fixtures_context(fixtures, ["client"]) as instances:
            client = instances["client"]

    The registered factory can be:
    - fn(...) or Class(...) resulting in the fixture object
    - fn(...) or Class(...) resulting in an object that implements the context manager protocol
    """

    def __init__(self):
        self._registry: dict[str, _FixtureDef] = {}

    def add(
        self,
        name: str,
        module_path: str,
        attr_name: str,
        parameters: dict[str, Any] | None = None,
        overwrite: bool = False,
        full_path: str | None = None,
    ) -> None:
        """
        Register a fixture lazily. The module is loaded on first use.

        Args:
            name: fixture name.
            module_path: Module path to import.
            attr_name: Attribute within the module.
            parameters: Keyword arguments for fixture instantiation.
            overwrite: Allow overwriting existing entry.
            full_path: Original dotted path.
        """
        if not overwrite and name in self._registry:
            raise ValueError(f"Fixture '{name}' already registered")

        self._registry[name] = _FixtureDef(
            module_path=module_path,
            attr_name=attr_name,
            full_path=full_path or f"{module_path}.{attr_name}",
            params=dict(parameters or {}),
        )

    def get_definition(self, name: str) -> _FixtureDef:
        try:
            return self._registry[name]
        except KeyError as e:
            raise KeyError(f"Unknown fixture '{name}'") from e

    def __contains__(self, name: str) -> bool:  # convenience
        return name in self._registry

    def __len__(self) -> int:
        return len(self._registry)


def build_fixtures(configs: dict[str, FixtureConfig]) -> Fixtures:
    """
    Construct a Fixtures registry from a dictionary of FixtureConfig objects.
    """
    fixtures = Fixtures()

    for name, cfg in configs.items():
        data = cfg.model_dump()

        # type is the dotted path to the function / context manager
        type_path: str = data.pop("type")
        if "." not in type_path:
            raise ValueError(
                f"Invalid fixture type path '{type_path}' (needs at least one dot)"
            )
        # other items are kwargs for fixture instantiation

        module_path, attr_name = type_path.rsplit(".", 1)
        fixtures.add(name, module_path, attr_name, data, full_path=type_path)

    return fixtures


def _get_injectable_params(
    factory: Factory, config_params: dict[str, Any]
) -> list[str]:
    """Return required parameter names not already covered by config_params.

    Only parameters without default values are returned. Parameters with defaults
    use their declared default (or a config value) and are never injected by the
    DI system.
    """
    try:
        sig = inspect.signature(factory)
    except (ValueError, TypeError):
        return []
    return [
        name
        for name, param in sig.parameters.items()
        if name != "self"
        and name not in config_params
        and param.kind
        not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
        and param.default is inspect.Parameter.empty
    ]


def _resolve_fixture_order(
    root_names: list[str],
    fixtures: Fixtures,
    runtime_names: frozenset[str],
    injectable_cache: dict[str, list[str]] | None = None,
) -> list[str]:
    """
    Return a topologically-sorted list of all fixture names (including transitive
    dependencies of root_names) with dependencies before dependents.

    If injectable_cache is provided, it is populated with the injectable parameter
    names for each resolved fixture, so the caller can reuse them without a second
    inspect.signature call.

    Raises ValueError on cycles or unresolvable parameters, KeyError if a fixture
    name is not registered.
    """
    order: list[str] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str, chain: list[str]) -> None:
        if name in visited:
            return
        if name in visiting:
            raise ValueError(
                f"Cycle detected in fixture dependencies: {' → '.join(chain + [name])}"
            )

        visiting.add(name)
        try:
            fixture_def = fixtures.get_definition(name)
            factory = fixture_def.get_factory()
            injectable = _get_injectable_params(factory, fixture_def.params)
            if injectable_cache is not None:
                injectable_cache[name] = injectable
            for dep in injectable:
                if dep in runtime_names:
                    continue
                if dep not in fixtures:
                    raise ValueError(
                        f"Fixture '{name}' requires parameter '{dep}', but '{dep}' is"
                        f" neither a registered fixture nor a runtime injectable"
                        f" (available runtime injectables: {sorted(runtime_names)})"
                    )
                visit(dep, chain + [name])
        finally:
            visiting.discard(name)

        visited.add(name)
        order.append(name)

    for name in root_names:
        visit(name, [])

    return order


def extract_test_fn_param_names(code: str, function_ref: str) -> list[str]:
    """
    Given Python source code as a string, find the function object `function_ref`
    and return the ordered list of its parameter names.
    This is not a generic implementation, it only handles cases that are found
    in ThinkingBox test code. Only positional and positional-or-keyword parameters
    are collected; keyword-only parameters (declared after a bare `*`) are not.

    If `function_ref` is a function, extract its parameters

    def `function_ref`(...)  # <-- this is the function to extract

    If `function_ref` is an assignment, then resolve it (one level only)

    `function_ref` = test_case_function  # find function node test_case_function

    Returns [] if the function cannot be found.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    target_func_name = None

    # Find assignment: `function_ref` = some_function
    for node in tree.body:
        if isinstance(node, ast.Assign):
            # Could be multiple targets: `function_ref` = x = target_function
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == function_ref:
                    if isinstance(node.value, ast.Name):
                        target_func_name = node.value.id
        elif isinstance(node, ast.AnnAssign):
            # Handle annotated assignment
            tgt = node.target
            if isinstance(tgt, ast.Name) and tgt.id == function_ref:
                if isinstance(node.value, ast.Name):
                    target_func_name = node.value.id
        if isinstance(node, ast.FunctionDef):
            if node.name == function_ref:
                target_func_name = function_ref

    if target_func_name is None:
        return []

    # Map function names to definition nodes
    func_defs = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            func_defs[node.name] = node

    fn_node = func_defs.get(target_func_name)
    if fn_node is None:
        return []

    args_obj = fn_node.args

    param_names: list[str] = []

    # Positional-only
    for a in args_obj.posonlyargs:
        param_names.append(a.arg)

    # Positional or keyword
    for a in args_obj.args:
        param_names.append(a.arg)

    # Preserve order but remove accidental duplicates
    seen = set()
    ordered_unique = []
    for name in param_names:
        if name not in seen:
            seen.add(name)
            ordered_unique.append(name)

    return ordered_unique


@contextlib.contextmanager
def fixtures_context(
    fixtures: Fixtures,
    names: list[str],
    runtime_context: dict[str, Any] | None = None,
    test_name: str | None = None,
):
    """
    Create fixture instances for a list of fixture names, injecting dependencies.

    Resolves the full dependency graph, instantiates in dependency order, and
    injects other fixture instances and runtime_context values as constructor
    arguments where the factory signature requires them.  Only the fixtures in
    `names` are yielded; transitive dependencies are instantiated internally.
    """
    runtime_context = runtime_context or {}
    runtime_names = frozenset(runtime_context.keys())
    test_name = test_name or "<unknown>"

    injectable_cache: dict[str, list[str]] = {}
    try:
        order = _resolve_fixture_order(names, fixtures, runtime_names, injectable_cache)
    except (ValueError, KeyError, ImportError) as e:
        raise RuntimeError(
            f"Failed to resolve fixture dependencies for test '{test_name}'"
        ) from e

    with contextlib.ExitStack() as stack:
        instantiated: dict[str, Any] = {}

        for name in order:
            fixture_def = fixtures.get_definition(name)
            factory = fixture_def.get_factory()

            kwargs: dict[str, Any] = dict(fixture_def.params)
            for dep in injectable_cache[name]:
                if dep in runtime_context:
                    kwargs[dep] = runtime_context[dep]
                elif dep in instantiated:
                    kwargs[dep] = instantiated[dep]
                else:
                    raise AssertionError(
                        f"Fixture '{name}' requires '{dep}' but it was not resolved."
                        " This is a bug in _resolve_fixture_order."
                    )

            try:
                instance = factory(**kwargs)
            except Exception as e:
                raise RuntimeError(
                    f"Error instantiating fixture '{name}' for test '{test_name}'"
                ) from e

            if hasattr(instance, "__enter__") and hasattr(instance, "__exit__"):
                instantiated[name] = stack.enter_context(instance)
            else:
                instantiated[name] = instance

        yield {name: instantiated[name] for name in names}


@contextlib.contextmanager
def fixtures_context_for_test_fn(
    fixtures: Fixtures | None,
    code: str,
    function_name: str,
    test_name: str | None = None,
    runtime_context: dict[str, Any] | None = None,
):
    """Create fixture instances for a test case function in a test script"""
    if fixtures is None:
        yield {}
        return
    names = extract_test_fn_param_names(code, function_name)
    names = [name for name in names if (name not in NON_FIXTURE_PARAMS)]
    with fixtures_context(
        fixtures, names, runtime_context=runtime_context, test_name=test_name
    ) as out:
        yield out
