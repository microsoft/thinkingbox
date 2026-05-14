# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Test case tagging and categorization types.

Enum values are loaded at import time from a YAML taxonomy file. The loader
prefers ``data/tag_taxonomy.yaml`` (gitignored; holds deployment-specific /
Microsoft-internal values) and falls back to ``data/tag_taxonomy.example.yaml``
(shipped in the public repo with fictitious values) when the former is absent.

Each YAML file has three required sections — ``domain``, ``eval``, ``category``
— each a mapping of ``UPPER_SNAKE_CASE`` identifier to its string slug. Schema
violations raise at import time so misconfigured taxonomies fail loudly.
"""

from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import BaseModel, Field, model_validator

_DATA_DIR = Path(__file__).parent / "data"
_TAXONOMY_FILE = _DATA_DIR / "tag_taxonomy.yaml"
_TAXONOMY_EXAMPLE = _DATA_DIR / "tag_taxonomy.example.yaml"

_REQUIRED_SECTIONS = ("domain", "eval", "category")


def _load_taxonomy() -> dict[str, dict[str, str]]:
    path = _TAXONOMY_FILE if _TAXONOMY_FILE.exists() else _TAXONOMY_EXAMPLE
    if not path.exists():
        raise FileNotFoundError(
            f"No tag taxonomy found. Expected {_TAXONOMY_FILE} "
            f"or {_TAXONOMY_EXAMPLE}."
        )
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(
            f"{path}: top-level must be a mapping, got {type(data).__name__}"
        )

    for section in _REQUIRED_SECTIONS:
        if section not in data:
            raise ValueError(f"{path}: missing required section '{section}'")
        members = data[section]
        if not isinstance(members, dict) or not members:
            raise ValueError(
                f"{path}: '{section}' must be a non-empty mapping of "
                f"NAME: value, got {type(members).__name__}"
            )
        for member, value in members.items():
            if (
                not isinstance(member, str)
                or not member.isidentifier()
                or not member.isupper()
            ):
                raise ValueError(
                    f"{path}: '{section}.{member}' is not a valid "
                    f"UPPER_SNAKE_CASE identifier"
                )
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"{path}: '{section}.{member}' value must be a non-empty string"
                )
    return data


_taxonomy = _load_taxonomy()


def _make_enum(name: str, members: dict[str, str]) -> type[Enum]:
    return Enum(name, members, type=str)  # type: ignore[return-value]


if TYPE_CHECKING:
    # Static stubs so type checkers see concrete Enum subclasses; the runtime
    # values are built dynamically from the loaded taxonomy below.
    class Domain(str, Enum): ...

    class Eval(str, Enum): ...

    class Category(str, Enum): ...

else:
    # Each test suite should have a single domain:* tag
    Domain = _make_enum("Domain", _taxonomy["domain"])

    # Each test suite may have at most one eval:* tag, encoding a 2-level hierarchy:
    #   eval:<top-level>:<sub-level>
    #
    # The top level drives the primary grouping in the Leaderboard ('Objective' selector).
    # The sub-level drives the secondary grouping within that objective.
    # Tags are mutually exclusive — at most one per scenario. Scenarios without an
    # eval:* tag are valid but will not appear in the Leaderboard.
    Eval = _make_enum("Eval", _taxonomy["eval"])

    # Category tags are multivalued, and a test suite might have several
    Category = _make_enum("Category", _taxonomy["category"])


class TestCaseTags(BaseModel):
    """
    Represents the set of tags associated with a test case.
    Fields:
        labels (list[str]): Arbitrary string labels for the test case.
        category (list[Category]): List of categories (e.g., 'user', 'xpia') assigned to the test case.
        domain (Domain | None): The domain to which the test case belongs (e.g., 'customer-service', 'erp').
        eval_type (Eval | None): The evaluation type as a compound tag (e.g., 'orchestration:tool-selection', 'knowledge-qa:tabular'). Mutually exclusive.
        skip (bool): If True, the test case should be skipped.
    Tag merging during hydration:
        When hydrating test cases, scenario tags act as defaults. If a test case specifies its own tags,
        those tags override the scenario-level tags. This allows for flexible inheritance and overriding
        of tag values between scenarios and individual test cases.
    """

    labels: list[str] = Field(default_factory=list)
    category: list[Category] = Field(default_factory=list)
    domain: Domain | None = None
    eval_type: Eval | None = None
    skip: bool = False

    @model_validator(mode="before")
    @classmethod
    def distribute_tags(cls, flat_tags):
        """
        Parse and distribute flat tag representations into structured fields for the TestCaseTags model.
        This validator processes a list of tags, where each tag is a string.
        Supported input formats:
          - Strings, which may be:
              * Simple labels (e.g., "foo")
              * Special prefixes:
                  - "domain:<value>" sets the 'domain' field (case-insensitive)
                  - "eval:<value>" sets the 'eval_type' field (case-insensitive)
                  - "category:<value>" adds to the 'category' list (case-insensitive)
                  - "skip" sets the 'skip' field to True (case-insensitive)
        All category, domain, and eval_type values are normalized to lowercase.
        Returns:
            dict: A dictionary with keys:
                - "category": list of category strings (lowercase)
                - "labels": list of label strings
                - "domain": domain string (lowercase) or None
                - "eval_type": eval string (lowercase) or None
                - "skip": boolean
        """
        if not isinstance(flat_tags, list):
            return flat_tags

        data: dict[str, Any] = {"category": [], "labels": []}
        for tag in flat_tags:
            if isinstance(tag, str):
                tag_lower = tag.lower()
                if tag_lower.startswith("domain:"):
                    data["domain"] = tag[7:].lower()  # len("domain:") == 7
                elif tag_lower.startswith("eval:"):
                    data["eval_type"] = tag[5:].lower()  # len("eval:") == 5
                elif tag_lower.startswith("category:"):
                    # Single category like category:user
                    data["category"].append(tag[9:].lower())  # len("category:") == 9
                elif tag_lower == "skip":
                    data["skip"] = True
                else:
                    data["labels"].append(tag)
            else:
                raise ValueError(
                    f"Invalid tag type: {type(tag).__name__}. "
                    f"Tags must be string but got {tag!r}"
                )

        return data
