"""Regression tests for langchain_classic root import compatibility."""

from __future__ import annotations

import warnings

import pytest
from langchain_core.prompts import PromptTemplate

import langchain_classic


def test_supported_root_import_warns_and_returns_symbol() -> None:
    """Supported compatibility imports should keep warning and routing behavior."""
    from langchain_classic.chains import ConversationChain

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        imported = getattr(langchain_classic, "ConversationChain")

    assert imported is ConversationChain
    assert len(caught) == 1
    assert (
        str(caught[0].message)
        == "Importing ConversationChain from langchain root module is no longer "
        "supported. Please use langchain_classic.chains.ConversationChain instead."
    )


def test_prompt_alias_warns_and_returns_prompt_template() -> None:
    """The legacy Prompt alias should still resolve to PromptTemplate."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        imported = getattr(langchain_classic, "Prompt")

    assert imported is PromptTemplate
    assert len(caught) == 1
    assert (
        str(caught[0].message)
        == "Importing Prompt from langchain root module is no longer supported. "
        "Please use langchain_core.prompts.PromptTemplate instead."
    )


def test_removed_root_import_raises_import_error() -> None:
    """Removed imports should keep the existing import error message."""
    with pytest.raises(ImportError, match="moved to langchain-experimental"):
        getattr(langchain_classic, "LLMBashChain")


def test_unknown_root_import_raises_attribute_error() -> None:
    """Unknown imports should continue to raise AttributeError."""
    with pytest.raises(AttributeError, match="Could not find: NotARealThing"):
        getattr(langchain_classic, "NotARealThing")
