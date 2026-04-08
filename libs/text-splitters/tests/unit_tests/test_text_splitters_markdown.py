"""Markdown text splitter tests."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from langchain_text_splitters.markdown import (
    ExperimentalMarkdownSyntaxTextSplitter,
    MarkdownHeaderTextSplitter,
)

def test_md_header_text_splitter_1() -> None:
    """Test markdown splitter by header: Case 1."""
    markdown_document = (
        "# Foo\n\n"
        "    ## Bar\n\n"
        "Hi this is Jim\n\n"
        "Hi this is Joe\n\n"
        " ## Baz\n\n"
        " Hi this is Molly"
    )
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
    )
    output = markdown_splitter.split_text(markdown_document)
    expected_output = [
        Document(
            page_content="Hi this is Jim  \nHi this is Joe",
            metadata={"Header 1": "Foo", "Header 2": "Bar"},
        ),
        Document(
            page_content="Hi this is Molly",
            metadata={"Header 1": "Foo", "Header 2": "Baz"},
        ),
    ]
    assert output == expected_output


def test_md_header_text_splitter_2() -> None:
    """Test markdown splitter by header: Case 2."""
    markdown_document = (
        "# Foo\n\n"
        "    ## Bar\n\n"
        "Hi this is Jim\n\n"
        "Hi this is Joe\n\n"
        " ### Boo \n\n"
        " Hi this is Lance \n\n"
        " ## Baz\n\n"
        " Hi this is Molly"
    )

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
    )
    output = markdown_splitter.split_text(markdown_document)
    expected_output = [
        Document(
            page_content="Hi this is Jim  \nHi this is Joe",
            metadata={"Header 1": "Foo", "Header 2": "Bar"},
        ),
        Document(
            page_content="Hi this is Lance",
            metadata={"Header 1": "Foo", "Header 2": "Bar", "Header 3": "Boo"},
        ),
        Document(
            page_content="Hi this is Molly",
            metadata={"Header 1": "Foo", "Header 2": "Baz"},
        ),
    ]
    assert output == expected_output


def test_md_header_text_splitter_3() -> None:
    """Test markdown splitter by header: Case 3."""
    markdown_document = (
        "# Foo\n\n"
        "    ## Bar\n\n"
        "Hi this is Jim\n\n"
        "Hi this is Joe\n\n"
        " ### Boo \n\n"
        " Hi this is Lance \n\n"
        " #### Bim \n\n"
        " Hi this is John \n\n"
        " ## Baz\n\n"
        " Hi this is Molly"
    )

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
    )
    output = markdown_splitter.split_text(markdown_document)

    expected_output = [
        Document(
            page_content="Hi this is Jim  \nHi this is Joe",
            metadata={"Header 1": "Foo", "Header 2": "Bar"},
        ),
        Document(
            page_content="Hi this is Lance",
            metadata={"Header 1": "Foo", "Header 2": "Bar", "Header 3": "Boo"},
        ),
        Document(
            page_content="Hi this is John",
            metadata={
                "Header 1": "Foo",
                "Header 2": "Bar",
                "Header 3": "Boo",
                "Header 4": "Bim",
            },
        ),
        Document(
            page_content="Hi this is Molly",
            metadata={"Header 1": "Foo", "Header 2": "Baz"},
        ),
    ]

    assert output == expected_output


def test_md_header_text_splitter_preserve_headers_1() -> None:
    """Test markdown splitter by header: Preserve Headers."""
    markdown_document = (
        "# Foo\n\n"
        "    ## Bat\n\n"
        "Hi this is Jim\n\n"
        "Hi Joe\n\n"
        "## Baz\n\n"
        "# Bar\n\n"
        "This is Alice\n\n"
        "This is Bob"
    )
    headers_to_split_on = [
        ("#", "Header 1"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )
    output = markdown_splitter.split_text(markdown_document)
    expected_output = [
        Document(
            page_content="# Foo  \n## Bat  \nHi this is Jim  \nHi Joe  \n## Baz",
            metadata={"Header 1": "Foo"},
        ),
        Document(
            page_content="# Bar  \nThis is Alice  \nThis is Bob",
            metadata={"Header 1": "Bar"},
        ),
    ]
    assert output == expected_output


def test_md_header_text_splitter_preserve_headers_2() -> None:
    """Test markdown splitter by header: Preserve Headers."""
    markdown_document = (
        "# Foo\n\n"
        "    ## Bar\n\n"
        "Hi this is Jim\n\n"
        "Hi this is Joe\n\n"
        "### Boo \n\n"
        "Hi this is Lance\n\n"
        "## Baz\n\n"
        "Hi this is Molly\n"
        "    ## Buz\n"
        "# Bop"
    )
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )
    output = markdown_splitter.split_text(markdown_document)
    expected_output = [
        Document(
            page_content="# Foo  \n## Bar  \nHi this is Jim  \nHi this is Joe",
            metadata={"Header 1": "Foo", "Header 2": "Bar"},
        ),
        Document(
            page_content="### Boo  \nHi this is Lance",
            metadata={"Header 1": "Foo", "Header 2": "Bar", "Header 3": "Boo"},
        ),
        Document(
            page_content="## Baz  \nHi this is Molly",
            metadata={"Header 1": "Foo", "Header 2": "Baz"},
        ),
        Document(
            page_content="## Buz",
            metadata={"Header 1": "Foo", "Header 2": "Buz"},
        ),
        Document(page_content="# Bop", metadata={"Header 1": "Bop"}),
    ]
    assert output == expected_output


@pytest.mark.parametrize("fence", [("```"), ("~~~")])
def test_md_header_text_splitter_fenced_code_block(fence: str) -> None:
    """Test markdown splitter by header: Fenced code block."""
    markdown_document = (
        f"# This is a Header\n\n{fence}\nfoo()\n# Not a header\nbar()\n{fence}"
    )

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
    )
    output = markdown_splitter.split_text(markdown_document)

    expected_output = [
        Document(
            page_content=f"{fence}\nfoo()\n# Not a header\nbar()\n{fence}",
            metadata={"Header 1": "This is a Header"},
        ),
    ]

    assert output == expected_output


@pytest.mark.parametrize(("fence", "other_fence"), [("```", "~~~"), ("~~~", "```")])
def test_md_header_text_splitter_fenced_code_block_interleaved(
    fence: str, other_fence: str
) -> None:
    """Test markdown splitter by header: Interleaved fenced code block."""
    markdown_document = (
        "# This is a Header\n\n"
        f"{fence}\n"
        "foo\n"
        "# Not a header\n"
        f"{other_fence}\n"
        "# Not a header\n"
        f"{fence}"
    )

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
    )
    output = markdown_splitter.split_text(markdown_document)

    expected_output = [
        Document(
            page_content=(
                f"{fence}\nfoo\n# Not a header\n{other_fence}\n# Not a header\n{fence}"
            ),
            metadata={"Header 1": "This is a Header"},
        ),
    ]

    assert output == expected_output


@pytest.mark.parametrize("characters", ["\ufeff"])
def test_md_header_text_splitter_with_invisible_characters(characters: str) -> None:
    """Test markdown splitter by header: Fenced code block."""
    markdown_document = f"{characters}# Foo\n\nfoo()\n{characters}## Bar\n\nbar()"

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
    )
    output = markdown_splitter.split_text(markdown_document)

    expected_output = [
        Document(
            page_content="foo()",
            metadata={"Header 1": "Foo"},
        ),
        Document(
            page_content="bar()",
            metadata={"Header 1": "Foo", "Header 2": "Bar"},
        ),
    ]

    assert output == expected_output


def test_md_header_text_splitter_with_custom_headers() -> None:
    """Test markdown splitter with custom header patterns like **Header**."""
    markdown_document = """**Chapter 1**

This is the content for chapter 1.

***Section 1.1***

This is the content for section 1.1.

**Chapter 2**

This is the content for chapter 2.

***Section 2.1***

This is the content for section 2.1.
"""

    headers_to_split_on = [
        ("**", "Bold Header"),
        ("***", "Bold Italic Header"),
    ]

    custom_header_patterns = {
        "**": 1,  # Level 1 headers
        "***": 2,  # Level 2 headers
    }
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        custom_header_patterns=custom_header_patterns,
    )
    output = markdown_splitter.split_text(markdown_document)

    expected_output = [
        Document(
            page_content="This is the content for chapter 1.",
            metadata={"Bold Header": "Chapter 1"},
        ),
        Document(
            page_content="This is the content for section 1.1.",
            metadata={"Bold Header": "Chapter 1", "Bold Italic Header": "Section 1.1"},
        ),
        Document(
            page_content="This is the content for chapter 2.",
            metadata={"Bold Header": "Chapter 2"},
        ),
        Document(
            page_content="This is the content for section 2.1.",
            metadata={"Bold Header": "Chapter 2", "Bold Italic Header": "Section 2.1"},
        ),
    ]

    assert output == expected_output


def test_md_header_text_splitter_mixed_headers() -> None:
    """Test markdown splitter with both standard and custom headers."""
    markdown_document = """# Standard Header 1

Content under standard header.

**Custom Header 1**

Content under custom header.

## Standard Header 2

Content under standard header 2.

***Custom Header 2***

Content under custom header 2.
"""

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("**", "Bold Header"),
        ("***", "Bold Italic Header"),
    ]

    custom_header_patterns = {
        "**": 1,  # Same level as #
        "***": 2,  # Same level as ##
    }

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        custom_header_patterns=custom_header_patterns,
    )
    output = markdown_splitter.split_text(markdown_document)

    expected_output = [
        Document(
            page_content="Content under standard header.",
            metadata={"Header 1": "Standard Header 1"},
        ),
        Document(
            page_content="Content under custom header.",
            metadata={"Bold Header": "Custom Header 1"},
        ),
        Document(
            page_content="Content under standard header 2.",
            metadata={
                "Bold Header": "Custom Header 1",
                "Header 2": "Standard Header 2",
            },
        ),
        Document(
            page_content="Content under custom header 2.",
            metadata={
                "Bold Header": "Custom Header 1",
                "Bold Italic Header": "Custom Header 2",
            },
        ),
    ]

    assert output == expected_output


EXPERIMENTAL_MARKDOWN_DOCUMENT = (
    "# My Header 1\n"
    "Content for header 1\n"
    "## Header 2\n"
    "Content for header 2\n"
    "### Header 3\n"
    "Content for header 3\n"
    "## Header 2 Again\n"
    "This should be tagged with Header 1 and Header 2 Again\n"
    "```python\n"
    "def func_definition():\n"
    "   print('Keep the whitespace consistent')\n"
    "```\n"
    "# Header 1 again\n"
    "We should also split on the horizontal line\n"
    "----\n"
    "This will be a new doc but with the same header metadata\n\n"
    "And it includes a new paragraph"
)


def test_experimental_markdown_syntax_text_splitter() -> None:
    """Test experimental markdown syntax splitter."""
    markdown_splitter = ExperimentalMarkdownSyntaxTextSplitter()
    output = markdown_splitter.split_text(EXPERIMENTAL_MARKDOWN_DOCUMENT)

    expected_output = [
        Document(
            page_content="Content for header 1\n",
            metadata={"Header 1": "My Header 1"},
        ),
        Document(
            page_content="Content for header 2\n",
            metadata={"Header 1": "My Header 1", "Header 2": "Header 2"},
        ),
        Document(
            page_content="Content for header 3\n",
            metadata={
                "Header 1": "My Header 1",
                "Header 2": "Header 2",
                "Header 3": "Header 3",
            },
        ),
        Document(
            page_content="This should be tagged with Header 1 and Header 2 Again\n",
            metadata={"Header 1": "My Header 1", "Header 2": "Header 2 Again"},
        ),
        Document(
            page_content=(
                "```python\ndef func_definition():\n   "
                "print('Keep the whitespace consistent')\n```\n"
            ),
            metadata={
                "Code": "python",
                "Header 1": "My Header 1",
                "Header 2": "Header 2 Again",
            },
        ),
        Document(
            page_content="We should also split on the horizontal line\n",
            metadata={"Header 1": "Header 1 again"},
        ),
        Document(
            page_content=(
                "This will be a new doc but with the same header metadata\n\n"
                "And it includes a new paragraph"
            ),
            metadata={"Header 1": "Header 1 again"},
        ),
    ]

    assert output == expected_output


def test_experimental_markdown_syntax_text_splitter_header_configuration() -> None:
    """Test experimental markdown syntax splitter."""
    headers_to_split_on = [("#", "Encabezamiento 1")]

    markdown_splitter = ExperimentalMarkdownSyntaxTextSplitter(
        headers_to_split_on=headers_to_split_on
    )
    output = markdown_splitter.split_text(EXPERIMENTAL_MARKDOWN_DOCUMENT)

    expected_output = [
        Document(
            page_content=(
                "Content for header 1\n"
                "## Header 2\n"
                "Content for header 2\n"
                "### Header 3\n"
                "Content for header 3\n"
                "## Header 2 Again\n"
                "This should be tagged with Header 1 and Header 2 Again\n"
            ),
            metadata={"Encabezamiento 1": "My Header 1"},
        ),
        Document(
            page_content=(
                "```python\ndef func_definition():\n   "
                "print('Keep the whitespace consistent')\n```\n"
            ),
            metadata={"Code": "python", "Encabezamiento 1": "My Header 1"},
        ),
        Document(
            page_content="We should also split on the horizontal line\n",
            metadata={"Encabezamiento 1": "Header 1 again"},
        ),
        Document(
            page_content=(
                "This will be a new doc but with the same header metadata\n\n"
                "And it includes a new paragraph"
            ),
            metadata={"Encabezamiento 1": "Header 1 again"},
        ),
    ]

    assert output == expected_output


def test_experimental_markdown_syntax_text_splitter_with_headers() -> None:
    """Test experimental markdown syntax splitter."""
    markdown_splitter = ExperimentalMarkdownSyntaxTextSplitter(strip_headers=False)
    output = markdown_splitter.split_text(EXPERIMENTAL_MARKDOWN_DOCUMENT)

    expected_output = [
        Document(
            page_content="# My Header 1\nContent for header 1\n",
            metadata={"Header 1": "My Header 1"},
        ),
        Document(
            page_content="## Header 2\nContent for header 2\n",
            metadata={"Header 1": "My Header 1", "Header 2": "Header 2"},
        ),
        Document(
            page_content="### Header 3\nContent for header 3\n",
            metadata={
                "Header 1": "My Header 1",
                "Header 2": "Header 2",
                "Header 3": "Header 3",
            },
        ),
        Document(
            page_content=(
                "## Header 2 Again\n"
                "This should be tagged with Header 1 and Header 2 Again\n"
            ),
            metadata={"Header 1": "My Header 1", "Header 2": "Header 2 Again"},
        ),
        Document(
            page_content=(
                "```python\ndef func_definition():\n   "
                "print('Keep the whitespace consistent')\n```\n"
            ),
            metadata={
                "Code": "python",
                "Header 1": "My Header 1",
                "Header 2": "Header 2 Again",
            },
        ),
        Document(
            page_content=(
                "# Header 1 again\nWe should also split on the horizontal line\n"
            ),
            metadata={"Header 1": "Header 1 again"},
        ),
        Document(
            page_content=(
                "This will be a new doc but with the same header metadata\n\n"
                "And it includes a new paragraph"
            ),
            metadata={"Header 1": "Header 1 again"},
        ),
    ]

    assert output == expected_output


def test_experimental_markdown_syntax_text_splitter_split_lines() -> None:
    """Test experimental markdown syntax splitter."""
    markdown_splitter = ExperimentalMarkdownSyntaxTextSplitter(return_each_line=True)
    output = markdown_splitter.split_text(EXPERIMENTAL_MARKDOWN_DOCUMENT)

    expected_output = [
        Document(
            page_content="Content for header 1", metadata={"Header 1": "My Header 1"}
        ),
        Document(
            page_content="Content for header 2",
            metadata={"Header 1": "My Header 1", "Header 2": "Header 2"},
        ),
        Document(
            page_content="Content for header 3",
            metadata={
                "Header 1": "My Header 1",
                "Header 2": "Header 2",
                "Header 3": "Header 3",
            },
        ),
        Document(
            page_content="This should be tagged with Header 1 and Header 2 Again",
            metadata={"Header 1": "My Header 1", "Header 2": "Header 2 Again"},
        ),
        Document(
            page_content="```python",
            metadata={
                "Code": "python",
                "Header 1": "My Header 1",
                "Header 2": "Header 2 Again",
            },
        ),
        Document(
            page_content="def func_definition():",
            metadata={
                "Code": "python",
                "Header 1": "My Header 1",
                "Header 2": "Header 2 Again",
            },
        ),
        Document(
            page_content="   print('Keep the whitespace consistent')",
            metadata={
                "Code": "python",
                "Header 1": "My Header 1",
                "Header 2": "Header 2 Again",
            },
        ),
        Document(
            page_content="```",
            metadata={
                "Code": "python",
                "Header 1": "My Header 1",
                "Header 2": "Header 2 Again",
            },
        ),
        Document(
            page_content="We should also split on the horizontal line",
            metadata={"Header 1": "Header 1 again"},
        ),
        Document(
            page_content="This will be a new doc but with the same header metadata",
            metadata={"Header 1": "Header 1 again"},
        ),
        Document(
            page_content="And it includes a new paragraph",
            metadata={"Header 1": "Header 1 again"},
        ),
    ]

    assert output == expected_output


EXPERIMENTAL_MARKDOWN_DOCUMENTS = [
    (
        "# My Header 1 From Document 1\n"
        "Content for header 1 from Document 1\n"
        "## Header 2 From Document 1\n"
        "Content for header 2 from Document 1\n"
        "```python\n"
        "def func_definition():\n"
        "   print('Keep the whitespace consistent')\n"
        "```\n"
        "# Header 1 again From Document 1\n"
        "We should also split on the horizontal line\n"
        "----\n"
        "This will be a new doc but with the same header metadata\n\n"
        "And it includes a new paragraph"
    ),
    (
        "# My Header 1 From Document 2\n"
        "Content for header 1 from Document 2\n"
        "## Header 2 From Document 2\n"
        "Content for header 2 from Document 2\n"
        "```python\n"
        "def func_definition():\n"
        "   print('Keep the whitespace consistent')\n"
        "```\n"
        "# Header 1 again From Document 2\n"
        "We should also split on the horizontal line\n"
        "----\n"
        "This will be a new doc but with the same header metadata\n\n"
        "And it includes a new paragraph"
    ),
]


def test_experimental_markdown_syntax_text_splitter_on_multi_files() -> None:
    """Test ExperimentalMarkdownSyntaxTextSplitter on multiple files.

    Test experimental markdown syntax splitter split on default called consecutively
    on two files.
    """
    markdown_splitter = ExperimentalMarkdownSyntaxTextSplitter()
    output = []
    for experimental_markdown_document in EXPERIMENTAL_MARKDOWN_DOCUMENTS:
        output += markdown_splitter.split_text(experimental_markdown_document)

    expected_output = [
        Document(
            page_content="Content for header 1 from Document 1\n",
            metadata={"Header 1": "My Header 1 From Document 1"},
        ),
        Document(
            page_content="Content for header 2 from Document 1\n",
            metadata={
                "Header 1": "My Header 1 From Document 1",
                "Header 2": "Header 2 From Document 1",
            },
        ),
        Document(
            page_content=(
                "```python\ndef func_definition():\n   "
                "print('Keep the whitespace consistent')\n```\n"
            ),
            metadata={
                "Code": "python",
                "Header 1": "My Header 1 From Document 1",
                "Header 2": "Header 2 From Document 1",
            },
        ),
        Document(
            page_content="We should also split on the horizontal line\n",
            metadata={"Header 1": "Header 1 again From Document 1"},
        ),
        Document(
            page_content=(
                "This will be a new doc but with the same header metadata\n\n"
                "And it includes a new paragraph"
            ),
            metadata={"Header 1": "Header 1 again From Document 1"},
        ),
        Document(
            page_content="Content for header 1 from Document 2\n",
            metadata={"Header 1": "My Header 1 From Document 2"},
        ),
        Document(
            page_content="Content for header 2 from Document 2\n",
            metadata={
                "Header 1": "My Header 1 From Document 2",
                "Header 2": "Header 2 From Document 2",
            },
        ),
        Document(
            page_content=(
                "```python\ndef func_definition():\n   "
                "print('Keep the whitespace consistent')\n```\n"
            ),
            metadata={
                "Code": "python",
                "Header 1": "My Header 1 From Document 2",
                "Header 2": "Header 2 From Document 2",
            },
        ),
        Document(
            page_content="We should also split on the horizontal line\n",
            metadata={"Header 1": "Header 1 again From Document 2"},
        ),
        Document(
            page_content=(
                "This will be a new doc but with the same header metadata\n\n"
                "And it includes a new paragraph"
            ),
            metadata={"Header 1": "Header 1 again From Document 2"},
        ),
    ]

    assert output == expected_output


def test_experimental_markdown_syntax_text_splitter_split_lines_on_multi_files() -> (
    None
):
    """Test ExperimentalMarkdownSyntaxTextSplitter split lines on multiple files.

    Test experimental markdown syntax splitter split on each line called consecutively
    on two files.
    """
    markdown_splitter = ExperimentalMarkdownSyntaxTextSplitter(return_each_line=True)
    output = []
    for experimental_markdown_document in EXPERIMENTAL_MARKDOWN_DOCUMENTS:
        output += markdown_splitter.split_text(experimental_markdown_document)
    expected_output = [
        Document(
            page_content="Content for header 1 from Document 1",
            metadata={"Header 1": "My Header 1 From Document 1"},
        ),
        Document(
            page_content="Content for header 2 from Document 1",
            metadata={
                "Header 1": "My Header 1 From Document 1",
                "Header 2": "Header 2 From Document 1",
            },
        ),
        Document(
            page_content="```python",
            metadata={
                "Code": "python",
                "Header 1": "My Header 1 From Document 1",
                "Header 2": "Header 2 From Document 1",
            },
        ),
        Document(
            page_content="def func_definition():",
            metadata={
                "Code": "python",
                "Header 1": "My Header 1 From Document 1",
                "Header 2": "Header 2 From Document 1",
            },
        ),
        Document(
            page_content="   print('Keep the whitespace consistent')",
            metadata={
                "Code": "python",
                "Header 1": "My Header 1 From Document 1",
                "Header 2": "Header 2 From Document 1",
            },
        ),
        Document(
            page_content="```",
            metadata={
                "Code": "python",
                "Header 1": "My Header 1 From Document 1",
                "Header 2": "Header 2 From Document 1",
            },
        ),
        Document(
            page_content="We should also split on the horizontal line",
            metadata={"Header 1": "Header 1 again From Document 1"},
        ),
        Document(
            page_content="This will be a new doc but with the same header metadata",
            metadata={"Header 1": "Header 1 again From Document 1"},
        ),
        Document(
            page_content="And it includes a new paragraph",
            metadata={"Header 1": "Header 1 again From Document 1"},
        ),
        Document(
            page_content="Content for header 1 from Document 2",
            metadata={"Header 1": "My Header 1 From Document 2"},
        ),
        Document(
            page_content="Content for header 2 from Document 2",
            metadata={
                "Header 1": "My Header 1 From Document 2",
                "Header 2": "Header 2 From Document 2",
            },
        ),
        Document(
            page_content="```python",
            metadata={
                "Code": "python",
                "Header 1": "My Header 1 From Document 2",
                "Header 2": "Header 2 From Document 2",
            },
        ),
        Document(
            page_content="def func_definition():",
            metadata={
                "Code": "python",
                "Header 1": "My Header 1 From Document 2",
                "Header 2": "Header 2 From Document 2",
            },
        ),
        Document(
            page_content="   print('Keep the whitespace consistent')",
            metadata={
                "Code": "python",
                "Header 1": "My Header 1 From Document 2",
                "Header 2": "Header 2 From Document 2",
            },
        ),
        Document(
            page_content="```",
            metadata={
                "Code": "python",
                "Header 1": "My Header 1 From Document 2",
                "Header 2": "Header 2 From Document 2",
            },
        ),
        Document(
            page_content="We should also split on the horizontal line",
            metadata={"Header 1": "Header 1 again From Document 2"},
        ),
        Document(
            page_content="This will be a new doc but with the same header metadata",
            metadata={"Header 1": "Header 1 again From Document 2"},
        ),
        Document(
            page_content="And it includes a new paragraph",
            metadata={"Header 1": "Header 1 again From Document 2"},
        ),
    ]

    assert output == expected_output


def test_experimental_markdown_syntax_text_splitter_with_header_on_multi_files() -> (
    None
):
    """Test ExperimentalMarkdownSyntaxTextSplitter with header on multiple files.

    Test experimental markdown splitter by header called consecutively on two files.
    """
    markdown_splitter = ExperimentalMarkdownSyntaxTextSplitter(strip_headers=False)
    output = []
    for experimental_markdown_document in EXPERIMENTAL_MARKDOWN_DOCUMENTS:
        output += markdown_splitter.split_text(experimental_markdown_document)

    expected_output = [
        Document(
            page_content="# My Header 1 From Document 1\n"
            "Content for header 1 from Document 1\n",
            metadata={"Header 1": "My Header 1 From Document 1"},
        ),
        Document(
            page_content="## Header 2 From Document 1\n"
            "Content for header 2 from Document 1\n",
            metadata={
                "Header 1": "My Header 1 From Document 1",
                "Header 2": "Header 2 From Document 1",
            },
        ),
        Document(
            page_content=(
                "```python\ndef func_definition():\n   "
                "print('Keep the whitespace consistent')\n```\n"
            ),
            metadata={
                "Code": "python",
                "Header 1": "My Header 1 From Document 1",
                "Header 2": "Header 2 From Document 1",
            },
        ),
        Document(
            page_content="# Header 1 again From Document 1\n"
            "We should also split on the horizontal line\n",
            metadata={"Header 1": "Header 1 again From Document 1"},
        ),
        Document(
            page_content=(
                "This will be a new doc but with the same header metadata\n\n"
                "And it includes a new paragraph"
            ),
            metadata={"Header 1": "Header 1 again From Document 1"},
        ),
        Document(
            page_content="# My Header 1 From Document 2\n"
            "Content for header 1 from Document 2\n",
            metadata={"Header 1": "My Header 1 From Document 2"},
        ),
        Document(
            page_content="## Header 2 From Document 2\n"
            "Content for header 2 from Document 2\n",
            metadata={
                "Header 1": "My Header 1 From Document 2",
                "Header 2": "Header 2 From Document 2",
            },
        ),
        Document(
            page_content=(
                "```python\ndef func_definition():\n   "
                "print('Keep the whitespace consistent')\n```\n"
            ),
            metadata={
                "Code": "python",
                "Header 1": "My Header 1 From Document 2",
                "Header 2": "Header 2 From Document 2",
            },
        ),
        Document(
            page_content="# Header 1 again From Document 2\n"
            "We should also split on the horizontal line\n",
            metadata={"Header 1": "Header 1 again From Document 2"},
        ),
        Document(
            page_content=(
                "This will be a new doc but with the same header metadata\n\n"
                "And it includes a new paragraph"
            ),
            metadata={"Header 1": "Header 1 again From Document 2"},
        ),
    ]
    assert output == expected_output


def test_experimental_markdown_syntax_text_splitter_header_config_on_multi_files() -> (
    None
):
    """Test ExperimentalMarkdownSyntaxTextSplitter header config on multiple files.

    Test experimental markdown splitter by header configuration called consecutively
    on two files.
    """
    headers_to_split_on = [("#", "Encabezamiento 1")]
    markdown_splitter = ExperimentalMarkdownSyntaxTextSplitter(
        headers_to_split_on=headers_to_split_on
    )
    output = []
    for experimental_markdown_document in EXPERIMENTAL_MARKDOWN_DOCUMENTS:
        output += markdown_splitter.split_text(experimental_markdown_document)

    expected_output = [
        Document(
            page_content="Content for header 1 from Document 1\n"
            "## Header 2 From Document 1\n"
            "Content for header 2 from Document 1\n",
            metadata={"Encabezamiento 1": "My Header 1 From Document 1"},
        ),
        Document(
            page_content=(
                "```python\ndef func_definition():\n   "
                "print('Keep the whitespace consistent')\n```\n"
            ),
            metadata={
                "Code": "python",
                "Encabezamiento 1": "My Header 1 From Document 1",
            },
        ),
        Document(
            page_content="We should also split on the horizontal line\n",
            metadata={"Encabezamiento 1": "Header 1 again From Document 1"},
        ),
        Document(
            page_content=(
                "This will be a new doc but with the same header metadata\n\n"
                "And it includes a new paragraph"
            ),
            metadata={"Encabezamiento 1": "Header 1 again From Document 1"},
        ),
        Document(
            page_content="Content for header 1 from Document 2\n"
            "## Header 2 From Document 2\n"
            "Content for header 2 from Document 2\n",
            metadata={"Encabezamiento 1": "My Header 1 From Document 2"},
        ),
        Document(
            page_content=(
                "```python\ndef func_definition():\n   "
                "print('Keep the whitespace consistent')\n```\n"
            ),
            metadata={
                "Code": "python",
                "Encabezamiento 1": "My Header 1 From Document 2",
            },
        ),
        Document(
            page_content="We should also split on the horizontal line\n",
            metadata={"Encabezamiento 1": "Header 1 again From Document 2"},
        ),
        Document(
            page_content=(
                "This will be a new doc but with the same header metadata\n\n"
                "And it includes a new paragraph"
            ),
            metadata={"Encabezamiento 1": "Header 1 again From Document 2"},
        ),
    ]

    assert output == expected_output


