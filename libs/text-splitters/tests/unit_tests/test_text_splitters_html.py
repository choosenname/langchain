"""HTML header and section splitter tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from langchain_core.documents import Document
from langchain_text_splitters.html import HTMLHeaderTextSplitter, HTMLSectionSplitter

if TYPE_CHECKING:
    from collections.abc import Callable

@pytest.fixture
def html_header_splitter_splitter_factory() -> Callable[
    [list[tuple[str, str]]], HTMLHeaderTextSplitter
]:
    """Fixture to create an `HTMLHeaderTextSplitter` instance with given headers.

    This factory allows dynamic creation of splitters with different headers.

    Returns:
        Factory function that takes a list of headers to split on and returns an
        `HTMLHeaderTextSplitter` instance.
    """

    def _create_splitter(
        headers_to_split_on: list[tuple[str, str]],
    ) -> HTMLHeaderTextSplitter:
        return HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    return _create_splitter


@pytest.mark.parametrize(
    ("headers_to_split_on", "html_input", "expected_documents", "test_case"),
    [
        (
            # Test Case 1: Split on h1 and h2
            [("h1", "Header 1"), ("h2", "Header 2")],
            """
            <html>
                <body>
                    <h1>Introduction</h1>
                    <p>This is the introduction.</p>
                    <h2>Background</h2>
                    <p>Background information.</p>
                    <h1>Conclusion</h1>
                    <p>Final thoughts.</p>
                </body>
            </html>
            """,
            [
                Document(
                    page_content="Introduction", metadata={"Header 1": "Introduction"}
                ),
                Document(
                    page_content="This is the introduction.",
                    metadata={"Header 1": "Introduction"},
                ),
                Document(
                    page_content="Background",
                    metadata={"Header 1": "Introduction", "Header 2": "Background"},
                ),
                Document(
                    page_content="Background information.",
                    metadata={"Header 1": "Introduction", "Header 2": "Background"},
                ),
                Document(
                    page_content="Conclusion", metadata={"Header 1": "Conclusion"}
                ),
                Document(
                    page_content="Final thoughts.", metadata={"Header 1": "Conclusion"}
                ),
            ],
            "Simple headers and paragraphs",
        ),
        (
            # Test Case 2: Nested headers with h1, h2, and h3
            [("h1", "Header 1"), ("h2", "Header 2"), ("h3", "Header 3")],
            """
            <html>
                <body>
                    <div>
                        <h1>Main Title</h1>
                        <div>
                            <h2>Subsection</h2>
                            <p>Details of subsection.</p>
                            <div>
                                <h3>Sub-subsection</h3>
                                <p>More details.</p>
                            </div>
                        </div>
                    </div>
                    <h1>Another Main Title</h1>
                    <p>Content under another main title.</p>
                </body>
            </html>
            """,
            [
                Document(
                    page_content="Main Title", metadata={"Header 1": "Main Title"}
                ),
                Document(
                    page_content="Subsection",
                    metadata={"Header 1": "Main Title", "Header 2": "Subsection"},
                ),
                Document(
                    page_content="Details of subsection.",
                    metadata={"Header 1": "Main Title", "Header 2": "Subsection"},
                ),
                Document(
                    page_content="Sub-subsection",
                    metadata={
                        "Header 1": "Main Title",
                        "Header 2": "Subsection",
                        "Header 3": "Sub-subsection",
                    },
                ),
                Document(
                    page_content="More details.",
                    metadata={
                        "Header 1": "Main Title",
                        "Header 2": "Subsection",
                        "Header 3": "Sub-subsection",
                    },
                ),
                Document(
                    page_content="Another Main Title",
                    metadata={"Header 1": "Another Main Title"},
                ),
                Document(
                    page_content="Content under another main title.",
                    metadata={"Header 1": "Another Main Title"},
                ),
            ],
            "Nested headers with h1, h2, and h3",
        ),
        (
            # Test Case 3: No headers
            [("h1", "Header 1")],
            """
            <html>
                <body>
                    <p>Paragraph one.</p>
                    <p>Paragraph two.</p>
                    <div>
                        <p>Paragraph three.</p>
                    </div>
                </body>
            </html>
            """,
            [
                Document(
                    page_content="Paragraph one.  \nParagraph two.  \nParagraph three.",
                    metadata={},
                )
            ],
            "No headers present",
        ),
        (
            # Test Case 4: Multiple headers of the same level
            [("h1", "Header 1")],
            """
            <html>
                <body>
                    <h1>Chapter 1</h1>
                    <p>Content of chapter 1.</p>
                    <h1>Chapter 2</h1>
                    <p>Content of chapter 2.</p>
                    <h1>Chapter 3</h1>
                    <p>Content of chapter 3.</p>
                </body>
            </html>
            """,
            [
                Document(page_content="Chapter 1", metadata={"Header 1": "Chapter 1"}),
                Document(
                    page_content="Content of chapter 1.",
                    metadata={"Header 1": "Chapter 1"},
                ),
                Document(page_content="Chapter 2", metadata={"Header 1": "Chapter 2"}),
                Document(
                    page_content="Content of chapter 2.",
                    metadata={"Header 1": "Chapter 2"},
                ),
                Document(page_content="Chapter 3", metadata={"Header 1": "Chapter 3"}),
                Document(
                    page_content="Content of chapter 3.",
                    metadata={"Header 1": "Chapter 3"},
                ),
            ],
            "Multiple headers of the same level",
        ),
        (
            # Test Case 5: Headers with no content
            [("h1", "Header 1"), ("h2", "Header 2")],
            """
            <html>
                <body>
                    <h1>Header 1</h1>
                    <h2>Header 2</h2>
                    <h1>Header 3</h1>
                </body>
            </html>
            """,
            [
                Document(page_content="Header 1", metadata={"Header 1": "Header 1"}),
                Document(
                    page_content="Header 2",
                    metadata={"Header 1": "Header 1", "Header 2": "Header 2"},
                ),
                Document(page_content="Header 3", metadata={"Header 1": "Header 3"}),
            ],
            "Headers with no associated content",
        ),
    ],
)
@pytest.mark.requires("bs4")
def test_html_header_text_splitter(
    html_header_splitter_splitter_factory: Callable[
        [list[tuple[str, str]]], HTMLHeaderTextSplitter
    ],
    headers_to_split_on: list[tuple[str, str]],
    html_input: str,
    expected_documents: list[Document],
    test_case: str,
) -> None:
    """Test the HTML header text splitter.

    Args:
        html_header_splitter_splitter_factory : Factory function to create the HTML
            header splitter.
        headers_to_split_on: List of headers to split on.
        html_input: The HTML input string to be split.
        expected_documents: List of expected Document objects.
        test_case: Description of the test case.

    Raises:
        AssertionError: If the number of documents or their content/metadata
            does not match the expected values.
    """
    splitter = html_header_splitter_splitter_factory(headers_to_split_on)
    docs = splitter.split_text(html_input)

    assert len(docs) == len(expected_documents), (
        f"Test Case '{test_case}' Failed: Number of documents mismatch. "
        f"Expected {len(expected_documents)}, got {len(docs)}."
    )
    for idx, (doc, expected) in enumerate(
        zip(docs, expected_documents, strict=False), start=1
    ):
        assert doc.page_content == expected.page_content, (
            f"Test Case '{test_case}' Failed at Document {idx}: "
            f"Content mismatch.\nExpected: {expected.page_content}"
            "\nGot: {doc.page_content}"
        )
        assert doc.metadata == expected.metadata, (
            f"Test Case '{test_case}' Failed at Document {idx}: "
            f"Metadata mismatch.\nExpected: {expected.metadata}\nGot: {doc.metadata}"
        )


@pytest.mark.parametrize(
    ("headers_to_split_on", "html_content", "expected_output", "test_case"),
    [
        (
            # Test Case A: Split on h1 and h2 with h3 in content
            [("h1", "Header 1"), ("h2", "Header 2"), ("h3", "Header 3")],
            """
            <!DOCTYPE html>
            <html>
            <body>
                <div>
                    <h1>Foo</h1>
                    <p>Some intro text about Foo.</p>
                    <div>
                        <h2>Bar main section</h2>
                        <p>Some intro text about Bar.</p>
                        <h3>Bar subsection 1</h3>
                        <p>Some text about the first subtopic of Bar.</p>
                        <h3>Bar subsection 2</h3>
                        <p>Some text about the second subtopic of Bar.</p>
                    </div>
                    <div>
                        <h2>Baz</h2>
                        <p>Some text about Baz</p>
                    </div>
                    <br>
                    <p>Some concluding text about Foo</p>
                </div>
            </body>
            </html>
            """,
            [
                Document(metadata={"Header 1": "Foo"}, page_content="Foo"),
                Document(
                    metadata={"Header 1": "Foo"},
                    page_content="Some intro text about Foo.",
                ),
                Document(
                    metadata={"Header 1": "Foo", "Header 2": "Bar main section"},
                    page_content="Bar main section",
                ),
                Document(
                    metadata={"Header 1": "Foo", "Header 2": "Bar main section"},
                    page_content="Some intro text about Bar.",
                ),
                Document(
                    metadata={
                        "Header 1": "Foo",
                        "Header 2": "Bar main section",
                        "Header 3": "Bar subsection 1",
                    },
                    page_content="Bar subsection 1",
                ),
                Document(
                    metadata={
                        "Header 1": "Foo",
                        "Header 2": "Bar main section",
                        "Header 3": "Bar subsection 1",
                    },
                    page_content="Some text about the first subtopic of Bar.",
                ),
                Document(
                    metadata={
                        "Header 1": "Foo",
                        "Header 2": "Bar main section",
                        "Header 3": "Bar subsection 2",
                    },
                    page_content="Bar subsection 2",
                ),
                Document(
                    metadata={
                        "Header 1": "Foo",
                        "Header 2": "Bar main section",
                        "Header 3": "Bar subsection 2",
                    },
                    page_content="Some text about the second subtopic of Bar.",
                ),
                Document(
                    metadata={"Header 1": "Foo", "Header 2": "Baz"}, page_content="Baz"
                ),
                Document(
                    metadata={"Header 1": "Foo"},
                    page_content=(
                        "Some text about Baz  \nSome concluding text about Foo"
                    ),
                ),
            ],
            "Test Case A: Split on h1, h2, and h3 with nested headers",
        ),
        (
            # Test Case B: Split on h1 only without any headers
            [("h1", "Header 1")],
            """
            <html>
                <body>
                    <p>Paragraph one.</p>
                    <p>Paragraph two.</p>
                    <p>Paragraph three.</p>
                </body>
            </html>
            """,
            [
                Document(
                    metadata={},
                    page_content="Paragraph one.  \nParagraph two.  \nParagraph three.",
                )
            ],
            "Test Case B: Split on h1 only without any headers",
        ),
    ],
)
@pytest.mark.requires("bs4")
def test_additional_html_header_text_splitter(
    html_header_splitter_splitter_factory: Callable[
        [list[tuple[str, str]]], HTMLHeaderTextSplitter
    ],
    headers_to_split_on: list[tuple[str, str]],
    html_content: str,
    expected_output: list[Document],
    test_case: str,
) -> None:
    """Test the HTML header text splitter.

    Args:
        html_header_splitter_splitter_factory: Factory function to create the HTML
            header splitter.
        headers_to_split_on: List of headers to split on.
        html_content: HTML content to be split.
        expected_output: Expected list of `Document` objects.
        test_case: Description of the test case.

    Raises:
        AssertionError: If the number of documents or their content/metadata
            does not match the expected output.
    """
    splitter = html_header_splitter_splitter_factory(headers_to_split_on)
    docs = splitter.split_text(html_content)

    assert len(docs) == len(expected_output), (
        f"{test_case} Failed: Number of documents mismatch. "
        f"Expected {len(expected_output)}, got {len(docs)}."
    )
    for idx, (doc, expected) in enumerate(
        zip(docs, expected_output, strict=False), start=1
    ):
        assert doc.page_content == expected.page_content, (
            f"{test_case} Failed at Document {idx}: "
            f"Content mismatch.\nExpected: {expected.page_content}\n"
            "Got: {doc.page_content}"
        )
        assert doc.metadata == expected.metadata, (
            f"{test_case} Failed at Document {idx}: "
            f"Metadata mismatch.\nExpected: {expected.metadata}\nGot: {doc.metadata}"
        )


@pytest.mark.parametrize(
    ("headers_to_split_on", "html_content", "expected_output", "test_case"),
    [
        (
            # Test Case C: Split on h1, h2, and h3 with no headers present
            [("h1", "Header 1"), ("h2", "Header 2"), ("h3", "Header 3")],
            """
            <html>
                <body>
                    <p>Just some random text without headers.</p>
                    <div>
                        <span>More text here.</span>
                    </div>
                </body>
            </html>
            """,
            [
                Document(
                    page_content="Just some random text without headers."
                    "  \nMore text here.",
                    metadata={},
                )
            ],
            "Test Case C: Split on h1, h2, and h3 without any headers",
        )
    ],
)
@pytest.mark.requires("bs4")
def test_html_no_headers_with_multiple_splitters(
    html_header_splitter_splitter_factory: Callable[
        [list[tuple[str, str]]], HTMLHeaderTextSplitter
    ],
    headers_to_split_on: list[tuple[str, str]],
    html_content: str,
    expected_output: list[Document],
    test_case: str,
) -> None:
    """Test HTML content splitting without headers using multiple splitters.

    Args:
        html_header_splitter_splitter_factory: Factory to create the HTML header
            splitter.
        headers_to_split_on: List of headers to split on.
        html_content: HTML content to be split.
        expected_output: Expected list of `Document` objects after splitting.
        test_case: Description of the test case.

    Raises:
        AssertionError: If the number of documents or their content/metadata
            does not match the expected output.
    """
    splitter = html_header_splitter_splitter_factory(headers_to_split_on)
    docs = splitter.split_text(html_content)

    assert len(docs) == len(expected_output), (
        f"{test_case} Failed: Number of documents mismatch. "
        f"Expected {len(expected_output)}, got {len(docs)}."
    )
    for idx, (doc, expected) in enumerate(
        zip(docs, expected_output, strict=False), start=1
    ):
        assert doc.page_content == expected.page_content, (
            f"{test_case} Failed at Document {idx}: "
            f"Content mismatch.\nExpected: {expected.page_content}\n"
            "Got: {doc.page_content}"
        )
        assert doc.metadata == expected.metadata, (
            f"{test_case} Failed at Document {idx}: "
            f"Metadata mismatch.\nExpected: {expected.metadata}\nGot: {doc.metadata}"
        )


@pytest.mark.requires("bs4")
@pytest.mark.requires("lxml")
def test_section_aware_happy_path_splitting_based_on_header_1_2() -> None:
    # arrange
    html_string = """<!DOCTYPE html>
            <html>
            <body>
                <div>
                    <h1>Foo</h1>
                    <p>Some intro text about Foo.</p>
                    <div>
                        <h2>Bar main section</h2>
                        <p>Some intro text about Bar.</p>
                        <h3>Bar subsection 1</h3>
                        <p>Some text about the first subtopic of Bar.</p>
                        <h3>Bar subsection 2</h3>
                        <p>Some text about the second subtopic of Bar.</p>
                    </div>
                    <div>
                        <h2>Baz</h2>
                        <p>Some text about Baz</p>
                    </div>
                    <br>
                    <p>Some concluding text about Foo</p>
                </div>
            </body>
            </html>"""

    sec_splitter = HTMLSectionSplitter(
        headers_to_split_on=[("h1", "Header 1"), ("h2", "Header 2")]
    )

    docs = sec_splitter.split_text(html_string)

    assert len(docs) == 3
    assert docs[0].metadata["Header 1"] == "Foo"
    assert docs[0].page_content == "Foo \n Some intro text about Foo."

    assert docs[1].page_content == (
        "Bar main section \n Some intro text about Bar. \n "
        "Bar subsection 1 \n Some text about the first subtopic of Bar. \n "
        "Bar subsection 2 \n Some text about the second subtopic of Bar."
    )
    assert docs[1].metadata["Header 2"] == "Bar main section"

    assert (
        docs[2].page_content
        == "Baz \n Some text about Baz \n \n \n Some concluding text about Foo"
    )
    # Baz \n Some text about Baz \n \n \n Some concluding text about Foo
    # Baz \n Some text about Baz \n \n Some concluding text about Foo
    assert docs[2].metadata["Header 2"] == "Baz"


@pytest.mark.requires("bs4")
@pytest.mark.requires("lxml")
def test_happy_path_splitting_based_on_header_with_font_size() -> None:
    # arrange
    html_string = """<!DOCTYPE html>
            <html>
            <body>
                <div>
                    <span style="font-size: 22px">Foo</span>
                    <p>Some intro text about Foo.</p>
                    <div>
                        <h2>Bar main section</h2>
                        <p>Some intro text about Bar.</p>
                        <h3>Bar subsection 1</h3>
                        <p>Some text about the first subtopic of Bar.</p>
                        <h3>Bar subsection 2</h3>
                        <p>Some text about the second subtopic of Bar.</p>
                    </div>
                    <div>
                        <h2>Baz</h2>
                        <p>Some text about Baz</p>
                    </div>
                    <br>
                    <p>Some concluding text about Foo</p>
                </div>
            </body>
            </html>"""

    sec_splitter = HTMLSectionSplitter(
        headers_to_split_on=[("h1", "Header 1"), ("h2", "Header 2")]
    )

    docs = sec_splitter.split_text(html_string)

    assert len(docs) == 3
    assert docs[0].page_content == "Foo \n Some intro text about Foo."
    assert docs[0].metadata["Header 1"] == "Foo"

    assert docs[1].page_content == (
        "Bar main section \n Some intro text about Bar. \n "
        "Bar subsection 1 \n Some text about the first subtopic of Bar. \n "
        "Bar subsection 2 \n Some text about the second subtopic of Bar."
    )
    assert docs[1].metadata["Header 2"] == "Bar main section"

    assert docs[2].page_content == (
        "Baz \n Some text about Baz \n \n \n Some concluding text about Foo"
    )
    assert docs[2].metadata["Header 2"] == "Baz"


@pytest.mark.requires("bs4")
@pytest.mark.requires("lxml")
def test_happy_path_splitting_based_on_header_with_whitespace_chars() -> None:
    # arrange
    html_string = """<!DOCTYPE html>
            <html>
            <body>
                <div>
                    <span style="font-size: 22px">\nFoo </span>
                    <p>Some intro text about Foo.</p>
                    <div>
                        <h2>Bar main section</h2>
                        <p>Some intro text about Bar.</p>
                        <h3>Bar subsection 1</h3>
                        <p>Some text about the first subtopic of Bar.</p>
                        <h3>Bar subsection 2</h3>
                        <p>Some text about the second subtopic of Bar.</p>
                    </div>
                    <div>
                        <h2>Baz</h2>
                        <p>Some text about Baz</p>
                    </div>
                    <br>
                    <p>Some concluding text about Foo</p>
                </div>
            </body>
            </html>"""

    sec_splitter = HTMLSectionSplitter(
        headers_to_split_on=[("h1", "Header 1"), ("h2", "Header 2")]
    )

    docs = sec_splitter.split_text(html_string)

    assert len(docs) == 3
    assert docs[0].page_content == "Foo  \n Some intro text about Foo."
    assert docs[0].metadata["Header 1"] == "Foo"

    assert docs[1].page_content == (
        "Bar main section \n Some intro text about Bar. \n "
        "Bar subsection 1 \n Some text about the first subtopic of Bar. \n "
        "Bar subsection 2 \n Some text about the second subtopic of Bar."
    )
    assert docs[1].metadata["Header 2"] == "Bar main section"

    assert docs[2].page_content == (
        "Baz \n Some text about Baz \n \n \n Some concluding text about Foo"
    )
    assert docs[2].metadata["Header 2"] == "Baz"


@pytest.mark.requires("bs4")
@pytest.mark.requires("lxml")
def test_happy_path_splitting_with_duplicate_header_tag() -> None:
    # arrange
    html_string = """<!DOCTYPE html>
        <html>
        <body>
            <div>
                <h1>Foo</h1>
                <p>Some intro text about Foo.</p>
                <div>
                    <h2>Bar main section</h2>
                    <p>Some intro text about Bar.</p>
                    <h3>Bar subsection 1</h3>
                    <p>Some text about the first subtopic of Bar.</p>
                    <h3>Bar subsection 2</h3>
                    <p>Some text about the second subtopic of Bar.</p>
                </div>
                <div>
                    <h2>Foo</h2>
                    <p>Some text about Baz</p>
                </div>
                <h1>Foo</h1>
                <br>
                <p>Some concluding text about Foo</p>
            </div>
        </body>
        </html>"""

    sec_splitter = HTMLSectionSplitter(
        headers_to_split_on=[("h1", "Header 1"), ("h2", "Header 2")]
    )

    docs = sec_splitter.split_text(html_string)

    assert len(docs) == 4
    assert docs[0].page_content == "Foo \n Some intro text about Foo."
    assert docs[0].metadata["Header 1"] == "Foo"

    assert docs[1].page_content == (
        "Bar main section \n Some intro text about Bar. \n "
        "Bar subsection 1 \n Some text about the first subtopic of Bar. \n "
        "Bar subsection 2 \n Some text about the second subtopic of Bar."
    )
    assert docs[1].metadata["Header 2"] == "Bar main section"

    assert docs[2].page_content == "Foo \n Some text about Baz"
    assert docs[2].metadata["Header 2"] == "Foo"

    assert docs[3].page_content == "Foo \n \n Some concluding text about Foo"
    assert docs[3].metadata["Header 1"] == "Foo"
