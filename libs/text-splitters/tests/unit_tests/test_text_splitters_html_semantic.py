"""Semantic HTML splitter tests."""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

import pytest
from langchain_core._api import suppress_langchain_beta_warning
from langchain_core.documents import Document
from langchain_text_splitters.html import HTMLSemanticPreservingSplitter

if TYPE_CHECKING:
    from bs4 import Tag


def custom_iframe_extractor(iframe_tag: Tag) -> str:
    iframe_src = iframe_tag.get("src", "")
    return f"[iframe:{iframe_src}]({iframe_src})"


@pytest.mark.requires("bs4")
def test_html_splitter_with_custom_extractor() -> None:
    """Test HTML splitting with a custom extractor."""
    html_content = """
    <h1>Section 1</h1>
    <p>This is an iframe:</p>
    <iframe src="http://example.com"></iframe>
    """
    with suppress_langchain_beta_warning():
        splitter = HTMLSemanticPreservingSplitter(
            headers_to_split_on=[("h1", "Header 1")],
            custom_handlers={"iframe": custom_iframe_extractor},
            max_chunk_size=1000,
        )
    documents = splitter.split_text(html_content)

    expected = [
        Document(
            page_content="This is an iframe: "
            "[iframe:http://example.com](http://example.com)",
            metadata={"Header 1": "Section 1"},
        ),
    ]

    assert documents == expected


@pytest.mark.requires("bs4")
def test_html_splitter_with_href_links() -> None:
    """Test HTML splitting with href links."""
    html_content = """
    <h1>Section 1</h1>
    <p>This is a link to <a href="http://example.com">example.com</a></p>
    """
    with suppress_langchain_beta_warning():
        splitter = HTMLSemanticPreservingSplitter(
            headers_to_split_on=[("h1", "Header 1")],
            preserve_links=True,
            max_chunk_size=1000,
        )
    documents = splitter.split_text(html_content)

    expected = [
        Document(
            page_content="This is a link to [example.com](http://example.com)",
            metadata={"Header 1": "Section 1"},
        ),
    ]

    assert documents == expected


@pytest.mark.requires("bs4")
def test_html_splitter_with_nested_elements() -> None:
    """Test HTML splitting with nested elements."""
    html_content = """
    <h1>Main Section</h1>
    <div>
        <p>Some text here.</p>
        <div>
            <p>Nested content.</p>
        </div>
    </div>
    """
    with suppress_langchain_beta_warning():
        splitter = HTMLSemanticPreservingSplitter(
            headers_to_split_on=[("h1", "Header 1")], max_chunk_size=1000
        )
    documents = splitter.split_text(html_content)

    expected = [
        Document(
            page_content="Some text here. Nested content.",
            metadata={"Header 1": "Main Section"},
        ),
    ]

    assert documents == expected


@pytest.mark.requires("bs4")
def test_html_splitter_with_preserved_elements() -> None:
    """Test HTML splitter with preserved elements.

    Test HTML splitting with preserved elements like <table>, <ul> with low chunk
    size.
    """
    html_content = """
    <h1>Section 1</h1>
    <table>
        <tr><td>Row 1</td></tr>
        <tr><td>Row 2</td></tr>
    </table>
    <ul>
        <li>Item 1</li>
        <li>Item 2</li>
    </ul>
    """
    with suppress_langchain_beta_warning():
        splitter = HTMLSemanticPreservingSplitter(
            headers_to_split_on=[("h1", "Header 1")],
            elements_to_preserve=["table", "ul"],
            max_chunk_size=50,  # Deliberately low to test preservation
        )
    documents = splitter.split_text(html_content)

    expected = [
        Document(
            page_content="Row 1 Row 2 Item 1 Item 2",
            metadata={"Header 1": "Section 1"},
        ),
    ]

    assert documents == expected  # Shouldn't split the table or ul


@pytest.mark.requires("bs4")
def test_html_splitter_with_nested_preserved_elements() -> None:
    """Test HTML splitter with preserved elements nested in containers.

    Test that preserved elements are correctly preserved even when they are
    nested inside other container elements like <section> or <article>.
    This is a regression test for issue #31569
    """
    html_content = """
    <article>
        <h1>Section 1</h1>
        <section>
            <p>Some context about the data:</p>
            <table>
                <tr><td>Col1</td><td>Col2</td></tr>
                <tr><td>Data1</td><td>Data2</td></tr>
            </table>
            <p>Conclusion about data.</p>
        </section>
    </article>
    """
    with suppress_langchain_beta_warning():
        splitter = HTMLSemanticPreservingSplitter(
            headers_to_split_on=[("h1", "Header 1")],
            elements_to_preserve=["table"],
            max_chunk_size=1000,
        )
    documents = splitter.split_text(html_content)

    # The table should be preserved in the output
    assert len(documents) == 1
    content = documents[0].page_content
    # Check that the table structure is maintained (not flattened)
    assert "Col1" in content
    assert "Col2" in content
    assert "Data1" in content
    assert "Data2" in content
    # Check metadata
    assert documents[0].metadata == {"Header 1": "Section 1"}


@pytest.mark.requires("bs4")
def test_html_splitter_with_nested_div_preserved() -> None:
    """Test HTML splitter preserving nested div elements.

    Nested div elements should be preserved when specified in elements_to_preserve
    """
    html_content = """
    <div>
        <h1>Header</h1>
        <p>outer text</p>
        <div>inner div content</div>
        <p>more outer text</p>
    </div>
    """
    with suppress_langchain_beta_warning():
        splitter = HTMLSemanticPreservingSplitter(
            headers_to_split_on=[("h1", "Header 1")],
            elements_to_preserve=["div"],
            max_chunk_size=1000,
        )
    documents = splitter.split_text(html_content)

    assert len(documents) == 1
    content = documents[0].page_content
    # The inner div content should be preserved
    assert "inner div content" in content
    assert "outer text" in content
    assert "more outer text" in content


@pytest.mark.requires("bs4")
def test_html_splitter_preserve_nested_in_paragraph() -> None:
    """Test preserving deeply nested elements (code inside paragraph).

    tests the case where a preserved element (<code>) is nested
    inside a non-container element (<p>)
    """
    html_content = "<p>before <code>KEEP_THIS</code> after</p>"
    with suppress_langchain_beta_warning():
        splitter = HTMLSemanticPreservingSplitter(
            headers_to_split_on=[],
            elements_to_preserve=["code"],
        )
    documents = splitter.split_text(html_content)

    assert len(documents) == 1
    content = documents[0].page_content
    # All text should be preserved
    assert "before" in content
    assert "KEEP_THIS" in content
    assert "after" in content


@pytest.mark.requires("bs4")
def test_html_splitter_with_no_further_splits() -> None:
    """Test HTML splitting that requires no further splits beyond sections."""
    html_content = """
    <h1>Section 1</h1>
    <p>Some content here.</p>
    <h1>Section 2</h1>
    <p>More content here.</p>
    """
    with suppress_langchain_beta_warning():
        splitter = HTMLSemanticPreservingSplitter(
            headers_to_split_on=[("h1", "Header 1")], max_chunk_size=1000
        )
    documents = splitter.split_text(html_content)

    expected = [
        Document(page_content="Some content here.", metadata={"Header 1": "Section 1"}),
        Document(page_content="More content here.", metadata={"Header 1": "Section 2"}),
    ]

    assert documents == expected  # No further splits, just sections


@pytest.mark.requires("bs4")
def test_html_splitter_with_small_chunk_size() -> None:
    """Test HTML splitting with a very small chunk size to validate chunking."""
    html_content = """
    <h1>Section 1</h1>
    <p>This is some long text that should be split into multiple chunks due to the
    small chunk size.</p>
    """
    with suppress_langchain_beta_warning():
        splitter = HTMLSemanticPreservingSplitter(
            headers_to_split_on=[("h1", "Header 1")], max_chunk_size=20, chunk_overlap=5
        )
    documents = splitter.split_text(html_content)

    expected = [
        Document(page_content="This is some long", metadata={"Header 1": "Section 1"}),
        Document(page_content="long text that", metadata={"Header 1": "Section 1"}),
        Document(page_content="that should be", metadata={"Header 1": "Section 1"}),
        Document(page_content="be split into", metadata={"Header 1": "Section 1"}),
        Document(page_content="into multiple", metadata={"Header 1": "Section 1"}),
        Document(page_content="chunks due to the", metadata={"Header 1": "Section 1"}),
        Document(page_content="the small chunk", metadata={"Header 1": "Section 1"}),
        Document(page_content="size.", metadata={"Header 1": "Section 1"}),
    ]

    assert documents == expected  # Should split into multiple chunks


@pytest.mark.requires("bs4")
def test_html_splitter_with_denylist_tags() -> None:
    """Test HTML splitting with denylist tag filtering."""
    html_content = """
    <h1>Section 1</h1>
    <p>This paragraph should be kept.</p>
    <span>This span should be removed.</span>
    """
    with suppress_langchain_beta_warning():
        splitter = HTMLSemanticPreservingSplitter(
            headers_to_split_on=[("h1", "Header 1")],
            denylist_tags=["span"],
            max_chunk_size=1000,
        )
    documents = splitter.split_text(html_content)

    expected = [
        Document(
            page_content="This paragraph should be kept.",
            metadata={"Header 1": "Section 1"},
        ),
    ]

    assert documents == expected


@pytest.mark.requires("bs4")
def test_html_splitter_with_external_metadata() -> None:
    """Test HTML splitting with external metadata integration."""
    html_content = """
    <h1>Section 1</h1>
    <p>This is some content.</p>
    """
    with suppress_langchain_beta_warning():
        splitter = HTMLSemanticPreservingSplitter(
            headers_to_split_on=[("h1", "Header 1")],
            external_metadata={"source": "example.com"},
            max_chunk_size=1000,
        )
    documents = splitter.split_text(html_content)

    expected = [
        Document(
            page_content="This is some content.",
            metadata={"Header 1": "Section 1", "source": "example.com"},
        ),
    ]

    assert documents == expected


@pytest.mark.requires("bs4")
def test_html_splitter_with_text_normalization() -> None:
    """Test HTML splitting with text normalization."""
    html_content = """
    <h1>Section 1</h1>
    <p>This is some TEXT that should be normalized!</p>
    """
    with suppress_langchain_beta_warning():
        splitter = HTMLSemanticPreservingSplitter(
            headers_to_split_on=[("h1", "Header 1")],
            normalize_text=True,
            max_chunk_size=1000,
        )
    documents = splitter.split_text(html_content)

    expected = [
        Document(
            page_content="this is some text that should be normalized",
            metadata={"Header 1": "Section 1"},
        ),
    ]

    assert documents == expected


@pytest.mark.requires("bs4")
def test_html_splitter_with_allowlist_tags() -> None:
    """Test HTML splitting with allowlist tag filtering."""
    html_content = """
    <h1>Section 1</h1>
    <p>This paragraph should be kept.</p>
    <span>This span should be kept.</span>
    <div>This div should be removed.</div>
    """
    with suppress_langchain_beta_warning():
        splitter = HTMLSemanticPreservingSplitter(
            headers_to_split_on=[("h1", "Header 1")],
            allowlist_tags=["p", "span"],
            max_chunk_size=1000,
        )
    documents = splitter.split_text(html_content)

    expected = [
        Document(
            page_content="This paragraph should be kept. This span should be kept.",
            metadata={"Header 1": "Section 1"},
        ),
    ]

    assert documents == expected


@pytest.mark.requires("bs4")
def test_html_splitter_with_mixed_preserve_and_filter() -> None:
    """Test HTML splitting with both preserved elements and denylist tags."""
    html_content = """
    <h1>Section 1</h1>
    <table>
        <tr>
            <td>Keep this table</td>
            <td>Cell contents kept, span removed
                <span>This span should be removed.</span>
            </td>
        </tr>
    </table>
    <p>This paragraph should be kept.</p>
    <span>This span should be removed.</span>
    """
    with suppress_langchain_beta_warning():
        splitter = HTMLSemanticPreservingSplitter(
            headers_to_split_on=[("h1", "Header 1")],
            elements_to_preserve=["table"],
            denylist_tags=["span"],
            max_chunk_size=1000,
        )
    documents = splitter.split_text(html_content)

    expected = [
        Document(
            page_content="Keep this table Cell contents kept, span removed"
            " This paragraph should be kept.",
            metadata={"Header 1": "Section 1"},
        ),
    ]

    assert documents == expected


@pytest.mark.requires("bs4")
def test_html_splitter_with_no_headers() -> None:
    """Test HTML splitting when there are no headers to split on."""
    html_content = """
    <p>This is content without any headers.</p>
    <p>It should still produce a valid document.</p>
    """
    with suppress_langchain_beta_warning():
        splitter = HTMLSemanticPreservingSplitter(
            headers_to_split_on=[],
            max_chunk_size=1000,
        )
    documents = splitter.split_text(html_content)

    expected = [
        Document(
            page_content="This is content without any headers. It should still produce"
            " a valid document.",
            metadata={},
        ),
    ]

    assert documents == expected


@pytest.mark.requires("bs4")
def test_html_splitter_with_media_preservation() -> None:
    """Test HTML splitter with media preservation.

    Test HTML splitting with media elements preserved and converted to Markdown-like
    links.
    """
    html_content = """
    <h1>Section 1</h1>
    <p>This is an image:</p>
    <img src="http://example.com/image.png" />
    <p>This is a video:</p>
    <video src="http://example.com/video.mp4"></video>
    <p>This is audio:</p>
    <audio src="http://example.com/audio.mp3"></audio>
    """
    with suppress_langchain_beta_warning():
        splitter = HTMLSemanticPreservingSplitter(
            headers_to_split_on=[("h1", "Header 1")],
            preserve_images=True,
            preserve_videos=True,
            preserve_audio=True,
            max_chunk_size=1000,
        )
    documents = splitter.split_text(html_content)

    expected = [
        Document(
            page_content="This is an image: ![image:http://example.com/image.png]"
            "(http://example.com/image.png) "
            "This is a video: ![video:http://example.com/video.mp4]"
            "(http://example.com/video.mp4) "
            "This is audio: ![audio:http://example.com/audio.mp3]"
            "(http://example.com/audio.mp3)",
            metadata={"Header 1": "Section 1"},
        ),
    ]

    assert documents == expected


@pytest.mark.requires("bs4")
def test_html_splitter_keep_separator_true() -> None:
    """Test HTML splitting with keep_separator=True."""
    html_content = """
    <h1>Section 1</h1>
    <p>This is some text. This is some other text.</p>
    """
    with suppress_langchain_beta_warning():
        splitter = HTMLSemanticPreservingSplitter(
            headers_to_split_on=[("h1", "Header 1")],
            max_chunk_size=10,
            separators=[". "],
            keep_separator=True,
        )
    documents = splitter.split_text(html_content)

    expected = [
        Document(
            page_content="This is some text",
            metadata={"Header 1": "Section 1"},
        ),
        Document(
            page_content=". This is some other text.",
            metadata={"Header 1": "Section 1"},
        ),
    ]

    assert documents == expected


@pytest.mark.requires("bs4")
def test_html_splitter_keep_separator_false() -> None:
    """Test HTML splitting with keep_separator=False."""
    html_content = """
    <h1>Section 1</h1>
    <p>This is some text. This is some other text.</p>
    """
    with suppress_langchain_beta_warning():
        splitter = HTMLSemanticPreservingSplitter(
            headers_to_split_on=[("h1", "Header 1")],
            max_chunk_size=10,
            separators=[". "],
            keep_separator=False,
        )
    documents = splitter.split_text(html_content)

    expected = [
        Document(
            page_content="This is some text",
            metadata={"Header 1": "Section 1"},
        ),
        Document(
            page_content="This is some other text.",
            metadata={"Header 1": "Section 1"},
        ),
    ]

    assert documents == expected


@pytest.mark.requires("bs4")
def test_html_splitter_keep_separator_start() -> None:
    """Test HTML splitting with keep_separator="start"."""
    html_content = """
    <h1>Section 1</h1>
    <p>This is some text. This is some other text.</p>
    """
    with suppress_langchain_beta_warning():
        splitter = HTMLSemanticPreservingSplitter(
            headers_to_split_on=[("h1", "Header 1")],
            max_chunk_size=10,
            separators=[". "],
            keep_separator="start",
        )
    documents = splitter.split_text(html_content)

    expected = [
        Document(
            page_content="This is some text",
            metadata={"Header 1": "Section 1"},
        ),
        Document(
            page_content=". This is some other text.",
            metadata={"Header 1": "Section 1"},
        ),
    ]

    assert documents == expected


@pytest.mark.requires("bs4")
def test_html_splitter_keep_separator_end() -> None:
    """Test HTML splitting with keep_separator="end"."""
    html_content = """
    <h1>Section 1</h1>
    <p>This is some text. This is some other text.</p>
    """
    with suppress_langchain_beta_warning():
        splitter = HTMLSemanticPreservingSplitter(
            headers_to_split_on=[("h1", "Header 1")],
            max_chunk_size=10,
            separators=[". "],
            keep_separator="end",
        )
    documents = splitter.split_text(html_content)

    expected = [
        Document(
            page_content="This is some text.",
            metadata={"Header 1": "Section 1"},
        ),
        Document(
            page_content="This is some other text.",
            metadata={"Header 1": "Section 1"},
        ),
    ]

    assert documents == expected


@pytest.mark.requires("bs4")
def test_html_splitter_keep_separator_default() -> None:
    """Test HTML splitting with keep_separator not set."""
    html_content = """
    <h1>Section 1</h1>
    <p>This is some text. This is some other text.</p>
    """
    with suppress_langchain_beta_warning():
        splitter = HTMLSemanticPreservingSplitter(
            headers_to_split_on=[("h1", "Header 1")],
            max_chunk_size=10,
            separators=[". "],
        )
    documents = splitter.split_text(html_content)

    expected = [
        Document(
            page_content="This is some text",
            metadata={"Header 1": "Section 1"},
        ),
        Document(
            page_content=". This is some other text.",
            metadata={"Header 1": "Section 1"},
        ),
    ]

    assert documents == expected


@pytest.mark.requires("bs4")
def test_html_splitter_preserved_elements_reverse_order() -> None:
    """Test HTML splitter with preserved elements and conflicting placeholders.

    This test validates that preserved elements are reinserted in reverse order
    to prevent conflicts when one placeholder might be a substring of another.
    """
    html_content = """
    <h1>Section 1</h1>
    <table>
        <tr><td>Table 1 content</td></tr>
    </table>
    <p>Some text between tables</p>
    <table>
        <tr><td>Table 10 content</td></tr>
    </table>
    <ul>
        <li>List item 1</li>
        <li>List item 10</li>
    </ul>
    """
    with suppress_langchain_beta_warning():
        splitter = HTMLSemanticPreservingSplitter(
            headers_to_split_on=[("h1", "Header 1")],
            elements_to_preserve=["table", "ul"],
            max_chunk_size=100,
        )
    documents = splitter.split_text(html_content)

    # Verify that all preserved elements are correctly reinserted
    # This would fail if placeholders were processed in forward order
    # when one placeholder is a substring of another
    assert len(documents) >= 1
    # Check that table content is preserved
    content = " ".join(doc.page_content for doc in documents)
    assert "Table 1 content" in content
    assert "Table 10 content" in content
    assert "List item 1" in content
    assert "List item 10" in content


@pytest.mark.requires("bs4")
def test_html_splitter_replacement_order() -> None:
    body = textwrap.dedent(
        """
        <p>Hello1</p>
        <p>Hello2</p>
        <p>Hello3</p>
        <p>Hello4</p>
        <p>Hello5</p>
        <p>Hello6</p>
        <p>Hello7</p>
        <p>Hello8</p>
        <p>Hello9</p>
        <p>Hello10</p>
        <p>Hello11</p>
        <p>Hello12</p>
        <p>Hello13</p>
        <p>Hello14</p>
        """
    )

    with suppress_langchain_beta_warning():
        splitter = HTMLSemanticPreservingSplitter(
            headers_to_split_on=[],
            elements_to_preserve=["p"],
        )
    documents = splitter.split_text(body)
    assert len(documents) == 1
    content = documents[0].page_content
    assert content == " ".join([f"Hello{i}" for i in range(1, 15)])


