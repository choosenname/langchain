"""Character and base text splitter tests."""

from __future__ import annotations

import re

import pytest
from langchain_core.documents import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_text_splitters.character import CharacterTextSplitter


def test_character_text_splitter() -> None:
    """Test splitting by character count."""
    text = "foo bar baz 123"
    splitter = CharacterTextSplitter(separator=" ", chunk_size=7, chunk_overlap=3)
    output = splitter.split_text(text)
    expected_output = ["foo bar", "bar baz", "baz 123"]
    assert output == expected_output


def test_character_text_splitter_empty_doc() -> None:
    """Test splitting by character count doesn't create empty documents."""
    text = "foo  bar"
    splitter = CharacterTextSplitter(separator=" ", chunk_size=2, chunk_overlap=0)
    output = splitter.split_text(text)
    expected_output = ["foo", "bar"]
    assert output == expected_output


def test_character_text_splitter_separtor_empty_doc() -> None:
    """Test edge cases are separators."""
    text = "f b"
    splitter = CharacterTextSplitter(separator=" ", chunk_size=2, chunk_overlap=0)
    output = splitter.split_text(text)
    expected_output = ["f", "b"]
    assert output == expected_output


def test_character_text_splitter_long() -> None:
    """Test splitting by character count on long words."""
    text = "foo bar baz a a"
    splitter = CharacterTextSplitter(separator=" ", chunk_size=3, chunk_overlap=1)
    output = splitter.split_text(text)
    expected_output = ["foo", "bar", "baz", "a a"]
    assert output == expected_output


def test_character_text_splitter_short_words_first() -> None:
    """Test splitting by character count when shorter words are first."""
    text = "a a foo bar baz"
    splitter = CharacterTextSplitter(separator=" ", chunk_size=3, chunk_overlap=1)
    output = splitter.split_text(text)
    expected_output = ["a a", "foo", "bar", "baz"]
    assert output == expected_output


def test_character_text_splitter_longer_words() -> None:
    """Test splitting by characters when splits not found easily."""
    text = "foo bar baz 123"
    splitter = CharacterTextSplitter(separator=" ", chunk_size=1, chunk_overlap=1)
    output = splitter.split_text(text)
    expected_output = ["foo", "bar", "baz", "123"]
    assert output == expected_output


# edge cases
def test_character_text_splitter_no_separator_in_text() -> None:
    """Text splitting where there is no separator but a single word."""
    text = "singleword"
    splitter = CharacterTextSplitter(separator=" ", chunk_size=10, chunk_overlap=0)
    output = splitter.split_text(text)
    expected_output = ["singleword"]
    assert output == expected_output


def test_character_text_splitter_handle_chunksize_equal_to_chunkoverlap() -> None:
    """Text splitting safe guards when chunk size is equal chunk overlap."""
    text = "hello"
    splitter = CharacterTextSplitter(separator=" ", chunk_size=5, chunk_overlap=5)
    output = splitter.split_text(text)
    expected_output = ["hello"]
    assert output == expected_output


def test_character_text_splitter_empty_input() -> None:
    """Test splitting safely where there is no input to process."""
    text = ""
    splitter = CharacterTextSplitter(separator=" ", chunk_size=5, chunk_overlap=0)
    output = splitter.split_text(text)
    expected_output: list[str] = []
    assert output == expected_output


def test_character_text_splitter_whitespace_only() -> None:
    """Test splitting safely where there is white space."""
    text = " "
    splitter = CharacterTextSplitter(separator=" ", chunk_size=5, chunk_overlap=0)
    output = splitter.split_text(text)
    expected_output: list[str] = []
    assert output == expected_output


@pytest.mark.parametrize(
    ("separator", "is_separator_regex"), [(re.escape("."), True), (".", False)]
)
def test_character_text_splitter_keep_separator_regex(
    *, separator: str, is_separator_regex: bool
) -> None:
    """Test CharacterTextSplitter keep separator regex.

    Test splitting by characters while keeping the separator
    that is a regex special character.
    """
    text = "foo.bar.baz.123"
    splitter = CharacterTextSplitter(
        separator=separator,
        chunk_size=1,
        chunk_overlap=0,
        keep_separator=True,
        is_separator_regex=is_separator_regex,
    )
    output = splitter.split_text(text)
    expected_output = ["foo", ".bar", ".baz", ".123"]
    assert output == expected_output


@pytest.mark.parametrize(
    ("separator", "is_separator_regex"), [(re.escape("."), True), (".", False)]
)
def test_character_text_splitter_keep_separator_regex_start(
    *, separator: str, is_separator_regex: bool
) -> None:
    """Test CharacterTextSplitter keep separator regex and put at start.

    Test splitting by characters while keeping the separator
    that is a regex special character and placing it at the start of each chunk.
    """
    text = "foo.bar.baz.123"
    splitter = CharacterTextSplitter(
        separator=separator,
        chunk_size=1,
        chunk_overlap=0,
        keep_separator="start",
        is_separator_regex=is_separator_regex,
    )
    output = splitter.split_text(text)
    expected_output = ["foo", ".bar", ".baz", ".123"]
    assert output == expected_output


@pytest.mark.parametrize(
    ("separator", "is_separator_regex"), [(re.escape("."), True), (".", False)]
)
def test_character_text_splitter_keep_separator_regex_end(
    *, separator: str, is_separator_regex: bool
) -> None:
    """Test CharacterTextSplitter keep separator regex and put at end.

    Test splitting by characters while keeping the separator
    that is a regex special character and placing it at the end of each chunk.
    """
    text = "foo.bar.baz.123"
    splitter = CharacterTextSplitter(
        separator=separator,
        chunk_size=1,
        chunk_overlap=0,
        keep_separator="end",
        is_separator_regex=is_separator_regex,
    )
    output = splitter.split_text(text)
    expected_output = ["foo.", "bar.", "baz.", "123"]
    assert output == expected_output


@pytest.mark.parametrize(
    ("separator", "is_separator_regex"), [(re.escape("."), True), (".", False)]
)
def test_character_text_splitter_discard_separator_regex(
    *, separator: str, is_separator_regex: bool
) -> None:
    """Test CharacterTextSplitter discard separator regex.

    Test splitting by characters discarding the separator
    that is a regex special character.
    """
    text = "foo.bar.baz.123"
    splitter = CharacterTextSplitter(
        separator=separator,
        chunk_size=1,
        chunk_overlap=0,
        keep_separator=False,
        is_separator_regex=is_separator_regex,
    )
    output = splitter.split_text(text)
    expected_output = ["foo", "bar", "baz", "123"]
    assert output == expected_output


def test_recursive_character_text_splitter_keep_separators() -> None:
    split_tags = [",", "."]
    query = "Apple,banana,orange and tomato."
    # start
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10,
        chunk_overlap=0,
        separators=split_tags,
        keep_separator="start",
    )
    result = splitter.split_text(query)
    assert result == ["Apple", ",banana", ",orange and tomato", "."]

    # end
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10,
        chunk_overlap=0,
        separators=split_tags,
        keep_separator="end",
    )
    result = splitter.split_text(query)
    assert result == ["Apple,", "banana,", "orange and tomato."]


def test_character_text_splitting_args() -> None:
    """Test invalid arguments."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Got a larger chunk overlap (4) than chunk size (2), should be smaller."
        ),
    ):
        CharacterTextSplitter(chunk_size=2, chunk_overlap=4)
    for invalid_size in (0, -1):
        with pytest.raises(ValueError, match="chunk_size must be > 0, got"):
            CharacterTextSplitter(chunk_size=invalid_size)
    with pytest.raises(ValueError, match="chunk_overlap must be >= 0, got -1"):
        CharacterTextSplitter(chunk_size=2, chunk_overlap=-1)


def test_merge_splits() -> None:
    """Test merging splits with a given separator."""
    splitter = CharacterTextSplitter(separator=" ", chunk_size=9, chunk_overlap=2)
    splits = ["foo", "bar", "baz"]
    expected_output = ["foo bar", "baz"]
    output = splitter._merge_splits(splits, separator=" ")
    assert output == expected_output


def test_create_documents() -> None:
    """Test create documents method."""
    texts = ["foo bar", "baz"]
    splitter = CharacterTextSplitter(separator=" ", chunk_size=3, chunk_overlap=0)
    docs = splitter.create_documents(texts)
    expected_docs = [
        Document(page_content="foo"),
        Document(page_content="bar"),
        Document(page_content="baz"),
    ]
    assert docs == expected_docs


def test_create_documents_with_metadata() -> None:
    """Test create documents with metadata method."""
    texts = ["foo bar", "baz"]
    splitter = CharacterTextSplitter(separator=" ", chunk_size=3, chunk_overlap=0)
    docs = splitter.create_documents(texts, [{"source": "1"}, {"source": "2"}])
    expected_docs = [
        Document(page_content="foo", metadata={"source": "1"}),
        Document(page_content="bar", metadata={"source": "1"}),
        Document(page_content="baz", metadata={"source": "2"}),
    ]
    assert docs == expected_docs


@pytest.mark.parametrize(
    ("splitter", "text", "expected_docs"),
    [
        (
            CharacterTextSplitter(
                separator=" ", chunk_size=7, chunk_overlap=3, add_start_index=True
            ),
            "foo bar baz 123",
            [
                Document(page_content="foo bar", metadata={"start_index": 0}),
                Document(page_content="bar baz", metadata={"start_index": 4}),
                Document(page_content="baz 123", metadata={"start_index": 8}),
            ],
        ),
        (
            RecursiveCharacterTextSplitter(
                chunk_size=6,
                chunk_overlap=0,
                separators=["\n\n", "\n", " ", ""],
                add_start_index=True,
            ),
            "w1 w1 w1 w1 w1 w1 w1 w1 w1",
            [
                Document(page_content="w1 w1", metadata={"start_index": 0}),
                Document(page_content="w1 w1", metadata={"start_index": 6}),
                Document(page_content="w1 w1", metadata={"start_index": 12}),
                Document(page_content="w1 w1", metadata={"start_index": 18}),
                Document(page_content="w1", metadata={"start_index": 24}),
            ],
        ),
    ],
)
def test_create_documents_with_start_index(
    splitter: TextSplitter, text: str, expected_docs: list[Document]
) -> None:
    """Test create documents method."""
    docs = splitter.create_documents([text])
    assert docs == expected_docs
    for doc in docs:
        s_i = doc.metadata["start_index"]
        assert text[s_i : s_i + len(doc.page_content)] == doc.page_content


def test_metadata_not_shallow() -> None:
    """Test that metadatas are not shallow."""
    texts = ["foo bar"]
    splitter = CharacterTextSplitter(separator=" ", chunk_size=3, chunk_overlap=0)
    docs = splitter.create_documents(texts, [{"source": "1"}])
    expected_docs = [
        Document(page_content="foo", metadata={"source": "1"}),
        Document(page_content="bar", metadata={"source": "1"}),
    ]
    assert docs == expected_docs
    docs[0].metadata["foo"] = 1
    assert docs[0].metadata == {"source": "1", "foo": 1}
    assert docs[1].metadata == {"source": "1"}


def test_iterative_text_splitter_keep_separator() -> None:
    chunk_size = 5
    output = __test_iterative_text_splitter(chunk_size=chunk_size, keep_separator=True)

    assert output == [
        "....5",
        "X..3",
        "Y...4",
        "X....5",
        "Y...",
    ]


def test_iterative_text_splitter_discard_separator() -> None:
    chunk_size = 5
    output = __test_iterative_text_splitter(chunk_size=chunk_size, keep_separator=False)

    assert output == [
        "....5",
        "..3",
        "...4",
        "....5",
        "...",
    ]


def __test_iterative_text_splitter(
    *, chunk_size: int, keep_separator: bool
) -> list[str]:
    chunk_size += 1 if keep_separator else 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,
        separators=["X", "Y"],
        keep_separator=keep_separator,
    )
    text = "....5X..3Y...4X....5Y..."
    output = splitter.split_text(text)
    for chunk in output:
        assert len(chunk) <= chunk_size, f"Chunk is larger than {chunk_size}"
    return output


def test_iterative_text_splitter() -> None:
    """Test iterative text splitter."""
    text = """Hi.\n\nI'm Harrison.\n\nHow? Are? You?\nOkay then f f f f.
This is a weird text to write, but gotta test the splittingggg some how.

Bye!\n\n-H."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=1)
    output = splitter.split_text(text)
    expected_output = [
        "Hi.",
        "I'm",
        "Harrison.",
        "How? Are?",
        "You?",
        "Okay then",
        "f f f f.",
        "This is a",
        "weird",
        "text to",
        "write,",
        "but gotta",
        "test the",
        "splitting",
        "gggg",
        "some how.",
        "Bye!",
        "-H.",
    ]
    assert output == expected_output


def test_split_documents() -> None:
    """Test split_documents."""
    splitter = CharacterTextSplitter(separator="", chunk_size=1, chunk_overlap=0)
    docs = [
        Document(page_content="foo", metadata={"source": "1"}),
        Document(page_content="bar", metadata={"source": "2"}),
        Document(page_content="baz", metadata={"source": "1"}),
    ]
    expected_output = [
        Document(page_content="f", metadata={"source": "1"}),
        Document(page_content="o", metadata={"source": "1"}),
        Document(page_content="o", metadata={"source": "1"}),
        Document(page_content="b", metadata={"source": "2"}),
        Document(page_content="a", metadata={"source": "2"}),
        Document(page_content="r", metadata={"source": "2"}),
        Document(page_content="b", metadata={"source": "1"}),
        Document(page_content="a", metadata={"source": "1"}),
        Document(page_content="z", metadata={"source": "1"}),
    ]
    assert splitter.split_documents(docs) == expected_output


def test_character_text_splitter_discard_regex_separator_on_merge() -> None:
    """Test that regex lookahead separator is not re-inserted when merging."""
    text = "SCE191 First chunk. SCE103 Second chunk."
    splitter = CharacterTextSplitter(
        separator=r"(?=SCE\d{3})",
        is_separator_regex=True,
        chunk_size=200,
        chunk_overlap=0,
        keep_separator=False,
    )
    output = splitter.split_text(text)
    assert output == ["SCE191 First chunk. SCE103 Second chunk."]


@pytest.mark.parametrize(
    ("separator", "is_regex", "text", "chunk_size", "expected"),
    [
        # 1) regex lookaround & split happens
        #   "abcmiddef" split by "(?<=mid)" → ["abcmid","def"], chunk_size=5 keeps both
        (r"(?<=mid)", True, "abcmiddef", 5, ["abcmid", "def"]),
        # 2) regex lookaround & no split
        #   chunk_size=100 merges back into ["abcmiddef"]
        (r"(?<=mid)", True, "abcmiddef", 100, ["abcmiddef"]),
        # 3) literal separator & split happens
        #   split on "mid" → ["abc","def"], chunk_size=3 keeps both
        ("mid", False, "abcmiddef", 3, ["abc", "def"]),
        # 4) literal separator & no split
        #   chunk_size=100 merges back into ["abcmiddef"]
        ("mid", False, "abcmiddef", 100, ["abcmiddef"]),
    ],
)
def test_character_text_splitter_chunk_size_effect(
    separator: str,
    *,
    is_regex: bool,
    text: str,
    chunk_size: int,
    expected: list[str],
) -> None:
    splitter = CharacterTextSplitter(
        separator=separator,
        is_separator_regex=is_regex,
        chunk_size=chunk_size,
        chunk_overlap=0,
        keep_separator=False,
    )
    assert splitter.split_text(text) == expected
