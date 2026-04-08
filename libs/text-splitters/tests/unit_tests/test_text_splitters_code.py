"""Language and code text splitter tests."""

from __future__ import annotations

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_text_splitters.jsx import JSFrameworkTextSplitter
from langchain_text_splitters.python import PythonCodeTextSplitter

FAKE_PYTHON_TEXT = """
class Foo:

    def bar():


def foo():

def testing_func():

def bar():
"""


def test_python_text_splitter() -> None:
    splitter = PythonCodeTextSplitter(chunk_size=30, chunk_overlap=0)
    splits = splitter.split_text(FAKE_PYTHON_TEXT)
    split_0 = """class Foo:\n\n    def bar():"""
    split_1 = """def foo():"""
    split_2 = """def testing_func():"""
    split_3 = """def bar():"""
    expected_splits = [split_0, split_1, split_2, split_3]
    assert splits == expected_splits


FAKE_JSX_TEXT = """
import React from 'react';
import OtherComponent from './OtherComponent';

function MyComponent() {
  const [count, setCount] = React.useState(0);

  const handleClick = () => {
    setCount(count + 1);
  };

  return (
    <div>
      <h1>Counter: {count}</h1>
      <button onClick={handleClick}>
        Increment
      </button>
      <OtherComponent />
    </div>
  );
}

export default MyComponent;
"""


def test_jsx_text_splitter() -> None:
    splitter = JSFrameworkTextSplitter(chunk_size=30, chunk_overlap=0)
    splits = splitter.split_text(FAKE_JSX_TEXT)

    expected_splits = [
        (
            "\nimport React from 'react';\n"
            "import OtherComponent from './OtherComponent';\n"
        ),
        "\nfunction MyComponent() {\n  const [count, setCount] = React.useState(0);",
        "\n\n  const handleClick = () => {\n    setCount(count + 1);\n  };",
        "return (",
        "<div>",
        "<h1>Counter: {count}</h1>\n      ",
        "<button onClick={handleClick}>\n        Increment\n      </button>\n      ",
        "<OtherComponent />\n    </div>\n  );\n}\n",
        "export default MyComponent;",
    ]
    assert [s.strip() for s in splits] == [s.strip() for s in expected_splits]


FAKE_VUE_TEXT = """
<template>
  <div>
    <h1>{{ title }}</h1>
    <button @click="increment">
      Count is: {{ count }}
    </button>
  </div>
</template>

<script>
export default {
  data() {
    return {
      title: 'Counter App',
      count: 0
    }
  },
  methods: {
    increment() {
      this.count++
    }
  }
}
</script>

<style>
button {
  color: blue;
}
</style>
"""


def test_vue_text_splitter() -> None:
    splitter = JSFrameworkTextSplitter(chunk_size=30, chunk_overlap=0)
    splits = splitter.split_text(FAKE_VUE_TEXT)

    expected_splits = [
        "<template>",
        "<div>",
        "<h1>{{ title }}</h1>",
        (
            '<button @click="increment">\n      Count is: {{ count }}\n'
            "    </button>\n  </div>\n</template>"
        ),
        "<script>",
        "export",
        (
            " default {\n  data() {\n    return {\n      title: 'Counter App',\n      "
            "count: 0\n    }\n  },\n  methods: {\n    increment() {\n      "
            "this.count++\n    }\n  }\n}\n</script>"
        ),
        "<style>\nbutton {\n  color: blue;\n}\n</style>",
    ]
    assert [s.strip() for s in splits] == [s.strip() for s in expected_splits]


FAKE_SVELTE_TEXT = """
<script>
  let count = 0

  function increment() {
    count += 1
  }
</script>

<main>
  <h1>Counter App</h1>
  <button on:click={increment}>
    Count is: {count}
  </button>
</main>

<style>
  button {
    color: blue;
  }
</style>
"""


def test_svelte_text_splitter() -> None:
    splitter = JSFrameworkTextSplitter(chunk_size=30, chunk_overlap=0)
    splits = splitter.split_text(FAKE_SVELTE_TEXT)

    expected_splits = [
        "<script>\n  let count = 0",
        "\n\n  function increment() {\n    count += 1\n  }\n</script>",
        "<main>",
        "<h1>Counter App</h1>",
        "<button on:click={increment}>\n    Count is: {count}\n  </button>\n</main>",
        "<style>\n  button {\n    color: blue;\n  }\n</style>",
    ]
    assert [s.strip() for s in splits] == [s.strip() for s in expected_splits]


def test_jsx_splitter_separator_not_mutated_across_calls() -> None:
    """Regression test: repeated split_text() calls must not mutate separators.

    Calling split_text() multiple times on the same JSFrameworkTextSplitter
    instance must not grow the internal separator list between calls.

    Before the fix, self._separators was overwritten with the full expanded list
    on every invocation, so a second call would start with the already-expanded
    list and append even more separators.
    """
    splitter = JSFrameworkTextSplitter(chunk_size=30, chunk_overlap=0)

    # Record separator count after constructing (should be 0 - no custom separators)
    initial_sep_count = len(splitter._separators)

    # Call split_text twice; the results should be identical for identical input
    splits_first = splitter.split_text(FAKE_JSX_TEXT)
    splits_second = splitter.split_text(FAKE_JSX_TEXT)

    assert splits_first == splits_second, (
        "split_text() must return identical results on repeated calls with the "
        "same input"
    )
    assert len(splitter._separators) == initial_sep_count, (
        "split_text() must not mutate self._separators between calls"
    )


CHUNK_SIZE = 16


def test_python_code_splitter() -> None:
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.PYTHON, chunk_size=CHUNK_SIZE, chunk_overlap=0
    )
    code = """
def hello_world():
    print("Hello, World!")

# Call the function
hello_world()
    """
    chunks = splitter.split_text(code)
    assert chunks == [
        "def",
        "hello_world():",
        'print("Hello,',
        'World!")',
        "# Call the",
        "function",
        "hello_world()",
    ]


def test_golang_code_splitter() -> None:
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.GO, chunk_size=CHUNK_SIZE, chunk_overlap=0
    )
    code = """
package main

import "fmt"

func helloWorld() {
    fmt.Println("Hello, World!")
}

func main() {
    helloWorld()
}
    """
    chunks = splitter.split_text(code)
    assert chunks == [
        "package main",
        'import "fmt"',
        "func",
        "helloWorld() {",
        'fmt.Println("He',
        "llo,",
        'World!")',
        "}",
        "func main() {",
        "helloWorld()",
        "}",
    ]


def test_rst_code_splitter() -> None:
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.RST, chunk_size=CHUNK_SIZE, chunk_overlap=0
    )
    code = """
Sample Document
===============

Section
-------

This is the content of the section.

Lists
-----

- Item 1
- Item 2
- Item 3

Comment
*******
Not a comment

.. This is a comment
    """
    chunks = splitter.split_text(code)
    assert chunks == [
        "Sample Document",
        "===============",
        "Section",
        "-------",
        "This is the",
        "content of the",
        "section.",
        "Lists",
        "-----",
        "- Item 1",
        "- Item 2",
        "- Item 3",
        "Comment",
        "*******",
        "Not a comment",
        ".. This is a",
        "comment",
    ]
    # Special test for special characters
    code = "harry\n***\nbabylon is"
    chunks = splitter.split_text(code)
    assert chunks == ["harry", "***\nbabylon is"]


def test_proto_file_splitter() -> None:
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.PROTO, chunk_size=CHUNK_SIZE, chunk_overlap=0
    )
    code = """
syntax = "proto3";

package example;

message Person {
    string name = 1;
    int32 age = 2;
    repeated string hobbies = 3;
}
    """
    chunks = splitter.split_text(code)
    assert chunks == [
        "syntax =",
        '"proto3";',
        "package",
        "example;",
        "message Person",
        "{",
        "string name",
        "= 1;",
        "int32 age =",
        "2;",
        "repeated",
        "string hobbies",
        "= 3;",
        "}",
    ]


def test_javascript_code_splitter() -> None:
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.JS, chunk_size=CHUNK_SIZE, chunk_overlap=0
    )
    code = """
function helloWorld() {
  console.log("Hello, World!");
}

// Call the function
helloWorld();
    """
    chunks = splitter.split_text(code)
    assert chunks == [
        "function",
        "helloWorld() {",
        'console.log("He',
        "llo,",
        'World!");',
        "}",
        "// Call the",
        "function",
        "helloWorld();",
    ]


def test_cobol_code_splitter() -> None:
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.COBOL, chunk_size=CHUNK_SIZE, chunk_overlap=0
    )
    code = """
IDENTIFICATION DIVISION.
PROGRAM-ID. HelloWorld.
DATA DIVISION.
WORKING-STORAGE SECTION.
01 GREETING           PIC X(12)   VALUE 'Hello, World!'.
PROCEDURE DIVISION.
DISPLAY GREETING.
STOP RUN.
    """
    chunks = splitter.split_text(code)
    assert chunks == [
        "IDENTIFICATION",
        "DIVISION.",
        "PROGRAM-ID.",
        "HelloWorld.",
        "DATA DIVISION.",
        "WORKING-STORAGE",
        "SECTION.",
        "01 GREETING",
        "PIC X(12)",
        "VALUE 'Hello,",
        "World!'.",
        "PROCEDURE",
        "DIVISION.",
        "DISPLAY",
        "GREETING.",
        "STOP RUN.",
    ]


def test_typescript_code_splitter() -> None:
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.TS, chunk_size=CHUNK_SIZE, chunk_overlap=0
    )
    code = """
function helloWorld(): void {
  console.log("Hello, World!");
}

// Call the function
helloWorld();
    """
    chunks = splitter.split_text(code)
    assert chunks == [
        "function",
        "helloWorld():",
        "void {",
        'console.log("He',
        "llo,",
        'World!");',
        "}",
        "// Call the",
        "function",
        "helloWorld();",
    ]


def test_java_code_splitter() -> None:
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.JAVA, chunk_size=CHUNK_SIZE, chunk_overlap=0
    )
    code = """
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}
    """
    chunks = splitter.split_text(code)
    assert chunks == [
        "public class",
        "HelloWorld {",
        "public",
        "static void",
        "main(String[]",
        "args) {",
        "System.out.prin",
        'tln("Hello,',
        'World!");',
        "}\n}",
    ]


def test_kotlin_code_splitter() -> None:
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.KOTLIN, chunk_size=CHUNK_SIZE, chunk_overlap=0
    )
    code = """
class HelloWorld {
    companion object {
        @JvmStatic
        fun main(args: Array<String>) {
            println("Hello, World!")
        }
    }
}
    """
    chunks = splitter.split_text(code)
    assert chunks == [
        "class",
        "HelloWorld {",
        "companion",
        "object {",
        "@JvmStatic",
        "fun",
        "main(args:",
        "Array<String>)",
        "{",
        'println("Hello,',
        'World!")',
        "}\n    }",
        "}",
    ]


def test_csharp_code_splitter() -> None:
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.CSHARP, chunk_size=CHUNK_SIZE, chunk_overlap=0
    )
    code = """
using System;
class Program
{
    static void Main()
    {
        int age = 30; // Change the age value as needed

        // Categorize the age without any console output
        if (age < 18)
        {
            // Age is under 18
        }
        else if (age >= 18 && age < 65)
        {
            // Age is an adult
        }
        else
        {
            // Age is a senior citizen
        }
    }
}
    """

    chunks = splitter.split_text(code)
    assert chunks == [
        "using System;",
        "class Program\n{",
        "static void",
        "Main()",
        "{",
        "int age",
        "= 30; // Change",
        "the age value",
        "as needed",
        "//",
        "Categorize the",
        "age without any",
        "console output",
        "if (age",
        "< 18)",
        "{",
        "//",
        "Age is under 18",
        "}",
        "else if",
        "(age >= 18 &&",
        "age < 65)",
        "{",
        "//",
        "Age is an adult",
        "}",
        "else",
        "{",
        "//",
        "Age is a senior",
        "citizen",
        "}\n    }",
        "}",
    ]


def test_cpp_code_splitter() -> None:
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.CPP, chunk_size=CHUNK_SIZE, chunk_overlap=0
    )
    code = """
#include <iostream>

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
    """
    chunks = splitter.split_text(code)
    assert chunks == [
        "#include",
        "<iostream>",
        "int main() {",
        "std::cout",
        '<< "Hello,',
        'World!" <<',
        "std::endl;",
        "return 0;\n}",
    ]


def test_scala_code_splitter() -> None:
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.SCALA, chunk_size=CHUNK_SIZE, chunk_overlap=0
    )
    code = """
object HelloWorld {
  def main(args: Array[String]): Unit = {
    println("Hello, World!")
  }
}
    """
    chunks = splitter.split_text(code)
    assert chunks == [
        "object",
        "HelloWorld {",
        "def",
        "main(args:",
        "Array[String]):",
        "Unit = {",
        'println("Hello,',
        'World!")',
        "}\n}",
    ]


def test_ruby_code_splitter() -> None:
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.RUBY, chunk_size=CHUNK_SIZE, chunk_overlap=0
    )
    code = """
def hello_world
  puts "Hello, World!"
end

hello_world
    """
    chunks = splitter.split_text(code)
    assert chunks == [
        "def hello_world",
        'puts "Hello,',
        'World!"',
        "end",
        "hello_world",
    ]


def test_php_code_splitter() -> None:
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.PHP, chunk_size=CHUNK_SIZE, chunk_overlap=0
    )
    code = """
<?php
function hello_world() {
    echo "Hello, World!";
}

hello_world();
?>
    """
    chunks = splitter.split_text(code)
    assert chunks == [
        "<?php",
        "function",
        "hello_world() {",
        "echo",
        '"Hello,',
        'World!";',
        "}",
        "hello_world();",
        "?>",
    ]


def test_swift_code_splitter() -> None:
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.SWIFT, chunk_size=CHUNK_SIZE, chunk_overlap=0
    )
    code = """
func helloWorld() {
    print("Hello, World!")
}

helloWorld()
    """
    chunks = splitter.split_text(code)
    assert chunks == [
        "func",
        "helloWorld() {",
        'print("Hello,',
        'World!")',
        "}",
        "helloWorld()",
    ]


def test_rust_code_splitter() -> None:
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.RUST, chunk_size=CHUNK_SIZE, chunk_overlap=0
    )
    code = """
fn main() {
    println!("Hello, World!");
}
    """
    chunks = splitter.split_text(code)
    assert chunks == ["fn main() {", 'println!("Hello', ",", 'World!");', "}"]


def test_r_code_splitter() -> None:
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.R, chunk_size=CHUNK_SIZE, chunk_overlap=0
    )
    code = """
library(dplyr)

my_func <- function(x) {
    return(x + 1)
}

if (TRUE) {
    print("Hello")
}
    """
    chunks = splitter.split_text(code)
    assert chunks == [
        "library(dplyr)",
        "my_func <-",
        "function(x) {",
        "return(x +",
        "1)",
        "}",
        "if (TRUE) {",
        'print("Hello")',
        "}",
    ]


def test_markdown_code_splitter() -> None:
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.MARKDOWN, chunk_size=CHUNK_SIZE, chunk_overlap=0
    )
    code = """
# Sample Document

## Section

This is the content of the section.

## Lists

- Item 1
- Item 2
- Item 3

### Horizontal lines

***********
____________
-------------------

#### Code blocks
```
This is a code block

# sample code
a = 1
b = 2
```
    """
    chunks = splitter.split_text(code)
    assert chunks == [
        "# Sample",
        "Document",
        "## Section",
        "This is the",
        "content of the",
        "section.",
        "## Lists",
        "- Item 1",
        "- Item 2",
        "- Item 3",
        "### Horizontal",
        "lines",
        "***********",
        "____________",
        "---------------",
        "----",
        "#### Code",
        "blocks",
        "```",
        "This is a code",
        "block",
        "# sample code",
        "a = 1\nb = 2",
        "```",
    ]
    # Special test for special characters
    code = "harry\n***\nbabylon is"
    chunks = splitter.split_text(code)
    assert chunks == ["harry", "***\nbabylon is"]


def test_latex_code_splitter() -> None:
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.LATEX, chunk_size=CHUNK_SIZE, chunk_overlap=0
    )
    code = """
Hi Harrison!
\\chapter{1}
"""
    chunks = splitter.split_text(code)
    assert chunks == ["Hi Harrison!", "\\chapter{1}"]


def test_html_code_splitter() -> None:
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.HTML, chunk_size=60, chunk_overlap=0
    )
    code = """
<h1>Sample Document</h1>
    <h2>Section</h2>
        <p id="1234">Reference content.</p>

    <h2>Lists</h2>
        <ul>
            <li>Item 1</li>
            <li>Item 2</li>
            <li>Item 3</li>
        </ul>

        <h3>A block</h3>
            <div class="amazing">
                <p>Some text</p>
                <p>Some more text</p>
            </div>
    """
    chunks = splitter.split_text(code)
    assert chunks == [
        "<h1>Sample Document</h1>\n    <h2>Section</h2>",
        '<p id="1234">Reference content.</p>',
        "<h2>Lists</h2>\n        <ul>",
        "<li>Item 1</li>\n            <li>Item 2</li>",
        "<li>Item 3</li>\n        </ul>",
        "<h3>A block</h3>",
        '<div class="amazing">',
        "<p>Some text</p>",
        "<p>Some more text</p>\n            </div>",
    ]


def test_solidity_code_splitter() -> None:
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.SOL, chunk_size=CHUNK_SIZE, chunk_overlap=0
    )
    code = """pragma solidity ^0.8.20;
  contract HelloWorld {
    function add(uint a, uint b) pure public returns(uint) {
      return  a + b;
    }
  }
  """
    chunks = splitter.split_text(code)
    assert chunks == [
        "pragma solidity",
        "^0.8.20;",
        "contract",
        "HelloWorld {",
        "function",
        "add(uint a,",
        "uint b) pure",
        "public",
        "returns(uint) {",
        "return  a",
        "+ b;",
        "}\n  }",
    ]


def test_lua_code_splitter() -> None:
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.LUA, chunk_size=CHUNK_SIZE, chunk_overlap=0
    )
    code = """
local variable = 10

function add(a, b)
    return a + b
end

if variable > 5 then
    for i=1, variable do
        while i < variable do
            repeat
                print(i)
                i = i + 1
            until i >= variable
        end
    end
end
    """
    chunks = splitter.split_text(code)
    assert chunks == [
        "local variable",
        "= 10",
        "function add(a,",
        "b)",
        "return a +",
        "b",
        "end",
        "if variable > 5",
        "then",
        "for i=1,",
        "variable do",
        "while i",
        "< variable do",
        "repeat",
        "print(i)",
        "i = i + 1",
        "until i >=",
        "variable",
        "end",
        "end\nend",
    ]


def test_haskell_code_splitter() -> None:
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.HASKELL, chunk_size=CHUNK_SIZE, chunk_overlap=0
    )
    code = """
        main :: IO ()
        main = do
          putStrLn "Hello, World!"

        -- Some sample functions
        add :: Int -> Int -> Int
        add x y = x + y
    """
    # Adjusted expected chunks to account for indentation and newlines
    expected_chunks = [
        "main ::",
        "IO ()",
        "main = do",
        "putStrLn",
        '"Hello, World!"',
        "--",
        "Some sample",
        "functions",
        "add :: Int ->",
        "Int -> Int",
        "add x y = x",
        "+ y",
    ]
    chunks = splitter.split_text(code)
    assert chunks == expected_chunks


def test_powershell_code_splitter_short_code() -> None:
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.POWERSHELL, chunk_size=60, chunk_overlap=0
    )
    code = """
# Check if a file exists
$filePath = "C:\\temp\\file.txt"
if (Test-Path $filePath) {
    # File exists
} else {
    # File does not exist
}
    """

    chunks = splitter.split_text(code)
    assert chunks == [
        '# Check if a file exists\n$filePath = "C:\\temp\\file.txt"',
        "if (Test-Path $filePath) {\n    # File exists\n} else {",
        "# File does not exist\n}",
    ]


def test_powershell_code_splitter_longer_code() -> None:
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.POWERSHELL, chunk_size=60, chunk_overlap=0
    )
    code = """
# Get a list of all processes and export to CSV
$processes = Get-Process
$processes | Export-Csv -Path "C:\\temp\\processes.csv" -NoTypeInformation

# Read the CSV file and display its content
$csvContent = Import-Csv -Path "C:\\temp\\processes.csv"
$csvContent | ForEach-Object {
    $_.ProcessName
}

# End of script
    """

    chunks = splitter.split_text(code)
    assert chunks == [
        "# Get a list of all processes and export to CSV",
        "$processes = Get-Process",
        '$processes | Export-Csv -Path "C:\\temp\\processes.csv"',
        "-NoTypeInformation",
        "# Read the CSV file and display its content",
        '$csvContent = Import-Csv -Path "C:\\temp\\processes.csv"',
        "$csvContent | ForEach-Object {\n    $_.ProcessName\n}",
        "# End of script",
    ]


FAKE_VISUALBASIC6_TEXT = """
Option Explicit

Public Function SumTwoIntegers(ByVal a As Integer, ByVal b As Integer) As Integer
    SumTwoIntegers = a + b
End Function

Public Sub Main()
    Dim i As Integer
    Dim limit As Integer

    i = 0
    limit = 50

    While i < limit
        i = SumTwoIntegers(i, 1)

        If i = limit \\ 2 Then
            MsgBox "Halfway there! i = " & i
        End If
    Wend

    MsgBox "Done! Final value of i: " & i
End Sub
"""


def test_visualbasic6_code_splitter() -> None:
    splitter = RecursiveCharacterTextSplitter.from_language(
        Language.VISUALBASIC6,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=0,
    )
    chunks = splitter.split_text(FAKE_VISUALBASIC6_TEXT)

    assert chunks == [
        "Option Explicit",
        "Public Function",
        "SumTwoIntegers(",
        "ByVal",
        "a As Integer,",
        "ByVal b As",
        "Integer) As",
        "Integer",
        "SumTwoIntegers",
        "= a + b",
        "End Function",
        "Public Sub",
        "Main()",
        "Dim i As",
        "Integer",
        "Dim limit",
        "As Integer",
        "i = 0",
        "limit = 50",
        "While i <",
        "limit",
        "i =",
        "SumTwoIntegers(",
        "i,",
        "1)",
        "If i =",
        "limit \\ 2 Then",
        'MsgBox "Halfway',
        'there! i = " &',
        "i",
        "End If",
        "Wend",
        "MsgBox",
        '"Done! Final',
        'value of i: " &',
        "i",
        "End Sub",
    ]



