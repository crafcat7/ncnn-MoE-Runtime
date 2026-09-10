#!/usr/bin/env python3
"""Check first-argument placement and clang-format without changing source files."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path


SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".comp"}

# Treat literals and comments as single tokens, including custom raw delimiters.
# This keeps embedded JSON/GLSL and commented-out examples out of the check.
LEXEMES = (
    r'(?P<raw>(?:u8|u|U|L)?R"(?P<delimiter>[^\s()\\]{0,16})\([\s\S]*?\)(?P=delimiter)")'
    r'|(?P<comment>//(?:\\\r?\n|[^\n])*|/\*[\s\S]*?\*/)'
    r'|(?P<literal>(?:u8|u|U|L)?(?:"(?:\\[\s\S]|[^"\\])*"|\'(?:\\[\s\S]|[^\'\\])*\'))'
    r'|(?P<number>(?:\d|\.\d)(?:[\w.]|\'\w|(?<=[eEpP])[+-])*)'
    r'|(?P<word>[A-Za-z_]\w*)'
    r'|(?P<splice>\\\r?\n)'
    r'|(?P<punctuation>[^\s])'
)
TOKENS = re.compile(r'(?P<directive>^[ \t]*\#(?:\\\r?\n|[^\n])*)|' + LEXEMES, re.MULTILINE)
MACRO_TOKENS = re.compile(LEXEMES, re.MULTILINE)


def source_tokens(text: str):
    for token in TOKENS.finditer(text):
        if token.lastgroup == "directive" and re.match(r"[ \t]*#[ \t]*define\b", token[0]):
            # Check macro bodies too; escaped newlines retain their physical lines.
            start = text.index("#", token.start(), token.end()) + 1
            yield from MACRO_TOKENS.finditer(text, start, token.end())
        else:
            yield token


def first_argument_breaks(text: str) -> list[tuple[int, int]]:
    errors = []
    opening = None
    for token in source_tokens(text):
        if token.lastgroup == "splice":
            continue
        if opening is not None and token.lastgroup not in {"comment", "directive"} and token[0] != ")":
            if "\n" in text[opening.end():token.start()]:
                offset = opening.start()
                line = text.count("\n", 0, offset) + 1
                column = offset - text.rfind("\n", 0, offset)
                errors.append((line, column))
        opening = token if token.lastgroup == "punctuation" and token[0] == "(" else None
    return errors


def source_files(root: Path) -> list[Path]:
    return sorted(path for directory in ("include", "src", "examples", "tests")
                  for path in (root / directory).rglob("*")
                  if path.is_file() and path.suffix in SOURCE_SUFFIXES)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--clang-format", default="clang-format")
    args = parser.parse_args()
    root = args.source_dir.resolve()
    files = source_files(root)
    if not files:
        parser.error(f"no C/C++ or shader sources found in {root}")

    failed = False
    for path in files:
        for line, column in first_argument_breaks(path.read_text(encoding="utf-8")):
            print(f"{path.relative_to(root)}:{line}:{column}: "
                  "keep the first argument on the opening-parenthesis line", file=sys.stderr)
            failed = True

    formatter = shutil.which(args.clang_format)
    if formatter is None:
        print(f"{args.clang_format} was not found; install clang-format 10 or newer", file=sys.stderr)
        return 1

    # clang-format owns continuation alignment; avoid duplicating its C++ parser.
    # Small batches keep the command below the Windows command-line limit.
    for start in range(0, len(files), 32):
        result = subprocess.run(
            [formatter, "--style=file", "--dry-run", "--Werror",
             *(str(path.relative_to(root)) for path in files[start:start + 32])],
            cwd=root,
            check=False,
        )
        failed |= result.returncode != 0

    if failed:
        return 1
    print(f"Format check passed: {len(files)} C/C++ and shader files; "
          "first arguments and continuation alignment are correct.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
