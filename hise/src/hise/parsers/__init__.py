"""Format parsers for HiSE structural-address extraction."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from html.parser import HTMLParser

from ..address import StructuralAddress

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def parse_markdown(markdown: str) -> list[tuple[str, StructuralAddress]]:
    """Return one structural address per non-empty Markdown line."""
    result: list[tuple[str, StructuralAddress]] = []
    heading_stack: list[str] = []
    sibling_counts: dict[str, int] = {}
    for position, line in enumerate(markdown.splitlines()):
        value = line.strip()
        if not value:
            continue
        match = _HEADING.match(value)
        if match:
            depth = len(match.group(1))
            heading_stack = heading_stack[: depth - 1]
            heading_stack.append(match.group(2))
            parent_path = "/".join(heading_stack[:-1])
            sibling_counts[parent_path] = sibling_counts.get(parent_path, 0) + 1
            sibling = sibling_counts[parent_path]
        else:
            depth = len(heading_stack)
            parent_path = "/".join(heading_stack)
            sibling_counts[parent_path] = sibling_counts.get(parent_path, 0) + 1
            sibling = sibling_counts[parent_path]
        result.append((value, StructuralAddress(position, depth, StructuralAddress.stable_parent_id(parent_path), sibling)))
    return result


def parse_table(rows: Sequence[Sequence[str]]) -> list[tuple[str, StructuralAddress]]:
    """Represent each table cell with row, column, and table-parent coordinates."""
    result: list[tuple[str, StructuralAddress]] = []
    parent_id = StructuralAddress.stable_parent_id("table")
    position = 0
    for row_index, row in enumerate(rows):
        for column_index, value in enumerate(row):
            result.append((str(value), StructuralAddress(position, 1, parent_id, row_index * 1000 + column_index)))
            position += 1
    return result


def parse_json(value: object, path: str = "root", position: int = 0, depth: int = 0) -> list[tuple[str, StructuralAddress]]:
    """Flatten JSON leaves while retaining key-path depth and parent identity."""
    result: list[tuple[str, StructuralAddress]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            result.extend(parse_json(child, child_path, position, depth + 1))
            position += len(result)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for sibling, child in enumerate(value):
            child_path = f"{path}[{sibling}]"
            result.extend(parse_json(child, child_path, position, depth + 1))
            position += len(result)
    else:
        parent = path.rsplit(".", 1)[0].rsplit("[", 1)[0]
        result.append((str(value), StructuralAddress(position, depth, StructuralAddress.stable_parent_id(parent), 0)))
    return result


class _DOMParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.depth = 0
        self.position = 0
        self.stack: list[str] = []
        self.items: list[tuple[str, StructuralAddress]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.stack.append(tag)
        self.depth += 1

    def handle_endtag(self, tag: str) -> None:
        if self.stack:
            self.stack.pop()
        self.depth = max(0, self.depth - 1)

    def handle_data(self, data: str) -> None:
        value = data.strip()
        if value:
            parent = "/".join(self.stack)
            self.items.append((value, StructuralAddress(self.position, self.depth, StructuralAddress.stable_parent_id(parent), 0)))
            self.position += 1


def parse_html(html: str) -> list[tuple[str, StructuralAddress]]:
    parser = _DOMParser()
    parser.feed(html)
    return parser.items


def parse_json_text(value: str) -> list[tuple[str, StructuralAddress]]:
    return parse_json(json.loads(value))

__all__ = [
    "parse_markdown",
    "parse_table",
    "parse_json",
    "parse_html",
    "parse_json_text",
]
