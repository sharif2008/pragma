#!/usr/bin/env python3
"""Remove Cursor Co-authored-by lines from stdin (for git filter-branch --msg-filter)."""
from __future__ import annotations

import re
import sys

_CURSOR_COAUTHOR = re.compile(
    r"^Co-authored-by:\s*Cursor\s*<cursoragent@cursor.com>\s*\n?",
    re.IGNORECASE | re.MULTILINE,
)

text = sys.stdin.read()
text = _CURSOR_COAUTHOR.sub("", text)
text = re.sub(r"\n{3,}", "\n\n", text).rstrip()
if text:
    sys.stdout.write(text + "\n")
