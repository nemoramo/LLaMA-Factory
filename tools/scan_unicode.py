#!/usr/bin/env python3

from __future__ import annotations

import pathlib
import unicodedata


def _is_suspicious(ch: str) -> bool:
    """Return True for Unicode chars that are typically invisible/dangerous in source code.

    GitHub warns on "hidden/bidirectional Unicode text" primarily for format/bidi control chars.
    This script flags:
      - All Unicode "Cf" (format) characters, including bidi marks and many zero-width chars
      - Non-ASCII whitespace (e.g. NBSP) which can be hard to spot in diffs/editors
      - Line/paragraph separators that can break tooling unexpectedly
    """
    cat = unicodedata.category(ch)
    if cat == "Cf":
        return True

    if ord(ch) > 0x7F and ch.isspace():
        return True

    # Line/paragraph separators (not ASCII, can appear invisible in some editors)
    if ch in ("\u2028", "\u2029"):
        return True

    return False


def main() -> int:
    root = pathlib.Path("src/llamafactory")
    if not root.exists():
        print(f"Not found: {root}")
        return 2

    findings: list[tuple[str, int, int, str, str]] = []
    for path in sorted(root.rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="strict")
        line = 1
        col = 0
        for ch in text:
            if ch == "\n":
                line += 1
                col = 0
                continue
            col += 1
            if _is_suspicious(ch):
                findings.append(
                    (
                        str(path),
                        line,
                        col,
                        f"U+{ord(ch):04X}",
                        unicodedata.name(ch, "UNKNOWN"),
                    )
                )

    if findings:
        print("Found suspicious unicode characters (hidden/bidi/format/whitespace):")
        for file, line, col, code, name in findings[:200]:
            print(f"{file}:{line}:{col}: {code} {name}")
        if len(findings) > 200:
            print(f"... and {len(findings) - 200} more")
        return 1

    print("OK: no suspicious unicode characters found in src/llamafactory/**/*.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

