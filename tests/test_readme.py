"""Execute every Python code block in README.md.

The README shipped for several releases with examples that could not run: a
`TypeError` from treating a dict as a list, a claimed output from a classifier
that had been deleted, and constructor keywords that no longer existed. None of
that was caught, because nothing ever ran the examples.

Each block runs in a **fresh namespace**, so a block that silently depends on a
variable defined in an earlier block fails here. That is deliberate: readers copy
one block, not the whole file.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parent.parent / "README.md"

# Fenced blocks tagged `python`. Blocks tagged `bash`, `text`, `toml` etc. are
# skipped, as is any block whose info string carries `skip-test` -- used for
# snippets that need network access or a user's own model.
_BLOCK = re.compile(r"^```python(?P<info>[^\n]*)\n(?P<code>.*?)^```", re.M | re.S)


def _blocks() -> list[tuple[int, str]]:
    """Extract runnable Python blocks with their 1-based line numbers.

    Returns
    -------
    list of (int, str)
        Line number of the opening fence and the block's source.
    """
    if not README.exists():  # pragma: no cover - README is part of the repo
        return []
    text = README.read_text()
    out = []
    for match in _BLOCK.finditer(text):
        if "skip-test" in match.group("info"):
            continue
        line_no = text.count("\n", 0, match.start()) + 1
        out.append((line_no, match.group("code")))
    return out


BLOCKS = _blocks()


def test_readme_has_python_blocks():
    """Guard against the extractor silently matching nothing."""
    assert BLOCKS, "no runnable python blocks found in README.md"


@pytest.mark.parametrize("line_no,code", BLOCKS, ids=[f"L{line}" for line, _ in BLOCKS])
def test_readme_block_runs(line_no, code):
    """Every README block must execute standalone without error.

    Parameters
    ----------
    line_no
        Line of the opening fence, so a failure points at the right place.
    code
        The block's source.
    """
    namespace: dict[str, object] = {"__name__": "__readme__"}
    try:
        exec(compile(code, f"README.md:{line_no}", "exec"), namespace)
    except Exception as exc:  # noqa: BLE001 - want the original error surfaced
        pytest.fail(
            f"README.md block at line {line_no} failed: "
            f"{type(exc).__name__}: {exc}\n\n{code}"
        )


# Lines of the form `#> expected output` record what the block above them prints.
# The README previously claimed an output produced by a classifier that had been
# deleted, so running the code is not enough -- the claimed numbers must match too.
_EXPECTED = re.compile(r"^\s*#>\s?(?P<text>.*)$", re.M)


@pytest.mark.parametrize(
    "line_no,code",
    [(line, code) for line, code in BLOCKS if _EXPECTED.search(code)],
    ids=[f"L{line}" for line, code in BLOCKS if _EXPECTED.search(code)],
)
def test_readme_block_output_matches(line_no, code, capsys):
    """A block annotated with `#>` lines must actually print them.

    Parameters
    ----------
    line_no
        Line of the opening fence.
    code
        The block's source.
    capsys
        pytest stdout capture.
    """
    expected = [m.group("text").rstrip() for m in _EXPECTED.finditer(code)]
    namespace: dict[str, object] = {"__name__": "__readme__"}
    exec(compile(code, f"README.md:{line_no}", "exec"), namespace)
    printed = [
        line.rstrip() for line in capsys.readouterr().out.splitlines() if line.strip()
    ]

    assert printed == expected, (
        f"README.md block at line {line_no} claims output it does not produce.\n"
        f"claimed:\n  " + "\n  ".join(expected) + "\nactual:\n  " + "\n  ".join(printed)
    )
