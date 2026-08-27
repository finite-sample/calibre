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
_BLOCK = re.compile(
    r"^```python(?P<info>[^\n]*)\n(?P<code>.*?)^```", re.MULTILINE | re.DOTALL
)


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


@pytest.mark.parametrize(
    ("line_no", "code"), BLOCKS, ids=[f"L{line}" for line, _ in BLOCKS]
)
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
    except Exception as exc:
        pytest.fail(
            f"README.md block at line {line_no} failed: "
            f"{type(exc).__name__}: {exc}\n\n{code}"
        )


# Lines of the form `#> expected output` record what the block above them prints.
# The README previously claimed an output produced by a classifier that had been
# deleted, so running the code is not enough -- the claimed numbers must match too.
# `# >` is accepted as well because `ruff format` normalizes comment spacing inside
# Markdown code blocks.
# `[ \t]` rather than `\s` throughout: `\s` matches a newline, so on a bare `#>`
# line recording a blank line of output the optional trailing `\s?` swallowed the
# line break and `.*` captured the *next* line instead. That silently corrupted
# every expectation following a blank one.
_EXPECTED = re.compile(r"^[ \t]*#[ \t]?>[ \t]?(?P<text>.*)$", re.MULTILINE)


@pytest.mark.parametrize(
    ("line_no", "code"), BLOCKS, ids=[f"L{line}" for line, _ in BLOCKS]
)
def test_readme_block_that_prints_declares_its_output(line_no, code):
    """A block that prints must say what it prints.

    Without this, ``test_readme_block_output_matches`` silently skips any block whose
    `#>` annotation was never written -- which is how an earlier strict-increment
    recipe came to
    print ``strictly increasing: False`` directly beneath prose promising the
    opposite. Executing the code is not enough; the claim has to be checked.

    Parameters
    ----------
    line_no
        Line of the opening fence.
    code
        The block's source.
    """
    if "print(" not in code:
        return
    assert _EXPECTED.search(code), (
        f"README.md block at line {line_no} calls print() but declares no expected "
        "output. Add `#> ...` lines recording exactly what it prints, so that "
        "test_readme_block_output_matches verifies them."
    )


@pytest.mark.parametrize(
    ("line_no", "code"),
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
    # Blank lines are dropped from both sides, not just from the printed output.
    # Filtering only one side made it impossible to declare the output of anything
    # that prints a blank line -- a multi-line report, say -- because the `#>`
    # marker recording that blank line had nothing to match against.
    expected = [
        text
        for m in _EXPECTED.finditer(code)
        if (text := m.group("text").rstrip()).strip()
    ]
    namespace: dict[str, object] = {"__name__": "__readme__"}
    exec(compile(code, f"README.md:{line_no}", "exec"), namespace)
    printed = [
        line.rstrip() for line in capsys.readouterr().out.splitlines() if line.strip()
    ]

    assert printed == expected, (
        f"README.md block at line {line_no} claims output it does not produce.\n"
        f"claimed:\n  " + "\n  ".join(expected) + "\nactual:\n  " + "\n  ".join(printed)
    )


def test_a_blank_expectation_does_not_swallow_the_next_line():
    """The `#>` extractor must not run two claimed lines together.

    `\\s` matches a newline, so the original pattern's optional trailing `\\s?`
    consumed the line break on a bare `#>` -- the marker for a blank line of
    output -- and `.*` then captured the following line. The corruption was
    silent: the expectation simply became wrong, and any block whose output
    contained a blank line could never be declared correctly.
    """
    code = "print(x)\n#> first\n#>\n#> second\n"
    assert [m.group("text").rstrip() for m in _EXPECTED.finditer(code)] == [
        "first",
        "",
        "second",
    ]


def test_readme_names_only_functions_that_exist():
    """Every ``plot_*`` / ``calibre.*`` name the README claims must be importable.

    Prose is not executed, so a plausible-sounding function name in a paragraph is
    invisible to the block tests above. Three invented names shipped in a draft of
    the plotting section this way.
    """
    import calibre
    import calibre.plots

    text = README.read_text()
    claimed = set(re.findall(r"`(plot_\w+)`", text))
    assert claimed, "no plot_* names found; the pattern has gone stale"

    exported = set(calibre.plots.__all__)
    assert claimed <= exported, (
        f"README names functions that calibre.plots does not export: "
        f"{sorted(claimed - exported)}"
    )
