"""Excessive code-comment verbosity heuristic (issue #143).

Detects *objective volume* signals in commentary a PR introduces, from the
optional unified diffs in :attr:`reviewgate.core.schemas.ChangedFile.patch`:

1. **Oversized consecutive comment blocks** -- a maximal run of consecutive
   newly-added full-line comment lines in one eligible file.
2. **Excessive total comment lines** -- newly-added full-line comment lines
   summed across all eligible files.
3. **Comment-heavy diff** -- the ratio of newly-added comment lines to
   newly-added non-blank source lines, guarded by a minimum sample size.

Explicit non-goals (issue #143): the heuristic never judges whether a
comment is useful, correct, well-written, or who or what wrote it. It
measures comment *volume*, never comment *value*.

Conservative-parsing contract (false negatives preferred):

* Only **added** patch lines are read. Deleted, context, and metadata lines
  (``@@`` hunks, ``+++`` headers, ``\\ No newline`` markers) are ignored, so
  the heuristic can never see repository content outside the PR's added
  lines.
* Only files the categorizer marks ``human_authored`` **and** ``source``,
  written in a supported language (by extension), are analyzed. Docs,
  generated, vendored, minified, snapshot, asset, lockfile, manifest, and
  unknown-language files are skipped.
* A line counts as a comment only when it is an *unmistakable full-line*
  comment (``#``, ``//``, or a ``/* ... */`` block line) at a lexical
  position where a comment can exist. A small single-pass scanner tracks
  string literals, ``/* */`` block comments, and Python triple-quoted
  strings so markers inside string content (``url = "https://example.com"``,
  ``pattern = "#[a-z]+"``) are never miscounted.
* Inline (trailing) comments are **not** counted in this MVP, and neither
  are Python docstrings: docstrings are string literals and may be runtime
  data. Both decisions are documented false-negative trade-offs.
* Blank lines terminate a block: a run of comment lines interrupted by a
  blank, code, or non-added diff line counts as separate, smaller blocks.

Pure: stdlib only, no I/O, no GitHub or LLM dependency (§4.1 boundary).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from pydantic import Field

from ._base import StrictModel
from .config import CodeCommentPolicy
from .schemas import ChangedFile, EngineWarning, FileCategoryRow, WarningSeverity

# Stable warning codes (issue #143). One code per volume dimension; severity
# distinguishes warn vs fail so downstream consumers can dedupe by code, the
# same convention as :mod:`reviewgate.core.size`.
WARN_CODE_OVERSIZED_BLOCK: Final[str] = "oversized_comment_block"
"""A newly-added consecutive full-line comment block reaches a threshold."""

WARN_CODE_EXCESSIVE_LINES: Final[str] = "excessive_comment_lines"
"""Newly-added comment lines across eligible files reach a threshold."""

WARN_CODE_COMMENT_HEAVY: Final[str] = "comment_heavy_diff"
"""The comment-to-source ratio of added lines reaches a threshold."""

_RATIO_PRECISION: Final[int] = 4
"""Decimal places kept for ``comment_ratio``; rounding once keeps evidence
byte-identical across runs and platforms."""

_SEVERITY_FAIL: Final[WarningSeverity] = "high"
_SEVERITY_WARN: Final[WarningSeverity] = "medium"

# --- language profiles ------------------------------------------------------
#
# Extension -> comment syntax profile. Extensions outside this map are not
# analyzed at all: guessing comment syntax for unsupported languages would
# risk exactly the false positives the issue forbids.

_SHELL_EXTENSIONS: Final[frozenset[str]] = frozenset({".sh", ".bash", ".zsh"})
_PYTHON_EXTENSIONS: Final[frozenset[str]] = frozenset({".py"})
_C_LIKE_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {
        ".js",
        ".jsx",
        ".mjs",
        ".cjs",
        ".ts",
        ".tsx",
        ".go",
        ".java",
        ".c",
        ".h",
        ".cpp",
        ".cc",
        ".cxx",
        ".hpp",
        ".hh",
        ".cs",
        ".rs",
    },
)


@dataclass(frozen=True)
class _Profile:
    """Comment-syntax profile applied by the line scanner."""

    hash_comments: bool  # `#` line comments (Python, Shell)
    c_comments: bool  # `//` line comments and `/* */` block comments
    triple_strings: bool  # Python triple-quoted strings


_PYTHON_PROFILE: Final[_Profile] = _Profile(
    hash_comments=True, c_comments=False, triple_strings=True
)
_SHELL_PROFILE: Final[_Profile] = _Profile(
    hash_comments=True, c_comments=False, triple_strings=False
)
_C_LIKE_PROFILE: Final[_Profile] = _Profile(
    hash_comments=False, c_comments=True, triple_strings=False
)


def _profile_for(filename: str) -> _Profile | None:
    """Return the comment profile for a path, or ``None`` if unsupported."""

    base = filename.rsplit("/", 1)[-1]
    dot = base.rfind(".")
    if dot <= 0:
        return None
    ext = base[dot:].lower()
    if ext in _PYTHON_EXTENSIONS:
        return _PYTHON_PROFILE
    if ext in _SHELL_EXTENSIONS:
        return _SHELL_PROFILE
    if ext in _C_LIKE_EXTENSIONS:
        return _C_LIKE_PROFILE
    return None


# --- conservative line scanner ----------------------------------------------


@dataclass
class _ScanState:
    """Lexical state carried across the added lines of one file patch."""

    in_block_comment: bool = False  # inside a `/* ... */` block (C-like)
    triple_delimiter: str | None = None  # inside `"""` / `'''` (Python)


_KIND_COMMENT: Final[str] = "comment"
_KIND_CODE: Final[str] = "code"
_KIND_BLANK: Final[str] = "blank"


def _scan_line(line: str, state: _ScanState, profile: _Profile) -> str:
    """Classify one added line as ``comment`` / ``code`` / ``blank``.

    Updates ``state`` in place so subsequent lines of the same file are
    scanned in the right context. Only whole-line comment forms are ever
    returned as :data:`_KIND_COMMENT`; trailing comments, string content,
    and anything ambiguous fall into :data:`_KIND_CODE` (the conservative
    bucket: non-blank, non-comment lines count as source lines).
    """

    stripped = line.strip()
    if not stripped:
        # A blank line never changes lexical state (a blank inside a block
        # comment or triple-quoted string leaves both open) and terminates
        # comment-block runs at the caller.
        return _KIND_BLANK

    i = 0
    n = len(line)
    while i < n and line[i] in (" ", "\t"):
        i += 1

    if state.triple_delimiter is not None:
        # Inside a Python triple-quoted string: string content is never a
        # comment (docstrings are string literals, possibly runtime data).
        i = _scan_triple_rest(line, i, state)
        if state.triple_delimiter is not None or i >= len(line):
            return _KIND_CODE
        # The string closed mid-line and real content follows; keep scanning
        # so a string reopened on the same line leaves state accurate for
        # later lines. The line itself still counts as code.
        _scan_normal(line, i, state, profile)
        return _KIND_CODE

    if state.in_block_comment:
        close = line.find("*/", i)
        if close == -1:
            return _KIND_COMMENT
        state.in_block_comment = False
        # Code after the close is possible (`*/ int x;`): keep scanning the
        # remainder for state fidelity, but the line is not a full-line
        # comment.
        _scan_normal(line, close + 2, state, profile)
        return _KIND_CODE if line[close + 2 :].strip() else _KIND_COMMENT

    if profile.hash_comments and line[i] == "#":
        if line[i : i + 2] == "#!":
            # Shebang: an interpreter directive, not commentary.
            return _KIND_CODE
        return _KIND_COMMENT

    if profile.c_comments and line[i : i + 2] == "//":
        return _KIND_COMMENT

    if profile.c_comments and line[i : i + 2] == "/*":
        close = line.find("*/", i + 2)
        if close == -1:
            state.in_block_comment = True
            return _KIND_COMMENT
        if not line[close + 2 :].strip():
            return _KIND_COMMENT
        _scan_normal(line, close + 2, state, profile)
        return _KIND_CODE

    return _scan_normal(line, i, state, profile)


def _scan_normal(line: str, i: int, state: _ScanState, profile: _Profile) -> str:
    """Walk a line from index ``i`` through normal (non-comment) context.

    Consumes string literals so comment markers inside them are ignored,
    opens ``/* */`` blocks and Python triple-quoted strings that continue
    onto later lines, and always returns :data:`_KIND_CODE` -- by the time
    this function runs, the line has already shown real (non-comment)
    content or its leading token was not an unmistakable comment form.
    """

    n = len(line)
    quote: str | None = None  # active single-line quote character
    while i < n:
        ch = line[i]
        if quote is not None:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in ("\"", "'"):
            if profile.triple_strings:
                delimiter = line[i : i + 3]
                if delimiter in ("\"\"\"", "'''"):
                    end = _find_triple_close(line, i + 3, delimiter)
                    if end == -1:
                        state.triple_delimiter = delimiter
                        return _KIND_CODE
                    i = end
                    continue
            quote = ch
            i += 1
            continue
        if profile.c_comments and ch == "/":
            if line[i : i + 2] == "//":
                # Trailing line comment: consumed, but not a full-line
                # comment, so the line stays in the code bucket.
                return _KIND_CODE
            if line[i : i + 2] == "/*":
                close = line.find("*/", i + 2)
                if close == -1:
                    state.in_block_comment = True
                    return _KIND_CODE
                i = close + 2
                continue
        if not profile.hash_comments and ch == "`":
            # Go raw strings / JS template literals: scan to the closing
            # backtick; escapes are not honored inside raw strings but the
            # difference is immaterial for comment detection.
            close = line.find("`", i + 1)
            i = n if close == -1 else close + 1
            continue
        i += 1
    return _KIND_CODE


def _find_triple_close(line: str, start: int, delimiter: str) -> int:
    """Index just past the first unescaped ``delimiter`` at/after ``start``.

    Returns ``-1`` when the delimiter does not terminate on this line, in
    which case the caller keeps the triple-quoted state open.
    """

    i = start
    n = len(line)
    while i < n:
        if line[i] == "\\":
            i += 2
            continue
        if line.startswith(delimiter, i):
            return i + len(delimiter)
        i += 1
    return -1


def _scan_triple_rest(line: str, i: int, state: _ScanState) -> int:
    """Consume the rest of a line inside a Python triple-quoted string.

    Returns the index to continue scanning from. If the string closes on
    this line, ``state`` returns to normal and the caller re-scans the
    remainder (which may open another string) so lexical state stays
    accurate for later lines.
    """

    delimiter = state.triple_delimiter
    if delimiter is None:  # pragma: no cover - guarded by caller
        return i
    end = _find_triple_close(line, i, delimiter)
    if end == -1:
        return len(line)
    state.triple_delimiter = None
    return end


# --- diff processing and aggregation -----------------------------------------


@dataclass(frozen=True)
class _FileTally:
    """Per-file added-line counts from one patch."""

    comment_lines: int
    code_lines: int
    largest_block: int


def _added_lines(patch: str) -> list[str | None]:
    """Extract added line contents from a unified diff patch.

    Returns one entry per patch line: the content of lines the PR adds,
    and ``None`` for every line that represents a real pre-existing file
    line or a hunk boundary (context lines, ``@@`` headers, ``+++``
    metadata, ``\\ No newline`` markers). Deleted lines are dropped
    entirely -- the PR removes them, so the surrounding added lines end up
    adjacent in the resulting file. Callers use ``None`` to terminate
    consecutive-comment-block runs, which keeps block counting about lines
    that are actually adjacent in the new file and makes the heuristic
    unable to observe anything outside the PR's added lines.
    """

    extracted: list[str | None] = []
    for line in patch.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            extracted.append(line[1:])
        elif line.startswith("-") and not line.startswith("---"):
            continue
        else:
            extracted.append(None)
    return extracted


def _tally_patch(patch: str, profile: _Profile) -> _FileTally:
    """Count added comment / code lines and the largest block in one patch."""

    state = _ScanState()
    comment_lines = 0
    code_lines = 0
    current_block = 0
    largest_block = 0
    for content in _added_lines(patch):
        if content is None:
            current_block = 0
            continue
        kind = _scan_line(content, state, profile)
        if kind == _KIND_BLANK:
            current_block = 0
        elif kind == _KIND_COMMENT:
            comment_lines += 1
            current_block += 1
            if current_block > largest_block:
                largest_block = current_block
        else:
            code_lines += 1
            current_block = 0
    return _FileTally(
        comment_lines=comment_lines,
        code_lines=code_lines,
        largest_block=largest_block,
    )


class CommentStats(StrictModel):
    """PR-level added-comment volume totals (issue #143 ``CommentStats``).

    ``comment_ratio`` is ``comment_lines_added / source_lines_added``
    rounded to :data:`_RATIO_PRECISION` places, with ``0.0`` when no
    non-blank lines were added; ``source_lines_added`` is the sum of
    ``comment_lines_added`` and ``code_lines_added``.
    """

    comment_lines_added: int = Field(
        ge=0,
        description="Newly-added full-line comment lines across eligible files.",
    )
    code_lines_added: int = Field(
        ge=0,
        description="Newly-added non-blank lines in eligible files that are not comments.",
    )
    largest_comment_block_lines: int = Field(
        ge=0,
        description="Largest consecutive full-line comment block in any eligible file.",
    )
    comment_ratio: float = Field(
        ge=0.0,
        le=1.0,
        description="comment_lines_added / source_lines_added, rounded deterministically.",
    )


def _eligible(file: ChangedFile, row: FileCategoryRow) -> _Profile | None:
    """Return the profile for files the heuristic may analyze, else ``None``.

    Eligibility reuses the categorizer's verdict rather than a second
    notion of "human-authored": the file must be categorised ``source``
    (docs, assets, configs, manifests, lockfiles are not), must not carry
    the ``docs`` label, must be ``human_authored`` (generated, vendored,
    minified, snapshot, lockfile rows are excluded), and its extension
    must map to a supported profile.
    """

    if not row.human_authored or "docs" in row.categories or "source" not in row.categories:
        return None
    return _profile_for(file.filename)


class CommentAnalysis(StrictModel):
    """Result of :func:`analyze_added_comments`."""

    stats: CommentStats = Field(description="PR-level added-comment volume totals.")
    warnings: list[EngineWarning] = Field(
        default_factory=list,
        description="Deterministic code-comment warnings in stable order.",
    )


def analyze_added_comments(
    files: list[ChangedFile],
    file_categories: list[FileCategoryRow],
    policy: CodeCommentPolicy,
) -> CommentAnalysis:
    """Analyze added comment volume and map it to deterministic warnings.

    Args:
        files: The engine's active (post-``ignored_paths``) changed files,
            in engine order.
        file_categories: The categorizer rows for exactly those files, in
            the same order (the pairing the engine already maintains).
        policy: The ``policy.code_comments`` block from
            :class:`reviewgate.core.config.CodeCommentPolicy`.

    Returns:
        A :class:`CommentAnalysis` whose ``warnings`` are ordered: one
        ``oversized_comment_block`` warning per eligible file that reaches
        a block threshold (input order), then ``excessive_comment_lines``,
        then ``comment_heavy_diff``. Thresholds are inclusive lower bounds,
        matching :func:`reviewgate.core.size.size_warnings`.

    Raises:
        ValueError: If ``files`` and ``file_categories`` lengths differ
            (contract drift between the engine and the categorizer).
    """

    if len(files) != len(file_categories):
        raise ValueError(
            "code_comments: files and file_categories must pair one-to-one "
            f"(got {len(files)} files vs {len(file_categories)} rows)"
        )

    tallies: list[tuple[ChangedFile, _FileTally]] = []
    total_comment = 0
    total_code = 0
    largest_block = 0
    for file, row in zip(files, file_categories, strict=True):
        profile = _eligible(file, row)
        if profile is None or file.patch is None:
            continue
        tally = _tally_patch(file.patch, profile)
        tallies.append((file, tally))
        total_comment += tally.comment_lines
        total_code += tally.code_lines
        if tally.largest_block > largest_block:
            largest_block = tally.largest_block

    stats = CommentStats(
        comment_lines_added=total_comment,
        code_lines_added=total_code,
        largest_comment_block_lines=largest_block,
        comment_ratio=_ratio(total_comment, total_comment + total_code),
    )

    warnings: list[EngineWarning] = []
    for file, tally in tallies:
        warning = _block_warning(file.filename, tally.largest_block, policy)
        if warning is not None:
            warnings.append(warning)
    warnings.extend(_total_warning(total_comment, policy))
    warnings.extend(_ratio_warning(stats, policy))

    return CommentAnalysis(stats=stats, warnings=warnings)


def _ratio(comment_lines: int, source_lines: int) -> float:
    """Deterministically rounded comment-to-source ratio."""

    if source_lines <= 0:
        return 0.0
    return round(comment_lines / source_lines, _RATIO_PRECISION)


def _block_warning(
    filename: str,
    block_lines: int,
    policy: CodeCommentPolicy,
) -> EngineWarning | None:
    """Build one per-file ``oversized_comment_block`` warning, or ``None``."""

    if block_lines <= 0:
        return None
    if block_lines >= policy.fail.max_block_lines:
        tier, severity, threshold = (
            "fail",
            _SEVERITY_FAIL,
            policy.fail.max_block_lines,
        )
    elif block_lines >= policy.warn.max_block_lines:
        tier, severity, threshold = (
            "warn",
            _SEVERITY_WARN,
            policy.warn.max_block_lines,
        )
    else:
        return None
    return EngineWarning(
        code=WARN_CODE_OVERSIZED_BLOCK,
        severity=severity,
        message=(
            f"Source file {filename} adds a {block_lines}-line consecutive "
            f"comment block, exceeding the configured {tier} threshold of "
            f"{threshold} lines."
        ),
        evidence={
            "filename": filename,
            "comment_block_lines": block_lines,
            "threshold": threshold,
            "tier": tier,
        },
    )


def _total_warning(
    comment_lines: int,
    policy: CodeCommentPolicy,
) -> list[EngineWarning]:
    """Build the PR-level ``excessive_comment_lines`` warning, if any."""

    if comment_lines >= policy.fail.max_total_lines:
        tier, severity, threshold = (
            "fail",
            _SEVERITY_FAIL,
            policy.fail.max_total_lines,
        )
    elif comment_lines >= policy.warn.max_total_lines:
        tier, severity, threshold = (
            "warn",
            _SEVERITY_WARN,
            policy.warn.max_total_lines,
        )
    else:
        return []
    return [
        EngineWarning(
            code=WARN_CODE_EXCESSIVE_LINES,
            severity=severity,
            message=(
                f"PR adds {comment_lines} comment lines across eligible "
                f"source files, exceeding the configured {tier} threshold "
                f"of {threshold} lines."
            ),
            evidence={
                "comment_lines_added": comment_lines,
                "threshold": threshold,
                "tier": tier,
            },
        ),
    ]


def _ratio_warning(
    stats: CommentStats,
    policy: CodeCommentPolicy,
) -> list[EngineWarning]:
    """Build the ``comment_heavy_diff`` warning, honoring the sample-size guard.

    The ratio never triggers when ``source_lines_added`` is below
    ``policy.min_added_source_lines``; block and total-volume dimensions
    stay active on small diffs.
    """

    source_lines = stats.comment_lines_added + stats.code_lines_added
    if source_lines < policy.min_added_source_lines:
        return []
    if stats.comment_ratio >= policy.fail.max_comment_ratio:
        tier, severity, threshold = (
            "fail",
            _SEVERITY_FAIL,
            policy.fail.max_comment_ratio,
        )
    elif stats.comment_ratio >= policy.warn.max_comment_ratio:
        tier, severity, threshold = (
            "warn",
            _SEVERITY_WARN,
            policy.warn.max_comment_ratio,
        )
    else:
        return []
    return [
        EngineWarning(
            code=WARN_CODE_COMMENT_HEAVY,
            severity=severity,
            message=(
                f"PR adds {stats.comment_lines_added} comment lines across "
                f"{source_lines} added source lines (comment ratio "
                f"{stats.comment_ratio}), exceeding the configured {tier} "
                f"threshold of {threshold}."
            ),
            evidence={
                "comment_lines_added": stats.comment_lines_added,
                "source_lines_added": source_lines,
                "comment_ratio": stats.comment_ratio,
                "threshold": threshold,
                "tier": tier,
            },
        ),
    ]


__all__ = [
    "WARN_CODE_COMMENT_HEAVY",
    "WARN_CODE_EXCESSIVE_LINES",
    "WARN_CODE_OVERSIZED_BLOCK",
    "CommentAnalysis",
    "CommentStats",
    "analyze_added_comments",
]
