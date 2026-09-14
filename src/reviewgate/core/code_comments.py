"""Excessive code-comment verbosity heuristic (issue #143).

Detects *objective volume* signals in commentary a PR introduces, from the
optional unified diffs in :attr:`reviewgate.core.schemas.ChangedFile.patch`:

1. **Oversized consecutive comment blocks** -- the PR-wide maximum run of
   consecutive newly-added full-line comment lines (filename in evidence).
2. **Excessive total comment lines** -- newly-added full-line comment lines
   summed across all eligible files.
3. **Comment-heavy diff** -- the ratio of newly-added comment lines to
   newly-added non-blank source lines, guarded by a minimum sample size.

Explicit non-goals (issue #143): the heuristic never judges whether a
comment is useful, correct, well-written, or who or what wrote it. It
measures comment *volume*, never comment *value*.

Conservative-parsing contract (false negatives preferred):

* Only **added** patch lines contribute to metrics. Deleted lines are
  dropped (they are gone in the post-image, so surrounding added lines
  become adjacent). Unchanged context lines are **scanned** so lexical
  state (open ``/* */`` blocks, strings, heredocs) stays accurate, but
  they are never tallied.
* A hunk whose new-file start line is greater than 1 does not establish
  that its first visible line is outside a pre-existing string, comment,
  or heredoc. Those hunks are skipped entirely (no scan, no tally):
  unknown entry state must not emit warnings. Hunks that start at new-file
  line 0 or 1 are known-normal and are analyzed.
* Only files the categorizer marks ``human_authored`` **and** ``source``,
  written in a supported language (by extension), are analyzed. Docs,
  generated, vendored, minified, snapshot, asset, lockfile, manifest, and
  unknown-language files are skipped. Supported extensions are those whose
  multiline string forms the scanner actually models: Python, Shell,
  JavaScript, TypeScript (not JSX/TSX), and Go. Other C-family suffixes
  are skipped rather than guessed.
* A line counts as a comment only when it is an *unmistakable full-line*
  comment (``#``, ``//``, or a ``/* ... */`` block line) at a lexical
  position where a comment can exist. A small single-pass scanner tracks
  string literals (including JS/Go backtick strings and shell quotes /
  heredocs), ``/* */`` block comments, and Python triple-quoted strings
  so markers inside string content (``url = "https://example.com"``,
  ``pattern = "#[a-z]+"``, template-literal / heredoc bodies) are never
  miscounted.
* Inline (trailing) comments are **not** counted in this MVP, and neither
  are Python docstrings: docstrings are string literals and may be runtime
  data. Both decisions are documented false-negative trade-offs.
* Blank lines terminate a block: a run of comment lines interrupted by a
  blank, code, or non-added diff line counts as separate, smaller blocks.

Each volume dimension emits at most one warning per PR (the same
convention as :mod:`reviewgate.core.size`): repeated evidence for one
metric must not masquerade as independent risk signals.

Pure: stdlib only, no I/O, no GitHub or LLM dependency (§4.1 boundary).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final, Literal

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
# Only extensions whose multiline string forms this scanner actually models.
# JSX/TSX text, Rust/C++/C#/Java raw strings and text blocks are omitted
# until they can be classified without false positives (issue #143).
_JS_GO_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {".js", ".mjs", ".cjs", ".ts", ".go"}
)


@dataclass(frozen=True)
class _Profile:
    """Comment-syntax profile applied by the line scanner."""

    hash_comments: bool  # `#` line comments (Python, Shell)
    c_comments: bool  # `//` line comments and `/* */` block comments
    triple_strings: bool  # Python triple-quoted strings
    backtick_strings: bool = False  # JS template literals / Go raw / shell `` ` ``
    shell_quoting: bool = False  # quotes and heredocs span lines without `\`


_PYTHON_PROFILE: Final[_Profile] = _Profile(
    hash_comments=True, c_comments=False, triple_strings=True
)
_SHELL_PROFILE: Final[_Profile] = _Profile(
    hash_comments=True,
    c_comments=False,
    triple_strings=False,
    backtick_strings=True,
    shell_quoting=True,
)
_JS_GO_PROFILE: Final[_Profile] = _Profile(
    hash_comments=False,
    c_comments=True,
    triple_strings=False,
    backtick_strings=True,
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
    if ext in _JS_GO_EXTENSIONS:
        return _JS_GO_PROFILE
    return None


# --- conservative line scanner ----------------------------------------------


@dataclass
class _ScanState:
    """Lexical state carried across post-image lines of one known hunk.

    Context lines update this state without contributing to metrics.
    Mid-file hunks are skipped rather than scanned from a default
    instance: unknown entry state must not emit warnings.
    """

    in_block_comment: bool = False  # inside a `/* ... */` block (C-like)
    triple_delimiter: str | None = None  # inside `"""` / `'''` (Python)
    in_string: str | None = None  # active quote: `"`, `'`, or `` ` ``
    string_raw: bool = False  # True: closer is unescaped (shell single quotes)
    heredoc_delimiter: str | None = None  # shell heredoc body until this word
    heredoc_strip: bool = False  # `<<-` strips leading tabs on the closer


_KIND_COMMENT: Final[str] = "comment"
_KIND_CODE: Final[str] = "code"
_KIND_BLANK: Final[str] = "blank"

_PatchKind = Literal["added", "context", "gap"]


@dataclass(frozen=True)
class _PatchLine:
    """One unified-diff line classified for scanning vs tallying."""

    kind: _PatchKind
    content: str  # empty for gap


def _scan_line(line: str, state: _ScanState, profile: _Profile) -> str:
    """Classify one post-image line as ``comment`` / ``code`` / ``blank``.

    Updates ``state`` in place so subsequent lines of the same file (added
    *or* context) are scanned in the right context. Only whole-line comment
    forms are ever returned as :data:`_KIND_COMMENT`; trailing comments,
    string content, and anything ambiguous fall into :data:`_KIND_CODE`
    (the conservative bucket: non-blank, non-comment lines count as source
    lines when the caller tallies an added line).
    """

    stripped = line.strip()
    if not stripped:
        # A blank line never changes lexical state (a blank inside a block
        # comment, string, or heredoc leaves it open) and terminates
        # comment-block runs at the caller.
        return _KIND_BLANK

    if state.heredoc_delimiter is not None:
        closer = line.lstrip("\t") if state.heredoc_strip else line
        if closer == state.heredoc_delimiter:
            state.heredoc_delimiter = None
            state.heredoc_strip = False
        return _KIND_CODE

    i = 0
    n = len(line)
    while i < n and line[i] in (" ", "\t"):
        i += 1

    if state.in_string is not None:
        i = _scan_string_rest(line, 0, state)
        if state.in_string is not None or i >= len(line):
            return _KIND_CODE
        _scan_normal(line, i, state, profile)
        return _KIND_CODE

    if state.triple_delimiter is not None:
        # Inside a Python triple-quoted string: string content is never a
        # comment (docstrings are string literals, possibly runtime data).
        i = _scan_triple_rest(line, i, state)
        if state.triple_delimiter is not None or i >= len(line):
            return _KIND_CODE
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
    opens ``/* */`` blocks, Python triple-quoted strings, backtick strings,
    and shell heredocs that continue onto later lines, and always returns
    :data:`_KIND_CODE` -- by the time this function runs, the line has
    already shown real (non-comment) content or its leading token was not
    an unmistakable comment form.
    """

    n = len(line)
    while i < n:
        ch = line[i]
        if profile.triple_strings and line[i : i + 3] in ('"""', "'''"):
            delimiter = line[i : i + 3]
            end = _find_triple_close(line, i + 3, delimiter)
            if end == -1:
                state.triple_delimiter = delimiter
                return _KIND_CODE
            i = end
            continue
        if ch in ("'", '"'):
            raw = profile.shell_quoting and ch == "'"
            close = _find_quote_close(line, i + 1, ch, raw=raw)
            if close == -1:
                if profile.shell_quoting or _ends_with_backslash(line):
                    state.in_string = ch
                    state.string_raw = raw
                return _KIND_CODE
            i = close + 1
            continue
        if profile.backtick_strings and ch == "`":
            close = _find_quote_close(line, i + 1, "`", raw=False)
            if close == -1:
                state.in_string = "`"
                state.string_raw = False
                return _KIND_CODE
            i = close + 1
            continue
        if profile.c_comments and ch == "/":
            if line[i : i + 2] == "//":
                return _KIND_CODE
            if line[i : i + 2] == "/*":
                close = line.find("*/", i + 2)
                if close == -1:
                    state.in_block_comment = True
                    return _KIND_CODE
                i = close + 2
                continue
        if profile.hash_comments and ch == "#":
            # Trailing hash comment: rest of the line is not code, so a
            # `# cat <<EOF` comment must not open a heredoc.
            return _KIND_CODE
        if profile.shell_quoting:
            opened = _try_open_heredoc(line, i, state)
            if opened is not None:
                i = opened
                continue
        i += 1
    return _KIND_CODE


def _find_quote_close(line: str, start: int, quote: str, *, raw: bool) -> int:
    """Index of the next closer, or ``-1`` if it does not appear on this line.

    When ``raw`` is false, a backslash skips the next character (JS template
    literals, Python/C strings, shell double quotes). Go raw strings have no
    escapes; honouring ``\\`` there is a false-negative (preferred).
    """

    i = start
    n = len(line)
    while i < n:
        if not raw and line[i] == "\\":
            i += 2
            continue
        if line[i] == quote:
            return i
        i += 1
    return -1


def _scan_string_rest(line: str, i: int, state: _ScanState) -> int:
    """Consume the rest of a line inside a carried-over string literal.

    Returns the index to continue scanning from. If the string closes on
    this line, ``state`` returns to normal so the caller can re-scan the
    remainder.
    """

    quote = state.in_string
    if quote is None:  # pragma: no cover - guarded by caller
        return i
    close = _find_quote_close(line, i, quote, raw=state.string_raw)
    if close == -1:
        return len(line)
    state.in_string = None
    state.string_raw = False
    return close + 1


def _ends_with_backslash(line: str) -> bool:
    """True when the line continues a quoted string with a trailing ``\\``."""

    stripped = line.rstrip(" \t")
    count = 0
    idx = len(stripped) - 1
    while idx >= 0 and stripped[idx] == "\\":
        count += 1
        idx -= 1
    return count % 2 == 1


def _try_open_heredoc(line: str, i: int, state: _ScanState) -> int | None:
    """Open a shell heredoc starting at ``i`` and return the resume index.

    Recognises ``<<EOF``, ``<<-EOF``, ``<<'EOF'``, ``<<"EOF"``, ``<<\\EOF``,
    and the same forms with whitespace before the delimiter. ``<<<``
    here-strings are ignored. Returns ``None`` when ``i`` is not a heredoc
    operator; the caller then advances one character as usual.
    """

    if line[i : i + 2] != "<<":
        return None
    if line[i : i + 3] == "<<<":
        return None
    j = i + 2
    strip = False
    if j < len(line) and line[j] == "-":
        strip = True
        j += 1
    while j < len(line) and line[j] in (" ", "\t"):
        j += 1
    if j >= len(line):
        return None
    quote = ""
    if line[j] in ("'", '"', "\\"):
        quote = line[j]
        j += 1
    start = j
    while j < len(line) and (line[j].isalnum() or line[j] == "_"):
        j += 1
    if j == start:
        return None
    if quote in ("'", '"'):
        if j >= len(line) or line[j] != quote:
            return None
        j += 1
        state.heredoc_delimiter = line[start : j - 1]
    else:
        state.heredoc_delimiter = line[start:j]
    state.heredoc_strip = strip
    return j


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


_HUNK_HEADER: Final[re.Pattern[str]] = re.compile(
    r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@"
)


def _hunk_entry_is_known(header: str) -> bool:
    """True when a hunk's first new-file line is the start of the file.

    New-file line 0 or 1 is the only case where "normal" lexical state is
    known. Any later start can sit inside a string, comment, or heredoc
    whose opener is outside Git's context window; those hunks must not
    emit comment metrics (issue #143: unknown syntax prefers false
    negatives).
    """

    match = _HUNK_HEADER.match(header)
    if match is None:
        return False
    return int(match.group(1)) <= 1


def _iter_patch_lines(patch: str) -> list[_PatchLine]:
    """Classify unified-diff lines into added / context / gap.

    Added lines are the only ones that contribute to metrics. Context
    lines (space prefix) are post-image content: the lexer must see them
    when the hunk's entry state is known. Deleted lines are dropped --
    they do not exist in the resulting file, so surrounding added lines
    become adjacent. Gap content is preserved so hunk headers can be
    parsed; a mid-file hunk is skipped rather than assumed normal.
    """

    extracted: list[_PatchLine] = []
    for line in patch.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            extracted.append(_PatchLine("gap", line))
        elif line.startswith("+"):
            extracted.append(_PatchLine("added", line[1:]))
        elif line.startswith("-"):
            continue
        elif line.startswith(" "):
            extracted.append(_PatchLine("context", line[1:]))
        else:
            extracted.append(_PatchLine("gap", line))
    return extracted


def _tally_patch(patch: str, profile: _Profile) -> _FileTally:
    """Count added comment / code lines and the largest block in one patch."""

    state = _ScanState()
    comment_lines = 0
    code_lines = 0
    current_block = 0
    largest_block = 0
    known = False
    for item in _iter_patch_lines(patch):
        if item.kind == "gap":
            state = _ScanState()
            current_block = 0
            if item.content.startswith("@@"):
                known = _hunk_entry_is_known(item.content)
            continue
        if not known:
            continue
        kind = _scan_line(item.content, state, profile)
        if item.kind != "added":
            current_block = 0
            continue
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
        A :class:`CommentAnalysis` whose ``warnings`` are ordered: at most
        one ``oversized_comment_block`` for the PR-wide maximum (filename
        of the first file that attains it, in input order), then
        ``excessive_comment_lines``, then ``comment_heavy_diff``.
        Thresholds are inclusive lower bounds, matching
        :func:`reviewgate.core.size.size_warnings`.

    Raises:
        ValueError: If ``files`` and ``file_categories`` lengths differ, or
            any paired ``filename`` values differ (the public API must not
            apply one file's category verdict to another file's patch).
    """

    if len(files) != len(file_categories):
        raise ValueError(
            "code_comments: files and file_categories must pair one-to-one "
            f"(got {len(files)} files vs {len(file_categories)} rows)"
        )

    total_comment = 0
    total_code = 0
    largest_block = 0
    largest_block_file: str | None = None
    for file, row in zip(files, file_categories, strict=True):
        if file.filename != row.filename:
            raise ValueError(
                "code_comments: files and file_categories must pair by "
                f"filename (got {file.filename!r} vs {row.filename!r})"
            )
        profile = _eligible(file, row)
        if profile is None or file.patch is None:
            continue
        tally = _tally_patch(file.patch, profile)
        total_comment += tally.comment_lines
        total_code += tally.code_lines
        if tally.largest_block > largest_block:
            largest_block = tally.largest_block
            largest_block_file = file.filename

    stats = CommentStats(
        comment_lines_added=total_comment,
        code_lines_added=total_code,
        largest_comment_block_lines=largest_block,
        comment_ratio=_ratio(total_comment, total_comment + total_code),
    )

    warnings: list[EngineWarning] = []
    if largest_block_file is not None:
        warning = _block_warning(largest_block_file, largest_block, policy)
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
    """Build the single PR-level ``oversized_comment_block`` warning, or ``None``."""

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
            f"PR adds a {block_lines}-line consecutive comment block in "
            f"{filename}, exceeding the configured {tier} threshold of "
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
