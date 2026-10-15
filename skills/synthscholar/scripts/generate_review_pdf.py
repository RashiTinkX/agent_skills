#!/usr/bin/env python3
"""Render an exported review (``review.md``) to PDF via pandoc.

``export_review.py`` writes the review as Markdown; this turns that file — and
optionally its sidecar exports as appendices — into one PDF. It does not
re-run or re-serialise anything: the PDF is a rendering of ``review.md``.

What a SynthScholar review needs beyond a plain pandoc call:

* **Title and numbering.** The review opens with ``# <title>`` and numbers its
  own sections (``## 1. Introduction``). The H1 becomes the PDF title and,
  when it is the only H1, the ``##`` sections become top-level.
  ``--number-sections`` is off by default (it would print "1 1. Introduction").
* **Tables.** Charting / narrative tables put line breaks in cells as
  ``<br>``, which TeX and Typst drop; they are rewritten to each engine's own
  line break. Tables wrap at the text width instead of running off the page.
* **Characters.** TeX's default font has no ``≥ μ κ α →``; they vanish with
  only a log line. On TeX engines the script picks an installed font that
  covers the characters the review actually uses (``fc-list``), maps
  ``✓ ✗ ⚠`` to TeX symbols, and reports any glyph still missing.
* **Citation gate.** If ``verify_citations.py`` has been run
  (``<stem>.citations.json`` next to the Markdown, or ``--citations``) and
  found errors, or its report is older than the Markdown, the build stops.
  ``--allow-citation-errors`` overrides; with no report at all it warns.
* **Verified reference list.** ``--references vancouver`` swaps the review's
  References list for ``<stem>.references.vancouver.md``, the list
  ``verify_citations.py --style vancouver`` rendered from registry metadata.
* **Appendices.** ``--include charting appraisal per-group narrative citations``
  appends ``<stem>.charting.md`` etc., each on a new page. A sidecar whose
  title already heads a section of the review (``appraisal`` duplicates its
  *Appendix C*) is skipped rather than printed twice.

Engines, best first: xelatex, lualatex, tectonic (self-contained TeX), pdflatex,
typst (no TeX at all), then the HTML engines weasyprint / wkhtmltopdf / prince.
``[@key]`` citations, if a hand-edited Markdown has them, are processed with
``--citeproc`` against a sibling or YAML-declared ``.bib`` and a CSL style
(``--citation-style``, fetched from the official CSL repository when needed).

Usage:
    python generate_review_pdf.py out/review.md
    python generate_review_pdf.py out/review.md --references vancouver \\
        --include per-group citations -o out/review.pdf
    python generate_review_pdf.py out/review.md --pdf-engine typst --dry-run
    python generate_review_pdf.py --check-deps

Needs ``pandoc`` and one engine; stdlib only otherwise.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional

# Engines pandoc can drive, best first. xelatex/lualatex give proper Unicode;
# tectonic is self-contained (fetches its own TeX packages on first run);
# pdflatex is last among TeX engines because its Unicode handling is weakest.
LATEX_ENGINES = ["xelatex", "lualatex", "tectonic", "pdflatex"]
UNICODE_TEX = {"xelatex", "lualatex", "tectonic"}   # take a system font (fontspec)
TYPST_ENGINES = ["typst"]
HTML_ENGINES = ["weasyprint", "wkhtmltopdf", "prince"]
ALL_ENGINES = LATEX_ENGINES + TYPST_ENGINES + HTML_ENGINES
CSL_URL = "https://raw.githubusercontent.com/citation-style-language/styles/master/{}.csl"
# Common names that are not file names in the CSL repository.
CSL_ALIASES = {"vancouver": "nlm-citation-sequence", "nlm": "nlm-citation-sequence",
               "ama": "american-medical-association", "chicago": "chicago-author-date",
               "harvard": "harvard-cite-them-right"}

# Sidecar exports that can be appended, by export_review.py's file suffix.
SIDECARS = {
    "charting": ".charting.md",
    "appraisal": ".appraisal.md",
    "per-group": ".per-group.md",
    "narrative": ".narrative.md",
    "citations": ".citations.md",   # verify_citations.py's report
}

# Symbols the exporters emit that most text fonts lack, mapped to TeX symbols.
TEX_SYMBOLS = {
    "✓": r"`$\checkmark$`{=latex}",
    "✔": r"`$\checkmark$`{=latex}",
    "✗": r"`$\times$`{=latex}",
    "✘": r"`$\times$`{=latex}",
    "⚠": r"`\textbf{!}`{=latex}",
    "⚡": "",
}
# Text fonts with wide coverage, best-looking first. The first installed one
# that covers every character the review uses becomes the TeX main font.
UNICODE_FONTS = ["STIX Two Text", "DejaVu Serif", "Noto Serif", "Libertinus Serif",
                 "FreeSerif", "STIXGeneral", "Arial Unicode MS", "DejaVu Sans", "Noto Sans"]
# One raw inline per output family; pandoc keeps only the one for the target.
LINE_BREAK = r"`\newline{}`{=latex}`#linebreak()`{=typst}"
PAGE_BREAK = ("\n```{=latex}\n\\newpage\n```\n\n```{=typst}\n#pagebreak()\n```\n\n"
              '```{=html}\n<div style="page-break-before: always"></div>\n```\n')
FENCE_OPEN = re.compile(r"^ {0,3}(`{3,}|~{3,})")
CODE_SPAN = re.compile(r"(?<!`)(`+)(?!`).*?(?<!`)\1(?!`)")
HEADING = re.compile(r"^(#{1,6})(\s+)(.*?)\s*#*\s*$")
RULE = re.compile(r"^ {0,3}(?:(?:-\s*){3,}|(?:\*\s*){3,}|(?:_\s*){3,})$")
# Pandoc wraps a pipe table (cells reflow to the text width) only when one of
# its lines is longer than --columns; review tables are long, so set it low.
WRAP_COLUMNS = 40

HTML_CSS = """
body { font-family: "Helvetica Neue", Arial, sans-serif; font-size: 10.5pt; line-height: 1.35; }
table { border-collapse: collapse; width: 100%; font-size: 8.5pt; margin: 0.6em 0; }
th, td { border: 1px solid #bbb; padding: 3px 5px; vertical-align: top; }
th { background: #f0f0f0; }
code, pre { font-size: 8.5pt; }
pre { white-space: pre-wrap; }
@page { size: A4; margin: 2cm; }
"""


# ─────────────────────────── tooling ────────────────────────────

def engine_family(engine: str) -> str:
    """``/Library/TeX/texbin/xelatex`` → ``xelatex``: options key off the name."""
    name = Path(engine).name.lower()
    return name[:-4] if name.endswith(".exe") else name


def find_engine(preferred: Optional[str] = None) -> Optional[str]:
    """First available PDF engine. An explicit choice is honoured or refused, not swapped."""
    if preferred:
        return preferred if shutil.which(preferred) else None
    return next((e for e in ALL_ENGINES if shutil.which(e)), None)


def csl_cache() -> Optional[Path]:
    """First writable cache directory, or None ($HOME is read-only in some sandboxes)."""
    xdg = os.environ.get("XDG_CACHE_HOME")
    candidates = [Path(xdg) / "csl"] if xdg else []
    candidates += [Path.home() / ".cache" / "csl", Path(tempfile.gettempdir()) / "csl-cache"]
    for d in candidates:
        try:
            d.mkdir(parents=True, exist_ok=True)
            return d
        except OSError:
            continue
    return None


def resolve_csl(style: str, near: Path) -> Optional[Path]:
    """Style name → CSL file: explicit path, sibling, cwd, cache, then the CSL repository."""
    if style.endswith(".csl"):
        p = Path(style)
        return p if p.is_file() else None
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", style):
        print(f"  warning: '{style}' is not a CSL style id; using pandoc's default style",
              file=sys.stderr)
        return None
    style = CSL_ALIASES.get(style, style)
    cache = csl_cache()
    for c in [near / f"{style}.csl", Path.cwd() / f"{style}.csl"] + ([cache / f"{style}.csl"] if cache else []):
        if c.is_file():
            return c
    try:
        with urllib.request.urlopen(CSL_URL.format(style), timeout=20) as r:
            body = r.read()
    except (urllib.error.URLError, OSError) as e:
        print(f"  warning: no CSL for '{style}' ({e}); using pandoc's default style — style ids "
              "are the file names at https://github.com/citation-style-language/styles", file=sys.stderr)
        return None
    if b"<style" not in body[:2000]:
        print(f"  warning: '{style}' did not download as a CSL style; using pandoc's default",
              file=sys.stderr)
        return None
    # Write to a temp file and rename, so an interrupted download never leaves a
    # truncated style in the cache to break every later build.
    dest_dir = cache or Path(tempfile.gettempdir())
    fd, tmp = tempfile.mkstemp(suffix=".csl", dir=dest_dir)
    with os.fdopen(fd, "wb") as f:
        f.write(body)
    target = dest_dir / f"{style}.csl"
    os.replace(tmp, target)
    print(f"  fetched CSL style '{style}' -> {target}", file=sys.stderr)
    return target


def find_bibliography(md: Path) -> Optional[Path]:
    for name in (f"{md.stem}.bib", "references.bib", "bibliography.bib"):
        p = md.parent / name
        if p.is_file():
            return p
    return None


def yaml_keys(text: str) -> set[str]:
    """Keys of a leading YAML block, so the author's own title/bibliography win."""
    m = re.match(r"^---[ \t]*\n(.*?)\n(?:---|\.\.\.)[ \t]*\n", text, re.S)
    return set(re.findall(r"^([A-Za-z][\w-]*)\s*:", m.group(1), re.M)) if m else set()


def _font_families(chars: str) -> list[str]:
    """Installed font families covering every character in ``chars`` (fontconfig)."""
    if not shutil.which("fc-list"):
        return []
    charset = " ".join(f"{ord(c):x}" for c in chars)
    try:
        out = subprocess.run(["fc-list", f":charset={charset}", "family"],
                             capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [fam for line in out.splitlines() for fam in line.split(",") if fam.strip()]


def pick_main_font(text: str) -> tuple[Optional[str], str]:
    """(font, chars it must cover) for a TeX engine, or (None, "") when Latin Modern will do.

    Latin Modern covers Latin-1 plus common typographic punctuation; anything
    beyond that (Greek, maths relations, arrows) needs a Unicode font.
    """
    needed = "".join(sorted({c for c in text if ord(c) > 0x17F and c not in TEX_SYMBOLS
                             and c not in "‘’“”–—…•′″€"
                             and unicodedata.category(c)[0] in "LSNP"}))
    if not needed:
        return None, ""
    have = set(_font_families(needed[:80]))
    return next((f for f in UNICODE_FONTS if f in have), None), needed


# ─────────────────────────── Markdown preparation ────────────────────────────

def _outside_fences(text: str, fn: Callable[[str], str]) -> str:
    """Apply ``fn`` to each line not inside a fenced code block.

    A fence closes only on the same character, at least as long as it opened,
    with nothing after it — so a ``` line inside a ~~~ block stays code.
    """
    out, fence = [], None
    for line in text.split("\n"):
        if fence is None:
            m = FENCE_OPEN.match(line)
            if m:
                fence = m.group(1)
                out.append(line)
            else:
                out.append(fn(line))
        else:
            out.append(line)
            s = line.strip()
            if s and set(s) == {fence[0]} and len(s) >= len(fence):
                fence = None
    return "\n".join(out)


def _outside_code_spans(line: str, fn: Callable[[str], str]) -> str:
    """Apply ``fn`` to the parts of a line outside inline `code` spans."""
    out, pos = [], 0
    for m in CODE_SPAN.finditer(line):
        out += [fn(line[pos:m.start()]), m.group()]
        pos = m.end()
    out.append(fn(line[pos:]))
    return "".join(out)


def headings(text: str) -> list[tuple[int, str, int]]:
    """(level, text, line index) of every ATX heading outside code fences."""
    found: list[tuple[int, str, int]] = []
    fence = None
    for i, line in enumerate(text.split("\n")):
        if fence is None:
            m = FENCE_OPEN.match(line)
            if m:
                fence = m.group(1)
            elif (h := HEADING.match(line)):
                found.append((len(h.group(1)), h.group(3), i))
        else:
            s = line.strip()
            if s and set(s) == {fence[0]} and len(s) >= len(fence):
                fence = None
    return found


def shift_headings(text: str, by: int) -> str:
    """Move every heading ``by`` levels (outside fences), clamped to 1–6."""
    return _outside_fences(text, lambda ln: re.sub(
        r"^(#{1,6})(?=\s)", lambda m: "#" * max(1, min(6, len(m.group(1)) + by)), ln))


def lift_title(text: str) -> tuple[str, Optional[str]]:
    """Take a leading ``# Title`` out of the body; (body, title or None).

    When that was the document's only H1, its ``##`` sections are promoted to
    top level. When other H1s follow, levels are left alone — promoting would
    push those H1s below level 1, where pandoc turns them into paragraphs.
    """
    hs = headings(text)
    lines = text.split("\n")
    first_text = next((i for i, ln in enumerate(lines) if ln.strip()), None)
    if not hs or hs[0][0] != 1 or hs[0][2] != first_text:
        return text, None
    title = hs[0][1]
    body = "\n".join(lines[:hs[0][2]] + lines[hs[0][2] + 1:])
    if not any(level == 1 for level, _, _ in hs[1:]):
        body = shift_headings(body, -1)
    return body, title


def neutralise_rules(text: str, keep_front_matter: bool) -> str:
    """Turn bare ``---`` rules into ``* * *``.

    The exporters separate sections with ``---``. Pandoc reads a ``---`` line
    followed by text ("---\\n*Generated: …*") as the start of a YAML block — and
    a sidecar appended mid-document is full of them — so the build fails with a
    YAML error. ``* * *`` is the same rule with no other reading. A genuine
    front-matter block at the very top of the review is kept.
    """
    lines = text.split("\n")
    start = 0
    if keep_front_matter and lines and lines[0].strip() == "---":
        close = next((i for i in range(1, len(lines)) if lines[i].strip() in ("---", "...")), None)
        if close is not None:
            start = close + 1
    head, rest = lines[:start], "\n".join(lines[start:])
    body = _outside_fences(rest, lambda ln: "\n* * *\n" if ln.strip() == "---" else ln)
    return "\n".join(head + [body]) if head else body


def for_engine(text: str, family: str) -> str:
    """Rewrite the HTML-isms and symbols an engine family can't print."""
    if family in HTML_ENGINES:
        return text

    def fix(seg: str) -> str:
        # a function, not a string: re.sub would read "\n" in "\newline" as a newline
        seg = re.sub(r"<br\s*/?>", lambda _: LINE_BREAK, seg, flags=re.I)
        seg = re.sub(r"</?(?:details|summary)[^>]*>", "", seg, flags=re.I)
        if family in LATEX_ENGINES:
            for sym, tex in TEX_SYMBOLS.items():
                seg = seg.replace(sym, tex)
        return seg
    return _outside_fences(text, lambda ln: _outside_code_spans(ln, fix))


def replace_references(text: str, styled: str) -> tuple[str, bool]:
    """Swap the list under ``## References`` for a styled one; (text, replaced?).

    Only the list is replaced: it ends at the next heading of any level or the
    next rule, so a footer or appendix after it survives even when References
    is the last ``##`` section.
    """
    lines = text.split("\n")
    start = next((i for i, ln in enumerate(lines) if re.match(r"^##\s+References\s*$", ln)), None)
    if start is None:
        return text, False
    end = next((i for i in range(start + 1, len(lines))
                if HEADING.match(lines[i]) or RULE.match(lines[i])), len(lines))
    body = styled.strip("\n").split("\n")
    if body and body[0].startswith("## "):  # verify_citations' "## References (style)"
        body = body[1:]
    new = ["## References", ""] + body + [""]
    return "\n".join(lines[:start] + new + lines[end:]), True


def _norm_title(s: str) -> str:
    s = re.sub(r"^appendix\s+[a-z0-9]+\s*[:.\-—]\s*", "", s.strip(), flags=re.I)
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def citation_gate(md: Path, report: Optional[Path], allow: bool) -> Optional[str]:
    """Refusal message if the citation report has errors or predates ``md``."""
    report = report or md.with_name(f"{md.stem}.citations.json")
    if not report.is_file():
        print(f"  warning: {report} not found — citations unverified; run "
              f"verify_citations.py on {md.stem}.json first (or point --citations at its report)",
              file=sys.stderr)
        return None
    try:
        summary = json.loads(report.read_text(encoding="utf-8")).get("summary") or {}
    except (OSError, json.JSONDecodeError) as e:
        return f"{report.name} is unreadable ({e}); re-run verify_citations.py"
    problems = []
    if report.stat().st_mtime < md.stat().st_mtime:
        problems.append(f"{report.name} is older than {md.name} — it describes an earlier "
                        "export; re-run verify_citations.py on the re-exported review.json")
    errors = int(summary.get("errors") or 0)
    if errors:
        codes = ", ".join(f"{k} {v}" for k, v in (summary.get("by_code") or {}).items())
        problems.append(f"{report.name} reports {errors} citation error(s) ({codes})")
    if problems and not allow:
        return ("\n  ".join(problems) + "\n  Fix them (see references/citation_verification.md) "
                "and re-export, or pass --allow-citation-errors to build anyway.")
    for p in problems:
        print(f"  warning: building anyway — {p}", file=sys.stderr)
    return None


def assemble(md: Path, family: str, references: Optional[str],
             include: list[str]) -> tuple[str, Optional[str], list[str]]:
    """(Markdown for pandoc, lifted title, notes). Raises ValueError."""
    text = neutralise_rules(md.read_text(encoding="utf-8"), keep_front_matter=True)
    notes: list[str] = []
    if references:
        styled = md.with_name(f"{md.stem}.references.{references}.md")
        if not styled.is_file():
            raise ValueError(f"{styled.name} not found — run "
                             f"verify_citations.py {md.stem}.json --style {references} first")
        text, done = replace_references(text, styled.read_text(encoding="utf-8"))
        if not done:
            raise ValueError(f"{md.name} has no '## References' section to replace")
        notes.append(f"references: {styled.name}")

    title = None
    if "title" not in yaml_keys(text):
        text, title = lift_title(text)
    present = {_norm_title(t) for _, t, _ in headings(text)}
    for name in include:
        side = md.with_name(md.stem + SIDECARS[name])
        if not side.is_file():
            raise ValueError(f"{side.name} not found — " + (
                "run verify_citations.py first" if name == "citations"
                else f"export it with export_review.py --formats {name}"))
        side_text = neutralise_rules(side.read_text(encoding="utf-8"), keep_front_matter=False)
        side_hs = headings(side_text)
        if side_hs and _norm_title(side_hs[0][1]) in present:
            notes.append(f"skipped {side.name} (already in {md.name} as “{side_hs[0][1]}”)")
            continue
        # Sidecars open with "# …": a top-level section, like the review's own.
        text += "\n" + PAGE_BREAK + "\n" + side_text + "\n"
        notes.append(f"appendix: {side.name}")
    return for_engine(text, family), title, notes


# ─────────────────────────── pandoc ────────────────────────────

def build_command(src: Path, md: Path, out: Path, engine: str, *, meta: Optional[Path],
                  citeproc: bool, csl: Optional[Path], bib: Optional[Path], toc: bool,
                  number_sections: bool, template: Optional[str], declared: set[str],
                  css: Optional[Path], mainfont: Optional[str]) -> list[str]:
    family = engine_family(engine)
    cmd = ["pandoc", str(src), "-f", "markdown", "-o", str(out), f"--pdf-engine={engine}",
           # images resolve relative to the review, not to the temp copy
           "--resource-path", f"{md.parent}{os.pathsep}.", "--standalone",
           f"--columns={WRAP_COLUMNS}"]
    if meta:
        cmd += ["--metadata-file", str(meta)]
    elif "title" not in declared:
        cmd += ["--metadata", f"title={md.stem.replace('_', ' ').replace('-', ' ').title()}"]

    if family in LATEX_ENGINES:
        cmd += ["-V", "geometry:margin=2cm", "-V", "fontsize=10pt", "-V", "linestretch=1.1",
                "-V", "colorlinks=true", "-V", "linkcolor=blue", "-V", "urlcolor=blue",
                "-V", "citecolor=blue", "-V", "papersize=a4"]
        if mainfont and family in UNICODE_TEX:
            cmd += ["-V", f"mainfont={mainfont}"]
    elif family in TYPST_ENGINES:
        cmd += ["-V", "fontsize=10pt", "-V", "papersize=a4"]
    elif css:
        cmd += ["--css", str(css), "--embed-resources"]

    if toc:
        cmd += ["--toc", "--toc-depth=2"]
    if number_sections:
        cmd.append("--number-sections")
    if citeproc:
        cmd += ["--citeproc", "-M", "link-citations=true"]
        if bib:
            cmd += ["--bibliography", str(bib)]
        if csl:
            cmd += ["--csl", str(csl)]
    if template:
        cmd += ["--template", template]
    return cmd


def _missing_glyphs(log: str) -> list[str]:
    """Characters TeX reported missing from the font, deduplicated."""
    found = re.findall(r"Missing character: There is no (\S+)", log)
    return sorted(set(found))


def generate_pdf(markdown_file: str, output_pdf: Optional[str] = None, *,
                 citation_style: str = "apa", references: Optional[str] = None,
                 include: Optional[list[str]] = None, template: Optional[str] = None,
                 toc: bool = True, number_sections: bool = False,
                 pdf_engine: Optional[str] = None, bibliography: Optional[str] = None,
                 citations: Optional[str] = None, allow_citation_errors: bool = False,
                 mainfont: Optional[str] = None, dry_run: bool = False,
                 keep_source: bool = False, timeout: int = 900) -> bool:
    md = Path(markdown_file)
    if not md.is_file():
        print(f"Error: markdown file not found: {md}", file=sys.stderr)
        return False
    if not shutil.which("pandoc"):
        print("Error: pandoc not found. Install with:\n"
              "  conda install -c conda-forge pandoc   # or: brew / apt install pandoc",
              file=sys.stderr)
        return False
    engine = find_engine(pdf_engine)
    if not engine:
        which = f"'{pdf_engine}'" if pdf_engine else "any of: " + ", ".join(ALL_ENGINES)
        print(f"Error: PDF engine not found ({which}).\n"
              "  conda install -c conda-forge tectonic   # self-contained TeX\n"
              "  conda install -c conda-forge typst      # no TeX at all\n"
              "  apt-get install texlive-xetex           # or: brew install --cask mactex",
              file=sys.stderr)
        return False
    family = engine_family(engine)
    if family not in ALL_ENGINES:
        print(f"  warning: '{engine}' is not an engine this script knows; "
              "no engine-specific options are applied", file=sys.stderr)

    refusal = citation_gate(md, Path(citations) if citations else None, allow_citation_errors)
    if refusal:
        print(f"Error: {refusal}", file=sys.stderr)
        return False

    try:
        text, title, notes = assemble(md, family, references, include or [])
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return False

    out = Path(output_pdf) if output_pdf else md.with_suffix(".pdf")
    out.parent.mkdir(parents=True, exist_ok=True)

    declared = yaml_keys(text)
    # citeproc only when the Markdown has pandoc citations — the exported review
    # carries its own rendered reference list, which citeproc would not touch.
    uses_cites = bool(re.search(r"\[@[\w:.#$%&\-+?<>~/]+", text))
    bib = None
    if bibliography:
        bib = Path(bibliography)
        if not bib.is_file():
            print(f"Error: bibliography not found: {bib}", file=sys.stderr)
            return False
    elif uses_cites and "bibliography" not in declared:
        bib = find_bibliography(md)
    citeproc = uses_cites and bool(bib or "bibliography" in declared)
    if uses_cites and not citeproc:
        print("  warning: [@key] citations but no bibliography (sibling .bib, YAML "
              "'bibliography:' or --bibliography); they will print as written", file=sys.stderr)
    csl = resolve_csl(citation_style, md.parent) if citeproc and "csl" not in declared else None

    if family in UNICODE_TEX and not mainfont:
        mainfont, needed = pick_main_font(text)
        if needed and not mainfont:
            print(f"  warning: no installed font found covering {needed[:40]!r}; those "
                  "characters may be missing — pass --mainfont", file=sys.stderr)
    elif family == "pdflatex" and pick_main_font(text)[1]:
        print("  warning: pdflatex cannot set this review's non-Latin characters; "
              "use xelatex, lualatex, tectonic or typst", file=sys.stderr)
    if mainfont and family in UNICODE_TEX:
        notes.append(f"font: {mainfont}")

    with tempfile.TemporaryDirectory(prefix="review-pdf-") as tmp:
        src = Path(tmp) / f"{md.stem}.pandoc.md"
        src.write_text(text, encoding="utf-8")
        meta = None
        if title:
            # A metadata file, not --metadata title=…: a title with a colon would
            # be read as YAML structure on the command line. JSON strings are YAML.
            meta = Path(tmp) / "meta.yaml"
            meta.write_text(f"title: {json.dumps(title, ensure_ascii=False)}\n", encoding="utf-8")
        css = None
        if family in HTML_ENGINES:
            css = Path(tmp) / "review.css"
            css.write_text(HTML_CSS, encoding="utf-8")
        cmd = build_command(src, md, out, engine, meta=meta, citeproc=citeproc, csl=csl, bib=bib,
                            toc=toc, number_sections=number_sections, template=template,
                            declared=declared, css=css, mainfont=mainfont)
        if keep_source:
            kept = out.with_suffix(".pandoc.md")
            kept.write_text(text, encoding="utf-8")
            notes.append(f"pandoc source: {kept}")
        print(f"Engine: {engine}" + "".join(f" | {n}" for n in notes), file=sys.stderr)
        if dry_run:
            print(" ".join(_shell_quote(c) for c in cmd))
            return True
        print(f"Writing: {out}", file=sys.stderr)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            print(f"Error: pandoc did not finish within {timeout}s "
                  "(tectonic's first run downloads its TeX bundle; retry, or raise --timeout)",
                  file=sys.stderr)
            return False

    if result.returncode != 0:
        print("Error: pandoc failed.", file=sys.stderr)
        # TeX dumps hundreds of log lines; the real error is near the end.
        for stream in (result.stdout, result.stderr):
            for line in [ln for ln in stream.splitlines() if ln.strip()][-15:]:
                print(f"  {line}", file=sys.stderr)
        return False
    missing = _missing_glyphs(result.stderr)
    # tectonic notes every system font it reads by absolute path; one line will do.
    font_paths = [ln for ln in result.stderr.splitlines() if "accessing absolute path" in ln]
    warnings = [ln for ln in result.stderr.splitlines()
                if ln.strip() and "Missing character" not in ln and ln not in font_paths]
    if font_paths:
        warnings.insert(0, f"note: {len(font_paths)} system font file(s) read; the PDF embeds them")
    for line in warnings[:10]:
        print(f"  {line}", file=sys.stderr)
    if len(warnings) > 10:
        print(f"  … {len(warnings) - 10} more warning line(s)", file=sys.stderr)
    if missing:
        print(f"  WARNING: {len(missing)} character(s) missing from the PDF's font: "
              f"{' '.join(missing[:30])} — pass --mainfont with a font that has them",
              file=sys.stderr)
    print(f"Done: {out} ({out.stat().st_size / 1024:.0f} KB)", file=sys.stderr)
    return True


def _shell_quote(s: str) -> str:
    return s if re.fullmatch(r"[\w@%+=:,./-]+", s) else "'" + s.replace("'", "'\\''") + "'"


def check_dependencies() -> bool:
    """Report what is installed; pandoc plus one engine is enough."""
    have_pandoc = bool(shutil.which("pandoc"))
    print(f"[{'ok' if have_pandoc else 'MISSING'}] pandoc")
    found = [e for e in ALL_ENGINES if shutil.which(e)]
    for e in found:
        print(f"[ok] {e}")
    if not found:
        print("[MISSING] no PDF engine (tried: " + ", ".join(ALL_ENGINES) + ")")
    if any(e in UNICODE_TEX for e in found):
        print(f"[{'ok' if shutil.which('fc-list') else '--'}] fc-list "
              "(picks a Unicode font for TeX engines; optional)")
    print("\nReady." if have_pandoc and found else "\nNot ready: install the items marked MISSING.")
    return have_pandoc and bool(found)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("markdown_file", nargs="?", help="exported review Markdown (review.md)")
    p.add_argument("-o", "--output", help="output PDF (default: next to the Markdown)")
    # Any style verify_citations.py can render: the file decides, not a list here.
    p.add_argument("--references", metavar="STYLE", type=lambda s: s if re.fullmatch(r"[a-z0-9-]+", s)
                   else p.error(f"bad style name {s!r}"),
                   help="replace the References list with <stem>.references.<STYLE>.md "
                        "(verify_citations.py --style STYLE)")
    p.add_argument("--include", nargs="+", choices=list(SIDECARS), default=[], metavar="NAME",
                   help="append sidecar exports as appendices: " + ", ".join(SIDECARS))
    p.add_argument("--citations", metavar="REPORT_JSON",
                   help="citation report to gate on (default: <stem>.citations.json beside the Markdown)")
    p.add_argument("--allow-citation-errors", action="store_true",
                   help="build even if the citation report has errors or is stale")
    p.add_argument("--citation-style", default="apa",
                   help="CSL style id or .csl path, for [@key] citations only (default: apa)")
    p.add_argument("--bibliography", help="explicit .bib for [@key] citations")
    p.add_argument("--pdf-engine", help=f"force an engine ({', '.join(ALL_ENGINES)}), name or path")
    p.add_argument("--mainfont", help="TeX main font (default: an installed font covering the text)")
    p.add_argument("--template", help="custom pandoc template")
    p.add_argument("--no-toc", action="store_true")
    p.add_argument("--number-sections", action="store_true",
                   help="pandoc section numbers (off: the review numbers its own)")
    p.add_argument("--keep-source", action="store_true",
                   help="also write the Markdown pandoc reads, next to the PDF")
    p.add_argument("--dry-run", action="store_true", help="print the pandoc command and stop")
    p.add_argument("--timeout", type=int, default=900, help="seconds before pandoc is stopped")
    p.add_argument("--check-deps", action="store_true", help="report tooling and exit")
    args = p.parse_args()

    if args.check_deps:
        return 0 if check_dependencies() else 1
    if not args.markdown_file:
        p.error("markdown_file is required (or use --check-deps)")
    ok = generate_pdf(args.markdown_file, args.output, citation_style=args.citation_style,
                      references=args.references, include=args.include, template=args.template,
                      toc=not args.no_toc, number_sections=args.number_sections,
                      pdf_engine=args.pdf_engine, bibliography=args.bibliography,
                      citations=args.citations, allow_citation_errors=args.allow_citation_errors,
                      mainfont=args.mainfont, dry_run=args.dry_run,
                      keep_source=args.keep_source, timeout=args.timeout)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
