#!/usr/bin/env python3
"""Render an exported review (``review.md``) to PDF via pandoc.

``export_review.py`` writes the review as Markdown; this turns that file — and
optionally its sidecar exports as appendices — into one PDF. It does not
re-run or re-serialise anything: the PDF is a rendering of ``review.md``.

What a SynthScholar review needs beyond a plain pandoc call:

* **Title and numbering.** The review opens with ``# <title>`` and numbers its
  own sections (``## 1. Introduction``). The H1 becomes the PDF title and the
  ``##`` sections become top-level, so ``--number-sections`` is off by default
  (it would print "1 1. Introduction").
* **Tables.** Charting / narrative tables put line breaks in cells as
  ``<br>``, which TeX and Typst drop; they are rewritten to each engine's own
  line break. Tables are set to wrap at the text width instead of running off
  the page.
* **Symbols.** ``✓ ✗ ⚠ ⚡`` (run summaries, compare tables) are missing from
  the default TeX fonts and vanish silently; on TeX engines they are mapped to
  TeX equivalents.
* **Citation gate.** If ``verify_citations.py`` has been run (``<stem>.citations.json``
  next to the Markdown) and found errors, the build stops — a PDF is the form a
  review leaves the building in. ``--allow-citation-errors`` overrides; with no
  report at all it warns.
* **Verified reference list.** ``--references vancouver`` swaps the review's
  References section for ``<stem>.references.vancouver.md``, the list
  ``verify_citations.py --style vancouver`` rendered from registry metadata.
* **Appendices.** ``--include charting appraisal per-group narrative citations``
  appends ``<stem>.charting.md`` etc., each on a new page, headings demoted to
  sit under the review's own.

Engines, best first: xelatex, lualatex, tectonic (self-contained TeX), pdflatex,
typst (no TeX at all), then the HTML engines weasyprint / wkhtmltopdf / prince.
``[@key]`` citations, if a hand-edited Markdown has them, are processed with
``--citeproc`` against a sibling ``.bib`` and a CSL style (``--citation-style``,
fetched from the official CSL repository when not held locally).

Usage:
    python generate_review_pdf.py out/review.md
    python generate_review_pdf.py out/review.md --references vancouver \\
        --include charting appraisal citations -o out/review.pdf
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
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

# Engines pandoc can drive, best first. xelatex/lualatex give proper Unicode;
# tectonic is self-contained (fetches its own TeX packages on first run);
# pdflatex is last among TeX engines because its Unicode handling is weakest.
LATEX_ENGINES = ["xelatex", "lualatex", "tectonic", "pdflatex"]
TYPST_ENGINES = ["typst"]
HTML_ENGINES = ["weasyprint", "wkhtmltopdf", "prince"]
ALL_ENGINES = LATEX_ENGINES + TYPST_ENGINES + HTML_ENGINES
CSL_URL = "https://raw.githubusercontent.com/citation-style-language/styles/master/{}.csl"

# Sidecar exports that can be appended, by export_review.py's file suffix.
SIDECARS = {
    "charting": ".charting.md",
    "appraisal": ".appraisal.md",
    "per-group": ".per-group.md",
    "narrative": ".narrative.md",
    "citations": ".citations.md",   # verify_citations.py's report
}
STYLES = ("apa", "nature", "vancouver", "ieee")

# Symbols the exporters emit that the default TeX fonts lack. Mapped to TeX so
# they print instead of disappearing with a "Missing character" log line.
TEX_SYMBOLS = {
    "✓": r"`$\checkmark$`{=latex}",
    "✔": r"`$\checkmark$`{=latex}",
    "✗": r"`$\times$`{=latex}",
    "✘": r"`$\times$`{=latex}",
    "⚠": r"`\textbf{!}`{=latex}",
    "⚡": "",
}
# One raw inline per output family; pandoc keeps only the one for the target.
LINE_BREAK = r"`\newline{}`{=latex}`#linebreak()`{=typst}"
PAGE_BREAK = ("\n```{=latex}\n\\newpage\n```\n\n```{=typst}\n#pagebreak()\n```\n\n"
              '```{=html}\n<div style="page-break-before: always"></div>\n```\n')
FENCE = re.compile(r"^\s*(```|~~~)")
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
    cache = csl_cache()
    for c in [near / f"{style}.csl", Path.cwd() / f"{style}.csl"] + ([cache / f"{style}.csl"] if cache else []):
        if c.is_file():
            return c
    try:
        with urllib.request.urlopen(CSL_URL.format(style), timeout=20) as r:
            body = r.read()
    except (urllib.error.URLError, OSError) as e:
        print(f"  warning: no CSL for '{style}' ({e}); using pandoc's default style", file=sys.stderr)
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


# ─────────────────────────── Markdown preparation ────────────────────────────

def _outside_fences(text: str, fn) -> str:
    """Apply ``fn`` to each line not inside a fenced code block."""
    out, fenced = [], False
    for line in text.split("\n"):
        if FENCE.match(line):
            fenced = not fenced
            out.append(line)
        else:
            out.append(line if fenced else fn(line))
    return "\n".join(out)


def neutralise_rules(text: str, keep_front_matter: bool) -> str:
    """Turn bare ``---`` rules into ``* * *``.

    The exporters separate sections with ``---``. Pandoc reads a ``---`` line
    followed by text ("---\n*Generated: …*") as the start of a YAML block — and
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


def demote_headings(text: str, by: int = 1) -> str:
    """``#`` → ``##`` etc., outside code fences (an appendix sits under the review)."""
    return _outside_fences(text, lambda ln: re.sub(r"^(#{1,6})(?=\s)", lambda m: "#" * min(6, len(m.group(1)) + by), ln))


def for_engine(text: str, engine: str) -> str:
    """Rewrite the HTML-isms and symbols an engine family can't print."""
    if engine in HTML_ENGINES:
        return text

    def fix(line: str) -> str:
        # a function, not a string: re.sub would read "\n" in "\newline" as a newline
        line = re.sub(r"<br\s*/?>", lambda _: LINE_BREAK, line, flags=re.I)
        line = re.sub(r"</?(?:details|summary)[^>]*>", "", line, flags=re.I)
        if engine in LATEX_ENGINES:
            for sym, tex in TEX_SYMBOLS.items():
                line = line.replace(sym, tex)
        return line
    return _outside_fences(text, fix)


def replace_references(text: str, styled: str) -> tuple[str, bool]:
    """Swap the body of ``## References`` for a styled list; (text, replaced?)."""
    lines = text.split("\n")
    start = next((i for i, ln in enumerate(lines) if re.match(r"^##\s+References\s*$", ln)), None)
    if start is None:
        return text, False
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    tail_rule = any(ln.strip() in ("---", "* * *") for ln in lines[start:end][-3:])
    body = styled.split("\n")
    if body and body[0].startswith("## "):  # verify_citations' "## References (style)"
        body = body[1:]
    new = ["## References", ""] + [ln for ln in body] + (["", "* * *", ""] if tail_rule else [""])
    return "\n".join(lines[:start] + new + lines[end:]), True


def first_heading_is_title(text: str) -> bool:
    """True when the document opens (after blank lines) with a single ``# `` heading."""
    for ln in text.split("\n"):
        if ln.strip():
            return bool(re.match(r"^#\s+\S", ln))
    return False


def citation_gate(md: Path, allow: bool) -> Optional[str]:
    """Refusal message if a citation report next to ``md`` has errors."""
    report = md.with_name(f"{md.stem}.citations.json")
    if not report.is_file():
        print(f"  warning: {report.name} not found — citations unverified; "
              f"run verify_citations.py on {md.stem}.json first", file=sys.stderr)
        return None
    try:
        summary = json.loads(report.read_text(encoding="utf-8")).get("summary") or {}
    except (OSError, json.JSONDecodeError) as e:
        return f"{report.name} is unreadable ({e}); re-run verify_citations.py"
    if report.stat().st_mtime < md.stat().st_mtime:
        print(f"  warning: {report.name} is older than {md.name}; re-run verify_citations.py "
              "after the last export", file=sys.stderr)
    errors = int(summary.get("errors") or 0)
    if errors and not allow:
        codes = ", ".join(f"{k} {v}" for k, v in (summary.get("by_code") or {}).items())
        return (f"{report.name} reports {errors} citation error(s) ({codes}).\n"
                "  Fix them (see references/citation_verification.md) and re-export, or pass "
                "--allow-citation-errors to build anyway.")
    if errors:
        print(f"  warning: building with {errors} unresolved citation error(s)", file=sys.stderr)
    return None


def assemble(md: Path, engine: str, references: Optional[str], include: list[str]) -> tuple[str, list[str]]:
    """The Markdown pandoc will read, plus notes for the user. Raises ValueError."""
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
    for name in include:
        side = md.with_name(md.stem + SIDECARS[name])
        if not side.is_file():
            raise ValueError(f"{side.name} not found — export it with "
                             f"export_review.py --formats {name}" if name != "citations"
                             else f"{side.name} not found — run verify_citations.py first")
        side_text = neutralise_rules(side.read_text(encoding="utf-8"), keep_front_matter=False)
        text += "\n" + PAGE_BREAK + "\n" + demote_headings(side_text) + "\n"
        notes.append(f"appendix: {side.name}")
    return for_engine(text, engine), notes


# ─────────────────────────── pandoc ────────────────────────────

def build_command(src: Path, md: Path, out: Path, engine: str, *, csl: Optional[Path],
                  bib: Optional[Path], toc: bool, number_sections: bool,
                  template: Optional[str], title_from_h1: bool, declared: set[str],
                  css: Optional[Path]) -> list[str]:
    cmd = ["pandoc", str(src), "-f", "markdown", "-o", str(out), f"--pdf-engine={engine}",
           # images resolve relative to the review, not to the temp copy
           "--resource-path", f"{md.parent}{os.pathsep}.", "--standalone",
           f"--columns={WRAP_COLUMNS}"]
    if title_from_h1:
        # The leading "# Title" becomes the document title; "## 1. …" become top level.
        cmd.append("--shift-heading-level-by=-1")
    elif "title" not in declared:
        cmd += ["--metadata", f"title={md.stem.replace('_', ' ').replace('-', ' ').title()}"]

    if engine in LATEX_ENGINES:
        cmd += ["-V", "geometry:margin=2cm", "-V", "fontsize=10pt", "-V", "linestretch=1.1",
                "-V", "colorlinks=true", "-V", "linkcolor=blue", "-V", "urlcolor=blue",
                "-V", "citecolor=blue", "-V", "papersize=a4"]
    elif engine in TYPST_ENGINES:
        cmd += ["-V", "fontsize=10pt", "-V", "papersize=a4"]
    elif css:
        cmd += ["--css", str(css), "--embed-resources"]

    if toc:
        cmd += ["--toc", "--toc-depth=2"]
    if number_sections:
        cmd.append("--number-sections")
    if bib:
        cmd += ["--citeproc", "--bibliography", str(bib), "-M", "link-citations=true"]
    if csl:
        cmd += ["--csl", str(csl)]
    if template:
        cmd += ["--template", template]
    return cmd


def generate_pdf(markdown_file: str, output_pdf: Optional[str] = None, *,
                 citation_style: str = "apa", references: Optional[str] = None,
                 include: Optional[list[str]] = None, template: Optional[str] = None,
                 toc: bool = True, number_sections: bool = False,
                 pdf_engine: Optional[str] = None, bibliography: Optional[str] = None,
                 allow_citation_errors: bool = False, dry_run: bool = False,
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

    refusal = citation_gate(md, allow_citation_errors)
    if refusal:
        print(f"Error: {refusal}", file=sys.stderr)
        return False

    try:
        text, notes = assemble(md, engine, references, include or [])
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return False

    out = Path(output_pdf) if output_pdf else md.with_suffix(".pdf")
    out.parent.mkdir(parents=True, exist_ok=True)

    declared = yaml_keys(text)
    title_from_h1 = "title" not in declared and first_heading_is_title(text)
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
    csl = resolve_csl(citation_style, md.parent) if (bib or "bibliography" in declared) and uses_cites else None

    with tempfile.TemporaryDirectory(prefix="review-pdf-") as tmp:
        src = Path(tmp) / f"{md.stem}.pandoc.md"
        src.write_text(text, encoding="utf-8")
        css = None
        if engine in HTML_ENGINES:
            css = Path(tmp) / "review.css"
            css.write_text(HTML_CSS, encoding="utf-8")
        cmd = build_command(src, md, out, engine, csl=csl, bib=bib, toc=toc,
                            number_sections=number_sections, template=template,
                            title_from_h1=title_from_h1, declared=declared, css=css)
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
    warnings = [ln for ln in result.stderr.splitlines() if ln.strip()]
    for line in warnings[:10]:
        print(f"  {line}", file=sys.stderr)
    if len(warnings) > 10:
        print(f"  … {len(warnings) - 10} more warning line(s)", file=sys.stderr)
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
    print("\nReady." if have_pandoc and found else "\nNot ready: install the items marked MISSING.")
    return have_pandoc and bool(found)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("markdown_file", nargs="?", help="exported review Markdown (review.md)")
    p.add_argument("-o", "--output", help="output PDF (default: next to the Markdown)")
    p.add_argument("--references", choices=STYLES,
                   help="replace the References section with <stem>.references.<style>.md "
                        "from verify_citations.py --style")
    p.add_argument("--include", nargs="+", choices=list(SIDECARS), default=[], metavar="NAME",
                   help="append sidecar exports as appendices: " + ", ".join(SIDECARS))
    p.add_argument("--allow-citation-errors", action="store_true",
                   help="build even if <stem>.citations.json reports errors")
    p.add_argument("--citation-style", default="apa",
                   help="CSL style id or .csl path, for [@key] citations only (default: apa)")
    p.add_argument("--bibliography", help="explicit .bib for [@key] citations")
    p.add_argument("--pdf-engine", help=f"force an engine ({', '.join(ALL_ENGINES)})")
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
                      allow_citation_errors=args.allow_citation_errors, dry_run=args.dry_run,
                      keep_source=args.keep_source, timeout=args.timeout)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
