#!/usr/bin/env python3
"""Flag AI-sounding and informal writing in a review's own prose.

The review is a formal scientific document; generated text drifts toward the
statistical-average voice (inflated significance, stacked hedges, "numerous
studies", "delve", a closing "paves the way"). This checker finds those
patterns sentence by sentence so they can be rewritten before export, following
``references/writing_style.md``. It does not rewrite anything itself.

Scope — only prose the review wrote, never what a study said:

* JSON (``review.json``): abstract, introduction, methods/results narrative,
  synthesis and themes, discussion, conclusion, limitations, bias and GRADE
  summaries, per-group answers, narrative-table cells. Evidence spans, charted
  answers, appraisal items, included articles, the screening log, the protocol,
  search strings and run metadata are skipped (``--all`` adds charting and
  appraisal text, which quote studies, as warnings only).
* Markdown (``review.md`` or any manuscript): the body up to the reference
  list, skipping code blocks, tables, headings and block quotes.

Severity: **error** = must not ship (chatbot artifacts, knowledge-cutoff
disclaimers, emojis, reader address). **warning** = rewrite (AI vocabulary,
stacked hedges, "significant" with no statistic, vague quantifiers, causal
overclaim, contractions, …). **info** = check (sentence-initial "Moreover",
dangling "This …", "novel").

Usage:
    python check_writing.py out/review.json
    python check_writing.py out/review.md
    python check_writing.py out/review.json --strict      # warnings fail too
    python check_writing.py out/review.json --all         # + charting / appraisal text

Writes ``<stem>.writing.md`` / ``.writing.json`` for a JSON input
(``<name>.writing.*`` for Markdown, so the two never overwrite each other) next to the input
(``--outdir`` to change). Exit 0 = no errors, 1 = errors (or warnings under
``--strict``), 2 = bad input. Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

# ─────────────────────────── scope ────────────────────────────

# Top-level review fields that hold the review's own prose.
PROSE_FIELDS = (
    "prisma_review", "synthesis_text", "bias_assessment", "limitations",
    "structured_abstract", "introduction_text", "conclusions_text",
    "grade_assessments", "per_group_analysis", "narrative_rows",
)
# Text quoted or extracted from studies: --all only, never errors.
QUOTED_FIELDS = ("data_charting_rubrics", "critical_appraisals", "structured_appraisal_results")
# Leaves that are inputs, identifiers or labels, not prose, wherever they occur.
SKIP_KEYS = {
    "question", "question_text", "question_id", "short_title", "theme", "query", "queries",
    "search_strategy", "label", "dimension", "source_id", "title", "pmid", "pmids",
    "doi", "references", "representative_pmids", "supporting_pmids", "id", "rating",
    "certainty", "overall", "judgement", "judgment", "status", "outcome", "domain_name",
}

# ─────────────────────────── patterns ────────────────────────────
# (code, severity, regex, guide pattern #, advice). Matched per sentence,
# case-insensitive unless the regex says otherwise.

W = r"\b"
PATTERNS: list[tuple[str, str, str, str, str]] = [
    # must not ship
    ("chatbot-artifact", "error",
     r"\b(i hope this helps|let me know if|feel free to|as an ai\b|as a language model|great question|"
     r"certainly[!,]|here is (the|a|an) (synthesis|summary|review|draft)|in this response|i'd be happy to)",
     "19", "delete it"),
    ("cutoff-disclaimer", "error",
     r"\b(as of my (last|knowledge)|my training data|based on (the )?available information|"
     r"while (specific )?details are limited|up to my knowledge)",
     "20", "delete it; the review states its search dates"),
    ("reader-address", "error", r"\b(you|your|you're|yourself)\b", "38",
     "state the implication; no reader address"),
    # AI vocabulary / promotional
    ("ai-vocabulary", "warning",
     r"\b(delve[sd]?|delving|tapestry|testament|landscape|underscor(e|es|ed|ing)|showcas(e|es|ed|ing)|"
     r"pivotal|intricate|multifaceted|realm|paramount|seamless(ly)?|holistic|synerg(y|ies|istic)|"
     r"leverag(e|es|ed|ing)|harness(es|ed|ing)?|myriad|plethora|ever-evolving|game-chang(er|ing)|"
     r"groundbreaking|cutting-edge|transformative|revolutioni[sz](e|es|ed|ing)|unprecedented|"
     r"crucial role|vital role|key role|shed(s|ding)? light|navigat(e|es|ing) the)\b",
     "7", "use the plain word"),
    ("copula-avoidance", "warning", r"\b(serves as|stands as|boasts|acts as a testament)\b", "8",
     "'is' / 'has'"),
    ("negative-parallelism", "warning", r"\bnot (just|only|merely)\b[^.;]{1,80}\b(but|it's|it is)\b",
     "9", "state the point directly"),
    ("ing-tail", "warning",
     r"[,—]\s*(highlighting|underscoring|emphasi[sz]ing|reflecting|showcasing|demonstrating|illustrating)"
     r"\s+(the|its|their|a|an|how)\b", "3", "cut the tail, or make it a sourced sentence"),
    ("vague-attribution", "warning",
     r"\b(experts (believe|agree|suggest)|it is (well[- ]known|widely (accepted|recognized|recognised|believed))|"
     r"(research|studies|evidence|literature) (has |have )?(consistently )?(shown|show|demonstrated|suggest(s|ed)?))\b",
     "5/26", "cite the studies and give the count"),
    ("generic-conclusion", "warning",
     r"\b(paves? the way|pave the way|open(s|ing)? (up )?new avenues|holds? (great |much |significant )?promise|"
     r"the future (looks|is) bright|exciting (new )?(avenues|opportunities|future)|"
     r"remains? to be seen|only time will tell)\b", "24", "name the gap and the study that would close it"),
    ("causal-overclaim", "warning",
     r"\b(prove[sd]?|proving|proven|conclusively|definitively|establish(es|ed)? that|confirm(s|ed)? that)\b",
     "25", "size the claim to the design ('is consistent with', 'was associated with')"),
    ("hype-adjective", "info", r"\b(novel|robust|powerful|remarkable|innovative|comprehensive)\b", "27",
     "say what is new or measured; 'robust' only in its statistical sense"),
    # quantification
    ("significant-without-test", "warning", r"\bsignificant(ly)?\b", "30",
     "give the test statistic, or the magnitude"),
    ("vague-quantifier", "warning",
     r"\b(many|numerous|several|various|a (large |small )?number of|a (wide |broad )?range of|"
     r"a (vast )?majority of|most|a few|a handful of|countless) (of the )?"
     r"(studies|trials|papers|articles|reports|participants|patients|samples|cohorts|datasets)\b",
     "31", "give the count: 'seven of 12 studies'"),
    ("unquantified-magnitude", "warning",
     r"\b(substantially|dramatically|markedly|considerably|greatly|vastly|drastically|massively|"
     r"significantly (higher|lower|better|worse|improved|reduced))\b", "29", "give the number"),
    ("numeric-style", "warning",
     r"(\bp\s*[<=>]\s*\.\d|\bp[<=>]\S|\b\d+(\.\d+)?(mg|kg|ml|mL|µg|ug|ms|Hz|kHz|nm|µm|um|mm|cm)\b|~\s?\d+\s?fold)",
     "32", "'p = 0.03', '5 mg', '10-fold'"),
    # hedging, filler, register
    ("stacked-hedge", "warning",
     r"\b(may|might|could|can)\s+(possibly|potentially|perhaps|conceivably|likely)\b|"
     r"\b(possibly|potentially|perhaps)\s+(may|might|could|suggest\w*|indicate\w*)\b|"
     r"\b(appears?|seems?) to (possibly|potentially|suggest)\b", "23", "keep one hedge"),
    ("filler", "warning",
     r"\b(in order to|due to the fact that|it is (important|worth|interesting) to (note|mention|highlight)|"
     r"it should be noted that|it is worth noting|needless to say|at the end of the day|"
     r"in terms of|the fact that|for all intents and purposes|first and foremost)\b",
     "22", "cut the filler ('to', 'because', or delete)"),
    ("contraction", "warning",
     r"\b(don't|doesn't|didn't|isn't|aren't|wasn't|weren't|can't|couldn't|won't|wouldn't|shouldn't|"
     r"it's|that's|there's|we're|they're|let's|we've|they've|i'm|hasn't|haven't)\b", "37",
     "write it out: 'did not'"),
    ("conversational", "warning",
     r"\b(looked into|came up with|figur(e|ed|ing) out|a lot of|lots of|pretty (much|close|good|clear)|"
     r"big deal|kind of|sort of|a bit|basically|actually|really|huge|tons of|okay|ok)\b", "37",
     "formal equivalent ('examined', 'developed', the count)"),
    ("anthropomorphism", "warning",
     r"\b(the (data|evidence|results?|literature|model) (tells?|says?|wants?|believes?|thinks?|knows?)|"
     r"tells us)\b", "33", "'the data indicate', 'the model assigns'"),
    ("open-list", "warning", r"(\betc\.?|\band so on\b|\band so forth\b|\bamong others\b)", "35",
     "name every item, or 'including' plus a table"),
    ("rhetorical-question", "warning", r"\?\s*$", "38", "make it a statement"),
    ("emoji", "error",
     "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF\U0001F900-\U0001F9FF]", "17",
     "remove it"),
    # style signals (info)
    ("sentence-initial-connective", "info",
     r"^(moreover|furthermore|additionally|in addition|notably|importantly|interestingly|overall|"
     r"ultimately|in conclusion|in summary)\b", "7", "often removable; vary or cut"),
    ("dangling-this", "info",
     r"^(this|these)\s+(is|are|was|were|suggests?|shows?|indicates?|highlights?|means|explains?|"
     r"demonstrates?|underscores?|reflects?|implies|confirms?|supports?)\b", "34",
     "name what 'this' refers to"),
]
COMPILED = [(c, s, re.compile(rx, re.I), n, a) for c, s, rx, n, a in PATTERNS]
# A sentence that carries its own statistic is allowed "significant(ly)".
STATISTIC = re.compile(
    r"(\bp\s*[<=>≤≥]|\bCI\b|confidence interval|\b(OR|HR|RR|SMD|MD|IRR|AUC|AUROC|β|beta|"
    r"χ2|χ²|chi-?square|I2|I²|r|ρ|d|g)\s*[=:]|\bt\(\d|\bF\(\d|\bz\s*=|statistically|"
    r"\b(test|regression|ANOVA|Wilcoxon|Mann-Whitney)\b)", re.I)
HAS_NUMBER = re.compile(r"\d")
EM_DASH = re.compile(r"—|(?<=\s)--(?=\s)")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")

SEVERITY_RANK = {"info": 0, "warning": 1, "error": 2}


def sentences(text: str) -> list[str]:
    flat = re.sub(r"\s+", " ", text or "").strip()
    return [s for s in SENTENCE_SPLIT.split(flat) if s]


def check_text(text: str, location: str, quoted: bool = False) -> list[dict]:
    """Findings for one prose field."""
    out: list[dict] = []
    for sent in sentences(text):
        for code, sev, rx, num, advice in COMPILED:
            m = rx.search(sent)
            if not m:
                continue
            if code == "significant-without-test" and STATISTIC.search(sent):
                continue
            if code in ("vague-quantifier", "unquantified-magnitude") and HAS_NUMBER.search(sent):
                continue   # the number is there; the word may just be context
            if code == "reader-address" and re.search(r"\byou(r)?\b", sent) is None:
                continue
            if quoted:
                sev = "info" if sev == "info" else "warning"   # quoted text: never an error
            out.append({"code": code, "severity": sev, "pattern": num, "advice": advice,
                        "match": m.group(0), "sentence": sent[:400], "location": location})
    dashes = len(EM_DASH.findall(text or ""))
    words = len((text or "").split())
    if dashes > 2 and dashes * 100 > words:
        out.append({"code": "em-dash-density", "severity": "warning", "pattern": "13",
                    "advice": "commas or separate sentences; at most one parenthetical dash per paragraph",
                    "match": f"{dashes} em dashes in {words} words", "sentence": "",
                    "location": location})
    return out


# ─────────────────────────── walking the inputs ────────────────────────────

def _leaves(node: Any, path: str) -> Iterable[tuple[str, str]]:
    if isinstance(node, dict):
        for k, v in node.items():
            if k in SKIP_KEYS:
                continue
            yield from _leaves(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _leaves(v, f"{path}[{i}]")
    elif isinstance(node, str) and len(node.split()) >= 3:
        yield path, node


def check_review(review: dict, include_quoted: bool) -> list[dict]:
    findings: list[dict] = []
    for field in PROSE_FIELDS:
        if field in review:
            for loc, text in _leaves(review[field], field):
                findings += check_text(text, loc)
    if include_quoted:
        for field in QUOTED_FIELDS:
            if field in review:
                for loc, text in _leaves(review[field], field):
                    findings += check_text(text, loc, quoted=True)
    return findings


def check_markdown(text: str) -> list[dict]:
    """The body up to the reference list: no code, tables, headings or quotes."""
    m = re.search(r"^#{1,3}\s*(?:\d+\.?\s*)?(References|Bibliography)\s*$", text, re.M | re.I)
    body = text[:m.start()] if m else text
    findings: list[dict] = []
    section, para, fence = "body", [], False

    def flush() -> None:
        if para:
            findings.extend(check_text(" ".join(para), section))
            para.clear()

    for line in body.split("\n"):
        if re.match(r"^\s*(```|~~~)", line):
            fence = not fence
            flush()
            continue
        if fence:
            continue
        h = re.match(r"^#{1,6}\s+(.*)", line)
        if h:
            flush()
            section = h.group(1).strip()
            continue
        if not line.strip() or line.lstrip().startswith(("|", ">", "<!--")) or re.match(r"^\s*(---|\*\*\*)\s*$", line):
            flush()
            continue
        para.append(re.sub(r"^\s*([-*+]|\d+\.)\s+", "", line))
    flush()
    return findings


# ─────────────────────────── report ────────────────────────────

def summarise(findings: list[dict]) -> dict:
    by_code: dict[str, int] = {}
    for f in findings:
        by_code[f["code"]] = by_code.get(f["code"], 0) + 1
    return {"errors": sum(f["severity"] == "error" for f in findings),
            "warnings": sum(f["severity"] == "warning" for f in findings),
            "info": sum(f["severity"] == "info" for f in findings),
            "by_code": dict(sorted(by_code.items(), key=lambda kv: -kv[1]))}


def render_markdown(findings: list[dict], summary: dict, source: str) -> str:
    out = [f"# Writing check — {Path(source).name}\n",
           f"**{summary['errors']} error(s), {summary['warnings']} warning(s), {summary['info']} note(s).** "
           "Rewrite flagged sentences in the review JSON following "
           "`references/writing_style.md`; never change numbers, citations or quoted text.\n"]
    if summary["by_code"]:
        out.append("| Pattern | Count |\n| --- | --- |")
        out += [f"| {k} | {v} |" for k, v in summary["by_code"].items()]
        out.append("")
    by_loc: dict[str, list[dict]] = {}
    for f in sorted(findings, key=lambda f: (-SEVERITY_RANK[f["severity"]], f["location"])):
        by_loc.setdefault(f["location"], []).append(f)
    for loc, fs in by_loc.items():
        out.append(f"## `{loc}`\n")
        for f in fs:
            line = f"- **{f['severity']}** `{f['code']}` (#{f['pattern']}): “{f['match']}” — {f['advice']}"
            out.append(line)
            if f["sentence"]:
                out.append(f"  > {f['sentence']}")
        out.append("")
    if not findings:
        out.append("No patterns flagged. Still read the prose once for claims sized past their "
                   "GRADE certainty and results reported without study counts.\n")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="review JSON (PRISMAReviewResult) or a Markdown file")
    ap.add_argument("--outdir", help="report directory (default: next to the input)")
    ap.add_argument("--all", action="store_true",
                    help="also check charting / appraisal text (quotes studies; warnings only)")
    ap.add_argument("--strict", action="store_true", help="warnings also fail the exit code")
    ap.add_argument("--quiet", action="store_true", help="no per-finding output on stderr")
    args = ap.parse_args()

    src = Path(args.input)
    if not src.is_file():
        print(f"no such file: {src}", file=sys.stderr)
        return 2
    text = src.read_text(encoding="utf-8")
    if src.suffix.lower() == ".json":
        try:
            review = json.loads(text)
        except json.JSONDecodeError as e:
            print(f"{src}: not valid JSON ({e})", file=sys.stderr)
            return 2
        if not isinstance(review, dict):
            print(f"{src}: not a review result", file=sys.stderr)
            return 2
        findings = check_review(review, args.all)
    else:
        findings = check_markdown(text)

    summary = summarise(findings)
    outdir = Path(args.outdir) if args.outdir else src.parent
    outdir.mkdir(parents=True, exist_ok=True)
    # review.json → review.writing.*; review.md → review.md.writing.* (no clash)
    base = outdir / (src.stem if src.suffix.lower() == ".json" else src.name)
    Path(f"{base}.writing.json").write_text(
        json.dumps({"file": str(src), "summary": summary, "findings": findings},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    Path(f"{base}.writing.md").write_text(render_markdown(findings, summary, str(src)), encoding="utf-8")

    if not args.quiet:
        for f in findings:
            if f["severity"] != "info":
                print(f"[{f['severity']:>7}] {f['code']:<28} {f['location']}: “{f['match']}”", file=sys.stderr)
    print(f"\n{summary['errors']} error(s), {summary['warnings']} warning(s), {summary['info']} note(s)"
          f" | report: {base}.writing.md", file=sys.stderr)
    return 1 if summary["errors"] or (args.strict and summary["warnings"]) else 0


if __name__ == "__main__":
    sys.exit(main())
