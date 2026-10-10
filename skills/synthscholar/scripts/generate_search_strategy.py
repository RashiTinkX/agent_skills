#!/usr/bin/env python3
"""Generate a deterministic PubMed search strategy from a research question.

The skill-side entry point to the search-strategy engine vendored from
``sensein/search-strategy-generator`` (``scripts/search_strategy/``). It runs the
same question -> concept blocks -> PubMed query pipeline the web app ran, without
the web app:

1. **Map** the question to MeSH-grounded concept blocks (``pipeline.build``).
2. **Compile** the blocks plus filters into a PubMed boolean query.
3. **Count** the hits on PubMed (``--count``), and optionally **fetch** the
   records to CSV / JSONL with an audit ``protocol.json`` (``--fetch``).

Four mapping modes (``--mode``):

  closure    (default) the LLM only marks WHERE each PICO-style concept sits in
             the question, sampled k times and majority-voted; vocabulary is then
             resolved from the MeSH index alone. Most model-independent.
  hybrid     a deterministic MeSH candidate slate; the LLM only selects ids.
  llm        the LLM proposes blocks and headings freely.
  mesh_only  no LLM at all: every maximal MeSH match becomes a block.

One-time setup — the MeSH index is built locally from NLM's RDF dump (the dump
and the index are large and are not committed):

    # download https://nlmpubs.nlm.nih.gov/projects/mesh/rdf/2026/mesh2026.nt.gz and gunzip
    python generate_search_strategy.py --build-index /path/to/mesh.nt

Usage:
    python generate_search_strategy.py "vocal biomarkers for depression"
    python generate_search_strategy.py "..." --mode mesh_only --count
    python generate_search_strategy.py "..." --date-from 2015 --species humans \\
        --languages english --count
    python generate_search_strategy.py "..." --fetch 5000 --outdir searches/
    python generate_search_strategy.py "..." --json > strategy.json

Environment:
    OPENROUTER_API_KEY   required by every mode except mesh_only
    OPENROUTER_MODEL     default model (else anthropic/claude-3.5-sonnet)
    NCBI_API_KEY         optional; raises the E-utilities rate limit
    NCBI_EMAIL           optional; NCBI asks for a contact address

Requires ``requests`` and ``httpx``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from search_strategy import build_index, pipeline, query_builder  # noqa: E402
from search_strategy.mesh_index import DEFAULT_DB  # noqa: E402
from search_strategy.openrouter_client import OpenRouterError  # noqa: E402
from search_strategy.pubmed import PubMed, to_csv, to_jsonl  # noqa: E402


def _filters(args: argparse.Namespace) -> dict:
    return {
        "date_from": args.date_from,
        "date_to": args.date_to,
        "languages": args.languages,
        "article_types_include": args.include_type,
        "article_types_exclude": args.exclude_type,
        "species": args.species,
        "exclude_terms": args.exclude_term,
    }



def _pubmed() -> PubMed:
    return PubMed(api_key=os.environ.get("NCBI_API_KEY", ""),
                  email=os.environ.get("NCBI_EMAIL", ""))


def _fetch(pm: PubMed, query: str, search: dict, max_records: int, outdir: Path) -> dict:
    """Mirror of the web app's /api/search: records + an audit protocol.json."""
    # fetch_query, not fetch_all: past 9,999 records NCBI's history server refuses
    # retstart, so the query has to be split by publication date.
    fetched = pm.fetch_query(query, max_records=max_records or None)
    articles = fetched["articles"]
    qhash = query_builder.query_hash(query)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    folder = outdir / f"{stamp}_{qhash}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "results.csv").write_text(to_csv(articles), encoding="utf-8")
    (folder / "results.jsonl").write_text(to_jsonl(articles), encoding="utf-8")
    protocol = {
        "query": query,
        "query_hash": qhash,
        "pubmed_translation": search["translation"],
        "total_count": search["count"],
        "fetched": len(articles),
        "capped": fetched["capped"],
        # audit trail for a >9,999-hit search: which date slices were walked, and
        # whether any records fell outside them
        "date_slices": [{k: s[k] for k in ("from", "to", "count", "truncated")}
                        for s in fetched["slices"]] if len(fetched["slices"]) > 1 else [],
        "records_unaccounted": fetched["missing"],
        "timestamp": stamp,
    }
    (folder / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    return {**protocol, "folder": str(folder)}



def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("question", nargs="?", default="", help="the research question")
    ap.add_argument("--mode", default="closure", choices=pipeline.MODES,
                    help="mapping mode (default: closure)")
    ap.add_argument("--model", default=None, help="model id (default: $OPENROUTER_MODEL)")
    ap.add_argument("--facet-runs", type=int, default=3, metavar="K",
                    help="closure mode: samples for self-consistency voting (default 3)")
    ap.add_argument("--prompt-version", default=None, choices=("v1", "v2"),
                    help="llm mode only")
    ap.add_argument("--domains", nargs="*", default=[],
                    help="domain-vocabulary clusters from data/domain_terms.json")
    ap.add_argument("--extra-context", default="", help="extra guidance for the model")
    ap.add_argument("--no-strict", action="store_true",
                    help="keep the model's free-text terms instead of index-derived synonyms")
    # filters
    ap.add_argument("--date-from", default="", help="YYYY or YYYY/MM/DD")
    ap.add_argument("--date-to", default="", help="YYYY or YYYY/MM/DD")
    ap.add_argument("--languages", nargs="*", default=[], help="e.g. english")
    ap.add_argument("--species", default="", choices=("", "humans", "animals"))
    ap.add_argument("--include-type", action="append", default=[], metavar="PT",
                    help="publication type to require, e.g. 'Randomized Controlled Trial'")
    ap.add_argument("--exclude-type", action="append", default=[], metavar="PT",
                    help="publication type to exclude, e.g. Review")
    ap.add_argument("--exclude-term", action="append", default=[], metavar="TERM",
                    help="Title/Abstract term to NOT")
    # PubMed
    ap.add_argument("--count", action="store_true", help="count hits on PubMed")
    ap.add_argument("--fetch", type=int, default=None, metavar="N",
                    help="fetch up to N records (0 = all) into --outdir")
    ap.add_argument("--outdir", default="searches", help="where --fetch writes (default searches/)")
    # outputs
    ap.add_argument("--json", action="store_true", help="print the full result as JSON")
    ap.add_argument("--build-index", default="", metavar="MESH_NT",
                    help="build the local MeSH index from an NLM mesh.nt dump and exit")
    args = ap.parse_args()

    if args.build_index:
        build_index.build(args.build_index, DEFAULT_DB)
        return 0
    if not args.question.strip():
        ap.error("a question is required (or use --build-index)")
    if not os.path.exists(DEFAULT_DB) or os.path.getsize(DEFAULT_DB) == 0:
        print(f"MeSH index not found at {DEFAULT_DB}.\n"
              "Download https://nlmpubs.nlm.nih.gov/projects/mesh/rdf/2026/mesh2026.nt.gz, gunzip it, "
              "then run:\n  python generate_search_strategy.py --build-index /path/to/mesh.nt",
              file=sys.stderr)
        return 2

    try:
        built = pipeline.build(
            args.question, domains=args.domains or None, mode=args.mode, model=args.model,
            extra_context=args.extra_context, prompt_version=args.prompt_version,
            strict=not args.no_strict, facet_runs=args.facet_runs,
        )
    except OpenRouterError as e:
        print(f"LLM call failed: {e}\n(--mode mesh_only needs no LLM)", file=sys.stderr)
        return 1

    filters = _filters(args)
    compiled = query_builder.compile_search(built["concepts"], filters)
    result = {
        "question": args.question,
        "mode": built["mode"],
        "model": built["model"],
        "notes": built["notes"],
        "concepts": built["concepts"],
        "dropped": built["dropped"],
        "filters": filters,
        **compiled,
    }

    if args.count or args.fetch is not None:
        pm = _pubmed()
        search = pm.search(compiled["query"])
        result.update(count=search["count"], translation=search["translation"],
                      warnings=search["warnings"], errors=search["errors"])
        if args.fetch is not None and search["count"]:
            result["fetch"] = _fetch(pm, compiled["query"], search, args.fetch,
                                     Path(args.outdir))

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print(f"mode: {result['mode']}   model: {result['model'] or '-'}")
    if result["notes"]:
        print(f"notes: {result['notes']}")
    for i, c in enumerate(result["concepts"], 1):
        mesh = "; ".join(c["mesh"]) or "-"
        print(f"  block {i} [{c['slot']}] {c['name']}: MeSH {mesh}  (+{len(c['freetext'])} text terms)")
    if result["dropped"]:
        print(f"  dropped (no exact MeSH match): {result['dropped']}")
    print(f"\nquery ({compiled['n_concepts']} blocks, {compiled['n_terms']} terms, "
          f"hash {compiled['hash']}):\n{compiled['query']}")
    if "count" in result:
        print(f"\nPubMed hits: {result['count']:,}")
        # esearch's warninglist / errorlist: {kind: [messages]}
        for kind, msgs in {**result["warnings"], **result["errors"]}.items():
            if msgs:
                print(f"  ! {kind}: {msgs}")
    if "fetch" in result:
        f = result["fetch"]
        print(f"fetched {f['fetched']:,} records -> {f['folder']}"
              + ("  (capped)" if f["capped"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
