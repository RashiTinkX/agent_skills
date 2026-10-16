"""
No-network regression checks for SynthScholar's search-strategy engine
(skills/synthscholar/scripts/search_strategy/): closure-mode facet voting,
deterministic MeSH closure, and coverage-gap recovery.

The voting / heuristic tests need nothing. The closure and coverage-gap tests
use the real local MeSH index (a static, offline fixture built with
``generate_search_strategy.py --build-index``); they skip when it is absent,
as on CI.
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import types
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "skills" / "synthscholar" / "scripts"))

try:
    import requests  # noqa: F401
except ImportError:
    # The engine only touches requests when it calls a model or PubMed, which
    # these tests never do; a placeholder lets the modules import on CI.
    sys.modules["requests"] = types.ModuleType("requests")

from search_strategy import closure, facet_pipeline, facets  # noqa: E402
from search_strategy.mesh_index import DEFAULT_DB, get_index  # noqa: E402

HAVE_INDEX = os.path.exists(DEFAULT_DB) and os.path.getsize(DEFAULT_DB) > 0
needs_index = unittest.skipUnless(
    HAVE_INDEX, "MeSH index not built (generate_search_strategy.py --build-index)")


class HeuristicFacetTests(unittest.TestCase):
    def test_splits_on_conjunctions_and_guesses_roles(self):
        q = "Effect of RNA sequencing on hippocampal neurons in Alzheimer Disease patients"
        names = [f["name"] for f in facets.heuristic_facets(q)]
        self.assertTrue(any("RNA sequencing" in n for n in names))
        self.assertTrue(any("Alzheimer Disease" in n for n in names))

    def test_deterministic_across_runs(self):
        q = "Effect of metformin on cardiovascular outcomes in type 2 diabetes patients"
        first = facets.heuristic_facets(q)
        for _ in range(5):
            self.assertEqual(facets.heuristic_facets(q), first)


class VotingTests(unittest.TestCase):
    def test_majority_span_wins_over_minority(self):
        # 2 of 3 runs agree on the same (start,end); the 3rd run disagrees
        # slightly -- majority should win, and agreement count is exact.
        runs = [
            [{"name": "RNA sequencing", "role": "method", "start": 5, "end": 6}],
            [{"name": "RNA sequencing", "role": "method", "start": 5, "end": 6}],
            [{"name": "the RNA sequencing step", "role": "method", "start": 4, "end": 7}],
        ]
        voted = facets._cluster_and_vote(runs, min_runs_agree=2)
        self.assertEqual(len(voted), 1)
        self.assertEqual((voted[0]["start"], voted[0]["end"]), (5, 6))
        self.assertEqual(voted[0]["agreement"], 3)  # all 3 runs overlap this span

    def test_below_threshold_cluster_is_dropped(self):
        runs = [
            [{"name": "a", "role": "other", "start": 0, "end": 0}],
            [],
            [],
        ]
        self.assertEqual(facets._cluster_and_vote(runs, min_runs_agree=2), [])

    def test_role_normalizer_rejects_echoed_enum(self):
        self.assertEqual(facets._normalize_pico_role(
            "population|intervention|comparator|outcome|method|context"), "other")
        self.assertEqual(facets._normalize_pico_role("Population"), "population")
        self.assertEqual(facets._normalize_pico_role("garbage"), "other")

    def test_no_key_falls_back_to_heuristic(self):
        q = "Does metformin reduce cardiovascular mortality in patients with type 2 diabetes?"
        old = os.environ.pop("OPENROUTER_API_KEY", None)
        try:
            seg = facets.segment(q, model="anthropic/claude-haiku-4.5", k=3)
        finally:
            if old is not None:
                os.environ["OPENROUTER_API_KEY"] = old
        self.assertEqual(seg["mode"], "heuristic")
        self.assertEqual(seg["facets"], facets.heuristic_facets(q))


@needs_index
class ClosureTests(unittest.TestCase):
    def setUp(self):
        self.ix = get_index()

    def test_generic_standalone_term_excluded(self):
        q = "Alzheimer Disease patients"
        facet = {"name": q, "role": "population", "start": 0, "end": 2}
        block = closure.build_facet(self.ix, q, facet)
        labels = [m["options"][0]["label"] for m in block["mesh"]]
        self.assertIn("Alzheimer Disease", labels)
        self.assertNotIn("Patients", labels)  # generic-term guard

    def test_epidemiological_generic_term_does_not_dilute_outcome(self):
        # "incidence" resolving to its own MeSH heading OR'd alongside
        # "Diarrhea" turned a specific outcome facet into
        # "Diarrhea OR Incidence-of-anything", diluting precision.
        q = "incidence of antibiotic-associated diarrhea"
        toks = closure.tokenize(q)
        facet = {"name": q, "role": "outcome", "start": 0, "end": len(toks) - 1}
        block = closure.build_facet(self.ix, q, facet)
        labels = [m["options"][0]["label"] for m in block["mesh"]]
        self.assertEqual(labels, ["Diarrhea"])

    def test_subsumption_pruning_keeps_broader_only(self):
        d_broad = self.ix.exact("Nervous System")
        d_narrow = self.ix.exact("Neurons")
        if not d_broad or not d_narrow:
            self.skipTest("index fixture missing expected descriptors")
        kept_labels = {d.label for d in closure.prune_subsumed(d_broad + d_narrow)}
        if closure._is_descendant(d_narrow[0], d_broad[0]):
            self.assertNotIn("Neurons", kept_labels)
            self.assertIn("Nervous System", kept_labels)

    def test_deterministic_repeat(self):
        q = "RNA sequencing of hippocampal neurons"
        facet = {"name": q, "role": "method", "start": 0, "end": 5}
        first = closure.build_facet(self.ix, q, facet)
        for _ in range(3):
            again = closure.build_facet(self.ix, q, facet)
            self.assertEqual(first["mesh"], again["mesh"])
            self.assertEqual(first["freetext"], again["freetext"])


@needs_index
class CoverageGapTests(unittest.TestCase):
    """closure.py is only as exhaustive as the union of facet spans it is
    given, so a phrase that never lands in any voted facet must be detected,
    not silently lost."""

    def setUp(self):
        self.ix = get_index()

    def test_detects_text_outside_every_facet(self):
        q = "Effect of RNA sequencing in Alzheimer Disease"
        toks = closure.tokenize(q)
        self.assertEqual(toks[2:4], ["RNA", "sequencing"])
        voted = [{"name": "RNA sequencing", "role": "method", "start": 2, "end": 3}]
        gaps = facet_pipeline.coverage_gaps(q, voted)
        self.assertEqual(len(gaps), 1)
        phrase = " ".join(toks[gaps[0]["start"]:gaps[0]["end"] + 1])
        self.assertIn("Alzheimer Disease", phrase)

    def test_fully_covered_question_has_no_gaps(self):
        q = "RNA sequencing of hippocampal neurons"
        toks = closure.tokenize(q)
        voted = [{"name": q, "role": "other", "start": 0, "end": len(toks) - 1}]
        self.assertEqual(facet_pipeline.coverage_gaps(q, voted), [])

    def test_gap_of_only_stopwords_is_not_reported(self):
        q = "RNA sequencing of the hippocampal neurons"
        # cover everything except "of the" in the middle -- not worth a gap
        voted = [{"start": 0, "end": 1}, {"start": 4, "end": 5}]
        self.assertEqual(facet_pipeline.coverage_gaps(q, voted), [])

    def test_resolve_coverage_gaps_recovers_missed_mesh_heading(self):
        q = "Effect of RNA sequencing in Alzheimer Disease"
        voted = [{"name": "RNA sequencing", "role": "method", "start": 2, "end": 3}]
        blocks = facet_pipeline.resolve_coverage_gaps(self.ix, q, voted)
        self.assertEqual(len(blocks), 1)
        labels = [m["options"][0]["label"] for m in blocks[0]["mesh"]]
        self.assertIn("Alzheimer Disease", labels)
        self.assertIn("coverage gap", blocks[0]["rationale"])

    def test_build_appends_gap_blocks_and_reports_them(self):
        q = "Effect of RNA sequencing in Alzheimer Disease"
        build = facet_pipeline._finish(
            q, self.ix, {"facets": [{"name": "RNA sequencing", "role": "method",
                                     "start": 2, "end": 3}], "mode": "heuristic", "runs": 0},
            model=None, min_agreement=0.5, explode=True, max_freetext=None)
        self.assertEqual(len(build["coverage_gaps"]), 1)
        self.assertTrue(any("Alzheimer" in c["name"] for c in build["concepts"]))
        self.assertIn("coverage gap", build["notes"])


@needs_index
class SelfTest(unittest.TestCase):
    def test_engine_selftest_passes(self):
        from search_strategy import selftest

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = selftest.main()
        self.assertEqual(code, 0, out.getvalue())


if __name__ == "__main__":
    unittest.main()
