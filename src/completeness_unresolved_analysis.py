from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Dict, List

import barnette_proof as bp
from completeness_frontier_scan import all_c2_occurrences
from pinch_ii_scan import all_pinch_ii_occurrences
from plantri_wrapper import iter_barnette_graph_rotations_via_plantri
from refined_c4_local import all_refined_c4_occurrences, graph_from_plantri_rotation


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLANTRI = ROOT / "plantri.exe"


@dataclass(frozen=True)
class UnresolvedGraphSummary:
    n_vertices: int
    raw_line: str
    c2_count: int
    pinch_count: int
    refined_c4_count: int
    c2_error_counts: Dict[str, int]
    pinch_error_counts: Dict[str, int]
    refined_c4_error_counts: Dict[str, int]
    face_lengths: List[int]


def face_lengths(G: bp.EmbeddedGraph) -> List[int]:
    darts_all = [(v, u) for v in G.adj for u in G.adj[v]]
    seen = set()
    lengths: List[int] = []
    for d in darts_all:
        if d in seen:
            continue
        orbit, end = G.trace_face_darts(d, steps=None)
        if end != d:
            raise AssertionError("face orbit did not close")
        for dart in orbit:
            seen.add(dart)
        lengths.append(len(orbit))
    return sorted(lengths)


def error_counter_for_rule(G: bp.EmbeddedGraph, step_type: str, occurrences: List[object]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for occ in occurrences:
        Gred, _, error = bp.try_reduce_certified(G, step_type, occ)
        if Gred is None:
            counter[error or "unknown"] += 1
    return counter


def has_any_certified(counter: Counter[str], occurrences: List[object]) -> bool:
    return bool(occurrences) and sum(counter.values()) < len(occurrences)


def analyze_graph(G: bp.EmbeddedGraph, raw_line: str) -> UnresolvedGraphSummary | None:
    occ2s = all_c2_occurrences(G)
    occps = all_pinch_ii_occurrences(G)
    occ4s = all_refined_c4_occurrences(G)

    c2_errors = error_counter_for_rule(G, "C2", occ2s)
    pinch_errors = error_counter_for_rule(G, "pinch(ii)", occps)
    refined_c4_errors = error_counter_for_rule(G, "refined_C4", occ4s)

    if has_any_certified(c2_errors, occ2s):
        return None
    if has_any_certified(pinch_errors, occps):
        return None
    if has_any_certified(refined_c4_errors, occ4s):
        return None

    return UnresolvedGraphSummary(
        n_vertices=len(G.adj),
        raw_line=raw_line,
        c2_count=len(occ2s),
        pinch_count=len(occps),
        refined_c4_count=len(occ4s),
        c2_error_counts=dict(c2_errors),
        pinch_error_counts=dict(pinch_errors),
        refined_c4_error_counts=dict(refined_c4_errors),
        face_lengths=face_lengths(G),
    )


def analyze_unresolved_frontier(plantri_path: Path, n_min: int, n_max: int) -> Dict[str, object]:
    unresolved: List[UnresolvedGraphSummary] = []
    n_counter = Counter()
    c2_error_totals: Counter[str] = Counter()
    pinch_error_totals: Counter[str] = Counter()
    refined_c4_error_totals: Counter[str] = Counter()
    face_length_patterns: Counter[str] = Counter()

    for n_vertices in range(n_min, n_max + 1, 2):
        for embedding in iter_barnette_graph_rotations_via_plantri(str(plantri_path), n_vertices):
            G = graph_from_plantri_rotation(embedding.rot)
            summary = analyze_graph(G, embedding.raw_line)
            if summary is None:
                continue
            unresolved.append(summary)
            n_counter[n_vertices] += 1
            c2_error_totals.update(summary.c2_error_counts)
            pinch_error_totals.update(summary.pinch_error_counts)
            refined_c4_error_totals.update(summary.refined_c4_error_counts)
            face_length_patterns[",".join(str(length) for length in summary.face_lengths)] += 1

    return {
        "plantri_path": str(plantri_path),
        "n_min": n_min,
        "n_max": n_max,
        "unresolved_graph_count": len(unresolved),
        "unresolved_counts_by_n": {str(n): n_counter[n] for n in sorted(n_counter)},
        "aggregate_c2_errors": dict(c2_error_totals),
        "aggregate_pinch_errors": dict(pinch_error_totals),
        "aggregate_refined_c4_errors": dict(refined_c4_error_totals),
        "face_length_patterns": dict(face_length_patterns),
        "graphs": [asdict(summary) for summary in unresolved],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze graphs with no currently certified completeness step."
    )
    parser.add_argument("--plantri", default=str(DEFAULT_PLANTRI))
    parser.add_argument("--n-min", type=int, default=8)
    parser.add_argument("--n-max", type=int, default=24)
    parser.add_argument(
        "--out",
        default=str(ROOT / "artifacts" / "completeness_unresolved_analysis.json"),
    )
    args = parser.parse_args()

    result = analyze_unresolved_frontier(Path(args.plantri), args.n_min, args.n_max)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
