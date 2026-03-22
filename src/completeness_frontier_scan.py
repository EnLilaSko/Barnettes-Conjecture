from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Dict, List

import barnette_proof as bp
from pinch_ii_scan import all_pinch_ii_occurrences
from plantri_wrapper import iter_barnette_graph_rotations_via_plantri
from refined_c4_local import all_refined_c4_occurrences, graph_from_plantri_rotation


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLANTRI = ROOT / "plantri.exe"


@dataclass(frozen=True)
class GraphSummary:
    n_vertices: int
    raw_line: str
    has_c2: bool
    has_pinch_ii: bool
    has_refined_c4: bool
    c2_certified: bool
    pinch_certified: bool
    refined_c4_certified: bool
    certified_kind: str
    unresolved_reason: str | None


def pattern_key(summary: GraphSummary) -> str:
    return (
        f"C2={int(summary.has_c2)}"
        f"|PINCH={int(summary.has_pinch_ii)}"
        f"|C4={int(summary.has_refined_c4)}"
    )


def all_c2_occurrences(G: bp.EmbeddedGraph) -> List[bp.OccC2]:
    seen = set()
    occurrences: List[bp.OccC2] = []
    for a, b in G.edges():
        if not (G.face_is_quad_from_dart((a, b)) and G.face_is_quad_from_dart((b, a))):
            continue
        left = G.trace_face_vertices((a, b), steps=4)
        right = G.trace_face_vertices((b, a), steps=4)
        occ = bp._canonicalize_adjacent_quads(G, left, right)
        if occ is None or occ in seen:
            continue
        seen.add(occ)
        occurrences.append(occ)
    return sorted(occurrences)


def any_certified_occurrence(G: bp.EmbeddedGraph, step_type: str, occurrences: List[object]) -> bool:
    for occ in occurrences:
        Gred, _, _ = bp.try_reduce_certified(G, step_type, occ)
        if Gred is not None:
            return True
    return False


def summarize_graph(G: bp.EmbeddedGraph, raw_line: str) -> GraphSummary:
    occ2s = all_c2_occurrences(G)
    occps = all_pinch_ii_occurrences(G)
    occ4s = all_refined_c4_occurrences(G)

    c2_certified = any_certified_occurrence(G, "C2", occ2s)
    pinch_certified = any_certified_occurrence(G, "pinch(ii)", occps)
    refined_c4_certified = any_certified_occurrence(G, "refined_C4", occ4s)

    certified_kind = "NONE"
    unresolved_reason = None
    if c2_certified:
        certified_kind = "C2"
    elif pinch_certified:
        certified_kind = "PINCH"
    elif refined_c4_certified:
        certified_kind = "C4"
    else:
        if occ4s:
            unresolved_reason = "detected_refined_c4_but_not_certified"
        elif occps:
            unresolved_reason = "detected_pinch_but_not_certified"
        elif occ2s:
            unresolved_reason = "detected_c2_but_not_certified"
        else:
            unresolved_reason = "no_detected_configuration"

    return GraphSummary(
        n_vertices=len(G.adj),
        raw_line=raw_line,
        has_c2=bool(occ2s),
        has_pinch_ii=bool(occps),
        has_refined_c4=bool(occ4s),
        c2_certified=c2_certified,
        pinch_certified=pinch_certified,
        refined_c4_certified=refined_c4_certified,
        certified_kind=certified_kind,
        unresolved_reason=unresolved_reason,
    )


def scan_completeness_frontier(plantri_path: Path, n_min: int, n_max: int) -> Dict[str, object]:
    summaries: List[GraphSummary] = []
    per_n_graph_counts: Dict[int, int] = defaultdict(int)
    per_n_certified_counts: Dict[int, Counter[str]] = defaultdict(Counter)
    per_n_pattern_counts: Dict[int, Counter[str]] = defaultdict(Counter)

    for n_vertices in range(n_min, n_max + 1, 2):
        for embedding in iter_barnette_graph_rotations_via_plantri(str(plantri_path), n_vertices):
            G = graph_from_plantri_rotation(embedding.rot)
            summary = summarize_graph(G, embedding.raw_line)
            summaries.append(summary)
            per_n_graph_counts[n_vertices] += 1
            per_n_certified_counts[n_vertices][summary.certified_kind] += 1
            per_n_pattern_counts[n_vertices][pattern_key(summary)] += 1

    certified_counter = Counter(summary.certified_kind for summary in summaries)
    detector_pattern_counter = Counter(pattern_key(summary) for summary in summaries)
    unresolved_counter = Counter(
        summary.unresolved_reason for summary in summaries if summary.unresolved_reason is not None
    )
    first_examples_by_pattern: Dict[str, object] = {}
    first_examples_by_certified_kind: Dict[str, object] = {}
    first_unresolved_by_reason: Dict[str, object] = {}

    for summary in summaries:
        patt = pattern_key(summary)
        first_examples_by_pattern.setdefault(patt, asdict(summary))
        first_examples_by_certified_kind.setdefault(summary.certified_kind, asdict(summary))
        if summary.unresolved_reason is not None:
            first_unresolved_by_reason.setdefault(summary.unresolved_reason, asdict(summary))

    return {
        "plantri_path": str(plantri_path),
        "n_min": n_min,
        "n_max": n_max,
        "scanned_graph_count": len(summaries),
        "certified_kind_counts": dict(certified_counter),
        "detector_pattern_counts": dict(detector_pattern_counter),
        "unresolved_reason_counts": dict(unresolved_counter),
        "per_n_graph_counts": {str(n): per_n_graph_counts[n] for n in sorted(per_n_graph_counts)},
        "per_n_certified_counts": {
            str(n): dict(per_n_certified_counts[n]) for n in sorted(per_n_certified_counts)
        },
        "per_n_detector_pattern_counts": {
            str(n): dict(per_n_pattern_counts[n]) for n in sorted(per_n_pattern_counts)
        },
        "first_examples_by_pattern": first_examples_by_pattern,
        "first_examples_by_certified_kind": first_examples_by_certified_kind,
        "first_unresolved_by_reason": first_unresolved_by_reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify Barnette graphs by detected local configurations and current certified witness."
    )
    parser.add_argument("--plantri", default=str(DEFAULT_PLANTRI))
    parser.add_argument("--n-min", type=int, default=8)
    parser.add_argument("--n-max", type=int, default=24)
    parser.add_argument(
        "--out",
        default=str(ROOT / "artifacts" / "completeness_frontier_scan.json"),
    )
    args = parser.parse_args()

    result = scan_completeness_frontier(Path(args.plantri), args.n_min, args.n_max)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
