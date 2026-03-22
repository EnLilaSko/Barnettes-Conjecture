from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

import barnette_proof as bp
from completeness_frontier_scan import all_c2_occurrences
from plantri_wrapper import iter_barnette_graph_rotations_via_plantri
from refined_c4_local import graph_from_plantri_rotation


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLANTRI = ROOT / "plantri.exe"


def connected_components_after_removal(adj: Dict[int, Set[int]], removed: Set[int]) -> List[List[int]]:
    nodes = [v for v in sorted(adj) if v not in removed]
    seen: Set[int] = set()
    components: List[List[int]] = []
    for start in nodes:
        if start in seen:
            continue
        comp = []
        queue = [start]
        seen.add(start)
        for v in queue:
            comp.append(v)
            for u in adj[v]:
                if u in removed or u in seen:
                    continue
                seen.add(u)
                queue.append(u)
        components.append(sorted(comp))
    return components


def all_two_cuts(adj: Dict[int, Set[int]]) -> List[Tuple[Tuple[int, int], List[int]]]:
    vertices = sorted(adj)
    cuts: List[Tuple[Tuple[int, int], List[int]]] = []
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            removed = {vertices[i], vertices[j]}
            components = connected_components_after_removal(adj, removed)
            if len(components) > 1:
                cuts.append(((vertices[i], vertices[j]), sorted(len(comp) for comp in components)))
    return cuts


@dataclass(frozen=True)
class C2ExceptionSummary:
    n_vertices: int
    raw_line: str
    occ: Dict[str, int]
    reduced_vertices: int
    reduced_edge_gadget_cut: bool
    all_two_cuts: List[Dict[str, object]]


def analyze_exception(G: bp.EmbeddedGraph, raw_line: str, occ: bp.OccC2) -> C2ExceptionSummary:
    H, rec = bp.reduce_C2(G, occ)
    two_cuts = all_two_cuts(H.adj)
    reduced_edge_cut = any(cut == tuple(sorted((rec.x, rec.y))) for cut, _ in two_cuts)
    cut_payload = [
        {"cut": list(cut), "component_sizes": component_sizes}
        for cut, component_sizes in two_cuts
    ]
    return C2ExceptionSummary(
        n_vertices=len(G.adj),
        raw_line=raw_line,
        occ=asdict(occ),
        reduced_vertices=len(H.adj),
        reduced_edge_gadget_cut=reduced_edge_cut,
        all_two_cuts=cut_payload,
    )


def scan_c2_exceptions(plantri_path: Path, n_min: int, n_max: int) -> Dict[str, object]:
    exceptions: List[C2ExceptionSummary] = []
    cut_counter: Counter[str] = Counter()
    counts_by_n: Counter[int] = Counter()
    cut_profile_counter: Counter[str] = Counter()
    gadget_cut_count = 0

    for n_vertices in range(n_min, n_max + 1, 2):
        for embedding in iter_barnette_graph_rotations_via_plantri(str(plantri_path), n_vertices):
            G = graph_from_plantri_rotation(embedding.rot)
            for occ in all_c2_occurrences(G):
                terminals = [occ.u1, occ.u4, occ.u5, occ.u6]
                if len(set(terminals)) != 4:
                    continue
                Gred, _, error = bp.try_reduce_certified(G, "C2", occ)
                if Gred is not None or error != "not 3-connected":
                    continue
                summary = analyze_exception(G, embedding.raw_line, occ)
                exceptions.append(summary)
                counts_by_n[summary.n_vertices] += 1
                if summary.reduced_edge_gadget_cut:
                    gadget_cut_count += 1
                profile_key = "|".join(
                    "-".join(str(size) for size in cut["component_sizes"])
                    for cut in summary.all_two_cuts
                )
                cut_profile_counter[profile_key] += 1
                for cut in summary.all_two_cuts:
                    cut_counter[",".join(str(v) for v in cut["cut"])] += 1

    return {
        "plantri_path": str(plantri_path),
        "n_min": n_min,
        "n_max": n_max,
        "exception_count": len(exceptions),
        "counts_by_n": {str(n): counts_by_n[n] for n in sorted(counts_by_n)},
        "gadget_edge_cut_count": gadget_cut_count,
        "cut_profile_counts": dict(cut_profile_counter),
        "two_cut_frequency": dict(cut_counter),
        "exceptions": [asdict(summary) for summary in exceptions],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze distinct-terminal C2 exceptions that fail by non-3-connectivity."
    )
    parser.add_argument("--plantri", default=str(DEFAULT_PLANTRI))
    parser.add_argument("--n-min", type=int, default=8)
    parser.add_argument("--n-max", type=int, default=24)
    parser.add_argument(
        "--out",
        default=str(ROOT / "artifacts" / "c2_exception_analysis.json"),
    )
    args = parser.parse_args()

    result = scan_c2_exceptions(Path(args.plantri), args.n_min, args.n_max)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
