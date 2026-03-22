from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Dict, List

import barnette_proof as bp
from plantri_wrapper import iter_barnette_graph_rotations_via_plantri
from refined_c4_local import graph_from_plantri_rotation


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLANTRI = ROOT / "plantri.exe"


@dataclass(frozen=True)
class OccurrenceSummary:
    n_vertices: int
    raw_line: str
    occ: Dict[str, int]
    boundary: List[int]
    boundary_edge_pattern: List[str]
    reduction_status: str
    reduction_error: str | None


def all_pinch_ii_occurrences(G: bp.EmbeddedGraph) -> List[bp.OccPinch]:
    found = set()
    for v1 in G.vertices():
        for v2 in sorted(G.adj[v1]):
            darts, end = G.trace_face_darts((v1, v2), steps=4)
            if end != (v1, v2):
                continue
            v2 = darts[0][1]
            v3 = darts[1][1]
            v4 = darts[2][1]
            if len({v1, v2, v3, v4}) != 4:
                continue

            u1 = G.third_neighbor(v1, {v2, v4})
            u2 = G.third_neighbor(v2, {v1, v3})
            u3 = G.third_neighbor(v3, {v2, v4})
            u4 = G.third_neighbor(v4, {v1, v3})
            if u1 != u3:
                continue

            w = u1
            t = G.third_neighbor(w, {v1, v3})
            if t in {u2, u4}:
                continue

            rs = sorted(G.adj[t] - {w})
            if len(rs) != 2:
                continue
            r, s = rs

            quad_edges = [(v1, v2), (v2, v3), (v3, v4), (v4, v1)]
            if any(G.other_face_is_quad(a, b) for a, b in quad_edges):
                continue

            p = G.rot[w][(G.pos[w][t] + 1) % 3]
            q = G.rot[p][(G.pos[p][w] + 1) % 3]
            epsilon = 0 if q == v2 else 1
            found.add(bp.OccPinch(v1, v2, v3, v4, w, t, r, s, u2, u4, epsilon))
    return sorted(found)


def summarize_occurrence(G: bp.EmbeddedGraph, raw_line: str, occ: bp.OccPinch) -> OccurrenceSummary:
    boundary_named = [("r", occ.r), ("s", occ.s), ("u2", occ.u2), ("u4", occ.u4)]
    edge_pattern = [
        f"{left_name}-{right_name}"
        for index, (left_name, left_vertex) in enumerate(boundary_named)
        for right_name, right_vertex in boundary_named[index + 1 :]
        if right_vertex in G.adj[left_vertex]
    ]
    Gred, _, error = bp.try_reduce_certified(G, "pinch(ii)", occ)
    status = "PASS" if Gred is not None else "FAIL"
    return OccurrenceSummary(
        n_vertices=len(G.adj),
        raw_line=raw_line,
        occ=asdict(occ),
        boundary=[occ.r, occ.s, occ.u2, occ.u4],
        boundary_edge_pattern=edge_pattern,
        reduction_status=status,
        reduction_error=error,
    )


def scan_pinch_ii(plantri_path: Path, n_min: int, n_max: int, limit: int | None) -> Dict[str, object]:
    summaries: List[OccurrenceSummary] = []
    scanned_graphs = 0
    graphs_with_occurrence = 0
    for n_vertices in range(n_min, n_max + 1, 2):
        for embedding in iter_barnette_graph_rotations_via_plantri(str(plantri_path), n_vertices):
            G = graph_from_plantri_rotation(embedding.rot)
            scanned_graphs += 1
            occurrences = all_pinch_ii_occurrences(G)
            if occurrences:
                graphs_with_occurrence += 1
            for occ in occurrences:
                summaries.append(summarize_occurrence(G, embedding.raw_line, occ))
                if limit is not None and len(summaries) >= limit:
                    break
            if limit is not None and len(summaries) >= limit:
                break
        if limit is not None and len(summaries) >= limit:
            break

    pass_records = [summary for summary in summaries if summary.reduction_status == "PASS"]
    fail_records = [summary for summary in summaries if summary.reduction_status == "FAIL"]
    error_counter = Counter(summary.reduction_error or "none" for summary in fail_records)
    boundary_edge_counter = Counter(",".join(summary.boundary_edge_pattern) or "none" for summary in summaries)

    return {
        "plantri_path": str(plantri_path),
        "n_min": n_min,
        "n_max": n_max,
        "scanned_graph_count": scanned_graphs,
        "graphs_with_occurrence": graphs_with_occurrence,
        "occurrence_count": len(summaries),
        "pass_count": len(pass_records),
        "fail_count": len(fail_records),
        "boundary_edge_patterns": dict(boundary_edge_counter),
        "reduction_errors": dict(error_counter),
        "first_pass": asdict(pass_records[0]) if pass_records else None,
        "first_fail": asdict(fail_records[0]) if fail_records else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan plantri graphs for pinch(ii) reducibility.")
    parser.add_argument("--plantri", default=str(DEFAULT_PLANTRI))
    parser.add_argument("--n-min", type=int, default=8)
    parser.add_argument("--n-max", type=int, default=24)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "pinch_ii_scan.json"))
    args = parser.parse_args()

    result = scan_pinch_ii(Path(args.plantri), args.n_min, args.n_max, args.limit)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
