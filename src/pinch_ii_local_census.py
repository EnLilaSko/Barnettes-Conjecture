from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Dict, List, Tuple

import barnette_proof as bp
from plantri_wrapper import iter_barnette_graph_rotations_via_plantri
from refined_c4_local import graph_from_plantri_rotation


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLANTRI = ROOT / "plantri.exe"


@dataclass(frozen=True)
class PinchedQuadSummary:
    n_vertices: int
    raw_line: str
    quad: List[int]
    opposite_equality: str
    occ: Dict[str, int]
    edge_isolated: bool
    adjacent_quad_edges: List[str]
    outer_face_lengths: List[int]
    category: str


def canonical_face(face: List[int]) -> Tuple[int, int, int, int]:
    variants: List[Tuple[int, int, int, int]] = []
    cycle = list(face)
    rev = list(reversed(face))
    for source in (cycle, rev):
        for offset in range(4):
            rotated = source[offset:] + source[:offset]
            variants.append(tuple(rotated))
    return min(variants)


def all_facial_quads(G: bp.EmbeddedGraph) -> List[List[int]]:
    seen = set()
    faces: List[List[int]] = []
    for v in G.vertices():
        for u in sorted(G.adj[v]):
            darts, end = G.trace_face_darts((v, u), steps=4)
            if end != (v, u):
                continue
            face = [tail for tail, _ in darts]
            canon = canonical_face(face)
            if canon in seen:
                continue
            seen.add(canon)
            faces.append(face)
    return faces


def rotate_face(face: List[int], offset: int) -> List[int]:
    return face[offset:] + face[:offset]


def other_face_length(G: bp.EmbeddedGraph, a: int, b: int) -> int:
    orbit, end = G.trace_face_darts((b, a), steps=None)
    if end != (b, a):
        raise AssertionError("face orbit did not close")
    return len(orbit)


def analyze_pinched_face(
    G: bp.EmbeddedGraph,
    raw_line: str,
    face: List[int],
) -> PinchedQuadSummary | None:
    v1, v2, v3, v4 = face
    u1 = G.third_neighbor(v1, {v2, v4})
    u2 = G.third_neighbor(v2, {v1, v3})
    u3 = G.third_neighbor(v3, {v2, v4})
    u4 = G.third_neighbor(v4, {v1, v3})

    equality_13 = u1 == u3
    equality_24 = u2 == u4
    if not equality_13 and not equality_24:
        return None

    oriented_face = face
    opposite_equality = "u1=u3"
    if not equality_13 and equality_24:
        oriented_face = rotate_face(face, 1)
        opposite_equality = "u2=u4"
        v1, v2, v3, v4 = oriented_face
        u1 = G.third_neighbor(v1, {v2, v4})
        u2 = G.third_neighbor(v2, {v1, v3})
        u3 = G.third_neighbor(v3, {v2, v4})
        u4 = G.third_neighbor(v4, {v1, v3})
    elif equality_13 and equality_24:
        opposite_equality = "both"

    w = u1
    t = G.third_neighbor(w, {v1, v3})
    rs = sorted(G.adj[t] - {w})
    if len(rs) != 2:
        raise AssertionError("expected cubic third-neighbor split at t")
    r, s = rs

    quad_edges = [(v1, v2), (v2, v3), (v3, v4), (v4, v1)]
    adjacent_quad_edges = [
        f"{a}-{b}"
        for a, b in quad_edges
        if G.other_face_is_quad(a, b)
    ]
    edge_isolated = not adjacent_quad_edges
    outer_face_lengths = [other_face_length(G, a, b) for a, b in quad_edges]

    if t == u2:
        category = "pinch_i_to_u2"
    elif t == u4:
        category = "pinch_i_to_u4"
    elif edge_isolated:
        category = "pinch_ii"
    else:
        category = "pinch_other_nonisolated"

    occ = {
        "v1": v1,
        "v2": v2,
        "v3": v3,
        "v4": v4,
        "w": w,
        "t": t,
        "r": r,
        "s": s,
        "u2": u2,
        "u4": u4,
    }
    return PinchedQuadSummary(
        n_vertices=len(G.adj),
        raw_line=raw_line,
        quad=list(oriented_face),
        opposite_equality=opposite_equality,
        occ=occ,
        edge_isolated=edge_isolated,
        adjacent_quad_edges=adjacent_quad_edges,
        outer_face_lengths=outer_face_lengths,
        category=category,
    )


def scan_pinched_quads(plantri_path: Path, n_min: int, n_max: int) -> Dict[str, object]:
    summaries: List[PinchedQuadSummary] = []
    scanned_graphs = 0
    graphs_with_pinched_quad = 0

    for n_vertices in range(n_min, n_max + 1, 2):
        for embedding in iter_barnette_graph_rotations_via_plantri(str(plantri_path), n_vertices):
            G = graph_from_plantri_rotation(embedding.rot)
            scanned_graphs += 1
            graph_summaries = [
                summary
                for face in all_facial_quads(G)
                for summary in [analyze_pinched_face(G, embedding.raw_line, face)]
                if summary is not None
            ]
            if graph_summaries:
                graphs_with_pinched_quad += 1
            summaries.extend(graph_summaries)

    category_counter = Counter(summary.category for summary in summaries)
    equality_counter = Counter(summary.opposite_equality for summary in summaries)
    adjacent_quad_counter = Counter(",".join(summary.adjacent_quad_edges) or "none" for summary in summaries)
    outer_length_counter = Counter(",".join(str(length) for length in summary.outer_face_lengths) for summary in summaries)
    first_examples: Dict[str, object] = {}
    first_order_by_category: Dict[str, int] = {}
    for summary in summaries:
        if summary.category not in first_examples:
            first_examples[summary.category] = asdict(summary)
            first_order_by_category[summary.category] = summary.n_vertices

    return {
        "plantri_path": str(plantri_path),
        "n_min": n_min,
        "n_max": n_max,
        "scanned_graph_count": scanned_graphs,
        "graphs_with_pinched_quad": graphs_with_pinched_quad,
        "pinched_quad_count": len(summaries),
        "category_counts": dict(category_counter),
        "opposite_equality_counts": dict(equality_counter),
        "adjacent_quad_patterns": dict(adjacent_quad_counter),
        "outer_face_length_patterns": dict(outer_length_counter),
        "first_order_by_category": first_order_by_category,
        "first_examples": first_examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify opposite-pinched facial quads in Barnette graphs.")
    parser.add_argument("--plantri", default=str(DEFAULT_PLANTRI))
    parser.add_argument("--n-min", type=int, default=8)
    parser.add_argument("--n-max", type=int, default=24)
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "pinch_ii_local_census.json"))
    args = parser.parse_args()

    result = scan_pinched_quads(Path(args.plantri), args.n_min, args.n_max)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
