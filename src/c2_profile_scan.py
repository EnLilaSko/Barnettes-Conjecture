from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Dict, List, Tuple

import barnette_proof as bp
from completeness_frontier_scan import all_c2_occurrences
from plantri_wrapper import iter_barnette_graph_rotations_via_plantri
from refined_c4_local import graph_from_plantri_rotation


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLANTRI = ROOT / "plantri.exe"


LEFT_BOUNDARY_EDGES = ("ab", "da", "cd")
RIGHT_BOUNDARY_EDGES = ("bf", "fe", "ec")


@dataclass(frozen=True)
class C2ProfileSummary:
    n_vertices: int
    raw_line: str
    occ: Dict[str, int]
    terminal_partition_key: List[int]
    terminal_edge_pattern: List[str]
    boundary_face_lengths: List[int]
    boundary_face_lengths_key: List[int]
    reduction_status: str
    reduction_error: str | None


def partition_key(terminals: List[int]) -> List[int]:
    mapping: Dict[int, int] = {}
    word: List[int] = []
    next_label = 0
    for value in terminals:
        if value not in mapping:
            mapping[value] = next_label
            next_label += 1
        word.append(mapping[value])
    candidates = []
    for source in (word, list(reversed(word))):
        for offset in range(len(source)):
            rotated = source[offset:] + source[:offset]
            candidates.append(rotated)
    return min(candidates)


def face_vertices(G: bp.EmbeddedGraph, start: Tuple[int, int]) -> List[int]:
    orbit, end = G.trace_face_darts(start, steps=None)
    if end != start:
        raise AssertionError("face orbit did not close")
    return [tail for tail, _ in orbit]


def external_face_length(
    G: bp.EmbeddedGraph,
    edge: Tuple[int, int],
    internal_face_vertices: List[int],
) -> int:
    candidates = [face_vertices(G, edge), face_vertices(G, (edge[1], edge[0]))]
    internal_set = set(internal_face_vertices)
    for face in candidates:
        if set(face) != internal_set:
            return len(face)
    raise AssertionError(f"could not identify external face for edge {edge}")


def boundary_profile(G: bp.EmbeddedGraph, occ: bp.OccC2) -> List[int]:
    left = [occ.a, occ.b, occ.c, occ.d]
    right = [occ.b, occ.c, occ.e, occ.f]
    edge_specs = [
        ((occ.a, occ.b), left),
        ((occ.b, occ.f), right),
        ((occ.f, occ.e), right),
        ((occ.e, occ.c), right),
        ((occ.c, occ.d), left),
        ((occ.d, occ.a), left),
    ]
    lengths = [
        external_face_length(G, edge, internal_face)
        for edge, internal_face in edge_specs
    ]
    return lengths


def canonical_length_key(lengths: List[int]) -> List[int]:
    reversed_lengths = [lengths[0]] + list(reversed(lengths[1:]))
    return min(lengths, reversed_lengths)


def terminal_edge_pattern(G: bp.EmbeddedGraph, occ: bp.OccC2) -> List[str]:
    names = [("u1", occ.u1), ("u4", occ.u4), ("u5", occ.u5), ("u6", occ.u6)]
    return [
        f"{left_name}-{right_name}"
        for index, (left_name, left_vertex) in enumerate(names)
        for right_name, right_vertex in names[index + 1 :]
        if right_vertex in G.adj[left_vertex]
    ]


def summarize_occurrence(G: bp.EmbeddedGraph, raw_line: str, occ: bp.OccC2) -> C2ProfileSummary:
    terminals = [occ.u1, occ.u4, occ.u5, occ.u6]
    lengths = boundary_profile(G, occ)
    Gred, _, error = bp.try_reduce_certified(G, "C2", occ)
    return C2ProfileSummary(
        n_vertices=len(G.adj),
        raw_line=raw_line,
        occ=asdict(occ),
        terminal_partition_key=partition_key(terminals),
        terminal_edge_pattern=terminal_edge_pattern(G, occ),
        boundary_face_lengths=lengths,
        boundary_face_lengths_key=canonical_length_key(lengths),
        reduction_status="PASS" if Gred is not None else "FAIL",
        reduction_error=error,
    )


def scan_c2_profiles(plantri_path: Path, n_min: int, n_max: int) -> Dict[str, object]:
    summaries: List[C2ProfileSummary] = []
    for n_vertices in range(n_min, n_max + 1, 2):
        for embedding in iter_barnette_graph_rotations_via_plantri(str(plantri_path), n_vertices):
            G = graph_from_plantri_rotation(embedding.rot)
            for occ in all_c2_occurrences(G):
                summaries.append(summarize_occurrence(G, embedding.raw_line, occ))

    profile_counter = Counter(
        ",".join(str(value) for value in summary.boundary_face_lengths_key) for summary in summaries
    )
    fail_profile_counter = Counter(
        ",".join(str(value) for value in summary.boundary_face_lengths_key)
        for summary in summaries
        if summary.reduction_status == "FAIL"
    )
    pass_profile_counter = Counter(
        ",".join(str(value) for value in summary.boundary_face_lengths_key)
        for summary in summaries
        if summary.reduction_status == "PASS"
    )
    distinct_fail_profile_counter = Counter(
        ",".join(str(value) for value in summary.boundary_face_lengths_key)
        for summary in summaries
        if summary.reduction_status == "FAIL" and summary.terminal_partition_key == [0, 1, 2, 3]
    )
    first_examples: Dict[str, object] = {}
    for summary in summaries:
        key = (
            f"P={','.join(str(v) for v in summary.terminal_partition_key)}"
            f"|L={','.join(str(v) for v in summary.boundary_face_lengths_key)}"
            f"|E={','.join(summary.terminal_edge_pattern) or 'none'}"
        )
        first_examples.setdefault(key, asdict(summary))

    return {
        "plantri_path": str(plantri_path),
        "n_min": n_min,
        "n_max": n_max,
        "occurrence_count": len(summaries),
        "boundary_face_profiles": dict(profile_counter),
        "pass_boundary_face_profiles": dict(pass_profile_counter),
        "fail_boundary_face_profiles": dict(fail_profile_counter),
        "distinct_terminal_fail_profiles": dict(distinct_fail_profile_counter),
        "first_examples": first_examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan local boundary-face profiles for C2 occurrences.")
    parser.add_argument("--plantri", default=str(DEFAULT_PLANTRI))
    parser.add_argument("--n-min", type=int, default=8)
    parser.add_argument("--n-max", type=int, default=24)
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "c2_profile_scan.json"))
    args = parser.parse_args()

    result = scan_c2_profiles(Path(args.plantri), args.n_min, args.n_max)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
