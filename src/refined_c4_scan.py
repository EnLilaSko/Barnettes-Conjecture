from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Dict, List

import barnette_proof as bp
from plantri_wrapper import iter_barnette_graph_rotations_via_plantri
from refined_c4_local import (
    all_refined_c4_occurrences,
    graph_from_plantri_rotation,
    occurrence_profile,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLANTRI = ROOT / "plantri.exe"


@dataclass(frozen=True)
class OccurrenceSummary:
    n_vertices: int
    raw_line: str
    occ: Dict[str, int]
    consecutive_terminal_edges: List[str]
    opposite_terminal_edges: List[str]
    outer_face_lengths: List[int]
    outer_face_length_key: List[int]
    general_boundary_count: int | None
    general_interface_word: List[int] | None
    general_interface_word_key: List[int] | None
    general_attachment_sizes: List[int] | None
    general_exterior_sizes: List[int] | None
    general_deleted_size: int | None
    general_patch_error: str | None
    hex_ring_pattern: List[int] | None
    hex_ring_pattern_key: List[int] | None
    hex_patch_error: str | None
    reduction_status: str
    reduction_error: str | None


def edge_name(a: str, b: str) -> str:
    return f"{a}-{b}"


def summarize_occurrence(G, raw_line: str, occ) -> OccurrenceSummary:
    profile = occurrence_profile(G, occ)
    named_terminals = [
        ("u1", occ.u1),
        ("u2", occ.u2),
        ("u3", occ.u3),
        ("u4", occ.u4),
    ]
    consecutive_pairs = [
        (named_terminals[0], named_terminals[1]),
        (named_terminals[1], named_terminals[2]),
        (named_terminals[2], named_terminals[3]),
        (named_terminals[3], named_terminals[0]),
    ]
    opposite_pairs = [
        (named_terminals[0], named_terminals[2]),
        (named_terminals[1], named_terminals[3]),
    ]

    consecutive_terminal_edges = [
        edge_name(left_name, right_name)
        for (left_name, left_vertex), (right_name, right_vertex) in consecutive_pairs
        if right_vertex in G.adj[left_vertex]
    ]
    opposite_terminal_edges = [
        edge_name(left_name, right_name)
        for (left_name, left_vertex), (right_name, right_vertex) in opposite_pairs
        if right_vertex in G.adj[left_vertex]
    ]

    Gred, _, error = bp.try_reduce_certified(G, "refined_C4", occ)
    status = "PASS" if Gred is not None else "FAIL"
    return OccurrenceSummary(
        n_vertices=len(G.adj),
        raw_line=raw_line,
        occ=asdict(occ),
        consecutive_terminal_edges=consecutive_terminal_edges,
        opposite_terminal_edges=opposite_terminal_edges,
        outer_face_lengths=list(profile.outer_face_lengths),
        outer_face_length_key=list(profile.outer_face_length_key),
        general_boundary_count=profile.general_boundary_count,
        general_interface_word=list(profile.general_interface_word) if profile.general_interface_word is not None else None,
        general_interface_word_key=list(profile.general_interface_word_key) if profile.general_interface_word_key is not None else None,
        general_attachment_sizes=list(profile.general_attachment_sizes) if profile.general_attachment_sizes is not None else None,
        general_exterior_sizes=list(profile.general_exterior_sizes) if profile.general_exterior_sizes is not None else None,
        general_deleted_size=profile.general_deleted_size,
        general_patch_error=profile.general_patch_error,
        hex_ring_pattern=list(profile.hex_ring_pattern) if profile.hex_ring_pattern is not None else None,
        hex_ring_pattern_key=list(profile.hex_ring_pattern_key) if profile.hex_ring_pattern_key is not None else None,
        hex_patch_error=profile.hex_patch_error,
        reduction_status=status,
        reduction_error=error,
    )


def scan_refined_c4(plantri_path: Path, n_min: int, n_max: int, limit: int | None) -> Dict[str, object]:
    summaries: List[OccurrenceSummary] = []
    scanned_graphs = 0
    for n_vertices in range(n_min, n_max + 1, 2):
        for embedding in iter_barnette_graph_rotations_via_plantri(str(plantri_path), n_vertices):
            G = graph_from_plantri_rotation(embedding.rot)
            scanned_graphs += 1
            for occ in all_refined_c4_occurrences(G):
                summaries.append(summarize_occurrence(G, embedding.raw_line, occ))
                if limit is not None and len(summaries) >= limit:
                    break
            if limit is not None and len(summaries) >= limit:
                break
        if limit is not None and len(summaries) >= limit:
            break

    pass_records = [summary for summary in summaries if summary.reduction_status == "PASS"]
    fail_records = [summary for summary in summaries if summary.reduction_status == "FAIL"]
    consecutive_counter = Counter(
        ",".join(summary.consecutive_terminal_edges) or "none" for summary in summaries
    )
    outer_length_counter = Counter(
        ",".join(str(length) for length in summary.outer_face_length_key) for summary in summaries
    )
    boundary_count_counter = Counter(
        str(summary.general_boundary_count) if summary.general_boundary_count is not None else "error"
        for summary in summaries
    )
    general_interface_counter = Counter(
        (
            f"B={summary.general_boundary_count}"
            f"|I={','.join(str(value) for value in summary.general_interface_word_key)}"
            f"|A={','.join(str(value) for value in summary.general_attachment_sizes)}"
            f"|E={','.join(str(value) for value in summary.general_exterior_sizes)}"
        )
        for summary in summaries
        if summary.general_interface_word_key is not None
        and summary.general_attachment_sizes is not None
        and summary.general_exterior_sizes is not None
    )
    family_counter = Counter(
        (
            f"L={','.join(str(length) for length in summary.outer_face_length_key)}"
            f"|B={summary.general_boundary_count}"
            f"|I={','.join(str(value) for value in summary.general_interface_word_key)}"
            f"|A={','.join(str(value) for value in summary.general_attachment_sizes)}"
            f"|E={','.join(str(value) for value in summary.general_exterior_sizes)}"
            if summary.general_interface_word_key is not None
            and summary.general_attachment_sizes is not None
            and summary.general_exterior_sizes is not None
            else f"L={','.join(str(length) for length in summary.outer_face_length_key)}|PATCH=error"
        )
        for summary in summaries
    )
    hex_ring_counter = Counter(
        ",".join(str(value) for value in summary.hex_ring_pattern_key)
        for summary in summaries
        if summary.hex_ring_pattern_key is not None
    )
    hex_patch_error_counter = Counter(
        summary.hex_patch_error or "none"
        for summary in summaries
    )
    general_patch_error_counter = Counter(
        summary.general_patch_error or "none"
        for summary in summaries
    )
    pass_counter = Counter(
        ",".join(summary.consecutive_terminal_edges) or "none" for summary in pass_records
    )
    fail_counter = Counter(
        ",".join(summary.consecutive_terminal_edges) or "none" for summary in fail_records
    )
    family_examples: Dict[str, object] = {}
    for summary in summaries:
        family_key = (
            f"L={','.join(str(length) for length in summary.outer_face_length_key)}"
            f"|B={summary.general_boundary_count}"
            f"|I={','.join(str(value) for value in summary.general_interface_word_key)}"
            f"|A={','.join(str(value) for value in summary.general_attachment_sizes)}"
            f"|E={','.join(str(value) for value in summary.general_exterior_sizes)}"
            if summary.general_interface_word_key is not None
            and summary.general_attachment_sizes is not None
            and summary.general_exterior_sizes is not None
            else f"L={','.join(str(length) for length in summary.outer_face_length_key)}|PATCH=error"
        )
        family_examples.setdefault(family_key, asdict(summary))

    return {
        "plantri_path": str(plantri_path),
        "n_min": n_min,
        "n_max": n_max,
        "scanned_graph_count": scanned_graphs,
        "occurrence_count": len(summaries),
        "pass_count": len(pass_records),
        "fail_count": len(fail_records),
        "outer_face_length_patterns": dict(outer_length_counter),
        "boundary_count_patterns": dict(boundary_count_counter),
        "general_interface_patterns": dict(general_interface_counter),
        "occurrence_families": dict(family_counter),
        "family_examples": family_examples,
        "hex_ring_patterns": dict(hex_ring_counter),
        "general_patch_errors": dict(general_patch_error_counter),
        "hex_patch_errors": dict(hex_patch_error_counter),
        "consecutive_terminal_edge_patterns": dict(consecutive_counter),
        "pass_patterns": dict(pass_counter),
        "fail_patterns": dict(fail_counter),
        "first_pass": asdict(pass_records[0]) if pass_records else None,
        "first_fail": asdict(fail_records[0]) if fail_records else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan plantri graphs for refined C4 reducibility.")
    parser.add_argument("--plantri", default=str(DEFAULT_PLANTRI))
    parser.add_argument("--n-min", type=int, default=8)
    parser.add_argument("--n-max", type=int, default=24)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "refined_c4_scan.json"))
    args = parser.parse_args()

    result = scan_refined_c4(Path(args.plantri), args.n_min, args.n_max, args.limit)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
