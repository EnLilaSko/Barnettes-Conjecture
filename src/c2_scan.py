from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Dict, List

import barnette_proof as bp
from completeness_frontier_scan import all_c2_occurrences
from plantri_wrapper import iter_barnette_graph_rotations_via_plantri
from refined_c4_local import graph_from_plantri_rotation


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLANTRI = ROOT / "plantri.exe"


@dataclass(frozen=True)
class C2OccurrenceSummary:
    n_vertices: int
    raw_line: str
    occ: Dict[str, int]
    terminal_vertices: List[int]
    terminal_partition_word: List[int]
    terminal_partition_key: List[int]
    terminal_edge_pattern: List[str]
    reduction_status: str
    reduction_error: str | None


def partition_word(values: List[int]) -> List[int]:
    mapping: Dict[int, int] = {}
    word: List[int] = []
    next_label = 0
    for value in values:
        if value not in mapping:
            mapping[value] = next_label
            next_label += 1
        word.append(mapping[value])
    return word


def canonical_partition_key(word: List[int]) -> List[int]:
    candidates = []
    for source in (word, list(reversed(word))):
        for offset in range(len(source)):
            rotated = source[offset:] + source[:offset]
            candidates.append(rotated)
    return min(candidates)


def summarize_occurrence(G: bp.EmbeddedGraph, raw_line: str, occ: bp.OccC2) -> C2OccurrenceSummary:
    terminals = [occ.u1, occ.u4, occ.u5, occ.u6]
    names = [("u1", occ.u1), ("u4", occ.u4), ("u5", occ.u5), ("u6", occ.u6)]
    edge_pattern = [
        f"{left_name}-{right_name}"
        for index, (left_name, left_vertex) in enumerate(names)
        for right_name, right_vertex in names[index + 1 :]
        if right_vertex in G.adj[left_vertex]
    ]
    word = partition_word(terminals)
    Gred, _, error = bp.try_reduce_certified(G, "C2", occ)
    return C2OccurrenceSummary(
        n_vertices=len(G.adj),
        raw_line=raw_line,
        occ=asdict(occ),
        terminal_vertices=terminals,
        terminal_partition_word=word,
        terminal_partition_key=canonical_partition_key(word),
        terminal_edge_pattern=edge_pattern,
        reduction_status="PASS" if Gred is not None else "FAIL",
        reduction_error=error,
    )


def scan_c2(plantri_path: Path, n_min: int, n_max: int) -> Dict[str, object]:
    summaries: List[C2OccurrenceSummary] = []
    scanned_graphs = 0
    graphs_with_c2 = 0

    for n_vertices in range(n_min, n_max + 1, 2):
        for embedding in iter_barnette_graph_rotations_via_plantri(str(plantri_path), n_vertices):
            G = graph_from_plantri_rotation(embedding.rot)
            scanned_graphs += 1
            occurrences = all_c2_occurrences(G)
            if occurrences:
                graphs_with_c2 += 1
            for occ in occurrences:
                summaries.append(summarize_occurrence(G, embedding.raw_line, occ))

    pass_records = [summary for summary in summaries if summary.reduction_status == "PASS"]
    fail_records = [summary for summary in summaries if summary.reduction_status == "FAIL"]
    partition_counter = Counter(
        ",".join(str(value) for value in summary.terminal_partition_key) for summary in summaries
    )
    pass_partition_counter = Counter(
        ",".join(str(value) for value in summary.terminal_partition_key) for summary in pass_records
    )
    fail_partition_counter = Counter(
        ",".join(str(value) for value in summary.terminal_partition_key) for summary in fail_records
    )
    edge_pattern_counter = Counter(
        ",".join(summary.terminal_edge_pattern) or "none" for summary in summaries
    )
    error_counter = Counter(summary.reduction_error or "none" for summary in fail_records)
    first_examples_by_partition: Dict[str, object] = {}
    for summary in summaries:
        key = ",".join(str(value) for value in summary.terminal_partition_key)
        first_examples_by_partition.setdefault(key, asdict(summary))
    distinct_terminal_fail_examples = [
        asdict(summary)
        for summary in fail_records
        if summary.terminal_partition_key == [0, 1, 2, 3]
    ]

    return {
        "plantri_path": str(plantri_path),
        "n_min": n_min,
        "n_max": n_max,
        "scanned_graph_count": scanned_graphs,
        "graphs_with_c2": graphs_with_c2,
        "occurrence_count": len(summaries),
        "pass_count": len(pass_records),
        "fail_count": len(fail_records),
        "terminal_partition_patterns": dict(partition_counter),
        "pass_partition_patterns": dict(pass_partition_counter),
        "fail_partition_patterns": dict(fail_partition_counter),
        "terminal_edge_patterns": dict(edge_pattern_counter),
        "reduction_errors": dict(error_counter),
        "first_examples_by_partition": first_examples_by_partition,
        "distinct_terminal_fail_examples": distinct_terminal_fail_examples,
        "first_pass": asdict(pass_records[0]) if pass_records else None,
        "first_fail": asdict(fail_records[0]) if fail_records else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan Barnette graphs for C2 reducibility.")
    parser.add_argument("--plantri", default=str(DEFAULT_PLANTRI))
    parser.add_argument("--n-min", type=int, default=8)
    parser.add_argument("--n-max", type=int, default=24)
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "c2_scan.json"))
    args = parser.parse_args()

    result = scan_c2(Path(args.plantri), args.n_min, args.n_max)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
