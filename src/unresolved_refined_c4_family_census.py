from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from refined_c4_local import all_refined_c4_occurrences, graph_from_plantri_ascii, occurrence_profile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UNRESOLVED = ROOT / "artifacts" / "completeness_unresolved_analysis_n30.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "unresolved_refined_c4_family_census_n30.json"


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def seq_key(values: Iterable[int] | None) -> str:
    if values is None:
        return "none"
    return ",".join(str(value) for value in values)


def family_key(profile) -> str:
    return (
        f"L={seq_key(profile.outer_face_length_key)}"
        f"|B={profile.general_boundary_count}"
        f"|I={seq_key(profile.general_interface_word_key)}"
        f"|A={seq_key(profile.general_attachment_sizes)}"
        f"|E={seq_key(profile.general_exterior_sizes)}"
    )


def interface_key(profile) -> str:
    return (
        f"B={profile.general_boundary_count}"
        f"|I={seq_key(profile.general_interface_word_key)}"
        f"|A={seq_key(profile.general_attachment_sizes)}"
        f"|E={seq_key(profile.general_exterior_sizes)}"
    )


def summarize_unresolved_refined_c4_families(unresolved_path: Path) -> Dict[str, object]:
    data = load_json(unresolved_path)
    graphs: List[Dict[str, object]] = data["graphs"]

    unresolved_refined = [graph for graph in graphs if int(graph.get("refined_c4_count", 0)) > 0]

    family_occurrence_counts: Counter[str] = Counter()
    family_graph_support: Counter[str] = Counter()
    interface_occurrence_counts: Counter[str] = Counter()
    interface_graph_support: Counter[str] = Counter()
    boundary_count_patterns: Counter[str] = Counter()
    outer_face_length_patterns: Counter[str] = Counter()
    deleted_size_patterns: Counter[str] = Counter()
    per_n_graph_counts: Counter[int] = Counter()
    per_n_occurrence_counts: Counter[int] = Counter()
    family_examples: Dict[str, object] = {}

    for graph_record in unresolved_refined:
        raw_line = str(graph_record["raw_line"])
        n_vertices = int(graph_record["n_vertices"])
        G = graph_from_plantri_ascii(raw_line)

        seen_family_keys = set()
        seen_interface_keys = set()
        occurrences = all_refined_c4_occurrences(G)
        for occ in occurrences:
            profile = occurrence_profile(G, occ)
            family = family_key(profile)
            interface = interface_key(profile)

            family_occurrence_counts[family] += 1
            interface_occurrence_counts[interface] += 1
            boundary_count_patterns[str(profile.general_boundary_count)] += 1
            outer_face_length_patterns[seq_key(profile.outer_face_length_key)] += 1
            deleted_size_patterns[str(profile.general_deleted_size)] += 1
            per_n_occurrence_counts[n_vertices] += 1

            if family not in seen_family_keys:
                family_graph_support[family] += 1
                seen_family_keys.add(family)
            if interface not in seen_interface_keys:
                interface_graph_support[interface] += 1
                seen_interface_keys.add(interface)

            family_examples.setdefault(
                family,
                {
                    "n_vertices": n_vertices,
                    "raw_line": raw_line,
                    "occ": {
                        "v1": occ.v1,
                        "v2": occ.v2,
                        "v3": occ.v3,
                        "v4": occ.v4,
                        "u1": occ.u1,
                        "u2": occ.u2,
                        "u3": occ.u3,
                        "u4": occ.u4,
                    },
                    "outer_face_length_key": list(profile.outer_face_length_key),
                    "general_boundary_count": profile.general_boundary_count,
                    "general_interface_word_key": (
                        list(profile.general_interface_word_key)
                        if profile.general_interface_word_key is not None
                        else None
                    ),
                    "general_attachment_sizes": (
                        list(profile.general_attachment_sizes)
                        if profile.general_attachment_sizes is not None
                        else None
                    ),
                    "general_exterior_sizes": (
                        list(profile.general_exterior_sizes)
                        if profile.general_exterior_sizes is not None
                        else None
                    ),
                    "general_deleted_size": profile.general_deleted_size,
                },
            )

        per_n_graph_counts[n_vertices] += 1

    return {
        "source_unresolved_artifact": str(unresolved_path.relative_to(ROOT)),
        "unresolved_graph_count": len(graphs),
        "unresolved_refined_c4_graph_count": len(unresolved_refined),
        "unresolved_refined_c4_occurrence_count": sum(family_occurrence_counts.values()),
        "per_n_refined_c4_graph_counts": {str(n): per_n_graph_counts[n] for n in sorted(per_n_graph_counts)},
        "per_n_refined_c4_occurrence_counts": {
            str(n): per_n_occurrence_counts[n] for n in sorted(per_n_occurrence_counts)
        },
        "boundary_count_patterns": dict(boundary_count_patterns),
        "outer_face_length_patterns": dict(outer_face_length_patterns),
        "deleted_size_patterns": dict(deleted_size_patterns),
        "family_occurrence_counts": dict(family_occurrence_counts),
        "family_graph_support": dict(family_graph_support),
        "interface_occurrence_counts": dict(interface_occurrence_counts),
        "interface_graph_support": dict(interface_graph_support),
        "family_examples": family_examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify refined-C4 occurrences inside the unresolved completeness frontier."
    )
    parser.add_argument(
        "--unresolved-artifact",
        type=Path,
        default=DEFAULT_UNRESOLVED,
        help="Path to a completeness_unresolved_analysis JSON artifact.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to write the refined-C4 frontier family census.",
    )
    args = parser.parse_args()

    payload = summarize_unresolved_refined_c4_families(args.unresolved_artifact.resolve())
    out_path = args.out.resolve()
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
