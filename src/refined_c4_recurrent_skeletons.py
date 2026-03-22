from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import networkx as nx


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "artifacts" / "refined_c4_k33_templates.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "refined_c4_recurrent_skeleton_classes.json"


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text())


def skeleton_graph(internal_edges: Sequence[Sequence[int]]) -> nx.Graph:
    graph = nx.Graph()
    vertices = sorted({vertex for edge in internal_edges for vertex in edge})
    graph.add_nodes_from(vertices)
    graph.add_edges_from(tuple(edge) for edge in internal_edges)
    return graph


def graph_invariants(internal_edges: Sequence[Sequence[int]]) -> Dict[str, object]:
    graph = skeleton_graph(internal_edges)
    cycle_basis = nx.cycle_basis(graph)
    return {
        "vertex_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "degree_sequence": sorted((degree for _, degree in graph.degree()), reverse=True),
        "leaf_vertices": sorted(vertex for vertex, degree in graph.degree() if degree == 1),
        "bridge_count": len(list(nx.bridges(graph))),
        "cycle_lengths": sorted(len(cycle) for cycle in cycle_basis),
    }


def summarize_recurrent_classes(data: Dict[str, object]) -> Dict[str, object]:
    classes = [
        skeleton
        for skeleton in data["global"]["internal_skeleton_classes"]
        if skeleton["count"] > 1
    ]

    labeled_classes: List[Dict[str, object]] = []
    for index, skeleton in enumerate(classes, start=1):
        label = f"R{index}"
        families = skeleton["families"]
        family_names = sorted(families)
        record = {
            "label": label,
            "count": skeleton["count"],
            "family_count": len(family_names),
            "families": families,
            "family_names": family_names,
            "cross_family": len(family_names) > 1,
            "all_three_families": len(family_names) == 3,
            "col_sum_profiles": skeleton["col_sum_profiles"],
            "representative_internal_edges": skeleton["representative_internal_edges"],
            "invariants": graph_invariants(skeleton["representative_internal_edges"]),
            "examples": skeleton["examples"],
        }
        labeled_classes.append(record)

    cross_family = [record for record in labeled_classes if record["cross_family"]]
    all_three = [record for record in labeled_classes if record["all_three_families"]]
    profile_support: Dict[str, List[str]] = {}
    for record in labeled_classes:
        for profile in sorted(record["col_sum_profiles"]):
            profile_support.setdefault(profile, []).append(record["label"])
    for labels in profile_support.values():
        labels.sort()

    return {
        "source_artifact": str(DEFAULT_INPUT.relative_to(ROOT)),
        "recurrent_class_count": len(labeled_classes),
        "recurrent_survivor_count": sum(record["count"] for record in labeled_classes),
        "cross_family_class_count": len(cross_family),
        "all_three_family_class_count": len(all_three),
        "labels_cross_family": [record["label"] for record in cross_family],
        "labels_all_three_families": [record["label"] for record in all_three],
        "distinct_profile_count": len(profile_support),
        "profile_support": profile_support,
        "classes": labeled_classes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract the recurrent refined-C4 8-vertex skeleton classes from the K3,3 template artifact."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to artifacts/refined_c4_k33_templates.json.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to write the recurrent-class JSON summary.",
    )
    args = parser.parse_args()

    payload = summarize_recurrent_classes(load_json(args.input))
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
