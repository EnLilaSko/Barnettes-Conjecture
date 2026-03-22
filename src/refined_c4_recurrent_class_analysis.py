from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import barnette_proof as bp

from refined_c4_gadget_search import (
    binary_matrices_with_margins,
    bipartition_coloring,
    forced_gadget_colors,
    matrix_is_canonical,
)
from refined_c4_k33_templates import recover_patch, short_family_id
from refined_c4_obstruction_analysis import (
    abstract_planarity,
    candidate_adjacency,
    graph_from_adjacency,
    kuratowski_core_data,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FAMILY_ARTIFACTS = (
    ROOT / "artifacts" / "refined_c4_b4_family_00123321_a2222_e1111_g8_obstruction.json",
    ROOT / "artifacts" / "refined_c4_b5_family_00112234_a12221_e21112_g8_obstruction.json",
    ROOT / "artifacts" / "refined_c4_b6_family_00112345_a111221_e222112_g8_obstruction.json",
)
DEFAULT_CLASS_ARTIFACT = ROOT / "artifacts" / "refined_c4_recurrent_skeleton_classes.json"


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text())


def boundary_support(matrix: Tuple[Tuple[int, ...], ...]) -> List[List[int]]:
    return [
        [row for row, row_values in enumerate(matrix) if row_values[column]]
        for column in range(len(matrix[0]))
    ]


def load_class_record(path: Path, label: str) -> Dict[str, object]:
    data = load_json(path)
    for record in data["classes"]:
        if record["label"] == label:
            return record
    raise ValueError(f"class label {label!r} not found in {path}")


def class_profile(record: Dict[str, object]) -> Tuple[int, ...]:
    profiles = list(record["col_sum_profiles"])
    if len(profiles) != 1:
        raise ValueError(f"class {record['label']} has multiple column profiles: {profiles}")
    return tuple(int(value) for value in profiles[0].split(","))


def family_artifact_map(paths: Sequence[Path]) -> Dict[str, Path]:
    return {short_family_id(path): path for path in paths}


def analyze_family(
    path: Path,
    class_label: str,
    col_sums: Tuple[int, ...],
    internal_edges: Tuple[Tuple[int, int], ...],
) -> Dict[str, object]:
    data = load_json(path)
    G, _, patch = recover_patch(data)
    boundary_colors = tuple(bipartition_coloring(G.adj)[vertex] for vertex in patch.boundary_vertices)
    matrices = [
        matrix
        for matrix in binary_matrices_with_margins(tuple(patch.boundary_attachment_sizes), col_sums)
        if matrix_is_canonical(matrix, col_sums)
        and forced_gadget_colors(matrix, boundary_colors) is not None
    ]

    result: Dict[str, object] = {
        "family": short_family_id(path),
        "artifact": str(path.relative_to(ROOT)),
        "class_label": class_label,
        "boundary_attachment_sizes": list(patch.boundary_attachment_sizes),
        "boundary_exterior_sizes": list(patch.boundary_exterior_sizes),
        "boundary_count": len(patch.boundary_vertices),
        "col_sums": list(col_sums),
        "internal_edges": [list(edge) for edge in internal_edges],
        "canonical_matrix_count": len(matrices),
        "bipartite_count": 0,
        "three_connected_count": 0,
        "abstract_planar_count": 0,
        "abstract_nonplanar_count": 0,
        "kuratowski_core_counts": {},
        "three_connected_candidates": [],
    }

    for matrix in matrices:
        adj = candidate_adjacency(G, patch, len(col_sums), matrix, internal_edges)
        if not bp.is_bipartite(adj):
            continue
        result["bipartite_count"] += 1
        if not bp.is_3_connected(adj):
            continue
        result["three_connected_count"] += 1

        planar = abstract_planarity(adj)
        kuratowski = None
        if planar:
            result["abstract_planar_count"] += 1
        else:
            result["abstract_nonplanar_count"] += 1
            graph = graph_from_adjacency(adj)
            kuratowski = kuratowski_core_data(graph)
            counts = result["kuratowski_core_counts"]
            counts[kuratowski["type"]] = counts.get(kuratowski["type"], 0) + 1

        result["three_connected_candidates"].append(
            {
                "matrix": [list(row) for row in matrix],
                "boundary_support": boundary_support(matrix),
                "abstract_planar": planar,
                "kuratowski_core": kuratowski,
            }
        )

    return result


def summarize_global(
    class_label: str,
    col_sums: Tuple[int, ...],
    internal_edges: Tuple[Tuple[int, int], ...],
    families: List[Dict[str, object]],
) -> Dict[str, object]:
    totals: Dict[str, object] = {
        "class_label": class_label,
        "family_count": len(families),
        "col_sums": list(col_sums),
        "internal_edges": [list(edge) for edge in internal_edges],
        "canonical_matrix_count": 0,
        "bipartite_count": 0,
        "three_connected_count": 0,
        "abstract_planar_count": 0,
        "abstract_nonplanar_count": 0,
        "kuratowski_core_counts": {},
    }
    for family in families:
        totals["canonical_matrix_count"] += family["canonical_matrix_count"]
        totals["bipartite_count"] += family["bipartite_count"]
        totals["three_connected_count"] += family["three_connected_count"]
        totals["abstract_planar_count"] += family["abstract_planar_count"]
        totals["abstract_nonplanar_count"] += family["abstract_nonplanar_count"]
        for key, value in family["kuratowski_core_counts"].items():
            totals["kuratowski_core_counts"][key] = totals["kuratowski_core_counts"].get(key, 0) + value
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a recurrent refined-C4 obstruction class across its supported families."
    )
    parser.add_argument("--label", required=True, help="Recurrent class label, e.g. R1 or R2.")
    parser.add_argument(
        "--class-artifact",
        type=Path,
        default=DEFAULT_CLASS_ARTIFACT,
        help="Path to artifacts/refined_c4_recurrent_skeleton_classes.json.",
    )
    parser.add_argument(
        "--family-artifact",
        action="append",
        dest="family_artifacts",
        help="Path to a refined-C4 family obstruction artifact (may be repeated).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Path to write the class-level JSON artifact.",
    )
    args = parser.parse_args()

    class_record = load_class_record(args.class_artifact, args.label)
    col_sums = class_profile(class_record)
    internal_edges = tuple(tuple(edge) for edge in class_record["representative_internal_edges"])
    family_paths = tuple(Path(path) for path in (args.family_artifacts or DEFAULT_FAMILY_ARTIFACTS))
    family_map = family_artifact_map(family_paths)
    selected_paths = [family_map[name] for name in class_record["family_names"]]

    families = [
        analyze_family(path, args.label, col_sums, internal_edges)
        for path in selected_paths
    ]
    out_path = (args.out if args.out is not None else (ROOT / "artifacts" / f"refined_c4_{args.label.lower()}_class.json")).resolve()
    payload = {
        "source_class_artifact": str(args.class_artifact.relative_to(ROOT)) if args.class_artifact.is_absolute() else str(args.class_artifact),
        "source_family_artifacts": [str(path.relative_to(ROOT)) if path.is_absolute() else str(path) for path in selected_paths],
        "class_label": args.label,
        "recurrent_class_record": class_record,
        "families": families,
        "global": summarize_global(args.label, col_sums, internal_edges, families),
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
