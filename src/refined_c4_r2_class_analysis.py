from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

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
DEFAULT_ARTIFACTS = (
    ROOT / "artifacts" / "refined_c4_b4_family_00123321_a2222_e1111_g8_obstruction.json",
    ROOT / "artifacts" / "refined_c4_b5_family_00112234_a12221_e21112_g8_obstruction.json",
    ROOT / "artifacts" / "refined_c4_b6_family_00112345_a111221_e222112_g8_obstruction.json",
)
R2_COL_SUMS = (0, 0, 1, 1, 1, 1, 2, 2)
R2_INTERNAL_EDGES = (
    (0, 2),
    (0, 3),
    (0, 4),
    (1, 2),
    (1, 3),
    (1, 5),
    (4, 6),
    (5, 7),
)


def boundary_support(matrix: Tuple[Tuple[int, ...], ...]) -> List[List[int]]:
    return [
        [row for row, row_values in enumerate(matrix) if row_values[column]]
        for column in range(len(matrix[0]))
    ]


def analyze_family(path: Path) -> Dict[str, object]:
    data = json.loads(path.read_text())
    G, _, patch = recover_patch(data)
    boundary_colors = tuple(bipartition_coloring(G.adj)[vertex] for vertex in patch.boundary_vertices)
    matrices = [
        matrix
        for matrix in binary_matrices_with_margins(tuple(patch.boundary_attachment_sizes), R2_COL_SUMS)
        if matrix_is_canonical(matrix, R2_COL_SUMS)
        and forced_gadget_colors(matrix, boundary_colors) is not None
    ]

    result: Dict[str, object] = {
        "family": short_family_id(path),
        "artifact": str(path.relative_to(ROOT)),
        "boundary_attachment_sizes": list(patch.boundary_attachment_sizes),
        "boundary_exterior_sizes": list(patch.boundary_exterior_sizes),
        "boundary_count": len(patch.boundary_vertices),
        "col_sums": list(R2_COL_SUMS),
        "internal_edges": [list(edge) for edge in R2_INTERNAL_EDGES],
        "canonical_matrix_count": len(matrices),
        "bipartite_count": 0,
        "three_connected_count": 0,
        "abstract_planar_count": 0,
        "abstract_nonplanar_count": 0,
        "kuratowski_core_counts": {},
        "three_connected_candidates": [],
    }

    for matrix in matrices:
        adj = candidate_adjacency(G, patch, len(R2_COL_SUMS), matrix, R2_INTERNAL_EDGES)
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


def summarize_global(families: List[Dict[str, object]]) -> Dict[str, object]:
    totals: Dict[str, object] = {
        "family_count": len(families),
        "col_sums": list(R2_COL_SUMS),
        "internal_edges": [list(edge) for edge in R2_INTERNAL_EDGES],
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
        description="Analyze the recurrent refined-C4 obstruction class R2 across the unresolved families."
    )
    parser.add_argument(
        "--artifact",
        action="append",
        dest="artifacts",
        help="Path to a refined-C4 family obstruction artifact (may be repeated).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "artifacts" / "refined_c4_r2_class.json",
        help="Path to write the R2 analysis artifact.",
    )
    args = parser.parse_args()

    artifact_paths = tuple(Path(path) for path in (args.artifacts or DEFAULT_ARTIFACTS))
    families = [analyze_family(path) for path in artifact_paths]
    payload = {
        "source_artifacts": [str(path.relative_to(ROOT)) if path.is_absolute() else str(path) for path in artifact_paths],
        "r2_class_label": "R2",
        "families": families,
        "global": summarize_global(families),
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
