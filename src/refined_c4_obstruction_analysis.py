from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, List, Tuple

import barnette_proof as bp
import networkx as nx
from networkx.algorithms.planarity import get_counterexample
from refined_c4_gadget_search import (
    apply_candidate,
    bipartition_coloring,
    binary_matrices_with_margins,
    canonical_orientations,
    forced_gadget_colors,
    internal_graph_is_canonical,
    internal_graphs_for_col_sums_with_forced_colors,
    matrix_is_canonical,
    prioritized_col_sum_sequences,
)
from refined_c4_local import all_refined_c4_occurrences, extract_general_patch, graph_from_plantri_ascii


ROOT = Path(__file__).resolve().parents[1]


def candidate_adjacency(
    G: bp.EmbeddedGraph,
    patch,
    gadget_vertices: int,
    matrix: Tuple[Tuple[int, ...], ...],
    internal_edges: Tuple[Tuple[int, int], ...],
) -> Dict[int, set[int]]:
    deleted = set(patch.deleted_vertices)
    adj = {
        vertex: {neighbor for neighbor in neighbors if neighbor not in deleted}
        for vertex, neighbors in G.adj.items()
        if vertex not in deleted
    }
    start = max(adj) + 1
    new_vertices = [start + i for i in range(gadget_vertices)]
    for new_vertex in new_vertices:
        adj[new_vertex] = set()

    for row, boundary_vertex in enumerate(patch.boundary_vertices):
        for column, new_vertex in enumerate(new_vertices):
            if matrix[row][column]:
                adj[boundary_vertex].add(new_vertex)
                adj[new_vertex].add(boundary_vertex)

    for left, right in internal_edges:
        a = new_vertices[left]
        b = new_vertices[right]
        adj[a].add(b)
        adj[b].add(a)
    return adj


def abstract_planarity(adj: Dict[int, set[int]]) -> bool:
    graph = nx.Graph()
    for vertex, neighbors in adj.items():
        for neighbor in neighbors:
            if vertex < neighbor:
                graph.add_edge(vertex, neighbor)
    return nx.check_planarity(graph)[0]


def graph_from_adjacency(adj: Dict[int, set[int]]) -> nx.Graph:
    graph = nx.Graph()
    for vertex, neighbors in adj.items():
        for neighbor in neighbors:
            if vertex < neighbor:
                graph.add_edge(vertex, neighbor)
    return graph


def suppress_degree_two_vertices(graph: nx.Graph) -> nx.Graph:
    H = nx.Graph()
    H.add_nodes_from(graph.nodes())
    H.add_edges_from(graph.edges())
    changed = True
    while changed:
        changed = False
        for vertex in list(H.nodes()):
            if H.has_node(vertex) and H.degree(vertex) == 2:
                left, right = list(H.neighbors(vertex))
                if left == right:
                    continue
                H.remove_node(vertex)
                H.add_edge(left, right)
                changed = True
                break
    return H


def kuratowski_core_data(graph: nx.Graph) -> Dict[str, object]:
    counterexample = get_counterexample(graph)
    core = suppress_degree_two_vertices(counterexample)
    K33 = nx.complete_bipartite_graph(3, 3)
    K5 = nx.complete_graph(5)
    if nx.is_isomorphic(core, K33):
        core_type = "K3,3"
    elif nx.is_isomorphic(core, K5):
        core_type = "K5"
    else:
        core_type = "other"
    return {
        "type": core_type,
        "nodes": core.number_of_nodes(),
        "degree_sequence": sorted(degree for _, degree in core.degree()),
    }


def analyze_family(seed_line: str, gadget_vertices: int) -> Dict[str, object]:
    G = graph_from_plantri_ascii(seed_line)
    occ = all_refined_c4_occurrences(G)[0]
    patch = extract_general_patch(G, occ)
    row_sums = tuple(patch.boundary_attachment_sizes)
    boundary_colors = tuple(bipartition_coloring(G.adj)[vertex] for vertex in patch.boundary_vertices)
    slot_templates = [
        list(itertools.permutations(range(size))) if size > 1 else [(0,)]
        for size in row_sums
    ]

    result: Dict[str, object] = {
        "seed_graph_line": seed_line,
        "seed_occurrence": {
            "v1": occ.v1,
            "v2": occ.v2,
            "v3": occ.v3,
            "v4": occ.v4,
            "u1": occ.u1,
            "u2": occ.u2,
            "u3": occ.u3,
            "u4": occ.u4,
        },
        "patch": {
            "boundary_vertices": list(patch.boundary_vertices),
            "interface_word_key": list(patch.interface_word_key),
            "boundary_attachment_sizes": list(patch.boundary_attachment_sizes),
            "boundary_exterior_sizes": list(patch.boundary_exterior_sizes),
        },
        "gadget_vertices": gadget_vertices,
        "profiles": [],
        "totals": {
            "matrix_count": 0,
            "graph_pairs": 0,
            "bipartite_pairs": 0,
            "three_connected_pairs": 0,
            "abstract_planar_pairs": 0,
            "abstract_nonplanar_pairs": 0,
            "embedding_survivors": 0,
            "kuratowski_core_counts": {},
        },
    }

    for col_sums in prioritized_col_sum_sequences(sum(row_sums), gadget_vertices):
        matrices = [
            matrix
            for matrix in binary_matrices_with_margins(row_sums, col_sums)
            if matrix_is_canonical(matrix, col_sums)
            and forced_gadget_colors(matrix, boundary_colors) is not None
        ]
        profile = {
            "col_sums": list(col_sums),
            "matrix_count": len(matrices),
            "graph_pairs": 0,
            "bipartite_pairs": 0,
            "three_connected_pairs": 0,
            "abstract_planar_pairs": 0,
            "abstract_nonplanar_pairs": 0,
            "embedding_survivors": 0,
            "survivors": [],
        }
        result["totals"]["matrix_count"] += len(matrices)

        for matrix in matrices:
            forced_colors = forced_gadget_colors(matrix, boundary_colors)
            if forced_colors is None:
                continue
            internal_graphs = [
                edges
                for edges in internal_graphs_for_col_sums_with_forced_colors(col_sums, forced_colors)
                if internal_graph_is_canonical(edges, col_sums)
            ]
            for internal_edges in internal_graphs:
                profile["graph_pairs"] += 1
                result["totals"]["graph_pairs"] += 1
                adj = candidate_adjacency(G, patch, gadget_vertices, matrix, internal_edges)
                if not bp.is_bipartite(adj):
                    continue
                profile["bipartite_pairs"] += 1
                result["totals"]["bipartite_pairs"] += 1
                if not bp.is_3_connected(adj):
                    continue
                profile["three_connected_pairs"] += 1
                result["totals"]["three_connected_pairs"] += 1
                is_planar = abstract_planarity(adj)
                graph = graph_from_adjacency(adj)
                kuratowski = None
                if is_planar:
                    profile["abstract_planar_pairs"] += 1
                    result["totals"]["abstract_planar_pairs"] += 1
                else:
                    profile["abstract_nonplanar_pairs"] += 1
                    result["totals"]["abstract_nonplanar_pairs"] += 1
                    kuratowski = kuratowski_core_data(graph)
                    counts = result["totals"]["kuratowski_core_counts"]
                    counts[kuratowski["type"]] = counts.get(kuratowski["type"], 0) + 1

                failure_counts: Dict[str, int] = {}
                success_count = 0
                if not is_planar:
                    failure_counts["abstract graph is nonplanar"] = 1
                else:
                    for boundary_slot_orders in itertools.product(*slot_templates):
                        for gadget_orders in itertools.product(canonical_orientations(), repeat=gadget_vertices):
                            try:
                                apply_candidate(G, patch, matrix, internal_edges, boundary_slot_orders, gadget_orders)
                                success_count += 1
                            except Exception as exc:
                                key = str(exc)
                                failure_counts[key] = failure_counts.get(key, 0) + 1

                if success_count:
                    profile["embedding_survivors"] += 1
                    result["totals"]["embedding_survivors"] += 1

                profile["survivors"].append(
                    {
                        "matrix": [list(row) for row in matrix],
                        "internal_edges": [list(edge) for edge in internal_edges],
                        "abstract_planar": is_planar,
                        "kuratowski_core": kuratowski,
                        "success_count": success_count,
                        "top_failures": [
                            {"reason": reason, "count": count}
                            for reason, count in sorted(
                                failure_counts.items(),
                                key=lambda item: (-item[1], item[0]),
                            )[:5]
                        ],
                    }
                )

        if (
            profile["matrix_count"]
            or profile["graph_pairs"]
            or profile["three_connected_pairs"]
            or profile["abstract_nonplanar_pairs"]
            or profile["embedding_survivors"]
        ):
            result["profiles"].append(profile)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze why a refined-C4 family resists the current gadget search.")
    parser.add_argument("--seed-line", required=True)
    parser.add_argument("--gadget-vertices", type=int, required=True)
    parser.add_argument(
        "--out",
        default=str(ROOT / "artifacts" / "refined_c4_obstruction_analysis.json"),
    )
    args = parser.parse_args()

    result = analyze_family(args.seed_line, args.gadget_vertices)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
