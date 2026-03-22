from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher
from networkx.algorithms.planarity import get_counterexample

from refined_c4_local import all_refined_c4_occurrences, extract_general_patch, graph_from_plantri_ascii
from refined_c4_obstruction_analysis import candidate_adjacency, graph_from_adjacency


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS = (
    ROOT / "artifacts" / "refined_c4_b4_family_00123321_a2222_e1111_g8_obstruction.json",
    ROOT / "artifacts" / "refined_c4_b5_family_00112234_a12221_e21112_g8_obstruction.json",
    ROOT / "artifacts" / "refined_c4_b6_family_00112345_a111221_e222112_g8_obstruction.json",
)
STANDARD_K33 = nx.complete_bipartite_graph(3, 3)
STANDARD_EDGES = tuple((left, right) for left in range(3) for right in range(3, 6))


def occurrence_key(occ) -> Tuple[int, ...]:
    return (occ.v1, occ.v2, occ.v3, occ.v4, occ.u1, occ.u2, occ.u3, occ.u4)


def recover_patch(data: Dict[str, object]):
    G = graph_from_plantri_ascii(data["seed_graph_line"])
    target_key = (
        data["seed_occurrence"]["v1"],
        data["seed_occurrence"]["v2"],
        data["seed_occurrence"]["v3"],
        data["seed_occurrence"]["v4"],
        data["seed_occurrence"]["u1"],
        data["seed_occurrence"]["u2"],
        data["seed_occurrence"]["u3"],
        data["seed_occurrence"]["u4"],
    )
    for occ in all_refined_c4_occurrences(G):
        if occurrence_key(occ) == target_key:
            return G, occ, extract_general_patch(G, occ)
    raise ValueError(f"could not recover refined-C4 occurrence {target_key}")


def suppress_degree_two_with_paths(graph: nx.Graph) -> nx.Graph:
    H = nx.Graph()
    H.add_nodes_from(graph.nodes())
    for left, right in graph.edges():
        H.add_edge(left, right, interior=tuple())

    changed = True
    while changed:
        changed = False
        for vertex in sorted(H.nodes()):
            if H.degree(vertex) != 2:
                continue
            left, right = sorted(H.neighbors(vertex))
            if H.has_edge(left, right):
                raise ValueError(
                    "degree-two suppression would create a parallel edge in the Kuratowski counterexample"
                )
            left_path = H[left][vertex]["interior"]
            right_path = H[right][vertex]["interior"]
            merged = tuple(sorted((*left_path, vertex, *right_path)))
            H.remove_node(vertex)
            H.add_edge(left, right, interior=merged)
            changed = True
            break
    return H


def classify_node(
    vertex: int,
    boundary_index: Dict[int, int],
    gadget_nodes: set[int],
) -> Tuple[str, int | None]:
    if vertex in boundary_index:
        return ("B", boundary_index[vertex])
    if vertex in gadget_nodes:
        return ("G", vertex)
    return ("A", None)


def edge_summary(
    members: Iterable[int],
    boundary_index: Dict[int, int],
    gadget_nodes: set[int],
) -> Dict[str, object]:
    boundary_positions = sorted(boundary_index[vertex] for vertex in members if vertex in boundary_index)
    gadget_count = sum(1 for vertex in members if vertex in gadget_nodes)
    ambient_count = sum(
        1
        for vertex in members
        if vertex not in gadget_nodes and vertex not in boundary_index
    )
    return {
        "boundary_positions": boundary_positions,
        "boundary_count": len(boundary_positions),
        "gadget_count": gadget_count,
        "ambient_count": ambient_count,
    }


def _tokenize_fine(
    raw: Tuple[str, int | None],
    gadget_names: Dict[int, int],
    ambient_counter: List[int],
) -> str:
    kind, value = raw
    if kind == "B":
        return f"B{value}"
    if kind == "G":
        if value not in gadget_names:
            gadget_names[value] = len(gadget_names)
        return f"G{gadget_names[value]}"
    index = ambient_counter[0]
    ambient_counter[0] += 1
    return f"A{index}"


def canonical_signature(
    core: nx.Graph,
    boundary_index: Dict[int, int],
    gadget_nodes: set[int],
) -> Dict[str, object]:
    if not nx.is_isomorphic(core, STANDARD_K33):
        raise ValueError("suppressed counterexample is not K3,3")

    best: Dict[str, object] | None = None
    matcher = GraphMatcher(core, STANDARD_K33)
    for mapping in matcher.isomorphisms_iter():
        inverse = {standard: source for source, standard in mapping.items()}
        gadget_names: Dict[int, int] = {}
        ambient_counter = [0]

        fine_nodes: List[str] = []
        coarse_nodes: List[str] = []
        for standard in range(6):
            raw = classify_node(inverse[standard], boundary_index, gadget_nodes)
            fine_nodes.append(_tokenize_fine(raw, gadget_names, ambient_counter))
            coarse_nodes.append(raw[0])

        fine_edges: List[Tuple[Tuple[int, ...], int, int]] = []
        coarse_edges: List[Tuple[int, int, int]] = []
        for left, right in STANDARD_EDGES:
            members = core[inverse[left]][inverse[right]]["interior"]
            summary = edge_summary(members, boundary_index, gadget_nodes)
            fine_edges.append(
                (
                    tuple(summary["boundary_positions"]),
                    summary["gadget_count"],
                    summary["ambient_count"],
                )
            )
            coarse_edges.append(
                (
                    summary["boundary_count"],
                    summary["gadget_count"],
                    summary["ambient_count"],
                )
            )

        fine_signature = tuple(fine_nodes + [str(item) for item in fine_edges])
        coarse_signature = tuple(coarse_nodes + [str(item) for item in coarse_edges])

        candidate = {
            "fine_signature": fine_signature,
            "coarse_signature": coarse_signature,
            "fine_template": {
                "branch_nodes": fine_nodes,
                "edge_paths": [
                    {
                        "edge": [left, right],
                        "boundary_positions": list(boundaries),
                        "gadget_count": gadget_count,
                        "ambient_count": ambient_count,
                    }
                    for (left, right), (boundaries, gadget_count, ambient_count) in zip(
                        STANDARD_EDGES,
                        fine_edges,
                    )
                ],
            },
            "coarse_template": {
                "branch_nodes": coarse_nodes,
                "edge_paths": [
                    {
                        "edge": [left, right],
                        "boundary_count": boundary_count,
                        "gadget_count": gadget_count,
                        "ambient_count": ambient_count,
                    }
                    for (left, right), (boundary_count, gadget_count, ambient_count) in zip(
                        STANDARD_EDGES,
                        coarse_edges,
                    )
                ],
            },
        }

        if best is None or (
            candidate["fine_signature"],
            candidate["coarse_signature"],
        ) < (
            best["fine_signature"],
            best["coarse_signature"],
        ):
            best = candidate

    if best is None:
        raise ValueError("could not canonicalize K3,3 signature")
    return best


def short_family_id(path: Path) -> str:
    stem = path.stem
    if stem.startswith("refined_c4_"):
        stem = stem[len("refined_c4_"):]
    return stem


def internal_skeleton_graph(internal_edges: Sequence[Sequence[int]], gadget_vertices: int) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(gadget_vertices))
    graph.add_edges_from(tuple(edge) for edge in internal_edges)
    return graph


def summarize_internal_skeleton_classes(survivors: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    runtime_groups: List[Dict[str, object]] = []
    for survivor in survivors:
        graph = internal_skeleton_graph(survivor["internal_edges"], len(survivor["col_sums"]))
        matched = None
        for group in runtime_groups:
            if nx.is_isomorphic(graph, group["graph"]):
                matched = group
                break
        if matched is None:
            matched = {
                "graph": graph,
                "count": 0,
                "families": {},
                "col_sum_profiles": {},
                "representative_internal_edges": survivor["internal_edges"],
                "examples": [],
            }
            runtime_groups.append(matched)
        matched["count"] += 1
        matched["families"][survivor["family"]] = matched["families"].get(survivor["family"], 0) + 1
        col_key = ",".join(str(value) for value in survivor["col_sums"])
        matched["col_sum_profiles"][col_key] = matched["col_sum_profiles"].get(col_key, 0) + 1
        if len(matched["examples"]) < 5:
            matched["examples"].append(
                {
                    "family": survivor["family"],
                    "col_sums": survivor["col_sums"],
                    "matrix": survivor["matrix"],
                    "internal_edges": survivor["internal_edges"],
                }
            )

    return [
        {
            "class_id": index + 1,
            "count": group["count"],
            "families": group["families"],
            "col_sum_profiles": group["col_sum_profiles"],
            "representative_internal_edges": group["representative_internal_edges"],
            "examples": group["examples"],
        }
        for index, group in enumerate(
            sorted(
                runtime_groups,
                key=lambda item: (-item["count"], item["representative_internal_edges"]),
            )
        )
    ]


def analyze_artifact(path: Path) -> Dict[str, object]:
    data = json.loads(path.read_text())
    G, _, patch = recover_patch(data)
    boundary_index = {
        vertex: idx for idx, vertex in enumerate(patch.boundary_vertices)
    }
    max_original = max(G.adj)
    gadget_nodes = set(range(max_original + 1, max_original + 1 + data["gadget_vertices"]))

    survivors: List[Dict[str, object]] = []
    for profile in data["profiles"]:
        for survivor in profile["survivors"]:
            adj = candidate_adjacency(
                G,
                patch,
                data["gadget_vertices"],
                tuple(tuple(row) for row in survivor["matrix"]),
                tuple(tuple(edge) for edge in survivor["internal_edges"]),
            )
            graph = graph_from_adjacency(adj)
            counterexample = get_counterexample(graph)
            core = suppress_degree_two_with_paths(counterexample)
            signature = canonical_signature(core, boundary_index, gadget_nodes)
            survivors.append(
                {
                    "family": short_family_id(path),
                    "artifact": str(path.relative_to(ROOT)),
                    "col_sums": profile["col_sums"],
                    "matrix": survivor["matrix"],
                    "internal_edges": survivor["internal_edges"],
                    "fine_signature": list(signature["fine_signature"]),
                    "coarse_signature": list(signature["coarse_signature"]),
                    "fine_template": signature["fine_template"],
                    "coarse_template": signature["coarse_template"],
                }
            )

    fine_groups: Dict[Tuple[str, ...], Dict[str, object]] = {}
    coarse_groups: Dict[Tuple[str, ...], Dict[str, object]] = {}
    for survivor in survivors:
        fine_key = tuple(survivor["fine_signature"])
        fine_group = fine_groups.setdefault(
            fine_key,
            {
                "count": 0,
                "families": {},
                "template": survivor["fine_template"],
                "examples": [],
            },
        )
        fine_group["count"] += 1
        fine_group["families"][survivor["family"]] = fine_group["families"].get(survivor["family"], 0) + 1
        if len(fine_group["examples"]) < 3:
            fine_group["examples"].append(
                {
                    "col_sums": survivor["col_sums"],
                    "matrix": survivor["matrix"],
                    "internal_edges": survivor["internal_edges"],
                }
            )

        coarse_key = tuple(survivor["coarse_signature"])
        coarse_group = coarse_groups.setdefault(
            coarse_key,
            {
                "count": 0,
                "families": {},
                "template": survivor["coarse_template"],
                "examples": [],
            },
        )
        coarse_group["count"] += 1
        coarse_group["families"][survivor["family"]] = coarse_group["families"].get(survivor["family"], 0) + 1
        if len(coarse_group["examples"]) < 3:
            coarse_group["examples"].append(
                {
                    "col_sums": survivor["col_sums"],
                    "matrix": survivor["matrix"],
                    "internal_edges": survivor["internal_edges"],
                }
            )

    internal_skeleton_classes = summarize_internal_skeleton_classes(survivors)
    return {
        "family": short_family_id(path),
        "artifact": str(path.relative_to(ROOT)),
        "survivor_count": len(survivors),
        "fine_template_count": len(fine_groups),
        "coarse_template_count": len(coarse_groups),
        "internal_skeleton_class_count": len(internal_skeleton_classes),
        "fine_templates": [
            {
                "template_id": index + 1,
                "count": group["count"],
                "families": group["families"],
                "template": group["template"],
                "examples": group["examples"],
            }
            for index, (_, group) in enumerate(
                sorted(
                    fine_groups.items(),
                    key=lambda item: (-item[1]["count"], item[0]),
                )
            )
        ],
        "coarse_templates": [
            {
                "template_id": index + 1,
                "count": group["count"],
                "families": group["families"],
                "template": group["template"],
                "examples": group["examples"],
            }
            for index, (_, group) in enumerate(
                sorted(
                    coarse_groups.items(),
                    key=lambda item: (-item[1]["count"], item[0]),
                )
            )
        ],
        "internal_skeleton_classes": internal_skeleton_classes,
        "survivors": survivors,
    }


def summarize_global(family_results: Sequence[Dict[str, object]]) -> Dict[str, object]:
    fine_groups: Dict[Tuple[str, ...], Dict[str, object]] = {}
    coarse_groups: Dict[Tuple[str, ...], Dict[str, object]] = {}

    for family in family_results:
        for survivor in family["survivors"]:
            fine_key = tuple(survivor["fine_signature"])
            fine_group = fine_groups.setdefault(
                fine_key,
                {
                    "count": 0,
                    "families": {},
                    "template": survivor["fine_template"],
                    "examples": [],
                },
            )
            fine_group["count"] += 1
            fine_group["families"][survivor["family"]] = fine_group["families"].get(survivor["family"], 0) + 1
            if len(fine_group["examples"]) < 5:
                fine_group["examples"].append(
                    {
                        "family": survivor["family"],
                        "col_sums": survivor["col_sums"],
                        "matrix": survivor["matrix"],
                        "internal_edges": survivor["internal_edges"],
                    }
                )

            coarse_key = tuple(survivor["coarse_signature"])
            coarse_group = coarse_groups.setdefault(
                coarse_key,
                {
                    "count": 0,
                    "families": {},
                    "template": survivor["coarse_template"],
                    "examples": [],
                },
            )
            coarse_group["count"] += 1
            coarse_group["families"][survivor["family"]] = coarse_group["families"].get(survivor["family"], 0) + 1
            if len(coarse_group["examples"]) < 5:
                coarse_group["examples"].append(
                    {
                        "family": survivor["family"],
                        "col_sums": survivor["col_sums"],
                        "matrix": survivor["matrix"],
                        "internal_edges": survivor["internal_edges"],
                    }
                )

    internal_skeleton_classes = summarize_internal_skeleton_classes(
        [survivor for family in family_results for survivor in family["survivors"]]
    )
    return {
        "survivor_count": sum(family["survivor_count"] for family in family_results),
        "fine_template_count": len(fine_groups),
        "coarse_template_count": len(coarse_groups),
        "internal_skeleton_class_count": len(internal_skeleton_classes),
        "internal_skeleton_classes": internal_skeleton_classes,
        "fine_templates": [
            {
                "template_id": index + 1,
                "count": group["count"],
                "families": group["families"],
                "template": group["template"],
                "examples": group["examples"],
            }
            for index, (_, group) in enumerate(
                sorted(
                    fine_groups.items(),
                    key=lambda item: (-item[1]["count"], item[0]),
                )
            )
        ],
        "coarse_templates": [
            {
                "template_id": index + 1,
                "count": group["count"],
                "families": group["families"],
                "template": group["template"],
                "examples": group["examples"],
            }
            for index, (_, group) in enumerate(
                sorted(
                    coarse_groups.items(),
                    key=lambda item: (-item[1]["count"], item[0]),
                )
            )
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Group refined-C4 K3,3 obstruction witnesses into canonical template classes."
    )
    parser.add_argument(
        "--artifact",
        action="append",
        dest="artifacts",
        help="Path to a refined-C4 obstruction artifact (may be repeated).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "artifacts" / "refined_c4_k33_templates.json",
        help="Path to the output JSON summary.",
    )
    args = parser.parse_args()

    artifact_paths = tuple(Path(path) for path in (args.artifacts or DEFAULT_ARTIFACTS))
    family_results = [analyze_artifact(path) for path in artifact_paths]
    payload = {
        "artifacts": [str(path.relative_to(ROOT)) if path.is_absolute() else str(path) for path in artifact_paths],
        "families": family_results,
        "global": summarize_global(family_results),
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
