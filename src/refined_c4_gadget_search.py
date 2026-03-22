from __future__ import annotations

import argparse
import itertools
import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import barnette_proof as bp
from barnette_proof import EmbeddedGraph
from plantri_wrapper import iter_barnette_graph_rotations_via_plantri
from refined_c4_local import (
    GeneralPatchDescriptor,
    all_refined_c4_occurrences,
    extract_general_patch,
    graph_from_plantri_ascii,
    graph_from_plantri_rotation,
    occurrence_profile,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLANTRI = ROOT / "plantri.exe"


def patch_is_supported(patch: GeneralPatchDescriptor) -> bool:
    return (
        sum(patch.boundary_attachment_sizes) <= 12
        and all(size in (1, 2) for size in patch.boundary_attachment_sizes)
    )


def canonical_orientations() -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    return ((0, 1, 2), (2, 1, 0))


def col_sum_sequences(total: int, length: int, max_entry: int = 3) -> Iterable[Tuple[int, ...]]:
    def rec(index: int, remaining: int, prev: int, acc: List[int]) -> Iterable[Tuple[int, ...]]:
        if index == length:
            if remaining == 0:
                yield tuple(acc)
            return

        slots_left = length - index - 1
        lower = max(prev, remaining - slots_left * max_entry)
        upper = min(max_entry, remaining)
        for value in range(lower, upper + 1):
            acc.append(value)
            yield from rec(index + 1, remaining - value, value, acc)
            acc.pop()

    yield from rec(0, total, 0, [])


def prioritized_col_sum_sequences(total: int, length: int) -> List[Tuple[int, ...]]:
    sequences = list(col_sum_sequences(total, length))
    sequences.sort(
        key=lambda seq: (
            sum(value in (0, 3) for value in seq),
            sum(abs(value - 1) for value in seq),
            seq,
        )
    )
    return sequences


@lru_cache(maxsize=None)
def equal_value_groups(values: Tuple[int, ...]) -> Tuple[Tuple[int, ...], ...]:
    groups: List[List[int]] = []
    current: List[int] = []
    current_value: Optional[int] = None
    for index, value in enumerate(values):
        if current_value is None or value == current_value:
            current.append(index)
            current_value = value
        else:
            groups.append(current)
            current = [index]
            current_value = value
    if current:
        groups.append(current)
    return tuple(tuple(group) for group in groups)


def matrix_is_canonical(matrix: Tuple[Tuple[int, ...], ...], col_sums: Tuple[int, ...]) -> bool:
    column_vectors = [
        tuple(row[column] for row in matrix)
        for column in range(len(col_sums))
    ]
    for group in equal_value_groups(col_sums):
        block = [column_vectors[index] for index in group]
        if block != sorted(block):
            return False
    return True


@lru_cache(maxsize=None)
def symmetry_permutations(col_sums: Tuple[int, ...]) -> Tuple[Tuple[int, ...], ...]:
    groups = equal_value_groups(col_sums)
    parts = [list(itertools.permutations(group)) for group in groups]
    permutations: List[Tuple[int, ...]] = []
    for choice in itertools.product(*parts):
        perm = list(range(len(col_sums)))
        for group, reordered in zip(groups, choice):
            for old_index, new_index in zip(group, reordered):
                perm[old_index] = new_index
        permutations.append(tuple(perm))
    return tuple(permutations)


def relabel_edge_set(edges: Tuple[Tuple[int, int], ...], permutation: Tuple[int, ...]) -> Tuple[Tuple[int, int], ...]:
    relabeled = []
    for left, right in edges:
        a = permutation[left]
        b = permutation[right]
        relabeled.append((a, b) if a < b else (b, a))
    return tuple(sorted(relabeled))


def internal_graph_is_canonical(edges: Tuple[Tuple[int, int], ...], col_sums: Tuple[int, ...]) -> bool:
    return edges == min(relabel_edge_set(edges, permutation) for permutation in symmetry_permutations(col_sums))


@lru_cache(maxsize=None)
def all_allowed_masks(n_vertices: int) -> Tuple[int, ...]:
    return tuple(
        sum(1 << other for other in range(n_vertices) if other != vertex)
        for vertex in range(n_vertices)
    )


def active_neighbor_count(
    vertex: int,
    degree_sequence: Tuple[int, ...],
    allowed_masks: Tuple[int, ...],
) -> int:
    mask = allowed_masks[vertex]
    count = 0
    for other, degree in enumerate(degree_sequence):
        if other == vertex or degree <= 0:
            continue
        if mask & (1 << other):
            count += 1
    return count


def degree_sequence_feasible(
    degree_sequence: Tuple[int, ...],
    allowed_masks: Tuple[int, ...],
) -> bool:
    if any(degree < 0 for degree in degree_sequence):
        return False
    if sum(degree_sequence) % 2 != 0:
        return False
    for vertex, degree in enumerate(degree_sequence):
        if degree == 0:
            continue
        if active_neighbor_count(vertex, degree_sequence, allowed_masks) < degree:
            return False
    return True


@lru_cache(maxsize=None)
def internal_graphs_for_degree_sequence(
    degree_sequence: Tuple[int, ...],
    allowed_masks: Tuple[int, ...],
) -> Tuple[Tuple[Tuple[int, int], ...], ...]:
    if not degree_sequence_feasible(degree_sequence, allowed_masks):
        return ()
    if all(degree == 0 for degree in degree_sequence):
        return ((),)

    n_vertices = len(degree_sequence)
    vertex = next(index for index, degree in enumerate(degree_sequence) if degree > 0)
    candidates = [
        other
        for other in range(vertex + 1, n_vertices)
        if degree_sequence[other] > 0 and (allowed_masks[vertex] & (1 << other))
    ]
    if len(candidates) < degree_sequence[vertex]:
        return ()

    results: List[Tuple[Tuple[int, int], ...]] = []
    for combo in itertools.combinations(candidates, degree_sequence[vertex]):
        next_degrees = list(degree_sequence)
        next_degrees[vertex] = 0
        valid = True
        for other in combo:
            next_degrees[other] -= 1
            if next_degrees[other] < 0:
                valid = False
                break
        if not valid:
            continue
        next_degrees_tuple = tuple(next_degrees)
        if not degree_sequence_feasible(next_degrees_tuple, allowed_masks):
            continue
        tail_graphs = internal_graphs_for_degree_sequence(next_degrees_tuple, allowed_masks)
        edge_prefix = tuple((vertex, other) for other in combo)
        for tail in tail_graphs:
            results.append(tuple(sorted(edge_prefix + tail)))
    return tuple(results)


@lru_cache(maxsize=None)
def binary_matrices_with_margins(row_sums: Tuple[int, ...], col_sums: Tuple[int, ...]) -> Tuple[Tuple[Tuple[int, ...], ...], ...]:
    if not row_sums:
        return ((),) if all(value == 0 for value in col_sums) else ()

    row_sum = row_sums[0]
    candidates: List[Tuple[Tuple[int, ...], ...]] = []
    positive_cols = [index for index, value in enumerate(col_sums) if value > 0]
    for cols in itertools.combinations(positive_cols, row_sum):
        next_col_sums = list(col_sums)
        row = [0] * len(col_sums)
        for col in cols:
            next_col_sums[col] -= 1
            row[col] = 1
        next_col_sums_tuple = tuple(next_col_sums)
        remaining_sum = sum(row_sums[1:])
        if remaining_sum != sum(next_col_sums_tuple):
            continue
        if any(value < 0 for value in next_col_sums_tuple):
            continue
        if any(value > len(row_sums) - 1 for value in next_col_sums_tuple):
            continue
        tails = binary_matrices_with_margins(row_sums[1:], next_col_sums_tuple)
        for tail in tails:
            candidates.append((tuple(row),) + tail)
    return tuple(candidates)


@lru_cache(maxsize=None)
def internal_graphs_for_col_sums(col_sums: Tuple[int, ...]) -> Tuple[Tuple[Tuple[int, int], ...], ...]:
    n_vertices = len(col_sums)
    degree_sequence = tuple(3 - value for value in col_sums)
    if any(degree < 0 or degree > n_vertices - 1 for degree in degree_sequence):
        return ()
    return internal_graphs_for_degree_sequence(degree_sequence, all_allowed_masks(n_vertices))


def bipartition_coloring(adj: Dict[int, Sequence[int]]) -> Dict[int, int]:
    color: Dict[int, int] = {}
    for start in sorted(adj):
        if start in color:
            continue
        color[start] = 0
        queue = [start]
        for vertex in queue:
            for neighbor in adj[vertex]:
                if neighbor not in color:
                    color[neighbor] = 1 - color[vertex]
                    queue.append(neighbor)
                elif color[neighbor] == color[vertex]:
                    raise ValueError("graph is not bipartite")
    return color


def forced_gadget_colors(
    matrix: Tuple[Tuple[int, ...], ...],
    boundary_colors: Tuple[int, ...],
) -> Optional[Tuple[Optional[int], ...]]:
    n_columns = len(matrix[0]) if matrix else 0
    forced: List[Optional[int]] = []
    for column in range(n_columns):
        attached_colors = {
            boundary_colors[row]
            for row in range(len(matrix))
            if matrix[row][column] == 1
        }
        if len(attached_colors) > 1:
            return None
        if not attached_colors:
            forced.append(None)
        else:
            forced.append(1 - next(iter(attached_colors)))
    return tuple(forced)


@lru_cache(maxsize=None)
def allowed_masks_for_forced_colors(forced_colors: Tuple[Optional[int], ...]) -> Tuple[int, ...]:
    n_vertices = len(forced_colors)
    masks: List[int] = []
    for vertex in range(n_vertices):
        mask = 0
        for other in range(n_vertices):
            if other == vertex:
                continue
            if (
                forced_colors[vertex] is not None
                and forced_colors[other] is not None
                and forced_colors[vertex] == forced_colors[other]
            ):
                continue
            mask |= 1 << other
        masks.append(mask)
    return tuple(masks)


def graph_respects_forced_colors(
    n_vertices: int,
    edges: Tuple[Tuple[int, int], ...],
    forced_colors: Tuple[Optional[int], ...],
) -> bool:
    adj = {vertex: set() for vertex in range(n_vertices)}
    for left, right in edges:
        adj[left].add(right)
        adj[right].add(left)

    color: Dict[int, int] = {
        vertex: assigned
        for vertex, assigned in enumerate(forced_colors)
        if assigned is not None
    }
    for start in range(n_vertices):
        if start in color:
            queue = [start]
        else:
            color[start] = 0
            queue = [start]
        for vertex in queue:
            for neighbor in adj[vertex]:
                if neighbor not in color:
                    color[neighbor] = 1 - color[vertex]
                    queue.append(neighbor)
                elif color[neighbor] == color[vertex]:
                    return False
    return True


@lru_cache(maxsize=None)
def internal_graphs_for_col_sums_with_forced_colors(
    col_sums: Tuple[int, ...],
    forced_colors: Tuple[Optional[int], ...],
) -> Tuple[Tuple[Tuple[int, int], ...], ...]:
    n_vertices = len(col_sums)
    degree_sequence = tuple(3 - value for value in col_sums)
    if any(degree < 0 or degree > n_vertices - 1 for degree in degree_sequence):
        return ()
    allowed_masks = allowed_masks_for_forced_colors(forced_colors)
    graphs = internal_graphs_for_degree_sequence(degree_sequence, allowed_masks)
    return tuple(
        edges
        for edges in graphs
        if graph_respects_forced_colors(n_vertices, edges, forced_colors)
    )


def boundary_permutation_for_pattern(
    source_pattern: Tuple[int, ...],
    target_pattern: Tuple[int, ...],
) -> Optional[Tuple[int, ...]]:
    source = tuple(source_pattern)
    boundary_count = max(max(source, default=-1), max(target_pattern, default=-1)) + 1
    for reverse in (False, True):
        base = tuple(reversed(source)) if reverse else source
        for shift in range(len(base)):
            transformed = base[shift:] + base[:shift]
            mapping: Dict[int, int] = {}
            normalized: List[int] = []
            next_id = 0
            for value in transformed:
                if value not in mapping:
                    mapping[value] = next_id
                    next_id += 1
                normalized.append(mapping[value])
            if tuple(normalized) == tuple(target_pattern):
                return tuple(mapping[index] for index in range(boundary_count))
    return None


def transport_candidate(
    candidate: Dict[str, object],
    boundary_permutation: Tuple[int, ...],
) -> Dict[str, object]:
    matrix = [tuple(int(value) for value in row) for row in candidate["matrix"]]
    boundary_slot_orders = [
        tuple(int(value) for value in order)
        for order in candidate["boundary_slot_orders"]
    ]
    gadget_orders = [tuple(int(value) for value in order) for order in candidate["gadget_orders"]]

    transported_rows: List[Optional[Tuple[int, ...]]] = [None] * len(matrix)
    transported_boundary_slot_orders: List[Optional[Tuple[int, ...]]] = [None] * len(matrix)
    for old_index, new_index in enumerate(boundary_permutation):
        transported_rows[new_index] = matrix[old_index]
        transported_boundary_slot_orders[new_index] = boundary_slot_orders[old_index]

    transported_gadget_orders: List[Tuple[int, int, int]] = []
    n_gadget_vertices = len(gadget_orders)
    for column in range(n_gadget_vertices):
        old_boundary_rows = [row for row in range(len(matrix)) if matrix[row][column] == 1]
        mapped_rows = [boundary_permutation[row] for row in old_boundary_rows]
        new_boundary_rows = sorted(mapped_rows)
        local_boundary_remap = {
            old_token: new_boundary_rows.index(mapped_rows[old_token])
            for old_token in range(len(old_boundary_rows))
        }
        transported_order = []
        for token in gadget_orders[column]:
            if token < len(old_boundary_rows):
                transported_order.append(local_boundary_remap[token])
            else:
                transported_order.append(token)
        transported_gadget_orders.append(tuple(transported_order))

    if any(row is None for row in transported_rows) or any(order is None for order in transported_boundary_slot_orders):
        raise AssertionError("incomplete candidate transport")

    return {
        "gadget_vertices": candidate["gadget_vertices"],
        "matrix": [list(row) for row in transported_rows],
        "internal_edges": candidate["internal_edges"],
        "boundary_slot_orders": [list(order) for order in transported_boundary_slot_orders],
        "gadget_orders": [list(order) for order in transported_gadget_orders],
        "n_after": candidate["n_after"],
    }


def apply_candidate(
    G: EmbeddedGraph,
    patch: GeneralPatchDescriptor,
    matrix: Tuple[Tuple[int, ...], ...],
    internal_edges: Tuple[Tuple[int, int], ...],
    boundary_slot_orders: Tuple[Tuple[int, ...], ...],
    gadget_orders: Tuple[Tuple[int, int, int], ...],
) -> EmbeddedGraph:
    n_gadget_vertices = len(gadget_orders)
    H = G.copy()
    new_vertices = [H.next_id + i for i in range(n_gadget_vertices)]
    H.next_id += n_gadget_vertices
    for new_vertex in new_vertices:
        H.create_empty_vertex(new_vertex)

    for vertex in patch.deleted_vertices:
        del H.adj[vertex]
        del H.rot[vertex]
        del H.pos[vertex]

    boundary_to_new: Dict[int, List[int]] = {}
    new_to_boundary: Dict[int, List[int]] = {new_vertex: [] for new_vertex in new_vertices}
    for row, boundary_vertex in enumerate(patch.boundary_vertices):
        attached = [new_vertices[col] for col in range(n_gadget_vertices) if matrix[row][col] == 1]
        boundary_to_new[boundary_vertex] = attached
        for new_vertex in attached:
            new_to_boundary[new_vertex].append(boundary_vertex)

    internal_neighbors: Dict[int, List[int]] = {new_vertex: [] for new_vertex in new_vertices}
    for left_idx, right_idx in internal_edges:
        left = new_vertices[left_idx]
        right = new_vertices[right_idx]
        internal_neighbors[left].append(right)
        internal_neighbors[right].append(left)
    for new_vertex in new_vertices:
        internal_neighbors[new_vertex].sort()

    for boundary_vertex, slot_order in zip(patch.boundary_vertices, boundary_slot_orders):
        attachments = boundary_to_new[boundary_vertex]
        if tuple(sorted(slot_order)) != tuple(range(len(attachments))):
            raise ValueError(f"invalid slot order {slot_order} for boundary {boundary_vertex}")
        replacement = list(G.rot[boundary_vertex])
        removed_neighbors = set(patch.boundary_deleted_neighbors[boundary_vertex])
        removed_positions = [index for index, neighbor in enumerate(replacement) if neighbor in removed_neighbors]
        if len(removed_positions) != len(attachments):
            raise ValueError(
                f"boundary {boundary_vertex} has removed positions {removed_positions} but attachments {attachments}"
            )
        for position, attachment_index in zip(removed_positions, slot_order):
            replacement[position] = attachments[attachment_index]
        H.set_vertex_rotation(boundary_vertex, replacement)

    for idx, new_vertex in enumerate(new_vertices):
        boundary_neighbors = new_to_boundary[new_vertex]
        local_neighbors = boundary_neighbors + internal_neighbors[new_vertex]
        if len(local_neighbors) != 3:
            raise ValueError(
                f"gadget vertex {idx} has boundary/internal split "
                f"{len(boundary_neighbors)}/{len(internal_neighbors[new_vertex])}, not degree 3"
            )
        sequence = [local_neighbors[token] for token in gadget_orders[idx]]
        H.set_vertex_rotation(new_vertex, sequence)

    H.validate_rotation_embedding()
    if any(len(H.adj[v]) != 3 for v in H.adj):
        raise AssertionError("not cubic")
    if not bp.is_bipartite(H.adj):
        raise AssertionError("not bipartite")
    if not bp.is_3_connected(H.adj):
        raise AssertionError("not 3-connected")
    return H


def search_candidate_for_occurrence(
    G: EmbeddedGraph,
    patch: GeneralPatchDescriptor,
    gadget_vertices: int,
    forced_col_sums: Optional[Tuple[int, ...]] = None,
) -> Optional[Dict[str, object]]:
    if not patch_is_supported(patch):
        return None

    row_sums = tuple(patch.boundary_attachment_sizes)
    slot_templates = [
        list(itertools.permutations(range(size))) if size > 1 else [(0,)]
        for size in row_sums
    ]

    if forced_col_sums is not None:
        col_sum_sequence_list = [forced_col_sums]
    else:
        col_sum_sequence_list = prioritized_col_sum_sequences(sum(row_sums), gadget_vertices)

    for col_sums in col_sum_sequence_list:
        candidate = search_candidate_for_col_sums(
            G,
            patch,
            gadget_vertices,
            col_sums,
            slot_templates,
        )
        if candidate is not None:
            return candidate
    return None


def search_candidate_for_col_sums(
    G: EmbeddedGraph,
    patch: GeneralPatchDescriptor,
    gadget_vertices: int,
    col_sums: Tuple[int, ...],
    slot_templates: Optional[List[List[Tuple[int, ...]]]] = None,
) -> Optional[Dict[str, object]]:
    for candidate in iter_candidates_for_col_sums(
        G,
        patch,
        gadget_vertices,
        col_sums,
        slot_templates,
    ):
        return candidate
    return None


def iter_candidates_for_col_sums(
    G: EmbeddedGraph,
    patch: GeneralPatchDescriptor,
    gadget_vertices: int,
    col_sums: Tuple[int, ...],
    slot_templates: Optional[List[List[Tuple[int, ...]]]] = None,
) -> Iterator[Dict[str, object]]:
    row_sums = tuple(patch.boundary_attachment_sizes)
    boundary_colors_map = bipartition_coloring(G.adj)
    boundary_colors = tuple(boundary_colors_map[vertex] for vertex in patch.boundary_vertices)
    if slot_templates is None:
        slot_templates = [
            list(itertools.permutations(range(size))) if size > 1 else [(0,)]
            for size in row_sums
        ]

    matrices = [
        matrix
        for matrix in binary_matrices_with_margins(row_sums, col_sums)
        if matrix_is_canonical(matrix, col_sums)
        and forced_gadget_colors(matrix, boundary_colors) is not None
    ]
    if not matrices:
        return
    for matrix in matrices:
        forced_colors = forced_gadget_colors(matrix, boundary_colors)
        if forced_colors is None:
            continue
        internal_graphs = [
            edges
            for edges in internal_graphs_for_col_sums_with_forced_colors(col_sums, forced_colors)
            if internal_graph_is_canonical(edges, col_sums)
        ]
        if not internal_graphs:
            continue
        for internal_edges in internal_graphs:
            for boundary_slot_orders in itertools.product(*slot_templates):
                for gadget_orders in itertools.product(canonical_orientations(), repeat=gadget_vertices):
                    try:
                        H = apply_candidate(G, patch, matrix, internal_edges, boundary_slot_orders, gadget_orders)
                    except Exception:
                        continue
                    yield {
                        "gadget_vertices": gadget_vertices,
                        "matrix": matrix,
                        "internal_edges": internal_edges,
                        "boundary_slot_orders": boundary_slot_orders,
                        "gadget_orders": gadget_orders,
                        "n_after": len(H.adj),
                    }


def search_space_summary(
    patch: GeneralPatchDescriptor,
    gadget_vertices: int,
    forced_col_sums: Optional[Tuple[int, ...]] = None,
) -> Dict[str, object]:
    row_sums = tuple(patch.boundary_attachment_sizes)
    slot_multiplier = 1
    for size in row_sums:
        if size > 1:
            slot_multiplier *= len(list(itertools.permutations(range(size))))
    orientation_multiplier = 2 ** gadget_vertices

    if forced_col_sums is not None:
        sequences = [forced_col_sums]
    else:
        sequences = prioritized_col_sum_sequences(sum(row_sums), gadget_vertices)

    per_sequence = []
    total_raw_candidates = 0
    for col_sums in sequences:
        all_internal_graphs = internal_graphs_for_col_sums(col_sums)
        all_matrices = binary_matrices_with_margins(row_sums, col_sums)
        internal_graphs = [edges for edges in all_internal_graphs if internal_graph_is_canonical(edges, col_sums)]
        matrices = [matrix for matrix in all_matrices if matrix_is_canonical(matrix, col_sums)]
        raw_candidates = len(internal_graphs) * len(matrices) * slot_multiplier * orientation_multiplier
        total_raw_candidates += raw_candidates
        per_sequence.append(
            {
                "col_sums": list(col_sums),
                "internal_graph_count": len(internal_graphs),
                "internal_graph_count_raw": len(all_internal_graphs),
                "matrix_count": len(matrices),
                "matrix_count_raw": len(all_matrices),
                "slot_multiplier": slot_multiplier,
                "orientation_multiplier": orientation_multiplier,
                "raw_candidate_count": raw_candidates,
            }
        )

    return {
        "row_sums": list(row_sums),
        "gadget_vertices": gadget_vertices,
        "slot_multiplier": slot_multiplier,
        "orientation_multiplier": orientation_multiplier,
        "per_sequence": per_sequence,
        "total_raw_candidate_count": total_raw_candidates,
    }


def sweep_col_sum_sequences(
    G: EmbeddedGraph,
    patch: GeneralPatchDescriptor,
    gadget_vertices: int,
) -> Dict[str, object]:
    if not patch_is_supported(patch):
        return {
            "supported": False,
            "candidate_found": False,
            "tested_sequences": [],
        }

    row_sums = tuple(patch.boundary_attachment_sizes)
    slot_templates = [
        list(itertools.permutations(range(size))) if size > 1 else [(0,)]
        for size in row_sums
    ]
    summaries = search_space_summary(patch, gadget_vertices)
    per_sequence_by_key = {
        tuple(entry["col_sums"]): entry
        for entry in summaries["per_sequence"]
    }

    tested_sequences: List[Dict[str, object]] = []
    for col_sums in prioritized_col_sum_sequences(sum(row_sums), gadget_vertices):
        sequence_summary = per_sequence_by_key[col_sums]
        candidate = search_candidate_for_col_sums(
            G,
            patch,
            gadget_vertices,
            col_sums,
            slot_templates,
        )
        tested = {
            "col_sums": list(col_sums),
            "raw_candidate_count": sequence_summary["raw_candidate_count"],
            "internal_graph_count": sequence_summary["internal_graph_count"],
            "matrix_count": sequence_summary["matrix_count"],
            "candidate_found": candidate is not None,
        }
        if candidate is not None:
            tested["candidate"] = candidate
            tested_sequences.append(tested)
            return {
                "supported": True,
                "candidate_found": True,
                "search_space": summaries,
                "tested_sequences": tested_sequences,
                "candidate_col_sums": list(col_sums),
                "candidate": candidate,
            }
        tested_sequences.append(tested)

    return {
        "supported": True,
        "candidate_found": False,
        "search_space": summaries,
        "tested_sequences": tested_sequences,
    }


def replay_candidate(
    G: EmbeddedGraph,
    patch: GeneralPatchDescriptor,
    candidate: Dict[str, object],
) -> EmbeddedGraph:
    matrix = tuple(tuple(int(value) for value in row) for row in candidate["matrix"])
    internal_edges = tuple(tuple(int(value) for value in edge) for edge in candidate["internal_edges"])
    boundary_slot_orders = tuple(
        tuple(int(value) for value in order)
        for order in candidate["boundary_slot_orders"]
    )
    gadget_orders = tuple(tuple(int(value) for value in order) for order in candidate["gadget_orders"])
    return apply_candidate(G, patch, matrix, internal_edges, boundary_slot_orders, gadget_orders)


def verify_candidate(
    plantri_path: Path,
    candidate: Dict[str, object],
    seed_pattern: Tuple[int, ...],
    target_pattern: Tuple[int, ...],
    target_boundary_count: int,
    target_attachment_sizes: Tuple[int, ...],
    target_exterior_sizes: Tuple[int, ...],
    target_outer_face_key: Tuple[int, ...] | None,
    n_min: int,
    n_max: int,
    limit: int | None,
) -> Dict[str, object]:
    scanned = 0
    skipped_other_shape = 0
    skipped_other_pattern = 0
    matched_pattern_count = 0
    passed = 0
    failures: List[Dict[str, object]] = []

    for n_vertices in range(n_min, n_max + 1, 2):
        for embedding in iter_barnette_graph_rotations_via_plantri(str(plantri_path), n_vertices):
            G = graph_from_plantri_rotation(embedding.rot)
            for occ in all_refined_c4_occurrences(G):
                scanned += 1
                profile = occurrence_profile(G, occ)
                if (
                    profile.general_boundary_count != target_boundary_count
                    or profile.general_attachment_sizes != target_attachment_sizes
                    or profile.general_exterior_sizes != target_exterior_sizes
                    or (
                        target_outer_face_key is not None
                        and profile.outer_face_length_key != target_outer_face_key
                    )
                ):
                    skipped_other_shape += 1
                    if limit is not None and scanned >= limit:
                        return {
                            "scanned_occurrences": scanned,
                            "skipped_other_shape": skipped_other_shape,
                            "skipped_other_pattern": skipped_other_pattern,
                            "matched_pattern_count": matched_pattern_count,
                            "pass_count": passed,
                            "failure": None,
                        }
                    continue
                if profile.general_interface_word_key != target_pattern:
                    skipped_other_pattern += 1
                    if limit is not None and scanned >= limit:
                        return {
                            "scanned_occurrences": scanned,
                            "skipped_other_shape": skipped_other_shape,
                            "skipped_other_pattern": skipped_other_pattern,
                            "matched_pattern_count": matched_pattern_count,
                            "pass_count": passed,
                            "failure": None,
                        }
                    continue
                try:
                    patch = extract_general_patch(G, occ)
                    permutation = boundary_permutation_for_pattern(seed_pattern, patch.interface_word)
                    if permutation is None:
                        raise ValueError(
                            f"could not transport seed pattern {seed_pattern} to occurrence pattern {patch.interface_word}"
                        )
                    transported_candidate = transport_candidate(candidate, permutation)
                    matched_pattern_count += 1
                    replay_candidate(G, patch, transported_candidate)
                    passed += 1
                except Exception as exc:
                    failures.append(
                        {
                            "n_vertices": n_vertices,
                            "raw_line": embedding.raw_line,
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
                            "error": str(exc),
                        }
                    )
                    return {
                        "scanned_occurrences": scanned,
                        "skipped_other_shape": skipped_other_shape,
                        "skipped_other_pattern": skipped_other_pattern,
                        "matched_pattern_count": matched_pattern_count,
                        "pass_count": passed,
                        "failure": failures[0],
                    }
                if limit is not None and scanned >= limit:
                    return {
                        "scanned_occurrences": scanned,
                        "skipped_other_shape": skipped_other_shape,
                        "skipped_other_pattern": skipped_other_pattern,
                        "matched_pattern_count": matched_pattern_count,
                        "pass_count": passed,
                        "failure": None,
                    }
    return {
        "scanned_occurrences": scanned,
        "skipped_other_shape": skipped_other_shape,
        "skipped_other_pattern": skipped_other_pattern,
        "matched_pattern_count": matched_pattern_count,
        "pass_count": passed,
        "failure": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Search for a refined C4 replacement gadget on a generalized patch family.")
    parser.add_argument("--plantri", default=str(DEFAULT_PLANTRI))
    parser.add_argument("--seed-line", default=None)
    parser.add_argument("--n-min", type=int, default=20)
    parser.add_argument("--n-max", type=int, default=20)
    parser.add_argument("--verify-max", type=int, default=24)
    parser.add_argument("--gadget-vertices", type=int, default=4)
    parser.add_argument("--col-sums", default=None, help="Comma-separated nondecreasing column sums for the gadget search")
    parser.add_argument(
        "--verify-seed-outer-face-key",
        action="store_true",
        help="When verifying a found candidate, restrict to occurrences with the seed outer-face length key.",
    )
    parser.add_argument("--count-only", action="store_true")
    parser.add_argument("--sweep-all-col-sums", action="store_true", help="Test every canonical column-sum profile and record the outcome for each.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "refined_c4_gadget_search.json"))
    args = parser.parse_args()

    first_graph = None
    first_occ = None
    first_patch = None
    first_line = None
    plantri_path = Path(args.plantri)
    forced_col_sums: Optional[Tuple[int, ...]] = None

    if args.col_sums:
        values = tuple(sorted(int(part) for part in args.col_sums.split(",") if part))
        if len(values) != args.gadget_vertices:
            raise SystemExit("--col-sums length must match --gadget-vertices")
        forced_col_sums = values

    if args.seed_line:
        first_line = args.seed_line
        first_graph = graph_from_plantri_ascii(args.seed_line)
        for occ in all_refined_c4_occurrences(first_graph):
            try:
                patch = extract_general_patch(first_graph, occ)
            except ValueError:
                continue
            if not patch_is_supported(patch):
                continue
            first_occ = occ
            first_patch = patch
            break
        if first_patch is None:
            raise SystemExit("No supported refined C4 occurrence found in the supplied seed graph.")
    else:
        for n_vertices in range(args.n_min, args.n_max + 1, 2):
            found = False
            for embedding in iter_barnette_graph_rotations_via_plantri(str(plantri_path), n_vertices):
                G = graph_from_plantri_rotation(embedding.rot)
                for occ in all_refined_c4_occurrences(G):
                    try:
                        patch = extract_general_patch(G, occ)
                    except ValueError:
                        continue
                    if not patch_is_supported(patch):
                        continue
                    first_graph = G
                    first_occ = occ
                    first_patch = patch
                    first_line = embedding.raw_line
                    found = True
                    break
                if found:
                    break
            if found:
                break

    if first_graph is None or first_occ is None or first_patch is None or first_line is None:
        raise SystemExit("No supported refined C4 occurrence found in the requested range.")

    result: Dict[str, object] = {
        "seed_graph_line": first_line,
        "seed_occurrence": {
            "v1": first_occ.v1,
            "v2": first_occ.v2,
            "v3": first_occ.v3,
            "v4": first_occ.v4,
            "u1": first_occ.u1,
            "u2": first_occ.u2,
            "u3": first_occ.u3,
            "u4": first_occ.u4,
        },
        "patch": {
            "deleted_vertices": list(first_patch.deleted_vertices),
            "boundary_vertices": list(first_patch.boundary_vertices),
            "interface_word": list(first_patch.interface_word),
            "interface_word_key": list(first_patch.interface_word_key),
            "boundary_attachment_sizes": list(first_patch.boundary_attachment_sizes),
            "boundary_exterior_sizes": list(first_patch.boundary_exterior_sizes),
        },
    }

    if args.count_only and args.sweep_all_col_sums:
        raise SystemExit("--count-only and --sweep-all-col-sums are mutually exclusive")

    if args.count_only:
        result["search_space"] = search_space_summary(first_patch, args.gadget_vertices, forced_col_sums)
    elif args.sweep_all_col_sums:
        if forced_col_sums is not None:
            raise SystemExit("--sweep-all-col-sums cannot be combined with --col-sums")
        sweep = sweep_col_sum_sequences(first_graph, first_patch, args.gadget_vertices)
        result["sweep"] = sweep
        result["candidate"] = sweep.get("candidate")

        if result["candidate"] is not None:
            result["verification"] = verify_candidate(
                plantri_path,
                result["candidate"],
                first_patch.interface_word,
                first_patch.interface_word_key,
                len(first_patch.boundary_vertices),
                first_patch.boundary_attachment_sizes,
                first_patch.boundary_exterior_sizes,
                occurrence_profile(first_graph, first_occ).outer_face_length_key if args.verify_seed_outer_face_key else None,
                args.n_min,
                args.verify_max,
                args.limit,
            )
    else:
        candidate = search_candidate_for_occurrence(first_graph, first_patch, args.gadget_vertices, forced_col_sums)
        result["candidate"] = candidate

        if candidate is not None:
            result["verification"] = verify_candidate(
                plantri_path,
                candidate,
                first_patch.interface_word,
                first_patch.interface_word_key,
                len(first_patch.boundary_vertices),
                first_patch.boundary_attachment_sizes,
                first_patch.boundary_exterior_sizes,
                occurrence_profile(first_graph, first_occ).outer_face_length_key if args.verify_seed_outer_face_key else None,
                args.n_min,
                args.verify_max,
                args.limit,
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
