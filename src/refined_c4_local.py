from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from barnette_proof import EmbeddedGraph, OccC4
from plantri_wrapper import parse_plantri_ascii_embedding


@dataclass(frozen=True)
class HexPatchDescriptor:
    deleted_vertices: Tuple[int, ...]
    boundary_vertices: Tuple[int, int, int, int]
    outside_neighbors: Dict[int, int]
    deleted_neighbors: Dict[int, Tuple[int, int]]
    ring_pattern: Tuple[int, ...]
    ring_pattern_key: Tuple[int, ...]


@dataclass(frozen=True)
class GeneralPatchDescriptor:
    deleted_vertices: Tuple[int, ...]
    interface_vertices: Tuple[int, ...]
    boundary_vertices: Tuple[int, ...]
    boundary_deleted_neighbors: Dict[int, Tuple[int, ...]]
    boundary_outside_neighbors: Dict[int, Tuple[int, ...]]
    interface_word: Tuple[int, ...]
    interface_word_key: Tuple[int, ...]
    boundary_attachment_sizes: Tuple[int, ...]
    boundary_exterior_sizes: Tuple[int, ...]
    face_path_lengths: Tuple[int, int, int, int]


@dataclass(frozen=True)
class RefinedC4Profile:
    outer_face_lengths: Tuple[int, int, int, int]
    outer_face_length_key: Tuple[int, int, int, int]
    general_boundary_count: Optional[int]
    general_interface_word: Optional[Tuple[int, ...]]
    general_interface_word_key: Optional[Tuple[int, ...]]
    general_attachment_sizes: Optional[Tuple[int, ...]]
    general_exterior_sizes: Optional[Tuple[int, ...]]
    general_deleted_size: Optional[int]
    general_patch_error: Optional[str]
    hex_ring_pattern: Optional[Tuple[int, ...]]
    hex_ring_pattern_key: Optional[Tuple[int, ...]]
    hex_patch_error: Optional[str]


def graph_from_plantri_rotation(rot: Dict[int, List[int]]) -> EmbeddedGraph:
    adj = {vertex: set(neighbors) for vertex, neighbors in rot.items()}
    return EmbeddedGraph(adj=adj, rot={vertex: list(neighbors) for vertex, neighbors in rot.items()})


def graph_from_plantri_ascii(line: str) -> EmbeddedGraph:
    parsed = parse_plantri_ascii_embedding(line)
    return graph_from_plantri_rotation(parsed.rot)


def canonicalize_occ(occ: OccC4) -> OccC4:
    vertices = [occ.v1, occ.v2, occ.v3, occ.v4]
    terminals = [occ.u1, occ.u2, occ.u3, occ.u4]
    candidates = []
    for start in range(4):
        forward = [(start + offset) % 4 for offset in range(4)]
        backward = [(start - offset) % 4 for offset in range(4)]
        for order in (forward, backward):
            candidates.append(
                OccC4(
                    vertices[order[0]],
                    vertices[order[1]],
                    vertices[order[2]],
                    vertices[order[3]],
                    terminals[order[0]],
                    terminals[order[1]],
                    terminals[order[2]],
                    terminals[order[3]],
                )
            )
    return min(candidates)


def all_refined_c4_occurrences(G: EmbeddedGraph) -> List[OccC4]:
    seen = set()
    occurrences: List[OccC4] = []
    for v1 in G.vertices():
        for v2 in sorted(G.adj[v1]):
            darts, end = G.trace_face_darts((v1, v2), steps=4)
            if end != (v1, v2):
                continue
            v2 = darts[0][1]
            v3 = darts[1][1]
            v4 = darts[2][1]
            if len({v1, v2, v3, v4}) != 4:
                continue

            u1 = G.third_neighbor(v1, {v2, v4})
            u2 = G.third_neighbor(v2, {v1, v3})
            u3 = G.third_neighbor(v3, {v2, v4})
            u4 = G.third_neighbor(v4, {v1, v3})
            if len({u1, u2, u3, u4}) != 4:
                continue

            quad_edges = [(v1, v2), (v2, v3), (v3, v4), (v4, v1)]
            if any(G.other_face_is_quad(a, b) for a, b in quad_edges):
                continue

            occ = canonicalize_occ(OccC4(v1, v2, v3, v4, u1, u2, u3, u4))
            if occ not in seen:
                seen.add(occ)
                occurrences.append(occ)
    return sorted(occurrences)


def face_vertices_with_start(G: EmbeddedGraph, start: Tuple[int, int]) -> List[int]:
    orbit, end = G.trace_face_darts(start, steps=None)
    if end != start:
        raise ValueError("face orbit did not close")
    return [tail for tail, _ in orbit]


def non_quad_face_for_edge(G: EmbeddedGraph, a: int, b: int) -> List[int]:
    faces = []
    for start in ((a, b), (b, a)):
        face = face_vertices_with_start(G, start)
        if len(face) != 4:
            faces.append(face)
    if len(faces) != 1:
        raise ValueError(f"expected exactly one non-quad face on edge {a}-{b}, got {faces}")
    return faces[0]


def orient_outer_face(face: Sequence[int], va: int, vb: int, u_left: int, u_right: int) -> Tuple[int, ...]:
    face_list = list(face)
    candidates = [face_list, list(reversed(face_list))]
    for candidate in candidates:
        n = len(candidate)
        for i in range(n):
            rotated = candidate[i:] + candidate[:i]
            if n >= 4 and rotated[0] == vb and rotated[1] == va and rotated[2] == u_left and rotated[-1] == u_right:
                return tuple(rotated)
    raise ValueError(
        f"could not orient outer face {face_list} for edge ({va},{vb}) with terminals ({u_left},{u_right})"
    )


def canonicalize_cycle(values: Sequence[int]) -> Tuple[int, ...]:
    seq = tuple(values)
    n = len(seq)
    candidates = []
    for start in range(n):
        candidates.append(seq[start:] + seq[:start])
    rev = tuple(reversed(seq))
    for start in range(n):
        candidates.append(rev[start:] + rev[:start])
    return min(candidates)


def normalize_first_seen_ids(values: Sequence[int]) -> Tuple[int, ...]:
    mapping: Dict[int, int] = {}
    normalized: List[int] = []
    next_id = 0
    for value in values:
        if value not in mapping:
            mapping[value] = next_id
            next_id += 1
        normalized.append(mapping[value])
    return tuple(normalized)


def canonicalize_pattern(values: Sequence[int]) -> Tuple[int, ...]:
    seq = tuple(values)
    n = len(seq)
    candidates = []
    for start in range(n):
        candidates.append(normalize_first_seen_ids(seq[start:] + seq[:start]))
    rev = tuple(reversed(seq))
    for start in range(n):
        candidates.append(normalize_first_seen_ids(rev[start:] + rev[:start]))
    return min(candidates)


def outer_faces_for_occurrence(G: EmbeddedGraph, occ: OccC4) -> Dict[str, Tuple[int, ...]]:
    return {
        "f12": orient_outer_face(non_quad_face_for_edge(G, occ.v1, occ.v2), occ.v1, occ.v2, occ.u1, occ.u2),
        "f23": orient_outer_face(non_quad_face_for_edge(G, occ.v2, occ.v3), occ.v2, occ.v3, occ.u2, occ.u3),
        "f34": orient_outer_face(non_quad_face_for_edge(G, occ.v3, occ.v4), occ.v3, occ.v4, occ.u3, occ.u4),
        "f41": orient_outer_face(non_quad_face_for_edge(G, occ.v4, occ.v1), occ.v4, occ.v1, occ.u4, occ.u1),
    }


def extract_general_patch(
    G: EmbeddedGraph,
    occ: OccC4,
    outer_faces: Optional[Dict[str, Tuple[int, ...]]] = None,
) -> GeneralPatchDescriptor:
    outer = outer_faces or outer_faces_for_occurrence(G, occ)
    face_paths = []
    interface_vertices: List[int] = []
    for label in ("f12", "f23", "f34", "f41"):
        face = list(outer[label])
        path = face[3:-1]
        if not path:
            raise ValueError(f"outer face has no interior path vertices: {face}")
        face_paths.append(tuple(path))
        interface_vertices.extend(path)

    deleted = {
        occ.v1, occ.v2, occ.v3, occ.v4,
        occ.u1, occ.u2, occ.u3, occ.u4,
        *interface_vertices,
    }

    boundary_sequence_vertices: List[int] = []
    path_to_boundary: Dict[int, int] = {}
    for interface_vertex in interface_vertices:
        outside = sorted(G.adj[interface_vertex] - deleted)
        if len(outside) == 0:
            continue
        if len(outside) != 1:
            raise ValueError(
                f"expected one outside neighbor for interface vertex {interface_vertex}, got {outside}"
            )
        boundary_sequence_vertices.append(interface_vertex)
        path_to_boundary[interface_vertex] = outside[0]

    if not boundary_sequence_vertices:
        raise ValueError("generalized patch has empty interface boundary")

    groups: Dict[int, List[int]] = {}
    first_seen: Dict[int, int] = {}
    for index, interface_vertex in enumerate(boundary_sequence_vertices):
        boundary_vertex = path_to_boundary[interface_vertex]
        groups.setdefault(boundary_vertex, []).append(interface_vertex)
        first_seen.setdefault(boundary_vertex, index)

    boundary_vertices = tuple(sorted(groups, key=lambda vertex: first_seen[vertex]))
    boundary_deleted_neighbors = {
        vertex: tuple(groups[vertex]) for vertex in boundary_vertices
    }

    boundary_outside_neighbors: Dict[int, Tuple[int, ...]] = {}
    for boundary_vertex, removed_neighbors in boundary_deleted_neighbors.items():
        removed_set = set(removed_neighbors)
        outside = tuple(sorted(G.adj[boundary_vertex] - removed_set))
        if len(removed_set) + len(outside) != 3:
            raise ValueError(
                f"unexpected boundary degree split at {boundary_vertex}: "
                f"removed={sorted(removed_set)} outside={list(outside)}"
            )
        boundary_outside_neighbors[boundary_vertex] = outside

    pattern_ids: Dict[int, int] = {}
    interface_word: List[int] = []
    next_id = 0
    for interface_vertex in boundary_sequence_vertices:
        boundary_vertex = path_to_boundary[interface_vertex]
        if boundary_vertex not in pattern_ids:
            pattern_ids[boundary_vertex] = next_id
            next_id += 1
        interface_word.append(pattern_ids[boundary_vertex])

    boundary_attachment_sizes = tuple(
        len(boundary_deleted_neighbors[vertex]) for vertex in boundary_vertices
    )
    boundary_exterior_sizes = tuple(
        len(boundary_outside_neighbors[vertex]) for vertex in boundary_vertices
    )

    return GeneralPatchDescriptor(
        deleted_vertices=tuple(sorted(deleted)),
        interface_vertices=tuple(boundary_sequence_vertices),
        boundary_vertices=boundary_vertices,
        boundary_deleted_neighbors=boundary_deleted_neighbors,
        boundary_outside_neighbors=boundary_outside_neighbors,
        interface_word=tuple(interface_word),
        interface_word_key=canonicalize_pattern(interface_word),
        boundary_attachment_sizes=boundary_attachment_sizes,
        boundary_exterior_sizes=boundary_exterior_sizes,
        face_path_lengths=tuple(len(path) for path in face_paths),
    )


def extract_hex_ring_patch(
    G: EmbeddedGraph,
    occ: OccC4,
    outer_faces: Optional[Dict[str, Tuple[int, ...]]] = None,
) -> HexPatchDescriptor:
    outer = outer_faces or outer_faces_for_occurrence(G, occ)
    for label in ("f12", "f23", "f34", "f41"):
        face = list(outer[label])
        if len(face) != 6:
            raise ValueError(f"adjacent outer face is not a hexagon: {face}")

    return extract_four_boundary_patch(G, occ, outer)


def extract_four_boundary_patch(
    G: EmbeddedGraph,
    occ: OccC4,
    outer_faces: Optional[Dict[str, Tuple[int, ...]]] = None,
) -> HexPatchDescriptor:
    general = extract_general_patch(G, occ, outer_faces)
    groups_debug = {
        vertex: list(general.boundary_deleted_neighbors[vertex])
        for vertex in general.boundary_vertices
    }
    if len(general.boundary_vertices) != 4:
        raise ValueError(f"expected four boundary vertices, got {groups_debug}")
    if any(size != 2 for size in general.boundary_attachment_sizes):
        raise ValueError(f"expected each boundary vertex to cover two ring vertices, got {groups_debug}")
    if any(size != 1 for size in general.boundary_exterior_sizes):
        raise ValueError(
            "expected each boundary vertex to have one outside neighbor, got "
            f"{ {vertex: list(general.boundary_outside_neighbors[vertex]) for vertex in general.boundary_vertices} }"
        )

    return HexPatchDescriptor(
        deleted_vertices=general.deleted_vertices,
        boundary_vertices=tuple(general.boundary_vertices),
        outside_neighbors={
            vertex: general.boundary_outside_neighbors[vertex][0]
            for vertex in general.boundary_vertices
        },
        deleted_neighbors={
            vertex: general.boundary_deleted_neighbors[vertex]
            for vertex in general.boundary_vertices
        },
        ring_pattern=general.interface_word,
        ring_pattern_key=general.interface_word_key,
    )


def occurrence_profile(G: EmbeddedGraph, occ: OccC4) -> RefinedC4Profile:
    outer = outer_faces_for_occurrence(G, occ)
    outer_face_lengths = tuple(len(outer[label]) for label in ("f12", "f23", "f34", "f41"))
    outer_face_length_key = canonicalize_cycle(outer_face_lengths)

    try:
        general = extract_general_patch(G, occ, outer)
        general_boundary_count = len(general.boundary_vertices)
        general_interface_word = general.interface_word
        general_interface_word_key = general.interface_word_key
        general_attachment_sizes = general.boundary_attachment_sizes
        general_exterior_sizes = general.boundary_exterior_sizes
        general_deleted_size = len(general.deleted_vertices)
        general_patch_error = None
    except ValueError as exc:
        general = None
        general_boundary_count = None
        general_interface_word = None
        general_interface_word_key = None
        general_attachment_sizes = None
        general_exterior_sizes = None
        general_deleted_size = None
        general_patch_error = str(exc)

    try:
        patch = extract_hex_ring_patch(G, occ, outer)
        hex_ring_pattern = patch.ring_pattern
        hex_ring_pattern_key = patch.ring_pattern_key
        hex_patch_error = None
    except ValueError as exc:
        hex_ring_pattern = None
        hex_ring_pattern_key = None
        hex_patch_error = str(exc)

    return RefinedC4Profile(
        outer_face_lengths=outer_face_lengths,
        outer_face_length_key=outer_face_length_key,
        general_boundary_count=general_boundary_count,
        general_interface_word=general_interface_word,
        general_interface_word_key=general_interface_word_key,
        general_attachment_sizes=general_attachment_sizes,
        general_exterior_sizes=general_exterior_sizes,
        general_deleted_size=general_deleted_size,
        general_patch_error=general_patch_error,
        hex_ring_pattern=hex_ring_pattern,
        hex_ring_pattern_key=hex_ring_pattern_key,
        hex_patch_error=hex_patch_error,
    )
