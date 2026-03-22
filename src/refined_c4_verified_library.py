from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

from refined_c4_gadget_search import (
    boundary_permutation_for_pattern,
    replay_candidate,
    transport_candidate,
)
from refined_c4_local import extract_general_patch, occurrence_profile


CandidateDict = Dict[str, Any]


@dataclass(frozen=True)
class CertifiedRefinedC4Family:
    name: str
    source: str
    outer_face_length_key: Tuple[int, ...]
    boundary_count: int
    interface_word_key: Tuple[int, ...]
    attachment_sizes: Tuple[int, ...]
    exterior_sizes: Tuple[int, ...]
    seed_pattern: Tuple[int, ...]
    candidate: CandidateDict


def _candidate(
    *,
    boundary_slot_orders: Iterable[Iterable[int]],
    gadget_orders: Iterable[Iterable[int]],
    gadget_vertices: int,
    internal_edges: Iterable[Iterable[int]],
    matrix: Iterable[Iterable[int]],
    n_after: int,
) -> CandidateDict:
    return {
        "boundary_slot_orders": [list(order) for order in boundary_slot_orders],
        "gadget_orders": [list(order) for order in gadget_orders],
        "gadget_vertices": gadget_vertices,
        "internal_edges": [list(edge) for edge in internal_edges],
        "matrix": [list(row) for row in matrix],
        "n_after": n_after,
    }


CERTIFIED_REFINED_C4_FAMILIES: Tuple[CertifiedRefinedC4Family, ...] = (
    CertifiedRefinedC4Family(
        name="L=6,6,6,6|B=6|I=0,0,1,1,2,3,4,5|A=1,2,2,1,1,1|E=2,1,1,2,2,2",
        source="artifacts/refined_c4_b6_family_00112345_found.json",
        outer_face_length_key=(6, 6, 6, 6),
        boundary_count=6,
        interface_word_key=(0, 0, 1, 1, 2, 3, 4, 5),
        attachment_sizes=(1, 2, 2, 1, 1, 1),
        exterior_sizes=(2, 1, 1, 2, 2, 2),
        seed_pattern=(0, 1, 1, 2, 2, 3, 4, 5),
        candidate=_candidate(
            boundary_slot_orders=((0,), (1, 0), (1, 0), (0,), (0,), (0,)),
            gadget_orders=((2, 1, 0), (0, 1, 2), (0, 1, 2), (0, 1, 2)),
            gadget_vertices=4,
            internal_edges=((0, 2), (1, 3)),
            matrix=(
                (0, 0, 0, 1),
                (0, 0, 1, 1),
                (1, 1, 0, 0),
                (0, 1, 0, 0),
                (1, 0, 0, 0),
                (0, 0, 1, 0),
            ),
            n_after=12,
        ),
    ),
    CertifiedRefinedC4Family(
        name="L=6,6,6,6|B=6|I=0,0,1,2,3,3,4,5|A=1,2,1,1,2,1|E=2,1,2,2,1,2",
        source="artifacts/refined_c4_b6_family_00123345.json",
        outer_face_length_key=(6, 6, 6, 6),
        boundary_count=6,
        interface_word_key=(0, 0, 1, 2, 3, 3, 4, 5),
        attachment_sizes=(1, 2, 1, 1, 2, 1),
        exterior_sizes=(2, 1, 2, 2, 1, 2),
        seed_pattern=(0, 1, 1, 2, 3, 4, 4, 5),
        candidate=_candidate(
            boundary_slot_orders=((0,), (1, 0), (0,), (0,), (1, 0), (0,)),
            gadget_orders=((2, 1, 0), (0, 1, 2), (2, 1, 0), (0, 1, 2)),
            gadget_vertices=4,
            internal_edges=((0, 1), (2, 3)),
            matrix=(
                (0, 0, 0, 1),
                (0, 1, 1, 0),
                (1, 0, 0, 0),
                (1, 0, 0, 0),
                (0, 1, 1, 0),
                (0, 0, 0, 1),
            ),
            n_after=12,
        ),
    ),
    CertifiedRefinedC4Family(
        name="L=6,6,6,6|B=6|I=0,0,1,2,3,4,5,1|A=1,1,1,1,2,2|E=2,2,2,2,1,1",
        source="artifacts/refined_c4_b6_family_0123451.json",
        outer_face_length_key=(6, 6, 6, 6),
        boundary_count=6,
        interface_word_key=(0, 0, 1, 2, 3, 4, 5, 1),
        attachment_sizes=(1, 1, 1, 1, 2, 2),
        exterior_sizes=(2, 2, 2, 2, 1, 1),
        seed_pattern=(0, 1, 2, 3, 4, 5, 5, 4),
        candidate=_candidate(
            boundary_slot_orders=((0,), (0,), (0,), (0,), (0, 1), (0, 1)),
            gadget_orders=((2, 1, 0), (0, 1, 2), (2, 1, 0), (0, 1, 2)),
            gadget_vertices=4,
            internal_edges=((0, 1), (2, 3)),
            matrix=(
                (0, 0, 0, 1),
                (0, 0, 1, 0),
                (0, 1, 0, 0),
                (1, 0, 0, 0),
                (0, 1, 1, 0),
                (1, 0, 0, 1),
            ),
            n_after=12,
        ),
    ),
    CertifiedRefinedC4Family(
        name="L=6,8,8,8|B=7|I=0,0,1,2,3,4,5,6,2,1|A=2,1,1,1,1,2,2|E=1,2,2,2,2,1,1",
        source="artifacts/refined_c4_b7_sum10_g8_outer6888.json",
        outer_face_length_key=(6, 8, 8, 8),
        boundary_count=7,
        interface_word_key=(0, 0, 1, 2, 3, 4, 5, 6, 2, 1),
        attachment_sizes=(2, 1, 1, 1, 1, 2, 2),
        exterior_sizes=(1, 2, 2, 2, 2, 1, 1),
        seed_pattern=(0, 1, 2, 3, 4, 0, 5, 6, 6, 5),
        candidate=_candidate(
            boundary_slot_orders=((1, 0), (0,), (0,), (0,), (0,), (0, 1), (1, 0)),
            gadget_orders=((2, 1, 0), (2, 1, 0), (0, 1, 2), (0, 1, 2), (0, 1, 2), (0, 1, 2), (0, 1, 2), (2, 1, 0)),
            gadget_vertices=8,
            internal_edges=((0, 1), (0, 2), (0, 5), (1, 3), (2, 6), (3, 4), (4, 7)),
            matrix=(
                (0, 0, 0, 0, 0, 0, 1, 1),
                (0, 0, 0, 0, 1, 0, 0, 0),
                (0, 0, 0, 1, 0, 0, 0, 0),
                (0, 0, 0, 0, 0, 1, 0, 0),
                (0, 0, 0, 0, 0, 1, 0, 0),
                (0, 0, 0, 0, 0, 0, 1, 1),
                (0, 1, 1, 0, 0, 0, 0, 0),
            ),
            n_after=16,
        ),
    ),
    CertifiedRefinedC4Family(
        name="L=6,8,6,10|B=8|I=0,0,1,2,3,4,4,3,5,6,7,1|A=1,1,1,2,2,1,2,2|E=2,2,2,1,1,2,1,1",
        source="artifacts/refined_c4_unresolved_family_search_top20.json",
        outer_face_length_key=(6, 8, 6, 10),
        boundary_count=8,
        interface_word_key=(0, 0, 1, 2, 3, 4, 4, 3, 5, 6, 7, 1),
        attachment_sizes=(1, 1, 1, 2, 2, 1, 2, 2),
        exterior_sizes=(2, 2, 2, 1, 1, 2, 1, 1),
        seed_pattern=(0, 1, 2, 3, 4, 4, 3, 5, 6, 7, 7, 6),
        candidate=_candidate(
            boundary_slot_orders=((0,), (0,), (0,), (1, 0), (1, 0), (0,), (0, 1), (0, 1)),
            gadget_orders=((0, 1, 2), (2, 1, 0), (0, 1, 2), (2, 1, 0), (2, 1, 0), (0, 1, 2), (2, 1, 0), (2, 1, 0)),
            gadget_vertices=8,
            internal_edges=((0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (6, 7)),
            matrix=(
                (0, 0, 0, 1, 0, 0, 0, 0),
                (0, 0, 1, 0, 0, 0, 0, 0),
                (0, 0, 0, 0, 0, 0, 0, 1),
                (0, 0, 0, 0, 0, 1, 1, 0),
                (0, 0, 0, 0, 1, 0, 0, 1),
                (0, 0, 0, 0, 0, 0, 1, 0),
                (0, 1, 0, 0, 0, 1, 0, 0),
                (1, 0, 0, 0, 1, 0, 0, 0),
            ),
            n_after=16,
        ),
    ),
    CertifiedRefinedC4Family(
        name="L=6,6,6,10|B=5|I=0,0,1,2,3,3,2,4|A=2,1,2,2,1|E=1,2,1,1,2",
        source="artifacts/refined_c4_unresolved_family_search_top20.json",
        outer_face_length_key=(6, 6, 6, 10),
        boundary_count=5,
        interface_word_key=(0, 0, 1, 2, 3, 3, 2, 4),
        attachment_sizes=(2, 1, 2, 2, 1),
        exterior_sizes=(1, 2, 1, 1, 2),
        seed_pattern=(0, 1, 2, 3, 3, 2, 4, 0),
        candidate=_candidate(
            boundary_slot_orders=((0, 1), (0,), (0, 1), (1, 0), (0,)),
            gadget_orders=((2, 1, 0), (0, 1, 2), (0, 1, 2), (2, 1, 0), (0, 1, 2), (0, 1, 2), (0, 1, 2), (2, 1, 0)),
            gadget_vertices=8,
            internal_edges=((0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 5), (4, 6), (5, 7)),
            matrix=(
                (0, 0, 0, 0, 0, 0, 1, 1),
                (0, 0, 0, 0, 0, 1, 0, 0),
                (0, 0, 0, 0, 0, 0, 1, 1),
                (0, 0, 0, 1, 1, 0, 0, 0),
                (0, 0, 1, 0, 0, 0, 0, 0),
            ),
            n_after=18,
        ),
    ),
    CertifiedRefinedC4Family(
        name="L=6,6,6,12|B=7|I=0,0,1,2,3,3,2,4,5,6|A=1,2,1,1,1,2,2|E=2,1,2,2,2,1,1",
        source="artifacts/refined_c4_unresolved_family_search_top20.json",
        outer_face_length_key=(6, 6, 6, 12),
        boundary_count=7,
        interface_word_key=(0, 0, 1, 2, 3, 3, 2, 4, 5, 6),
        attachment_sizes=(1, 2, 1, 1, 1, 2, 2),
        exterior_sizes=(2, 1, 2, 2, 2, 1, 1),
        seed_pattern=(0, 1, 1, 2, 3, 4, 5, 6, 6, 5),
        candidate=_candidate(
            boundary_slot_orders=((0,), (0, 1), (0,), (0,), (0,), (1, 0), (0, 1)),
            gadget_orders=((2, 1, 0), (0, 1, 2), (0, 1, 2), (2, 1, 0), (0, 1, 2), (2, 1, 0)),
            gadget_vertices=6,
            internal_edges=((0, 1), (0, 2), (1, 3), (4, 5)),
            matrix=(
                (0, 0, 0, 0, 0, 1),
                (0, 0, 0, 1, 1, 0),
                (0, 1, 0, 0, 0, 0),
                (1, 0, 0, 0, 0, 0),
                (0, 0, 1, 0, 0, 0),
                (0, 0, 0, 1, 1, 0),
                (0, 0, 1, 0, 0, 1),
            ),
            n_after=14,
        ),
    ),
    CertifiedRefinedC4Family(
        name="L=6,6,6,12|B=7|I=0,0,1,2,3,4,5,6,2,1|A=1,1,1,2,2,2,1|E=2,2,2,1,1,1,2",
        source="artifacts/refined_c4_unresolved_family_search_top20.json",
        outer_face_length_key=(6, 6, 6, 12),
        boundary_count=7,
        interface_word_key=(0, 0, 1, 2, 3, 4, 5, 6, 2, 1),
        attachment_sizes=(1, 1, 1, 2, 2, 2, 1),
        exterior_sizes=(2, 2, 2, 1, 1, 1, 2),
        seed_pattern=(0, 1, 2, 3, 4, 5, 5, 4, 3, 6),
        candidate=_candidate(
            boundary_slot_orders=((0,), (0,), (0,), (1, 0), (1, 0), (1, 0), (0,)),
            gadget_orders=((0, 1, 2), (0, 1, 2), (0, 1, 2), (2, 1, 0), (0, 1, 2), (2, 1, 0)),
            gadget_vertices=6,
            internal_edges=((0, 1), (0, 2), (1, 3), (4, 5)),
            matrix=(
                (0, 0, 0, 0, 0, 1),
                (0, 0, 0, 0, 1, 0),
                (0, 0, 0, 1, 0, 0),
                (0, 1, 1, 0, 0, 0),
                (1, 0, 0, 1, 0, 0),
                (0, 0, 1, 0, 1, 0),
                (0, 0, 0, 0, 0, 1),
            ),
            n_after=14,
        ),
    ),
    CertifiedRefinedC4Family(
        name="L=6,6,6,12|B=7|I=0,0,1,2,3,4,5,6,2,1|A=1,2,2,2,1,1,1|E=2,1,1,1,2,2,2",
        source="artifacts/refined_c4_unresolved_family_search_top20.json",
        outer_face_length_key=(6, 6, 6, 12),
        boundary_count=7,
        interface_word_key=(0, 0, 1, 2, 3, 4, 5, 6, 2, 1),
        attachment_sizes=(1, 2, 2, 2, 1, 1, 1),
        exterior_sizes=(2, 1, 1, 1, 2, 2, 2),
        seed_pattern=(0, 1, 2, 3, 3, 2, 1, 4, 5, 6),
        candidate=_candidate(
            boundary_slot_orders=((0,), (0, 1), (1, 0), (1, 0), (0,), (0,), (0,)),
            gadget_orders=((2, 1, 0), (0, 1, 2), (2, 1, 0), (2, 1, 0), (0, 1, 2), (2, 1, 0)),
            gadget_vertices=6,
            internal_edges=((0, 1), (0, 2), (1, 3), (4, 5)),
            matrix=(
                (0, 0, 0, 0, 0, 1),
                (0, 0, 0, 1, 1, 0),
                (0, 0, 1, 0, 0, 1),
                (0, 0, 0, 1, 1, 0),
                (0, 1, 0, 0, 0, 0),
                (1, 0, 0, 0, 0, 0),
                (0, 0, 1, 0, 0, 0),
            ),
            n_after=14,
        ),
    ),
    CertifiedRefinedC4Family(
        name="L=6,6,6,12|B=8|I=0,0,1,2,3,4,4,3,5,6,7,1|A=2,2,1,1,1,2,2,1|E=1,1,2,2,2,1,1,2",
        source="artifacts/refined_c4_unresolved_family_search_top20.json",
        outer_face_length_key=(6, 6, 6, 12),
        boundary_count=8,
        interface_word_key=(0, 0, 1, 2, 3, 4, 4, 3, 5, 6, 7, 1),
        attachment_sizes=(2, 2, 1, 1, 1, 2, 2, 1),
        exterior_sizes=(1, 1, 2, 2, 2, 1, 1, 2),
        seed_pattern=(0, 1, 2, 3, 4, 5, 6, 6, 5, 7, 1, 0),
        candidate=_candidate(
            boundary_slot_orders=((0, 1), (1, 0), (0,), (0,), (0,), (1, 0), (0, 1), (0,)),
            gadget_orders=((0, 1, 2), (0, 1, 2), (0, 1, 2), (0, 1, 2), (2, 1, 0), (2, 1, 0)),
            gadget_vertices=6,
            internal_edges=((0, 1), (0, 2), (3, 4)),
            matrix=(
                (0, 0, 0, 0, 1, 1),
                (0, 0, 1, 1, 0, 0),
                (1, 0, 0, 0, 0, 0),
                (0, 1, 0, 0, 0, 0),
                (0, 0, 0, 0, 0, 1),
                (0, 0, 0, 0, 1, 1),
                (0, 1, 0, 1, 0, 0),
                (0, 0, 1, 0, 0, 0),
            ),
            n_after=14,
        ),
    ),
)


def match_family_for_occurrence(
    G: Any,
    occ: Any,
) -> Optional[Tuple[CertifiedRefinedC4Family, CandidateDict]]:
    patch = extract_general_patch(G, occ)
    profile = occurrence_profile(G, occ)

    for family in CERTIFIED_REFINED_C4_FAMILIES:
        if profile.outer_face_length_key != family.outer_face_length_key:
            continue
        if profile.general_boundary_count != family.boundary_count:
            continue
        if profile.general_interface_word_key != family.interface_word_key:
            continue
        if profile.general_attachment_sizes != family.attachment_sizes:
            continue
        if profile.general_exterior_sizes != family.exterior_sizes:
            continue

        permutation = boundary_permutation_for_pattern(family.seed_pattern, patch.interface_word)
        if permutation is None:
            continue
        transported = transport_candidate(family.candidate, permutation)
        return family, transported

    return None


def reduce_with_certified_family(
    G: Any,
    occ: Any,
) -> Optional[Tuple[Any, Dict[str, Any]]]:
    patch = extract_general_patch(G, occ)
    matched = match_family_for_occurrence(G, occ)
    if matched is None:
        return None

    family, transported_candidate = matched
    H = replay_candidate(G, patch, transported_candidate)
    return H, {
        "kind": "refined_C4_family",
        "family_name": family.name,
        "source": family.source,
        "delta_n": len(H.adj) - len(G.adj),
        "gadget_vertices": int(transported_candidate["gadget_vertices"]),
    }
