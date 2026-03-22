from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from plantri_wrapper import iter_barnette_graph_rotations_via_plantri
from refined_c4_gadget_search import (
    boundary_permutation_for_pattern,
    prioritized_col_sum_sequences,
    iter_candidates_for_col_sums,
    replay_candidate,
    transport_candidate,
)
from refined_c4_local import (
    all_refined_c4_occurrences,
    extract_general_patch,
    graph_from_plantri_ascii,
    graph_from_plantri_rotation,
    occurrence_profile,
)
from refined_c4_unresolved_family_search import occ_matches


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLANTRI = ROOT / "plantri.exe"
DEFAULT_CENSUS = ROOT / "artifacts" / "unresolved_refined_c4_family_census_n30_after_refined_c4_library.json"
DEFAULT_OUT = ROOT / "artifacts" / "refined_c4_universal_family_search.json"


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


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_col_sums(text: Optional[str]) -> Optional[Tuple[int, ...]]:
    if text is None:
        return None
    return tuple(int(part) for part in text.split(",") if part.strip())


def normalize(value):
    if isinstance(value, dict):
        return {str(key): normalize(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    return value


def recover_seed_occurrence(example: Dict[str, object]):
    G = graph_from_plantri_ascii(str(example["raw_line"]))
    payload = example["occ"]
    for occ in all_refined_c4_occurrences(G):
        if occ_matches(occ, payload):
            return G, occ
    raise ValueError("could not recover seed occurrence")


def collect_matching_occurrences(
    family_name: str,
    plantri_path: Path,
    n_min: int,
    n_max: int,
) -> List[Dict[str, object]]:
    matches: List[Dict[str, object]] = []
    for n_vertices in range(n_min, n_max + 1, 2):
        for embedding in iter_barnette_graph_rotations_via_plantri(str(plantri_path), n_vertices):
            G = graph_from_plantri_rotation(embedding.rot)
            for occ in all_refined_c4_occurrences(G):
                profile = occurrence_profile(G, occ)
                if family_key(profile) != family_name:
                    continue
                patch = extract_general_patch(G, occ)
                matches.append(
                    {
                        "n_vertices": n_vertices,
                        "raw_line": embedding.raw_line,
                        "occ": occ,
                        "patch": patch,
                    }
                )
    return matches


def verify_candidate_on_occurrences(
    seed_pattern: Tuple[int, ...],
    candidate: Dict[str, object],
    occurrences: Sequence[Dict[str, object]],
) -> Tuple[bool, Optional[Dict[str, object]]]:
    for record in occurrences:
        patch = record["patch"]
        permutation = boundary_permutation_for_pattern(seed_pattern, patch.interface_word)
        if permutation is None:
            return False, {
                "error": "could not transport candidate to target pattern",
                "n_vertices": int(record["n_vertices"]),
                "raw_line": record["raw_line"],
            }
        try:
            transported = transport_candidate(candidate, permutation)
            replay_candidate(graph_from_plantri_ascii(record["raw_line"]), patch, transported)
        except Exception as exc:
            occ = record["occ"]
            return False, {
                "error": str(exc),
                "n_vertices": int(record["n_vertices"]),
                "raw_line": record["raw_line"],
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
            }
    return True, None


def search_universal_candidate(
    family_name: str,
    census_path: Path,
    plantri_path: Path,
    n_min: int,
    n_max: int,
    gadget_vertices: int,
    col_sums: Optional[Tuple[int, ...]],
    candidate_limit: Optional[int],
) -> Dict[str, object]:
    census = load_json(census_path)
    family_examples = census["family_examples"]
    if family_name not in family_examples:
        raise ValueError(f"family not found in census: {family_name}")

    seed_graph, seed_occ = recover_seed_occurrence(family_examples[family_name])
    seed_patch = extract_general_patch(seed_graph, seed_occ)
    occurrences = collect_matching_occurrences(family_name, plantri_path, n_min, n_max)
    slot_templates = [
        list(itertools.permutations(range(size))) if size > 1 else [(0,)]
        for size in seed_patch.boundary_attachment_sizes
    ]

    tested = 0
    first_failure = None
    universal_candidate = None

    col_sum_sequences = [col_sums] if col_sums is not None else list(
        prioritized_col_sum_sequences(sum(seed_patch.boundary_attachment_sizes), gadget_vertices)
    )

    for col_sum_sequence in col_sum_sequences:
        for candidate in iter_candidates_for_col_sums(
            seed_graph,
            seed_patch,
            gadget_vertices,
            col_sum_sequence,
            slot_templates,
        ):
            tested += 1
            ok, failure = verify_candidate_on_occurrences(seed_patch.interface_word, candidate, occurrences)
            if ok:
                universal_candidate = candidate
                col_sums = col_sum_sequence
                break
            if first_failure is None:
                first_failure = failure
            if candidate_limit is not None and tested >= candidate_limit:
                break
        if universal_candidate is not None:
            break
        if candidate_limit is not None and tested >= candidate_limit:
            break

    return {
        "family_name": family_name,
        "source_census": str(census_path.relative_to(ROOT)),
        "plantri_path": str(plantri_path),
        "n_min": n_min,
        "n_max": n_max,
        "gadget_vertices": gadget_vertices,
        "col_sums": list(col_sums) if col_sums is not None else None,
        "matching_occurrence_count": len(occurrences),
        "seed_pattern": list(seed_patch.interface_word),
        "candidates_tested": tested,
        "candidate_limit": candidate_limit,
        "universal_candidate_found": universal_candidate is not None,
        "universal_candidate": normalize(universal_candidate),
        "first_failure": normalize(first_failure),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search a coarse refined-C4 family for a gadget that works on all occurrences in range."
    )
    parser.add_argument("--family", required=True)
    parser.add_argument("--census", type=Path, default=DEFAULT_CENSUS)
    parser.add_argument("--plantri", type=Path, default=DEFAULT_PLANTRI)
    parser.add_argument("--n-min", type=int, default=8)
    parser.add_argument("--n-max", type=int, default=30)
    parser.add_argument("--gadget-vertices", type=int, default=8)
    parser.add_argument("--col-sums", default=None)
    parser.add_argument("--candidate-limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    payload = search_universal_candidate(
        args.family,
        args.census.resolve(),
        args.plantri.resolve(),
        args.n_min,
        args.n_max,
        args.gadget_vertices,
        parse_col_sums(args.col_sums),
        args.candidate_limit,
    )
    out_path = args.out.resolve()
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
