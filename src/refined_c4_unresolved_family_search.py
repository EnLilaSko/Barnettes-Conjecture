from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from refined_c4_gadget_search import search_candidate_for_occurrence, verify_candidate
from refined_c4_local import (
    all_refined_c4_occurrences,
    extract_general_patch,
    graph_from_plantri_ascii,
    occurrence_profile,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CENSUS = ROOT / "artifacts" / "unresolved_refined_c4_family_census_n30.json"
DEFAULT_PLANTRI = ROOT / "plantri.exe"
DEFAULT_OUTPUT = ROOT / "artifacts" / "refined_c4_unresolved_family_search.json"


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_sizes(text: str) -> List[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def normalize(value):
    if isinstance(value, dict):
        return {str(key): normalize(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    return value


def occ_matches(occ, payload: Dict[str, int]) -> bool:
    return (
        occ.v1 == int(payload["v1"])
        and occ.v2 == int(payload["v2"])
        and occ.v3 == int(payload["v3"])
        and occ.v4 == int(payload["v4"])
        and occ.u1 == int(payload["u1"])
        and occ.u2 == int(payload["u2"])
        and occ.u3 == int(payload["u3"])
        and occ.u4 == int(payload["u4"])
    )


def choose_families(
    data: Dict[str, object],
    top: int | None,
    family_names: Sequence[str] | None,
) -> List[Tuple[str, int]]:
    counts = data["family_occurrence_counts"]
    items = sorted(
        ((str(name), int(count)) for name, count in counts.items()),
        key=lambda item: (-item[1], item[0]),
    )
    if family_names:
        wanted = set(family_names)
        selected = [item for item in items if item[0] in wanted]
        missing = sorted(wanted - {name for name, _ in selected})
        if missing:
            raise ValueError(f"family name(s) not found in census: {missing}")
        return selected
    if top is None:
        return items
    return items[:top]


def search_family(
    family_name: str,
    family_count: int,
    example: Dict[str, object],
    plantri_path: Path,
    n_min: int,
    verify_max: int,
    gadget_sizes: Sequence[int],
) -> Dict[str, object]:
    raw_line = str(example["raw_line"])
    occ_payload = example["occ"]
    G = graph_from_plantri_ascii(raw_line)

    chosen_occ = None
    for occ in all_refined_c4_occurrences(G):
        if occ_matches(occ, occ_payload):
            chosen_occ = occ
            break
    if chosen_occ is None:
        raise ValueError(f"could not recover seed occurrence for family {family_name}")

    patch = extract_general_patch(G, chosen_occ)
    profile = occurrence_profile(G, chosen_occ)
    size_results: List[Dict[str, object]] = []

    for gadget_vertices in gadget_sizes:
        candidate = search_candidate_for_occurrence(G, patch, gadget_vertices)
        verification = None
        if candidate is not None:
            verification = verify_candidate(
                plantri_path,
                candidate,
                patch.interface_word,
                patch.interface_word_key,
                len(patch.boundary_vertices),
                patch.boundary_attachment_sizes,
                patch.boundary_exterior_sizes,
                profile.outer_face_length_key,
                n_min,
                verify_max,
                None,
            )
        size_results.append(
            {
                "gadget_vertices": gadget_vertices,
                "candidate_found": candidate is not None,
                "candidate": normalize(candidate),
                "verification": normalize(verification),
            }
        )

    return {
        "family_name": family_name,
        "occurrence_count": family_count,
        "seed_example": normalize(example),
        "size_results": size_results,
    }


def summarize(results: Sequence[Dict[str, object]]) -> Dict[str, object]:
    found = 0
    verified = 0
    per_size_found: Dict[str, int] = {}
    per_size_verified: Dict[str, int] = {}
    for result in results:
        for entry in result["size_results"]:
            size_key = str(entry["gadget_vertices"])
            if entry["candidate_found"]:
                found += 1
                per_size_found[size_key] = per_size_found.get(size_key, 0) + 1
            verification = entry.get("verification")
            if verification and verification.get("failure") is None and verification.get("matched_pattern_count", 0) > 0:
                verified += 1
                per_size_verified[size_key] = per_size_verified.get(size_key, 0) + 1
    return {
        "family_count": len(results),
        "candidate_result_count": found,
        "verified_result_count": verified,
        "per_size_found": per_size_found,
        "per_size_verified": per_size_verified,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search unresolved refined-C4 frontier families for 6/8-vertex gadgets."
    )
    parser.add_argument("--census", type=Path, default=DEFAULT_CENSUS)
    parser.add_argument("--plantri", type=Path, default=DEFAULT_PLANTRI)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--family", action="append", help="Specific family name to search.")
    parser.add_argument("--sizes", default="6,8", help="Comma-separated gadget sizes to try.")
    parser.add_argument("--n-min", type=int, default=8)
    parser.add_argument("--verify-max", type=int, default=30)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    data = load_json(args.census.resolve())
    selected = choose_families(data, args.top, args.family)
    family_examples = data["family_examples"]
    gadget_sizes = parse_sizes(args.sizes)

    results = [
        search_family(
            family_name,
            family_count,
            family_examples[family_name],
            args.plantri.resolve(),
            args.n_min,
            args.verify_max,
            gadget_sizes,
        )
        for family_name, family_count in selected
    ]

    payload = {
        "source_census": str(args.census.resolve().relative_to(ROOT)),
        "selected_families": [family_name for family_name, _ in selected],
        "gadget_sizes": gadget_sizes,
        "results": results,
        "summary": summarize(results),
    }
    out_path = args.out.resolve()
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
