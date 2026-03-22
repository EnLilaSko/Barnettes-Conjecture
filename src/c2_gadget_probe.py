from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Dict, List

import barnette_proof as bp
from completeness_frontier_scan import all_c2_occurrences
from plantri_wrapper import iter_barnette_graph_rotations_via_plantri
from refined_c4_gadget_search import (
    apply_candidate,
    prioritized_col_sum_sequences,
    search_candidate_for_col_sums,
)
from refined_c4_local import GeneralPatchDescriptor, graph_from_plantri_rotation


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLANTRI = ROOT / "plantri.exe"


def make_c2_patch(occ: bp.OccC2) -> GeneralPatchDescriptor:
    boundary_vertices = (occ.u1, occ.u6, occ.u5, occ.u4)
    return GeneralPatchDescriptor(
        deleted_vertices=(occ.a, occ.b, occ.c, occ.d, occ.e, occ.f),
        interface_vertices=boundary_vertices,
        boundary_vertices=boundary_vertices,
        boundary_deleted_neighbors={
            occ.u1: (occ.a,),
            occ.u6: (occ.f,),
            occ.u5: (occ.e,),
            occ.u4: (occ.d,),
        },
        boundary_outside_neighbors={
            occ.u1: tuple(sorted(set())),
            occ.u6: tuple(sorted(set())),
            occ.u5: tuple(sorted(set())),
            occ.u4: tuple(sorted(set())),
        },
        interface_word=(0, 1, 2, 3),
        interface_word_key=(0, 1, 2, 3),
        boundary_attachment_sizes=(1, 1, 1, 1),
        boundary_exterior_sizes=(0, 0, 0, 0),
        face_path_lengths=(1, 1, 1, 1),
    )


def find_distinct_terminal_c2_exceptions(
    plantri_path: Path,
    n_min: int,
    n_max: int,
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    seen = set()
    for n_vertices in range(n_min, n_max + 1, 2):
        for embedding in iter_barnette_graph_rotations_via_plantri(str(plantri_path), n_vertices):
            G = graph_from_plantri_rotation(embedding.rot)
            for occ in all_c2_occurrences(G):
                terminals = (occ.u1, occ.u4, occ.u5, occ.u6)
                if len(set(terminals)) != 4:
                    continue
                Gred, _, error = bp.try_reduce_certified(G, "C2", occ)
                if Gred is not None or error != "not 3-connected":
                    continue
                key = (embedding.raw_line, occ)
                if key in seen:
                    continue
                seen.add(key)
                out.append({"graph": G, "raw_line": embedding.raw_line, "occ": occ})
    return out


def serialize_candidate(candidate: Dict[str, object]) -> Dict[str, object]:
    return {
        "gadget_vertices": candidate["gadget_vertices"],
        "matrix": [list(row) for row in candidate["matrix"]],
        "internal_edges": [list(edge) for edge in candidate["internal_edges"]],
        "boundary_slot_orders": [list(order) for order in candidate["boundary_slot_orders"]],
        "gadget_orders": [list(order) for order in candidate["gadget_orders"]],
        "n_after": candidate["n_after"],
    }


def probe_gadget_size(
    plantri_path: Path,
    n_min: int,
    n_max: int,
    gadget_vertices: int,
) -> Dict[str, object]:
    exceptions = find_distinct_terminal_c2_exceptions(plantri_path, n_min, n_max)
    results: List[Dict[str, object]] = []
    for item in exceptions:
        G = item["graph"]
        occ = item["occ"]
        patch = make_c2_patch(occ)
        candidate = None
        for col_sums in prioritized_col_sum_sequences(sum(patch.boundary_attachment_sizes), gadget_vertices):
            candidate = search_candidate_for_col_sums(G, patch, gadget_vertices, col_sums)
            if candidate is not None:
                break
        result = {
            "n_vertices": len(G.adj),
            "raw_line": item["raw_line"],
            "occ": asdict(occ),
            "candidate_found": candidate is not None,
            "candidate": serialize_candidate(candidate) if candidate is not None else None,
        }
        results.append(result)

    return {
        "plantri_path": str(plantri_path),
        "n_min": n_min,
        "n_max": n_max,
        "gadget_vertices": gadget_vertices,
        "exception_count": len(exceptions),
        "candidate_count": sum(1 for result in results if result["candidate_found"]),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe larger C2 gadgets on distinct-terminal exceptions.")
    parser.add_argument("--plantri", default=str(DEFAULT_PLANTRI))
    parser.add_argument("--n-min", type=int, default=20)
    parser.add_argument("--n-max", type=int, default=24)
    parser.add_argument("--gadget-vertices", type=int, default=6)
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "c2_gadget_probe.json"))
    args = parser.parse_args()

    result = probe_gadget_size(
        Path(args.plantri),
        args.n_min,
        args.n_max,
        args.gadget_vertices,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
