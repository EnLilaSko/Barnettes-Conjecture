from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from barnette_proof import detect_refined_C4
from refined_c4_local import graph_from_plantri_ascii, occurrence_profile, outer_faces_for_occurrence


def describe_occurrence(G, occ) -> Dict[str, object]:
    profile = occurrence_profile(G, occ)
    quad = [occ.v1, occ.v2, occ.v3, occ.v4]
    outer = {label: list(face) for label, face in outer_faces_for_occurrence(G, occ).items()}

    neighborhood_vertices = sorted(set(quad + [occ.u1, occ.u2, occ.u3, occ.u4] + [v for face in outer.values() for v in face]))
    neighborhood_edges = []
    seen = set()
    for v in neighborhood_vertices:
        for u in G.adj[v]:
            if u in neighborhood_vertices:
                edge = tuple(sorted((u, v)))
                if edge not in seen:
                    seen.add(edge)
                    neighborhood_edges.append(list(edge))

    return {
        "occurrence": {
            "v1": occ.v1,
            "v2": occ.v2,
            "v3": occ.v3,
            "v4": occ.v4,
            "u1": occ.u1,
            "u2": occ.u2,
            "u3": occ.u3,
            "u4": occ.u4,
        },
        "quad": quad,
        "outer_faces": outer,
        "outer_face_lengths": list(profile.outer_face_lengths),
        "outer_face_length_key": list(profile.outer_face_length_key),
        "general_boundary_count": profile.general_boundary_count,
        "general_interface_word": list(profile.general_interface_word) if profile.general_interface_word is not None else None,
        "general_interface_word_key": list(profile.general_interface_word_key) if profile.general_interface_word_key is not None else None,
        "general_attachment_sizes": list(profile.general_attachment_sizes) if profile.general_attachment_sizes is not None else None,
        "general_exterior_sizes": list(profile.general_exterior_sizes) if profile.general_exterior_sizes is not None else None,
        "general_deleted_size": profile.general_deleted_size,
        "general_patch_error": profile.general_patch_error,
        "hex_ring_pattern": list(profile.hex_ring_pattern) if profile.hex_ring_pattern is not None else None,
        "hex_ring_pattern_key": list(profile.hex_ring_pattern_key) if profile.hex_ring_pattern_key is not None else None,
        "hex_patch_error": profile.hex_patch_error,
        "neighborhood_vertices": neighborhood_vertices,
        "neighborhood_edges": sorted(neighborhood_edges),
        "rotation": {str(v): list(G.rot[v]) for v in neighborhood_vertices},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump one refined C4 neighborhood from a plantri line.")
    parser.add_argument("--line", required=True, help="Raw plantri -a line")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    G = graph_from_plantri_ascii(args.line)
    occ = detect_refined_C4(G)
    if occ is None:
        raise SystemExit("No refined C4 occurrence found in the supplied graph.")

    payload = describe_occurrence(G, occ)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
