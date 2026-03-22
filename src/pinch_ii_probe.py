from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import barnette_proof as bp


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "pinch_ii_custom_probe.json"


def pairings(items: Sequence[int]) -> Iterable[Tuple[Tuple[int, int], Tuple[int, int]]]:
    if len(items) != 4:
        raise ValueError("expected exactly four boundary vertices")
    a, b, c, d = items
    yield ((a, b), (c, d))
    yield ((a, c), (b, d))
    yield ((a, d), (b, c))


def analyze_candidate(
    G: bp.EmbeddedGraph,
    occ: bp.OccPinch,
    left: Tuple[int, int],
    right: Tuple[int, int],
) -> Dict[str, object]:
    H = G.copy()
    x = H.next_id
    y = x + 1
    H.next_id += 2
    H.create_empty_vertex(x)
    H.create_empty_vertex(y)

    old_neighbor = {
        occ.r: occ.t,
        occ.s: occ.t,
        occ.u2: occ.v2,
        occ.u4: occ.v4,
    }

    for vertex in left:
        H.replace_neighbor(vertex, old_neighbor[vertex], x)
    for vertex in right:
        H.replace_neighbor(vertex, old_neighbor[vertex], y)

    for vertex in (occ.v1, occ.v2, occ.v3, occ.v4, occ.w, occ.t):
        del H.adj[vertex]
        del H.rot[vertex]
        del H.pos[vertex]

    H.set_vertex_rotation(x, [left[0], left[1], y])
    H.set_vertex_rotation(y, [right[0], right[1], x])

    result: Dict[str, object] = {
        "left": list(left),
        "right": list(right),
        "x": x,
        "y": y,
    }
    try:
        H.assert_consistent()
        result["consistent"] = True
    except Exception as exc:
        result["consistent"] = False
        result["error"] = f"consistency: {exc}"
        return result

    try:
        H.validate_rotation_embedding()
        result["embedding_ok"] = True
    except Exception as exc:
        result["embedding_ok"] = False
        result["embedding_error"] = str(exc)

    try:
        bp.validate_in_Q(H)
        result["in_Q"] = True
    except Exception as exc:
        result["in_Q"] = False
        result["q_error"] = str(exc)

    result["rotation_updates"] = {str(v): H.rot[v] for v in (occ.r, occ.s, occ.u2, occ.u4, x, y)}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe all two-vertex surgeries for the custom pinch(ii) witness."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to write the probe JSON.",
    )
    args = parser.parse_args()

    G = bp.make_custom_pinch_example()
    occ = bp.detect_C_pinch_ii(G)
    if occ is None:
        raise ValueError("custom pinch example does not contain a detected pinch(ii) occurrence")

    boundary = [occ.r, occ.s, occ.u2, occ.u4]
    candidates: List[Dict[str, object]] = []
    for pair_left, pair_right in pairings(boundary):
        for left in itertools.permutations(pair_left):
            for right in itertools.permutations(pair_right):
                candidates.append(analyze_candidate(G, occ, left, right))

    payload = {
        "boundary": boundary,
        "occurrence": {
            "v1": occ.v1,
            "v2": occ.v2,
            "v3": occ.v3,
            "v4": occ.v4,
            "w": occ.w,
            "t": occ.t,
            "r": occ.r,
            "s": occ.s,
            "u2": occ.u2,
            "u4": occ.u4,
            "epsilon": occ.epsilon,
        },
        "candidate_count": len(candidates),
        "embedding_pass_count": sum(1 for candidate in candidates if candidate.get("embedding_ok")),
        "q_pass_count": sum(1 for candidate in candidates if candidate.get("in_Q")),
        "candidates": candidates,
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
