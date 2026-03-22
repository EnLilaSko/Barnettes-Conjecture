from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import barnette_proof as bp


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "pinch_ii_expansion_probe.json"


def reverse_cyclic(order: List[int]) -> List[int]:
    return [order[0], order[2], order[1]]


def insert_with_choice(rot: List[int], new_neighbor: int, choice: int) -> List[int]:
    if len(rot) != 2:
        raise ValueError("expected degree-2 boundary rotation after edge removal")
    if choice == 0:
        return [rot[0], new_neighbor, rot[1]]
    if choice == 1:
        return [rot[0], rot[1], new_neighbor]
    raise ValueError("invalid insertion choice")


def build_candidate(
    x: int,
    y: int,
    new_orders: Dict[str, List[int]],
    insert_choices: Dict[str, int],
) -> bp.EmbeddedGraph:
    base = bp.make_prism(8)
    r, s = bp._neighbors_around(base, x, y)
    i = base.pos[y][x]
    u2 = base.rot[y][(i + 1) % 3]
    u4 = base.rot[y][(i - 1) % 3]

    H = base.copy()
    H.remove_vertex(x)
    H.remove_vertex(y)

    v1 = H.next_id
    v2 = v1 + 1
    v3 = v1 + 2
    v4 = v1 + 3
    w = v1 + 4
    t = v1 + 5
    H.next_id += 6
    for vertex in (v1, v2, v3, v4, w, t):
        H.create_empty_vertex(vertex)

    name_to_vertex = {
        "v1": v1,
        "v2": v2,
        "v3": v3,
        "v4": v4,
        "w": w,
        "t": t,
        "r": r,
        "s": s,
        "u2": u2,
        "u4": u4,
    }

    for name in ("v1", "v2", "v3", "v4", "w", "t"):
        H.set_vertex_rotation(name_to_vertex[name], [name_to_vertex[token] for token in new_orders[name]])

    H.set_vertex_rotation(r, insert_with_choice(H.rot[r], t, insert_choices["r"]))
    H.adj[r].add(t)
    H.set_vertex_rotation(s, insert_with_choice(H.rot[s], t, insert_choices["s"]))
    H.adj[s].add(t)
    H.set_vertex_rotation(u2, insert_with_choice(H.rot[u2], v2, insert_choices["u2"]))
    H.adj[u2].add(v2)
    H.set_vertex_rotation(u4, insert_with_choice(H.rot[u4], v4, insert_choices["u4"]))
    H.adj[u4].add(v4)

    H.assert_consistent()
    return H


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe local rotation choices for the pinch(ii) expansion on prism(8)."
    )
    parser.add_argument("--x", type=int, default=0, help="Tail endpoint of the expanded edge.")
    parser.add_argument("--y", type=int, default=8, help="Head endpoint of the expanded edge.")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to write the probe JSON.",
    )
    args = parser.parse_args()

    canonical = {
        "v1": ["v4", "v2", "w"],
        "v2": ["v1", "v3", "u2"],
        "v3": ["v2", "v4", "w"],
        "v4": ["v3", "v1", "u4"],
        "w": ["v1", "t", "v3"],
        "t": ["r", "s", "w"],
    }
    insertion_keys = ("r", "s", "u2", "u4")
    new_vertex_keys = ("v1", "v2", "v3", "v4", "w", "t")

    results: List[Dict[str, object]] = []
    for reverse_mask in range(1 << len(new_vertex_keys)):
        new_orders: Dict[str, List[int]] = {}
        for index, key in enumerate(new_vertex_keys):
            order = list(canonical[key])
            if (reverse_mask >> index) & 1:
                order = reverse_cyclic(order)
            new_orders[key] = order

        for insert_mask in range(1 << len(insertion_keys)):
            insert_choices = {
                key: (insert_mask >> index) & 1
                for index, key in enumerate(insertion_keys)
            }
            result: Dict[str, object] = {
                "reverse_mask": reverse_mask,
                "insert_mask": insert_mask,
                "new_orders": new_orders,
                "insert_choices": insert_choices,
            }
            try:
                H = build_candidate(args.x, args.y, new_orders, insert_choices)
            except Exception as exc:
                result["status"] = "construction_error"
                result["error"] = str(exc)
                results.append(result)
                continue

            occ = bp.detect_C_pinch_ii(H)
            result["detected"] = occ is not None
            if occ is None:
                result["status"] = "no_occurrence"
                results.append(result)
                continue

            try:
                bp.validate_in_Q(H)
                result["expanded_in_Q"] = True
            except Exception as exc:
                result["expanded_in_Q"] = False
                result["expanded_error"] = str(exc)

            try:
                reduced, _ = bp.reduce_pinch(H, occ)
                bp.validate_in_Q(reduced)
                result["reduced_in_Q"] = True
            except Exception as exc:
                result["reduced_in_Q"] = False
                result["reduced_error"] = str(exc)

            result["status"] = "valid_example" if result.get("expanded_in_Q") and result.get("reduced_in_Q") else "reject"
            if result["status"] == "valid_example":
                result["occurrence"] = {
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
                }
                result["rotation"] = {str(v): H.rot[v] for v in sorted(H.adj)}
            results.append(result)

    payload = {
        "x": args.x,
        "y": args.y,
        "candidate_count": len(results),
        "valid_examples": [result for result in results if result["status"] == "valid_example"],
        "results": results,
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
