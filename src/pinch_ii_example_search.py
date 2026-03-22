from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import barnette_proof as bp


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "pinch_ii_example_search.json"


def analyze_directed_edge(x: int, y: int) -> Dict[str, object]:
    base = bp.make_prism(8)
    result: Dict[str, object] = {"x": x, "y": y}
    try:
        H = bp.expand_pinch_from_edge(base, x, y)
    except Exception as exc:
        result["status"] = "expand_error"
        result["error"] = str(exc)
        return result

    try:
        bp.validate_in_Q(H)
        result["expanded_in_Q"] = True
    except Exception as exc:
        result["expanded_in_Q"] = False
        result["expanded_error"] = str(exc)

    occ = bp.detect_C_pinch_ii(H)
    result["detected"] = occ is not None
    if occ is None:
        result["status"] = "no_occurrence"
        return result

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

    try:
        reduced, _ = bp.reduce_pinch(H, occ)
        bp.validate_in_Q(reduced)
        result["reduced_in_Q"] = True
    except Exception as exc:
        result["reduced_in_Q"] = False
        result["reduced_error"] = str(exc)

    if result.get("expanded_in_Q") and result.get("reduced_in_Q"):
        result["status"] = "valid_example"
    elif result.get("expanded_in_Q"):
        result["status"] = "bad_reduction"
    else:
        result["status"] = "bad_expansion"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search prism(8) directed edges for a valid pinch(ii) example."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to write the search JSON.",
    )
    args = parser.parse_args()

    base = bp.make_prism(8)
    directed_edges = [(x, y) for x in base.vertices() for y in sorted(base.adj[x])]
    results: List[Dict[str, object]] = [analyze_directed_edge(x, y) for x, y in directed_edges]

    payload = {
        "directed_edge_count": len(directed_edges),
        "valid_examples": [result for result in results if result["status"] == "valid_example"],
        "results": results,
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
