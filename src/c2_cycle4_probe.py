from __future__ import annotations

import argparse
import itertools
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Dict, List, Sequence

import barnette_proof as bp
from completeness_frontier_scan import all_c2_occurrences
from plantri_wrapper import iter_barnette_graph_rotations_via_plantri
from refined_c4_local import graph_from_plantri_rotation


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLANTRI = ROOT / "plantri.exe"


@dataclass(frozen=True)
class Cycle4Result:
    n_vertices: int
    raw_line: str
    occ: Dict[str, int]
    terminal_order: List[str]
    reduction_status: str
    reduction_error: str | None


TERMINAL_NAMES = ("u1", "u6", "u5", "u4")


def cyclic_orders(items: Sequence[str]) -> List[List[str]]:
    orders = []
    for source in (list(items), list(reversed(items))):
        for offset in range(len(source)):
            rotated = source[offset:] + source[:offset]
            if rotated not in orders:
                orders.append(rotated)
    return orders


def reduce_c2_cycle4(G: bp.EmbeddedGraph, occ: bp.OccC2, terminal_order: Sequence[str]) -> bp.EmbeddedGraph:
    H = G.copy()
    terminal_to_old_neighbor = {
        "u1": (occ.u1, occ.a),
        "u4": (occ.u4, occ.d),
        "u5": (occ.u5, occ.e),
        "u6": (occ.u6, occ.f),
    }

    x0 = H.next_id
    x1 = x0 + 1
    x2 = x0 + 2
    x3 = x0 + 3
    H.next_id += 4
    new_vertices = [x0, x1, x2, x3]
    for vertex in new_vertices:
        H.create_empty_vertex(vertex)

    attachment_vertex_for_name = {
        name: new_vertices[index] for index, name in enumerate(terminal_order)
    }
    for name, (terminal, old_neighbor) in terminal_to_old_neighbor.items():
        H.replace_neighbor(terminal, old_neighbor, attachment_vertex_for_name[name])

    for vertex in (occ.a, occ.b, occ.c, occ.d, occ.e, occ.f):
        del H.adj[vertex]
        del H.rot[vertex]
        del H.pos[vertex]

    for index, vertex in enumerate(new_vertices):
        terminal_name = terminal_order[index]
        terminal, _ = terminal_to_old_neighbor[terminal_name]
        left = new_vertices[(index - 1) % 4]
        right = new_vertices[(index + 1) % 4]
        H.set_vertex_rotation(vertex, [terminal, right, left])

    H.assert_consistent()
    return H


def scan_cycle4_probe(plantri_path: Path, n_min: int, n_max: int) -> Dict[str, object]:
    results: List[Cycle4Result] = []
    order_counter: Counter[str] = Counter()
    error_counter: Counter[str] = Counter()
    success_graphs = 0
    tested_graphs = 0

    candidate_orders = cyclic_orders(TERMINAL_NAMES)

    for n_vertices in range(n_min, n_max + 1, 2):
        for embedding in iter_barnette_graph_rotations_via_plantri(str(plantri_path), n_vertices):
            G = graph_from_plantri_rotation(embedding.rot)
            graph_had_success = False
            for occ in all_c2_occurrences(G):
                terminals = [occ.u1, occ.u4, occ.u5, occ.u6]
                if len(set(terminals)) != 4:
                    continue
                Gred, _, error = bp.try_reduce_certified(G, "C2", occ)
                if Gred is not None or error != "not 3-connected":
                    continue
                tested_graphs += 1
                for order in candidate_orders:
                    try:
                        H = reduce_c2_cycle4(G, occ, order)
                        bp.validate_in_Q(H)
                        status = "PASS"
                        reduction_error = None
                        graph_had_success = True
                        order_counter[",".join(order)] += 1
                    except Exception as exc:
                        status = "FAIL"
                        reduction_error = str(exc)
                        error_counter[reduction_error] += 1
                    results.append(
                        Cycle4Result(
                            n_vertices=len(G.adj),
                            raw_line=embedding.raw_line,
                            occ=asdict(occ),
                            terminal_order=list(order),
                            reduction_status=status,
                            reduction_error=reduction_error,
                        )
                    )
            if graph_had_success:
                success_graphs += 1

    pass_records = [result for result in results if result.reduction_status == "PASS"]
    fail_records = [result for result in results if result.reduction_status == "FAIL"]

    return {
        "plantri_path": str(plantri_path),
        "n_min": n_min,
        "n_max": n_max,
        "tested_exception_occurrences": tested_graphs,
        "success_graphs": success_graphs,
        "pass_count": len(pass_records),
        "fail_count": len(fail_records),
        "successful_orders": dict(order_counter),
        "failure_errors": dict(error_counter),
        "first_pass": asdict(pass_records[0]) if pass_records else None,
        "first_fail": asdict(fail_records[0]) if fail_records else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe a 4-cycle replacement gadget on distinct-terminal C2 exceptions."
    )
    parser.add_argument("--plantri", default=str(DEFAULT_PLANTRI))
    parser.add_argument("--n-min", type=int, default=8)
    parser.add_argument("--n-max", type=int, default=24)
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "c2_cycle4_probe.json"))
    args = parser.parse_args()

    result = scan_cycle4_probe(Path(args.plantri), args.n_min, args.n_max)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
