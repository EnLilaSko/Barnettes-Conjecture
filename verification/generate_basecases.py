# verification/generate_basecases.py
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

# Add src to path if needed (though usually we run from root)
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.plantri_wrapper import iter_barnette_graph_rotations_via_plantri
from src.barnette_proof import EmbeddedGraph, validate_in_Q, find_hamiltonian_cycle

def embedded_from_rotation(rot: Dict[int, List[int]]) -> EmbeddedGraph:
    adj = {v: set(nbs) for v, nbs in rot.items()}
    return EmbeddedGraph(adj=adj, rot=rot)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plantri", default="plantri", help="path to plantri executable")
    ap.add_argument("--nmax", type=int, default=14, help="max vertex count (even)")
    ap.add_argument("--out", default="artifacts/basecases.jsonl", help="output jsonl")
    ap.add_argument("--lift-out", default="artifacts/lift_library.json", help="output lift library json")
    ap.add_argument("--c", type=int, default=3, help="connectivity mode in plantri (-c#)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    from src.barnette_proof import EmbeddedGraph, validate_in_Q, find_hamiltonian_cycle, get_lift_cache_json

    with open(args.out, "w", encoding="utf-8") as f:
        for N in range(8, args.nmax + 1, 2):
            print(f"Generating and solving Barnette graphs with N={N}...")
            count = 0
            for pg in iter_barnette_graph_rotations_via_plantri(args.plantri, N, connectivity=args.c):
                G = embedded_from_rotation(pg.rot)
                validate_in_Q(G)  # safety check
                C = find_hamiltonian_cycle(G, debug=False)
                # Store a compact, reproducible record
                rec = {
                    "n": N,
                    "graph_hash": G.canonical_hash(),
                    "cycle_edges": C.edges_as_sorted_pairs(),
                }
                f.write(json.dumps(rec, sort_keys=True) + "\n")
                count += 1
            print(f"  Found {count} graphs.")

    if args.lift_out:
        print(f"Saving lift library to {args.lift_out}...")
        with open(args.lift_out, "w", encoding="utf-8") as f:
            f.write(get_lift_cache_json())

if __name__ == "__main__":
    main()
