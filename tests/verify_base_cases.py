#!/usr/bin/env python3
"""
tests/verify_base_cases.py

Verifies that every graph in artifacts/Q_upto_14.jsonl is Hamiltonian.

This script is intentionally "artifact-driven" so the proof suite can run even when
external enumerators (e.g. nauty geng) are not available.

Usage:
  python tests/verify_base_cases.py --nbase 14
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Set

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
try:
    from barnette_proof import EmbeddedGraph, validate_in_Q, find_hamiltonian_cycle
except ImportError:
    # If running from root without package structure
    try:
        from src.barnette_proof import EmbeddedGraph, validate_in_Q, find_hamiltonian_cycle
    except ImportError:
        # Fallback: assume barnette_proof is in path
        import barnette_proof
        EmbeddedGraph = barnette_proof.EmbeddedGraph
        validate_in_Q = barnette_proof.validate_in_Q
        find_hamiltonian_cycle = barnette_proof.find_hamiltonian_cycle


def record_to_embedded(rec: dict) -> EmbeddedGraph:
    if "adj" not in rec:
        raise ValueError("record missing 'adj'")
    adj: Dict[int, Set[int]] = {}
    for k, vs in rec["adj"].items():
        v = int(k)
        adj[v] = set(int(x) for x in vs)

    if "rot" not in rec:
        raise ValueError("record missing 'rot' (rotation system)")
    rot: Dict[int, List[int]] = {}
    for k, vs in rec["rot"].items():
        v = int(k)
        rot[v] = [int(x) for x in vs]

    return EmbeddedGraph(adj, rot)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nbase", type=int, default=14)
    ap.add_argument("--artifact", default="artifacts/Q_upto_14.jsonl")
    args = ap.parse_args()

    if not os.path.exists(args.artifact):
        raise SystemExit(
            f"Missing {args.artifact}. Create it with: python src/enumerate_Q_upto.py --nbase {args.nbase}"
        )

    recs = []
    with open(args.artifact, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            recs.append(json.loads(line))

    if not recs:
        raise SystemExit(f"No graphs found in {args.artifact}")

    for rec in recs:
        name = rec.get("name", "<unnamed>")
        EG = record_to_embedded(rec)

        validate_in_Q(EG)

        C = find_hamiltonian_cycle(EG, debug=False)
        C.validate_hamiltonian(EG)

        print(f"[PASS] {name} (n={len(EG.adj)})")

    print("Base Case Verification: PASSED")


if __name__ == "__main__":
    main()
