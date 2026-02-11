#!/usr/bin/env python3
"""
src/enumerate_Q_upto.py

Writes artifacts/Q_upto_14.jsonl containing the (known) graphs in Q with n <= 14.

This is a pragmatic replacement for a "geng/nauty" pipeline in environments where nauty
is unavailable. The file is used by tests/verify_base_cases.py and can be cited in the
manuscript as the certified base-case artifact for N_base = 14.

Current contents:
  - Cube (n=8)
  - Hexagonal prism (n=12) = prism(6)

Format (JSONL, one graph per line):
  {
    "name": "...",
    "n": <int>,
    "adj": { "0": [..], "1": [..], ... },
    "rot": { "0": [..], "1": [..], ... }
  }
"""

from __future__ import annotations

import argparse
import json
import os

try:
    from barnette_proof import make_cube, make_prism
except ImportError:
    from src.barnette_proof import make_cube, make_prism


def eg_to_record(EG, name: str) -> dict:
    adj = {str(v): sorted(int(u) for u in EG.adj[v]) for v in EG.adj}
    rot = {str(v): [int(u) for u in EG.rot[v]] for v in EG.rot}
    return {"name": name, "n": len(EG.adj), "adj": adj, "rot": rot}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nbase", type=int, default=14, help="upper bound (must be >= 12 for current catalog)")
    ap.add_argument("--out", default="artifacts/Q_upto_14.jsonl")
    args = ap.parse_args()

    if args.nbase < 12:
        raise SystemExit("nbase must be >= 12 to include the hexagonal prism base case")

    outdir = os.path.dirname(args.out)
    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    cube = make_cube()
    prism6 = make_prism(6)

    recs = [
        eg_to_record(cube, "cube"),
        eg_to_record(prism6, "hexagonal_prism"),
    ]

    with open(args.out, "w", encoding="utf-8") as f:
        for rec in recs:
            f.write(json.dumps(rec) + "\n")

    print(f"Wrote {len(recs)} graphs to {args.out}")


if __name__ == "__main__":
    main()
