#!/usr/bin/env python3
"""
Enumerate all graphs in Q (cubic, bipartite, planar, 3-connected)
up to a given even n_max, using nauty's geng as the generator.

This produces a canonical artifact (graph6 list) that can be checked
independently and used by verify_base_cases.py.

Dependencies:
  - nauty 'geng' available on PATH (or pass --geng-bin).
  - networkx
"""

from __future__ import annotations
import argparse
import shutil
import subprocess
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import networkx as nx


def run_geng_graph6(n: int, geng_bin: str = "geng") -> Iterable[str]:
    """
    Generate connected bipartite 3-regular graphs on n vertices in graph6.

    Flags:
      -c  : connected
      -b  : bipartite
      -d3 : minimum degree 3
      -D3 : maximum degree 3

    geng prints graph6 lines (sometimes with header lines starting with '>').
    """
    if n % 2 != 0:
        return
    cmd = [geng_bin, "-c", "-b", "-d3", "-D3", str(n)]
    p = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in p.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith(">"):
            continue
        yield line


def graph6_to_nx(g6: str) -> nx.Graph:
    G = nx.from_graph6_bytes(g6.encode("ascii"))
    # networkx returns nodes 0..n-1 already, but make it explicit
    H = nx.convert_node_labels_to_integers(G, first_label=0, ordering="sorted")
    return H


def is_planar(G: nx.Graph) -> bool:
    ok, _ = nx.check_planarity(G, counterexample=False)
    return bool(ok)


def is_3connected(G: nx.Graph) -> bool:
    if G.number_of_nodes() < 4:
        return False
    return nx.node_connectivity(G) >= 3


def in_Q(G: nx.Graph) -> bool:
    # cubic + bipartite are guaranteed by geng flags, but keep defensive checks
    if any(d != 3 for _, d in G.degree()):
        return False
    if not nx.is_bipartite(G):
        return False
    if not is_planar(G):
        return False
    if not is_3connected(G):
        return False
    return True


@dataclass(frozen=True)
class QGraph:
    n: int
    g6: str


def enumerate_Q_upto(n_max: int, geng_bin: str = "geng") -> List[QGraph]:
    out: List[QGraph] = []
    for n in range(4, n_max + 1):
        if n % 2 == 1:
            continue
        try:
            # We catch CalledProcessError if user forgot to install geng or path is wrong
            # but here run_geng_graph6 calls subprocess.run with check=True
            for g6 in run_geng_graph6(n, geng_bin=geng_bin):
                G = graph6_to_nx(g6)
                if in_Q(G):
                    out.append(QGraph(n=n, g6=g6))
        except FileNotFoundError:
             # This will be caught in main if shutil fails, but good to be robust
             pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmax", type=int, default=14)
    ap.add_argument("--geng-bin", default="geng")
    ap.add_argument("--out", default="artifacts/Q_upto_nmax.graph6")
    args = ap.parse_args()

    if shutil.which(args.geng_bin) is None:
        # Fallback check: maybe it's in current dir or specific path?
        if not os.path.exists(args.geng_bin):
             print(f"[error] '{args.geng_bin}' not found on PATH used by shutil.which")
             # We proceed to try running it, maybe subprocess finds it?
             # actually better to just warn and fail if not found
             # but user might have alias
             pass

    try:
        Qs = enumerate_Q_upto(args.nmax, geng_bin=args.geng_bin)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"[fatal] Could not run geng: {e}")
        print("Please ensure nauty is installed and 'geng' is in PATH or passed via --geng-bin")
        return

    # write artifact
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for q in Qs:
            f.write(q.g6 + "\n")

    # report
    by_n = {}
    for q in Qs:
        by_n[q.n] = by_n.get(q.n, 0) + 1
    print(f"[ok] enumerated |Q| up to n={args.nmax}: total={len(Qs)}")
    for n in sorted(by_n):
        print(f"  n={n}: {by_n[n]} graphs")


if __name__ == "__main__":
    main()
