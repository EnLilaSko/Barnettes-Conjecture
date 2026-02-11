#!/usr/bin/env python3
"""
src/enumerate_local_types.py

Enumerate radius-3 *rotation-system* neighborhoods of a *facial* 4-cycle
compatible with cubicity + bipartiteness, and (optionally) search for
completions to full graphs in Q up to a vertex bound.

Key correction vs the earlier draft:
  - The earlier generator forced a tree-like outward expansion (dist2 slots),
    which *cannot* realize the Cube neighborhood around a 4-face, because the
    Cube needs "horizontal" edges among the distance-1 neighbors.
  - This version allows those horizontal edges (and more generally, any edge
    additions consistent with bipartite+cubic+planar+radius constraints),
    so the Cube local type (n=8) is generated and is extendible.

Outputs (JSONL):
  - artifacts/local_types.jsonl
  - artifacts/extendible_witnesses.jsonl
  - artifacts/obstruction_witnesses.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Set

import networkx as nx

# -------------------------
# Import your proof objects
# -------------------------
try:
    from barnette_proof import EmbeddedGraph, verify_completeness
except ImportError:
    from src.barnette_proof import EmbeddedGraph, verify_completeness


# =============================================================================
# Basic graph predicates (NetworkX)
# =============================================================================

def is_cubic(G: nx.Graph) -> bool:
    return all(d == 3 for _, d in G.degree())

def is_bipartite(G: nx.Graph) -> bool:
    return nx.is_bipartite(G)

def is_planar(G: nx.Graph) -> bool:
    ok, _ = nx.check_planarity(G, counterexample=False)
    return bool(ok)

def is_3connected(G: nx.Graph) -> bool:
    if G.number_of_nodes() < 4:
        return False
    return nx.node_connectivity(G) >= 3

def in_Q(G: nx.Graph) -> bool:
    return is_cubic(G) and is_bipartite(G) and is_planar(G) and is_3connected(G)


# =============================================================================
# EmbeddedGraph conversion + "root face is facial" check
# =============================================================================

def _embedding_neighbors_clockwise(emb: nx.PlanarEmbedding, v: int) -> List[int]:
    # NetworkX API differs slightly between versions; support both.
    if hasattr(emb, "neighbors_cw_order"):
        return list(emb.neighbors_cw_order(v))
    return list(emb.neighbors(v))  # typically clockwise order

def root_quad_is_face(G: nx.Graph, root: Tuple[int, int, int, int]) -> bool:
    ok, emb = nx.check_planarity(G, counterexample=False)
    if not ok:
        return False

    cycle = list(root)
    edges = [(cycle[i], cycle[(i + 1) % 4]) for i in range(4)]
    if any(not G.has_edge(a, b) for a, b in edges):
        return False

    target = cycle
    target_rev = [cycle[0], cycle[3], cycle[2], cycle[1]]

    # Try traversing each directed edge of the root 4-cycle;
    # if any face traversal gives that 4-cycle, accept.
    darts = edges + [(b, a) for (a, b) in edges]
    for (u, v) in darts:
        try:
            face = list(emb.traverse_face(u, v))
        except Exception:
            continue
        if len(face) == 4 and (face == target or face == target_rev):
            return True
    return False

def nx_to_embedded(G: nx.Graph) -> EmbeddedGraph:
    ok, emb = nx.check_planarity(G, counterexample=False)
    if not ok:
        raise ValueError("Graph is not planar")

    adj: Dict[int, Set[int]] = {int(v): set(int(u) for u in G.neighbors(v)) for v in G.nodes()}
    rot: Dict[int, List[int]] = {int(v): [int(u) for u in _embedding_neighbors_clockwise(emb, v)] for v in G.nodes()}
    return EmbeddedGraph(adj, rot)


# =============================================================================
# Data model: LocalType (graph + stubs + bipartite colors + distances)
# =============================================================================

@dataclass(frozen=True)
class LocalType:
    """
    Radius-3 neighborhood around a rooted facial 4-cycle.

    - vertices are labeled 0..n-1 in this local object
    - root_face is (0,1,2,3) in cyclic order
    - dist[v] is graph distance to the root-face vertex set {0,1,2,3}
    - vertices with dist <= 2 are fully cubic *inside* the neighborhood (degree==3)
    - vertices with dist == 3 may have missing degree slots, recorded as "stubs"
    """
    n: int
    edges: List[Tuple[int, int]]
    root_face: Tuple[int, int, int, int]
    stubs: List[Tuple[int, int]]
    color: List[int]
    dist: List[int]

    def to_nx_partial(self) -> nx.Graph:
        G = nx.Graph()
        G.add_nodes_from(range(self.n))
        G.add_edges_from(self.edges)
        return G

    def canonical_key(self) -> str:
        ed = sorted((min(u, v), max(u, v)) for u, v in self.edges)
        return json.dumps(
            {"root": self.root_face, "edges": ed, "color": self.color, "dist": self.dist},
            separators=(",", ":"),
        )


# =============================================================================
# Radius computation
# =============================================================================

def multi_source_dist(G: nx.Graph, sources: List[int]) -> Dict[int, int]:
    dist = {v: 10**9 for v in G.nodes()}
    for s in sources:
        dist[s] = 0
    q = list(sources)
    while q:
        v = q.pop(0)
        for u in G.neighbors(v):
            if dist[u] > dist[v] + 1:
                dist[u] = dist[v] + 1
                q.append(u)
    return dist


# =============================================================================
# (A) Enumerate local types
# =============================================================================

def generate_radius3_local_types(n_cap: int) -> List[LocalType]:
    """
    Backtracking generator for rooted radius-3 neighborhoods of a facial 4-cycle.

    Notes:
      - We *do not* pre-impose a tree-like structure on the distance-1 layer.
        Horizontal edges (like the Cube's opposite 4-face) are allowed and required.
      - We maintain bipartite coloring seeded by the root cycle.
      - We require planarity throughout, and at finalization we also require the root
        4-cycle is facial in the planarity embedding returned by NetworkX.
    """
    root = (0, 1, 2, 3)
    roots = [0, 1, 2, 3]

    # Seed bipartition on the root 4-cycle: 0/2 in part 0, 1/3 in part 1
    base_color: Dict[int, int] = {0: 0, 1: 1, 2: 0, 3: 1}

    # Start with the 4-cycle
    base = nx.Graph()
    base.add_nodes_from(range(4))
    base.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])

    types: Dict[str, LocalType] = {}

    def ok_partial(G: nx.Graph) -> bool:
        if any(G.degree(v) > 3 for v in G.nodes()):
            return False
        if not nx.is_bipartite(G):
            return False
        return is_planar(G)

    def finalize(G: nx.Graph, color: Dict[int, int], next_id: int) -> None:
        dist = multi_source_dist(G, roots)
        if max(dist.values()) > 3:
            return

        internal = [v for v in range(next_id) if dist[v] <= 2]
        boundary = [v for v in range(next_id) if dist[v] == 3]

        if any(G.degree(v) != 3 for v in internal):
            return
        if any(G.degree(v) > 3 for v in boundary):
            return

        if not root_quad_is_face(G, root):
            return

        stubs: List[Tuple[int, int]] = []
        for v in boundary:
            for k in range(3 - G.degree(v)):
                stubs.append((v, k))

        edge_list = sorted((min(a, b), max(a, b)) for (a, b) in G.edges())
        color_list = [int(color[v]) for v in range(next_id)]
        dist_list = [int(dist[v]) for v in range(next_id)]

        lt = LocalType(
            n=next_id,
            edges=edge_list,
            root_face=root,
            stubs=stubs,
            color=color_list,
            dist=dist_list,
        )
        types[lt.canonical_key()] = lt

    def fill_degrees(G: nx.Graph, color: Dict[int, int], next_id: int) -> None:
        if next_id > n_cap:
            return
        if not ok_partial(G):
            return

        dist = multi_source_dist(G, roots)
        if max(dist.values()) > 3:
            return

        internal = [v for v in range(next_id) if dist[v] <= 2]

        v_need: Optional[int] = None
        for v in sorted(internal):
            if G.degree(v) < 3:
                v_need = v
                break

        if v_need is None:
            finalize(G, color, next_id)
            return

        need_color = 1 - color[v_need]

        for u in range(next_id):
            if u == v_need:
                continue
            if color.get(u) != need_color:
                continue
            if G.has_edge(v_need, u):
                continue
            if G.degree(u) >= 3:
                continue
            if G.degree(v_need) >= 3:
                continue

            G2 = G.copy()
            G2.add_edge(v_need, u)
            if ok_partial(G2):
                fill_degrees(G2, color, next_id)

        if next_id < n_cap:
            u = next_id
            G2 = G.copy()
            G2.add_node(u)
            G2.add_edge(v_need, u)
            color2 = dict(color)
            color2[u] = need_color

            if ok_partial(G2):
                fill_degrees(G2, color2, next_id + 1)

    def attach_third_neighbors(i: int, G: nx.Graph, color: Dict[int, int], next_id: int) -> None:
        if next_id > n_cap:
            return
        if i == 4:
            fill_degrees(G, color, next_id)
            return

        qi = roots[i]
        needed_color = 1 - base_color[qi]

        for u in range(4, next_id):
            if color.get(u) != needed_color:
                continue
            if G.has_edge(qi, u):
                continue
            if G.degree(u) >= 3:
                continue
            if G.degree(qi) >= 3:
                continue

            G2 = G.copy()
            G2.add_edge(qi, u)
            attach_third_neighbors(i + 1, G2, color, next_id)

        if next_id < n_cap:
            u = next_id
            G2 = G.copy()
            G2.add_node(u)
            G2.add_edge(qi, u)
            color2 = dict(color)
            color2[u] = needed_color
            attach_third_neighbors(i + 1, G2, color2, next_id + 1)

    attach_third_neighbors(0, base, dict(base_color), 4)
    return list(types.values())


# =============================================================================
# (B) Completion search to full graphs in Q (within a vertex cap)
# =============================================================================

def complete_to_Q(local: LocalType, n_max: int, max_witnesses: int = 1) -> List[nx.Graph]:
    """
    Given a local type (with boundary stubs), try to complete it into a full cubic bipartite planar
    3-connected graph with <= n_max vertices.
    """
    G0 = local.to_nx_partial()

    if all(G0.degree(v) == 3 for v in G0.nodes()) and in_Q(G0):
        return [G0.copy()]

    color = {v: local.color[v] for v in range(local.n)}

    if any(G0.degree(v) > 3 for v in G0.nodes()):
        return []
    if not nx.is_bipartite(G0):
        return []

    stubs: List[int] = []
    for v in range(local.n):
        stubs.extend([v] * (3 - G0.degree(v)))

    witnesses: List[nx.Graph] = []

    def backtrack(G: nx.Graph, color_map: Dict[int, int], stub_list: List[int], next_vid: int) -> None:
        nonlocal witnesses
        if len(witnesses) >= max_witnesses:
            return
        if next_vid > n_max:
            return
        if not stub_list:
            if in_Q(G):
                witnesses.append(G.copy())
            return

        v = stub_list[0]
        rest = stub_list[1:]

        seen_u: Set[int] = set()
        for j, u in enumerate(rest):
            if u in seen_u:
                continue
            seen_u.add(u)

            if color_map[v] == color_map[u]:
                continue
            if v == u or G.has_edge(v, u):
                continue
            if G.degree(v) >= 3 or G.degree(u) >= 3:
                continue

            G.add_edge(v, u)
            if nx.is_bipartite(G) and is_planar(G):
                backtrack(G, color_map, rest[:j] + rest[j + 1 :], next_vid)
            G.remove_edge(v, u)

        if next_vid < n_max:
            w = next_vid
            G.add_node(w)
            color2 = dict(color_map)
            color2[w] = 1 - color_map[v]
            G.add_edge(v, w)

            new_stubs = rest + [w, w]
            if nx.is_bipartite(G) and is_planar(G):
                backtrack(G, color2, new_stubs, next_vid + 1)

            G.remove_edge(v, w)
            G.remove_node(w)

    backtrack(G0.copy(), color, stubs, local.n)
    return witnesses


# =============================================================================
# (C) Driver
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncap", type=int, default=14, help="max vertices in local type generation")
    ap.add_argument("--nmax", type=int, default=14, help="max vertices in completion to Q")
    ap.add_argument("--out-types", default="artifacts/local_types.jsonl")
    ap.add_argument("--out-extendible", default="artifacts/extendible_witnesses.jsonl")
    ap.add_argument("--out-obstructions", default="artifacts/obstruction_witnesses.jsonl")
    ap.add_argument("--max-witnesses-per-type", type=int, default=1)
    args = ap.parse_args()

    for fpath in [args.out_types, args.out_extendible, args.out_obstructions]:
        d = os.path.dirname(fpath)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

    locals_ = generate_radius3_local_types(n_cap=args.ncap)
    print(f"[enumerate] generated {len(locals_)} candidate local types (<= {args.ncap} vertices).")

    with open(args.out_types, "w", encoding="utf-8") as f:
        for lt in locals_:
            f.write(json.dumps(asdict(lt)) + "\n")

    extendible = 0
    obstructions = 0

    with open(args.out_extendible, "w", encoding="utf-8") as f_ext, open(args.out_obstructions, "w", encoding="utf-8") as f_obs:
        for idx, lt in enumerate(locals_):
            if idx % 10 == 0:
                print(f"... processing type {idx+1}/{len(locals_)}", end="\r")

            witnesses = complete_to_Q(lt, n_max=args.nmax, max_witnesses=args.max_witnesses_per_type)
            if not witnesses:
                continue

            extendible += 1
            Gw = witnesses[0]

            f_ext.write(json.dumps({
                "local_type_key": lt.canonical_key(),
                "n_local": lt.n,
                "n_witness": Gw.number_of_nodes(),
                "edges_witness": sorted((int(u), int(v)) for u, v in Gw.edges()),
            }) + "\n")

            try:
                EG = nx_to_embedded(Gw)
                verify_completeness(EG)
                failed = False
                note = ""
            except Exception as e:
                failed = True
                note = repr(e)

            if failed:
                obstructions += 1
                f_obs.write(json.dumps({
                    "local_type_key": lt.canonical_key(),
                    "n_local": lt.n,
                    "witness_n": Gw.number_of_nodes(),
                    "witness_edges": sorted((int(u), int(v)) for u, v in Gw.edges()),
                    "note": f"verify_completeness failed: {note}",
                }) + "\n")

    print("")
    print(f"[filter] extendible types (<= {args.nmax} vertices): {extendible}")
    print(f"[result] obstruction witnesses found: {obstructions}")
    print("[done] wrote:")
    print(f"  - {args.out_types}")
    print(f"  - {args.out_extendible}")
    print(f"  - {args.out_obstructions}")


if __name__ == "__main__":
    main()
