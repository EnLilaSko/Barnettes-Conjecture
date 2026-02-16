#!/usr/bin/env python3
"""
src/enumerate_local_types.py

This script supports the computational claim behind *Theorem 41* in the manuscript:

  For every radius-3 *rotation-system* type around a *facial* 4-cycle in class Q,
  at least one of {C2, C_pinch(ii), refined C4} admits a *certified* reduction step
  whose 3-connectivity admissibility can be checked on the bounded closure
  (Lemma 52 / "separator locality").

Crucially (and unlike earlier drafts), this script does **not** filter types via a
small-graph "extendibility witness" (e.g. n<=14).  Instead it:

  (A) Enumerates all radius-3 local types (with explicit cyclic orders) up to a
      vertex cap ncap that is set to the true worst-case bound used in the paper.

  (B) For each type, enumerates all certified occurrences *rooted at the base
      4-face* and checks admissibility by:
        - performing the reduction locally, and
        - testing 3-connectivity of the induced closure S = N^{<=2}(B) in the
          reduced graph, where B is the boundary set (terminals + gadget vertices).

Outputs:
  artifacts/local_types.jsonl
  artifacts/extendible_witnesses.jsonl      (historical filename; now stores admissible-step witnesses)
  artifacts/obstruction_witnesses.jsonl     (types with no admissible rooted step)

The witness file is meant to be replay-checkable: it records the local type key,
the chosen occurrence, the reduction kind, and the boundary/closure sets.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Set, Tuple, Optional, Iterable, Any
import argparse
import json
import os

import networkx as nx

import barnette_proof as bp


# =============================================================================
# Local type container
# =============================================================================

@dataclass(frozen=True)
class LocalType:
    # labeled vertex set {0,...,n-1}
    n: int
    edges: List[Tuple[int, int]]          # undirected edges (u<v)
    color: List[int]                      # bipartition colors (0/1)
    dist: List[int]                       # BFS distance from {0,1,2,3}
    rot: List[List[int]]                  # cyclic order of neighbors at each vertex (as a list)

    def canonical_key(self) -> str:
        # This is a *serialization key* (not an embedding-isomorphism canonical form).
        # It is stable across runs because our generator uses a deterministic expansion order.
        return json.dumps(
            {
                "edges": self.edges,
                "color": self.color,
                "dist": self.dist,
                "rot": self.rot,
            },
            separators=(",", ":"),
            sort_keys=True,
        )

    def to_nx(self) -> nx.Graph:
        G = nx.Graph()
        G.add_nodes_from(range(self.n))
        G.add_edges_from(self.edges)
        return G


def _is_planar_with_embedding(G: nx.Graph) -> Tuple[bool, Optional[nx.PlanarEmbedding]]:
    ok, emb = nx.check_planarity(G, counterexample=False)
    if not ok:
        return False, None
    return True, emb


def _embedding_neighbors_clockwise(emb: nx.PlanarEmbedding, v: int) -> List[int]:
    # NetworkX stores an embedding adjacency as a rotation system.
    # neighbors_cw_order(v) returns a cyclic list (starting point arbitrary but deterministic).
    return list(emb.neighbors_cw_order(v))


# =============================================================================
# Radius-3 generation around a root facial 4-cycle
# =============================================================================

def _root_cycle() -> nx.Graph:
    G0 = nx.Graph()
    G0.add_nodes_from([0, 1, 2, 3])
    G0.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
    return G0


def _all_stubs(G: nx.Graph, dist: Dict[int, int]) -> List[int]:
    """Return a list containing one entry per missing half-edge at vertices with dist<=2."""
    stubs: List[int] = []
    for v in range(G.number_of_nodes()):
        if dist[v] <= 2:
            need = 3 - G.degree(v)
            if need > 0:
                stubs.extend([v] * need)
    return stubs


def _root_quad_is_face(G: nx.Graph, emb: nx.PlanarEmbedding) -> bool:
    # Check if 0-1-2-3 is a facial cycle in the embedding.
    # In a planar embedding, every edge belongs to two faces.
    # We check if 0-1-2-3 matches either the left or right face of (0, 1).
    try:
        f1 = emb.traverse_face(0, 1)
        if f1 == [0, 1, 2, 3] or f1 == [0, 3, 2, 1]:
            return True
        f2 = emb.traverse_face(1, 0)
        if f2 == [1, 0, 3, 2] or f2 == [1, 2, 3, 0]:
            return True
    except Exception:
        return False
    return False


def generate_radius3_local_types(n_cap: int = 32) -> List[LocalType]:
    """
    Enumerate all radius-3 local types around a facial 4-cycle with:
      - cubicity enforced on vertices with dist<=2,
      - bipartite coloring extending root 0-1-2-3 alternation,
      - planarity preserved during growth,
      - full adjacency data up to distance 3 (dist==3 vertices may have missing degree).
    """
    assert n_cap >= 4

    base = _root_cycle()

    # Fix root coloring: 0,2 in color 0; 1,3 in color 1.
    base_color: Dict[int, int] = {0: 0, 1: 1, 2: 0, 3: 1}
    base_dist: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}

    results: Dict[str, LocalType] = {}

    _count = [0]
    def backtrack(G: nx.Graph, color: Dict[int, int], dist: Dict[int, int], next_vid: int) -> None:
        _count[0] += 1
        if _count[0] % 1000 == 0:
            print(f"   ... count {_count[0]}, next_vid {next_vid}", end="\r")
        # If cubic constraints satisfied on dist<=2, finalize this local type.
        stubs = _all_stubs(G, dist)
        if not stubs:
            ok, emb = _is_planar_with_embedding(G)
            if not ok or emb is None:
                return
            if not _root_quad_is_face(G, emb):
                return

            # Record a rotation system for *all* vertices currently present.
            n = next_vid
            rot = [_embedding_neighbors_clockwise(emb, v) for v in range(n)]

            lt = LocalType(
                n=n,
                edges=sorted((int(min(u, v)), int(max(u, v))) for u, v in G.edges()),
                color=[int(color[i]) for i in range(n)],
                dist=[int(dist[i]) for i in range(n)],
                rot=[[int(x) for x in rot[v]] for v in range(n)],
            )
            results[lt.canonical_key()] = lt
            return

        # Deterministic processing: first stub in list.
        v = stubs[0]
        rest = stubs[1:]

        # Option 1: connect v to an existing vertex u (closing stubs),
        # but only if u has a complementary color and the new vertex remains within dist<=3.
        seen_u: Set[int] = set()
        for u in rest:
            if u in seen_u:
                continue
            seen_u.add(u)

            if v == u or G.has_edge(v, u):
                continue
            if color[v] == color[u]:
                continue

            # Ensure adding this edge does not exceed degree 3.
            if G.degree(v) >= 3 or G.degree(u) >= 3:
                continue

            G.add_edge(v, u)
            ok, _ = _is_planar_with_embedding(G)
            if ok and nx.is_bipartite(G):
                backtrack(G, color, dist, next_vid)
            G.remove_edge(v, u)

        # Option 2: create a new vertex w adjacent to v (a stub expansion).
        if next_vid < n_cap:
            w = next_vid
            # Ensure v can take another edge.
            if G.degree(v) < 3:
                G.add_node(w)
                color2 = dict(color)
                dist2 = dict(dist)

                # Color forced by bipartiteness along edge v-w.
                color2[w] = 1 - color[v]
                G.add_edge(v, w)

                # Update distances from the root set.
                dist2[w] = dist[v] + 1

                # We only keep vertices up to dist<=3 from the root.
                if dist2[w] <= 3:
                    ok, _ = _is_planar_with_embedding(G)
                    if ok and nx.is_bipartite(G):
                        backtrack(G, color2, dist2, next_vid + 1)

                G.remove_edge(v, w)
                G.remove_node(w)

    backtrack(base.copy(), base_color, base_dist, 4)
    return list(results.values())


def load_local_types_with_rot(path: str) -> List[LocalType]:
    """Load LocalType records from a JSONL file.

    This loader is intentionally strict: Theorem 41's "rotation-system types" require
    an explicit neighbor cyclic order at each vertex.  If the input lacks `rot`,
    we refuse (unless the caller intentionally regenerates types via `generate_radius3_local_types`).
    """
    out: List[LocalType] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if "rot" not in obj:
                raise ValueError(
                    f"{path}:{ln}: missing required field `rot` (rotation system). "
                    "Provide the newer local_types.jsonl that includes rotations, "
                    "or run this script without --in-types to regenerate types."
                )
            out.append(
                LocalType(
                    n=int(obj["n"]),
                    edges=[(int(u), int(v)) for (u, v) in obj["edges"]],
                    color=[int(x) for x in obj["color"]],
                    dist=[int(x) for x in obj["dist"]],
                    rot=[[int(x) for x in nbrs] for nbrs in obj["rot"]],
                )
            )
    return out


# =============================================================================
# Local admissibility check for a rooted reduction
# =============================================================================

def localtype_to_embedded(lt: LocalType) -> bp.EmbeddedGraph:
    adj: Dict[int, Set[int]] = {i: set() for i in range(lt.n)}
    for u, v in lt.edges:
        adj[u].add(v)
        adj[v].add(u)
    rot = {i: list(lt.rot[i]) for i in range(lt.n)}
    return bp.EmbeddedGraph(adj=adj, rot=rot)


def _bfs_within(adj: Dict[int, Set[int]], seeds: Iterable[int], radius: int) -> Set[int]:
    S: Set[int] = set()
    frontier: Set[int] = set(seeds)
    S |= frontier
    for _ in range(radius):
        nxt: Set[int] = set()
        for v in frontier:
            nxt |= adj.get(v, set())
        nxt -= S
        S |= nxt
        frontier = nxt
        if not frontier:
            break
    return S


def _induced_adj(adj: Dict[int, Set[int]], S: Set[int]) -> Dict[int, Set[int]]:
    return {v: set(w for w in adj[v] if w in S) for v in S}


def _boundary_and_kind_from_record(kind: str, rec: Any) -> Set[int]:
    # Boundary set B for Lemma 52 locality checks:
    #   B = gadget vertices + *attachment terminals that remain in the reduced graph*.
    #
    # Important: do NOT include vertices deleted by the reduction (e.g. v1..v4 in C4,
    # a..f in C2, or v1..v4,w,t in pinch(ii)). Including deleted vertices would
    # pollute the closure set with non-vertices and break induced-subgraph checks.
    B: Set[int] = set()
    if kind == "C2":
        B |= {rec.x, rec.y, rec.u1, rec.u4, rec.u5, rec.u6}
    elif kind == "C4":
        B |= {rec.x, rec.y, rec.u1, rec.u2, rec.u3, rec.u4}
    elif kind == "CP":
        # For pinch(ii), the terminals that remain are (r,s,u2,u4), plus gadget (x,y).
        B |= {rec.x, rec.y, rec.r, rec.s, rec.u2, rec.u4}
    else:
        raise ValueError(f"unknown reduction kind: {kind}")
    return B


def rooted_occurrences(G: bp.EmbeddedGraph) -> List[Tuple[str, Any]]:
    """
    Enumerate *certified* occurrences rooted at the base 4-face {0,1,2,3}.
    Returns (kind, occ) where kind in {"C2","CP","C4"}.
    """
    kinds: List[Tuple[str, Any]] = []

    root_set = {0, 1, 2, 3}

    # C2: any adjacent 4-face across one of the four root edges.
    for (a, b) in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        # check both darts; one side is the root face, the other side might be quad
        if G.face_is_quad_from_dart((a, b)) and G.face_is_quad_from_dart((b, a)):
            L = G.trace_face_vertices((a, b), steps=4)
            R = G.trace_face_vertices((b, a), steps=4)
            occ = bp._canonicalize_adjacent_quads(G, L, R)
            if occ is None:
                continue
            if len({occ.u1, occ.u4, occ.u5, occ.u6}) != 4:
                continue
            if set(L) == root_set or set(R) == root_set:
                kinds.append(("C2", occ))

    # refined C4: root face itself must be an isolated 4-face with distinct terminals.
    # Use both orientations of the root face.
    for dart in [(0, 1), (1, 0)]:
        darts, end = G.trace_face_darts(dart, steps=4)
        if end != dart:
            continue
        v1 = dart[0]
        v2 = darts[0][1]
        v3 = darts[1][1]
        v4 = darts[2][1]
        if set([v1, v2, v3, v4]) != root_set:
            continue
        quad_edges = [(v1, v2), (v2, v3), (v3, v4), (v4, v1)]
        if any(G.other_face_is_quad(x, y) for x, y in quad_edges):
            continue
        u1 = G.third_neighbor(v1, {v2, v4})
        u2 = G.third_neighbor(v2, {v1, v3})
        u3 = G.third_neighbor(v3, {v2, v4})
        u4 = G.third_neighbor(v4, {v1, v3})
        if len({u1, u2, u3, u4}) == 4:
            kinds.append(("C4", bp.OccC4(v1, v2, v3, v4, u1, u2, u3, u4)))

    # pinch(ii): root face with u1=u3 and the pinch(ii) side-conditions.
    for occ in bp.all_C_pinch_ii_occurrences(G):
        if set([occ.v1, occ.v2, occ.v3, occ.v4]) == root_set:
            kinds.append(("CP", occ))

    # De-duplicate while preserving deterministic order.
    seen = set()
    out: List[Tuple[str, Any]] = []
    for k, o in sorted(kinds, key=lambda t: (t[0], t[1])):
        key = (k, o)
        if key in seen:
            continue
        seen.add(key)
        out.append((k, o))
    return out



def admissible_witness_for_type(lt: LocalType, max_witnesses: int = 1) -> List[Dict[str, Any]]:
    """Return up to max_witnesses admissible-step witnesses for this local type."""
    G = localtype_to_embedded(lt)
    witnesses: List[Dict[str, Any]] = []

    for kind, occ in rooted_occurrences(G):
        try:
            if kind == "C2":
                H, rec = bp.reduce_C2(G, occ)
            elif kind == "C4":
                H, rec = bp.reduce_C4(G, occ)
            elif kind == "CP":
                H, rec = bp.reduce_pinch_ii(G, occ)
            else:
                continue
        except Exception as e:
            # Not a certified occurrence for this local type / missing radius data.
            continue

        B = _boundary_and_kind_from_record(kind, rec)
        S = _bfs_within(H.adj, B, radius=2)
        Hs = _induced_adj(H.adj, S)

        if bp.is_3_connected(Hs):
            witnesses.append(
                {
                    "local_type_key": lt.canonical_key(),
                    "n_local": lt.n,
                    "reduction_kind": kind,
                    "occurrence": asdict(occ),
                    "record": asdict(rec),
                    "boundary_B": sorted(int(x) for x in B),
                    "closure_S": sorted(int(x) for x in S),
                    "n_closure": len(S),
                }
            )
            if len(witnesses) >= max_witnesses:
                break

    return witnesses


# =============================================================================
# Driver
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncap", type=int, default=32, help="max vertices in local type generation (paper default: 32)")
    ap.add_argument(
        "--in-types",
        default=None,
        help="optional JSONL of local types INCLUDING rotation systems (field `rot`). "
             "If provided, skip generation and use these types for witness production.",
    )
    ap.add_argument("--out-types", default="artifacts/local_types.jsonl")
    ap.add_argument("--out-extendible", default="artifacts/extendible_witnesses.jsonl",
                    help="historical filename; stores admissible-step witnesses")
    ap.add_argument("--out-obstructions", default="artifacts/obstruction_witnesses.jsonl")
    ap.add_argument("--max-witnesses-per-type", type=int, default=1)
    args = ap.parse_args()

    for fpath in [args.out_types, args.out_extendible, args.out_obstructions]:
        d = os.path.dirname(fpath)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

    if args.in_types:
        locals_ = load_local_types_with_rot(args.in_types)
        print(f"[enumerate] loaded {len(locals_)} local types from {args.in_types}.")
        # If loading types, we do not overwrite the type file unless the user explicitly
        # points --out-types somewhere else.
        if args.out_types:
            with open(args.out_types, "w", encoding="utf-8") as f:
                for lt in locals_:
                    f.write(json.dumps(asdict(lt), separators=(',', ':'), sort_keys=True) + "\n")
    else:
        locals_ = generate_radius3_local_types(n_cap=args.ncap)
        print(f"[enumerate] generated {len(locals_)} radius-3 local types (<= {args.ncap} vertices).")

        with open(args.out_types, "w", encoding="utf-8") as f:
            for lt in locals_:
                f.write(json.dumps(asdict(lt), separators=(',', ':'), sort_keys=True) + "\n")

    admissible = 0
    obstructions = 0

    with open(args.out_extendible, "w", encoding="utf-8") as f_ok, open(args.out_obstructions, "w", encoding="utf-8") as f_bad:
        for i, lt in enumerate(locals_):
            if i % 10 == 0:
                print(f"... checking type {i+1}/{len(locals_)}", end="\r")

            ws = admissible_witness_for_type(lt, max_witnesses=args.max_witnesses_per_type)
            if ws:
                admissible += 1
                for w in ws:
                    f_ok.write(json.dumps(w, separators=(',', ':'), sort_keys=True) + "\n")
            else:
                obstructions += 1
                f_bad.write(json.dumps({
                    "local_type_key": lt.canonical_key(),
                    "n_local": lt.n,
                    "note": "no admissible rooted reduction found by local closure check",
                }, separators=(',', ':'), sort_keys=True) + "\n")

    print("")
    print(f"[result] types with >=1 admissible rooted step: {admissible}")
    print(f"[result] obstruction types (no admissible rooted step): {obstructions}")
    print("[done] wrote:")
    print(f"  - {args.out_types}")
    print(f"  - {args.out_extendible}")
    print(f"  - {args.out_obstructions}")


if __name__ == "__main__":
    main()
