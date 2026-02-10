#!/usr/bin/env python3
"""
Enumerate radius-3 rotation-system neighborhoods of a facial 4-cycle
compatible with cubicity + bipartiteness, and filter those extendible
to 3-connected instances.

This is intended to back the manuscript claim (Lemma 41-style) that
"all local obstruction types are finite and occur <= N_base".

High-level pipeline:
  (A) Generate local neighborhoods N (partial graphs with stubs) around a rooted 4-face.
  (B) For each N, attempt completions to full graphs G in Q with |V(G)| <= N_max.
  (C) For each completion G, run verify_completeness(G) (or equivalent) and record witnesses.

Outputs:
  - local_types.jsonl: serialized neighborhoods (with root face + stubs)
  - extendible_witnesses.jsonl: one completion witness per extendible type
  - obstruction_witnesses.jsonl: any completion that appears to have no admissible move

NOTE:
  This script is deliberately conservative and small-trust:
  it treats your reducer as untrusted and relies on your checker-ish
  predicates (planarity/bipartite/3conn) + verify_completeness outcomes.
"""

from __future__ import annotations

import argparse
import json
import itertools
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Set

import networkx as nx

# -------------------------
# Adapt these imports to your repo
# -------------------------
try:
    # If barnette_proof.py is at repo root or PYTHONPATH includes src:
    from barnette_proof import EmbeddedGraph, verify_completeness, CompletenessWitness
except ImportError:
    try:
        # If it lives under src/ and running from root:
        from src.barnette_proof import EmbeddedGraph, verify_completeness, CompletenessWitness
    except ImportError:
        # Fallback for when running from inside src/ without package structure
        import sys
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from barnette_proof import EmbeddedGraph, verify_completeness, CompletenessWitness


# -------------------------
# Small helpers
# -------------------------

def is_cubic(G: nx.Graph) -> bool:
    return all(d == 3 for _, d in G.degree())

def is_bipartite(G: nx.Graph) -> bool:
    return nx.is_bipartite(G)

def is_planar(G: nx.Graph) -> bool:
    ok, _ = nx.check_planarity(G, counterexample=False)
    return bool(ok)

def is_3connected(G: nx.Graph) -> bool:
    # For small graphs this is fine. If you have a faster 3-conn routine, use it.
    if G.number_of_nodes() < 4:
        return False
    return nx.node_connectivity(G) >= 3

def in_Q(G: nx.Graph) -> bool:
    return is_cubic(G) and is_bipartite(G) and is_planar(G) and is_3connected(G)

def nx_to_embedded(G: nx.Graph) -> EmbeddedGraph:
    """
    Convert a NetworkX graph into your EmbeddedGraph.
    Uses nx.check_planarity to determine a valid rotation system.
    """
    is_planar, embedding = nx.check_planarity(G)
    if not is_planar:
        raise ValueError("Graph is not planar, cannot convert to EmbeddedGraph")
    
    # nx.PlanarEmbedding uses a half-edge structure where neighbors are ordered
    adj = {}
    rot = {}
    for v in G.nodes():
        adj[v] = set(G.neighbors(v))
        # embedding.neighbors(v) iterates neighbors in clockwise order
        rot[v] = list(embedding.neighbors(v))
        
    return EmbeddedGraph(adj, rot)


# -------------------------
# Data model: local neighborhood with stubs
# -------------------------

@dataclass(frozen=True)
class LocalType:
    """
    Partial neighborhood around a rooted facial 4-cycle.

    vertices: list of vertex ids (0..n-1 in this local object)
    edges: list of undirected edges among vertices
    root_face: the 4-cycle vertices in cyclic order [q0,q1,q2,q3]
    stubs: list of half-edges (v, stub_id) representing missing degree slots at v
           (these are "to outside the neighborhood")
    color: bipartition coloring on vertices: 0/1
    """
    n: int
    edges: List[Tuple[int, int]]
    root_face: Tuple[int, int, int, int]
    stubs: List[Tuple[int, int]]
    color: List[int]

    def to_nx_partial(self) -> nx.Graph:
        G = nx.Graph()
        G.add_nodes_from(range(self.n))
        for u, v in self.edges:
            G.add_edge(u, v)
        return G

    def degree_deficit(self, v: int) -> int:
        G = self.to_nx_partial()
        return 3 - G.degree(v)

    def canonical_key(self) -> str:
        # Rooted key: keep root_face fixed, hash edges + colors.
        # This is not full isomorphism canonization, but is stable and reproducible.
        ed = sorted((min(u, v), max(u, v)) for u, v in self.edges)
        return json.dumps({"root": self.root_face, "edges": ed, "color": self.color}, separators=(",", ":"))


# -------------------------
# (A) Generate local radius-3 neighborhoods (partial)
# -------------------------

def generate_radius3_local_types(n_cap: int) -> List[LocalType]:
    """
    Enumerate partial graphs reachable within distance <=3 of a rooted 4-cycle,
    under cubic + bipartite constraints, keeping total vertices <= n_cap.

    This generator is intentionally *simple*: it grows outward from the 4-cycle and
    allows identifications to keep n small. It does NOT attempt to generate every
    combinatorial map on the sphere; instead it targets the object described in the
    manuscript: a "radius-3 neighborhood" shape with stubs.
    """

    # Root face vertices:
    q0, q1, q2, q3 = 0, 1, 2, 3
    root = (q0, q1, q2, q3)

    # Bipartition: alternate on cycle
    # q0,q2 in color 0; q1,q3 in color 1
    base_color = {q0: 0, q1: 1, q2: 0, q3: 1}

    # Start graph: 4-cycle
    base_edges = {(q0, q1), (q1, q2), (q2, q3), (q3, q0)}

    # Each qi needs one external neighbor (distance 1), but we allow identifications
    # consistent with bipartition (q0/q2 share color; their neighbors share color 1).
    # We'll create u0..u3 sequentially with optional merging.
    local_types: Dict[str, LocalType] = {}

    def add_edge(edges: Set[Tuple[int, int]], u: int, v: int) -> None:
        if u == v:
            raise ValueError("loop edge")
        a, b = (u, v) if u < v else (v, u)
        edges.add((a, b))

    def build_from_assignment(u_map: Dict[int, int], next_vid: int) -> None:
        # u_map maps qi -> ui
        edges = set(base_edges)
        color = dict(base_color)

        # assign colors and add edges qi-ui
        for qi, ui in u_map.items():
            if ui not in color:
                color[ui] = 1 - color[qi]
            if color[ui] != 1 - color[qi]:
                return
            add_edge(edges, qi, ui)

        # Early prune: degrees must be <=3
        G = nx.Graph()
        G.add_nodes_from(range(next_vid))
        G.add_edges_from(edges)
        if any(G.degree(v) > 3 for v in G.nodes()):
            return
        if not nx.is_bipartite(G):
            return

        # Expand outward within radius 3 in a very controlled way:
        # For each ui (dist1), give it 2 more neighbors (possibly shared), then for those dist2
        # give them remaining neighbors (possibly shared) but stop growth at n_cap.
        # We track BFS layers implicitly by construction.
        dist1 = sorted(set(u_map.values()))

        # create "slots" for dist2 neighbors (two per dist1 vertex)
        dist2_slots: List[Tuple[int, int]] = []
        for ui in dist1:
            dist2_slots.append((ui, 0))
            dist2_slots.append((ui, 1))

        # We will decide each slot attaches either to an existing dist2 vertex (merge),
        # or a new vertex.
        def backtrack_dist2(i: int, edges2: Set[Tuple[int, int]], color2: Dict[int, int], next_id: int,
                            dist2_vertices: List[int], slot_to_vertex: Dict[Tuple[int, int], int]) -> None:
            if next_id > n_cap:
                return
            if i == len(dist2_slots):
                # Now assign dist3 stubs only; we won't explicitly create dist3 vertices here.
                # Any remaining degree deficits at dist2 vertices and dist1 vertices become stubs.
                Gp = nx.Graph()
                Gp.add_nodes_from(range(next_id))
                Gp.add_edges_from(edges2)

                # enforce degrees <=3 and bipartite
                if any(Gp.degree(v) > 3 for v in Gp.nodes()):
                    return
                if not nx.is_bipartite(Gp):
                    return

                # stubs are missing degree slots on vertices at distance 3 boundary, but since
                # we didn't explicitly create dist3 layer, we treat any deficit on non-cycle vertices
                # as stubs to outside.
                stubs: List[Tuple[int, int]] = []
                for v in range(next_id):
                    deficit = 3 - Gp.degree(v)
                    for k in range(deficit):
                        stubs.append((v, k))

                # Build LocalType
                edge_list = sorted((min(a, b), max(a, b)) for a, b in edges2)
                color_list = [color2[v] for v in range(next_id)]
                lt = LocalType(
                    n=next_id,
                    edges=edge_list,
                    root_face=root,
                    stubs=stubs,
                    color=color_list,
                )
                key = lt.canonical_key()
                local_types[key] = lt
                return

            (ui, slot) = dist2_slots[i]
            needed_color = 1 - color2[ui]

            # Option 1: attach to an existing dist2 vertex of correct color (merge)
            for v in dist2_vertices:
                if color2[v] != needed_color:
                    continue
                # add edge ui-v if not present, and if doesn't violate degree cap
                a, b = (ui, v) if ui < v else (v, ui)
                if (a, b) in edges2:
                    continue
                # quick degree cap check
                # compute current degrees for ui and v
                # (small, so rebuild is OK)
                Gtmp = nx.Graph()
                Gtmp.add_nodes_from(range(next_id))
                Gtmp.add_edges_from(edges2)
                if Gtmp.degree(ui) >= 3 or Gtmp.degree(v) >= 3:
                    continue
                edges_next = set(edges2)
                edges_next.add((a, b))
                slot_to_vertex_next = dict(slot_to_vertex)
                slot_to_vertex_next[(ui, slot)] = v
                backtrack_dist2(i + 1, edges_next, color2, next_id, dist2_vertices, slot_to_vertex_next)

            # Option 2: create a new dist2 vertex
            if next_id < n_cap:
                v = next_id
                color_next = dict(color2)
                color_next[v] = needed_color
                edges_next = set(edges2)
                add_edge(edges_next, ui, v)
                dist2_next = dist2_vertices + [v]
                slot_to_vertex_next = dict(slot_to_vertex)
                slot_to_vertex_next[(ui, slot)] = v
                backtrack_dist2(i + 1, edges_next, color_next, next_id + 1, dist2_next, slot_to_vertex_next)

        backtrack_dist2(
            i=0,
            edges2=edges,
            color2=color,
            next_id=next_vid,
            dist2_vertices=[],
            slot_to_vertex={},
        )

    # Enumerate identifications among u0..u3 consistent with bipartite constraints
    # q0,q2 neighbors must be color 1; q1,q3 neighbors must be color 0.
    # We grow u assignments sequentially with optional reuse.
    def backtrack_u(i: int, u_vertices: List[int], u_map: Dict[int, int], next_vid: int) -> None:
        qs = [q0, q1, q2, q3]
        if i == 4:
            build_from_assignment(u_map, next_vid)
            return
        qi = qs[i]
        needed_color = 1 - base_color[qi]

        # reuse existing u vertex with correct color
        for u in u_vertices:
            # color(u) is determined by first attachment; check compatibility
            # Here we ensure q parity consistent:
            if needed_color == (1 - base_color[qi]):
                u_map2 = dict(u_map)
                u_map2[qi] = u
                backtrack_u(i + 1, u_vertices, u_map2, next_vid)

        # create a new u vertex
        if next_vid < n_cap:
            u_new = next_vid
            u_map2 = dict(u_map)
            u_map2[qi] = u_new
            backtrack_u(i + 1, u_vertices + [u_new], u_map2, next_vid + 1)

    backtrack_u(0, [], {}, 4)

    return list(local_types.values())


# -------------------------
# (B) Completion search to full graphs in Q (within a vertex cap)
# -------------------------

def complete_to_Q(local: LocalType, n_max: int, max_witnesses: int = 1) -> List[nx.Graph]:
    """
    Given a local partial type (with degree deficits = stubs), try to complete it into
    a full cubic bipartite planar 3-connected graph with <= n_max vertices.

    We complete abstractly (ignoring rotation) and then filter by Q.
    This is feasible only because n_max is small (<=14 default).
    """
    G0 = local.to_nx_partial()

    # Extract bipartite colors from local.color
    color = {v: local.color[v] for v in range(local.n)}

    # Ensure current partial is consistent:
    if any(G0.degree(v) > 3 for v in G0.nodes()):
        return []
    if not nx.is_bipartite(G0):
        return []

    # Build list of "open stubs": for each vertex v, we need (3 - deg(v)) new incident edges.
    stubs: List[int] = []
    for v in range(local.n):
        stubs.extend([v] * (3 - G0.degree(v)))

    # Completion backtracking:
    witnesses: List[nx.Graph] = []

    def backtrack(G: nx.Graph, color_map: Dict[int, int], stub_list: List[int], next_vid: int) -> None:
        nonlocal witnesses
        if len(witnesses) >= max_witnesses:
            return
        if next_vid > n_max:
            return
        if not stub_list:
            # full degree satisfied
            if in_Q(G):
                witnesses.append(G.copy())
            return

        # pick first stub endpoint
        v = stub_list[0]
        rest = stub_list[1:]

        # Option 1: connect v to another existing stub endpoint u (pair stubs)
        # choose u from rest where color differs and edge not already present
        seen_u: Set[int] = set()
        for j, u in enumerate(rest):
            if u in seen_u:
                continue
            seen_u.add(u)
            if color_map.get(v, None) is None or color_map.get(u, None) is None:
                continue
            if color_map[v] == color_map[u]:
                continue
            if v == u or G.has_edge(v, u):
                continue
            # degree cap
            if G.degree(v) >= 3 or G.degree(u) >= 3:
                continue

            G.add_edge(v, u)
            new_rest = rest[:j] + rest[j+1:]
            # quick pruning: bipartite + planarity partial
            # check_planarity is relatively expensive, maybe prune every few steps?
            # For n<=14 it's fast enough to check frequently to prune branches.
            if nx.is_bipartite(G) and nx.check_planarity(G, counterexample=False)[0]:
                backtrack(G, color_map, new_rest, next_vid)
            G.remove_edge(v, u)

        # Option 2: add a new vertex w and connect v-w, leaving w with 2 more stubs
        if next_vid < n_max:
            w = next_vid
            G.add_node(w)
            # assign bipartite color
            color_map2 = dict(color_map)
            color_map2[w] = 1 - color_map[v]
            G.add_edge(v, w)

            # w needs 2 more incident edges later
            new_stubs = rest + [w, w]

            if nx.is_bipartite(G) and nx.check_planarity(G, counterexample=False)[0]:
                backtrack(G, color_map2, new_stubs, next_vid + 1)

            # undo
            G.remove_edge(v, w)
            G.remove_node(w)

    # Initialize colors for existing nodes if missing (shouldn't be)
    if set(color.keys()) != set(G0.nodes()):
        # fallback: compute bipartition
        try:
            part = nx.bipartite.color(G0)
            color = {int(k): int(v) for k, v in part.items()}
        except Exception:
            return []

    backtrack(G0.copy(), color, stubs, local.n)
    return witnesses


# -------------------------
# (C) Main driver: enumerate local types, filter extendible, and test completeness
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncap", type=int, default=14, help="max vertices in local type generation")
    ap.add_argument("--nmax", type=int, default=14, help="max vertices in completion to Q")
    ap.add_argument("--out-types", default="artifacts/local_types.jsonl")
    ap.add_argument("--out-extendible", default="artifacts/extendible_witnesses.jsonl")
    ap.add_argument("--out-obstructions", default="artifacts/obstruction_witnesses.jsonl")
    ap.add_argument("--max-witnesses-per-type", type=int, default=1)
    args = ap.parse_args()

    # Ensure output dirs exist
    for fpath in [args.out_types, args.out_extendible, args.out_obstructions]:
        d = os.path.dirname(fpath)
        if d and not os.path.exists(d):
            os.makedirs(d)

    locals_ = generate_radius3_local_types(n_cap=args.ncap)
    print(f"[enumerate] generated {len(locals_)} candidate local types (<= {args.ncap} vertices).")

    # Write all local types
    with open(args.out_types, "w", encoding="utf-8") as f:
        for lt in locals_:
            f.write(json.dumps(asdict(lt)) + "\n")

    extendible = 0
    obstructions = 0

    with open(args.out_extendible, "w", encoding="utf-8") as f_ext, open(args.out_obstructions, "w", encoding="utf-8") as f_obs:
        for idx, lt in enumerate(locals_):
            # Optional: print progress
            if idx % 10 == 0:
                print(f"... processing type {idx+1}/{len(locals_)}", end="\r")

            witnesses = complete_to_Q(lt, n_max=args.nmax, max_witnesses=args.max_witnesses_per_type)
            if not witnesses:
                continue

            extendible += 1

            # Record one extendible witness
            Gw = witnesses[0]
            f_ext.write(json.dumps({
                "local_type_key": lt.canonical_key(),
                "n_local": lt.n,
                "n_witness": Gw.number_of_nodes(),
                "edges_witness": sorted((int(u), int(v)) for u, v in Gw.edges()),
            }) + "\n")

            # Now apply your instance completeness checker on the witness
            try:
                EG = nx_to_embedded(Gw)
                wit = verify_completeness(EG)
                # If we got here, a configuration was found (witness returned)
                failed = False
            except (AssertionError, ValueError, IndexError) as e:
                # verify_completeness raises AssertionError if no config found or invariant broken
                # ValueError might come from embedding failure (though we checked planarity)
                failed = True
                note = str(e)
            except Exception as e:
                failed = True
                note = f"Unexpected error: {str(e)}"

            if failed:
                obstructions += 1
                f_obs.write(json.dumps({
                    "local_type_key": lt.canonical_key(),
                    "n_local": lt.n,
                    "witness_n": Gw.number_of_nodes(),
                    "witness_edges": sorted((int(u), int(v)) for u, v in Gw.edges()),
                    "note": f"verify_completeness failed: {note if 'note' in locals() else 'unknown'}",
                }) + "\n")

    print("") # clear progress line
    print(f"[filter] extendible types (<= {args.nmax} vertices): {extendible}")
    print(f"[result] obstruction witnesses found: {obstructions}")
    print("[done] wrote:")
    print(f"  - {args.out_types}")
    print(f"  - {args.out_extendible}")
    print(f"  - {args.out_obstructions}")


if __name__ == "__main__":
    main()
