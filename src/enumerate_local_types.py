#!/usr/bin/env python3
"""
Enumerate radius-3 rotation-system neighborhoods of a facial 4-cycle
compatible with cubicity + bipartiteness, and filter those extendible
to 3-connected instances (within N_max).

Outputs:
  - artifacts/local_types.jsonl
  - artifacts/extendible_witnesses.jsonl
  - artifacts/obstruction_witnesses.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Set, Optional

import networkx as nx

# -------------------------
# Adapt these imports to your repo
# -------------------------
try:
    from barnette_proof import EmbeddedGraph, verify_completeness
except ImportError:
    try:
        from src.barnette_proof import EmbeddedGraph, verify_completeness
    except ImportError:
        import sys
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from barnette_proof import EmbeddedGraph, verify_completeness


# -------------------------
# Basic predicates
# -------------------------

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

def nx_to_embedded(G: nx.Graph) -> EmbeddedGraph:
    """
    Convert a NetworkX graph into your EmbeddedGraph using a planar embedding
    returned by nx.check_planarity. For 3-connected planar graphs, embedding is
    unique up to reflection, so any returned embedding is acceptable.
    """
    ok, embedding = nx.check_planarity(G)
    if not ok:
        raise ValueError("Graph is not planar, cannot convert to EmbeddedGraph")

    adj: Dict[int, Set[int]] = {}
    rot: Dict[int, List[int]] = {}
    for v in G.nodes():
        adj[v] = set(G.neighbors(v))
        rot[v] = list(embedding.neighbors(v))
    return EmbeddedGraph(adj, rot)

def cyclic_equal(a: List[int], b: List[int]) -> bool:
    """Equality up to cyclic shift."""
    if len(a) != len(b):
        return False
    if not a:
        return True
    s = a + a
    for i in range(len(a)):
        if s[i:i+len(a)] == b:
            return True
    return False

def face_is_root(face: List[int], root: Tuple[int,int,int,int]) -> bool:
    """Check whether `face` equals root cycle up to rotation and reversal."""
    if len(face) != 4:
        return False
    r = list(root)
    return cyclic_equal(face, r) or cyclic_equal(face, list(reversed(r)))

def has_facial_root_cycle(G: nx.Graph, root: Tuple[int,int,int,int]) -> bool:
    """
    Verify the 4-cycle `root` appears as a facial cycle in *some* planar embedding
    returned by nx.check_planarity. For 3-connected planar graphs this is fine.
    """
    ok, emb = nx.check_planarity(G, counterexample=False)
    if not ok:
        return False

    # Iterate directed edges and traverse faces.
    seen_darts: Set[Tuple[int,int]] = set()
    for u, v in G.edges():
        for a, b in [(u, v), (v, u)]:
            if (a, b) in seen_darts:
                continue
            try:
                f = emb.traverse_face(a, b)
            except Exception:
                continue
            # mark all darts on this face as seen
            for i in range(len(f)):
                x, y = f[i], f[(i+1) % len(f)]
                seen_darts.add((x, y))
            if face_is_root(f, root):
                return True
    return False


# -------------------------
# Data model: local neighborhood with stubs
# -------------------------

@dataclass(frozen=True)
class LocalType:
    n: int
    edges: List[Tuple[int, int]]
    root_face: Tuple[int, int, int, int]
    stubs: List[Tuple[int, int]]
    color: List[int]

    def to_nx_partial(self) -> nx.Graph:
        G = nx.Graph()
        G.add_nodes_from(range(self.n))
        G.add_edges_from(self.edges)
        return G

    def canonical_key(self) -> str:
        ed = sorted((min(u, v), max(u, v)) for u, v in self.edges)
        return json.dumps({"root": self.root_face, "edges": ed, "color": self.color}, separators=(",", ":"))


# -------------------------
# (A) Generate local radius-3 neighborhoods (partial)
# -------------------------

def generate_radius3_local_types(n_cap: int) -> List[LocalType]:
    """
    Enumerate partial neighborhoods around a rooted 4-face (q0q1q2q3),
    under subcubic + bipartite constraints, allowing *horizontal edges*
    between same-layer vertices (this is essential to include the Cube).

    This is a conservative generator: it explores small partial graphs with
    stubs and deduplicates them by a rooted key. It does not claim to
    enumerate all combinatorial maps; it enumerates the local "types" used
    by the manuscript's computational Lemma 41 pipeline.
    """

    q0, q1, q2, q3 = 0, 1, 2, 3
    root = (q0, q1, q2, q3)

    base_color = {q0: 0, q1: 1, q2: 0, q3: 1}
    base_edges = {(0, 1), (1, 2), (2, 3), (0, 3)}

    local_types: Dict[str, LocalType] = {}

    def add_edge(edges: Set[Tuple[int, int]], u: int, v: int) -> None:
        if u == v:
            return
        a, b = (u, v) if u < v else (v, u)
        edges.add((a, b))

    def build_from_assignment(u_map: Dict[int, int], next_vid: int) -> None:
        edges = set(base_edges)
        color = dict(base_color)

        # add attachments qi-ui
        for qi, ui in u_map.items():
            if ui not in color:
                color[ui] = 1 - color[qi]
            if color[ui] != 1 - color[qi]:
                return
            add_edge(edges, qi, ui)

        # early prune: subcubic + bipartite
        G = nx.Graph()
        G.add_nodes_from(range(next_vid))
        G.add_edges_from(edges)
        if any(G.degree(v) > 3 for v in G.nodes()):
            return
        if not nx.is_bipartite(G):
            return

        dist1 = sorted(set(u_map.values()))

        # each dist1 vertex needs two more incident edges (unless it already merged)
        slots: List[int] = []
        for ui in dist1:
            need = 3 - G.degree(ui)
            if need < 0:
                return
            slots.extend([ui] * need)

        def backtrack_slots(i: int, edges2: Set[Tuple[int, int]], color2: Dict[int, int], next_id: int) -> None:
            if next_id > n_cap:
                return

            # rebuild graph for degree checks (small, OK)
            Gtmp = nx.Graph()
            Gtmp.add_nodes_from(range(next_id))
            Gtmp.add_edges_from(edges2)
            if any(Gtmp.degree(v) > 3 for v in Gtmp.nodes()):
                return
            if not nx.is_bipartite(Gtmp):
                return

            if i == len(slots):
                # record as a LocalType with stubs = remaining deficits everywhere
                stubs: List[Tuple[int, int]] = []
                for v in range(next_id):
                    deficit = 3 - Gtmp.degree(v)
                    for k in range(deficit):
                        stubs.append((v, k))

                edge_list = sorted((min(a, b), max(a, b)) for a, b in edges2)
                color_list = [color2[v] for v in range(next_id)]
                lt = LocalType(
                    n=next_id,
                    edges=edge_list,
                    root_face=root,
                    stubs=stubs,
                    color=color_list,
                )
                local_types[lt.canonical_key()] = lt
                return

            ui = slots[i]
            needed_color = 1 - color2[ui]

            # Candidate targets: ANY existing vertex (including other dist1 vertices),
            # as long as it matches bipartite color and has free degree.
            for v in range(next_id):
                if v == ui:
                    continue
                if color2.get(v, None) is None:
                    continue
                if color2[v] != needed_color:
                    continue
                a, b = (ui, v) if ui < v else (v, ui)
                if (a, b) in edges2:
                    continue
                if Gtmp.degree(ui) >= 3 or Gtmp.degree(v) >= 3:
                    continue

                edges_next = set(edges2)
                edges_next.add((a, b))
                backtrack_slots(i + 1, edges_next, color2, next_id)

            # Option: create new vertex and attach ui-new
            if next_id < n_cap:
                v = next_id
                color_next = dict(color2)
                color_next[v] = needed_color
                edges_next = set(edges2)
                add_edge(edges_next, ui, v)
                backtrack_slots(i + 1, edges_next, color_next, next_id + 1)

        backtrack_slots(0, edges, color, next_vid)

    # enumerate possible identifications for the four distance-1 neighbors
    def backtrack_u(i: int, u_vertices: List[int], u_map: Dict[int, int], next_vid: int) -> None:
        qs = [q0, q1, q2, q3]
        if i == 4:
            build_from_assignment(u_map, next_vid)
            return
        qi = qs[i]

        # reuse existing
        for u in u_vertices:
            u_map2 = dict(u_map)
            u_map2[qi] = u
            backtrack_u(i + 1, u_vertices, u_map2, next_vid)

        # create new
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

def complete_to_Q(local: LocalType, n_max: int, max_witnesses: int = 1,
                  require_facial_root: bool = True) -> List[nx.Graph]:
    """
    Complete a local partial type to a full graph in Q with <= n_max vertices.
    Exhaustive backtracking for small n_max (<=14).
    """
    G0 = local.to_nx_partial()
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
        if len(witnesses) >= max_witnesses:
            return
        if next_vid > n_max:
            return

        if not stub_list:
            if in_Q(G):
                if (not require_facial_root) or has_facial_root_cycle(G, local.root_face):
                    witnesses.append(G.copy())
            return

        v = stub_list[0]
        rest = stub_list[1:]

        # pair v with some other existing stub endpoint u
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
            ok = nx.is_bipartite(G) and nx.check_planarity(G, counterexample=False)[0]
            if ok:
                new_rest = rest[:j] + rest[j+1:]
                backtrack(G, color_map, new_rest, next_vid)
            G.remove_edge(v, u)

        # create new vertex w and connect v-w
        if next_vid < n_max:
            w = next_vid
            G.add_node(w)
            color_map2 = dict(color_map)
            color_map2[w] = 1 - color_map[v]
            G.add_edge(v, w)

            ok = nx.is_bipartite(G) and nx.check_planarity(G, counterexample=False)[0]
            if ok:
                backtrack(G, color_map2, rest + [w, w], next_vid + 1)

            G.remove_edge(v, w)
            G.remove_node(w)

    backtrack(G0.copy(), color, stubs, local.n)
    return witnesses


# -------------------------
# (C) Main driver
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncap", type=int, default=14)
    ap.add_argument("--nmax", type=int, default=14)
    ap.add_argument("--out-types", default="artifacts/local_types.jsonl")
    ap.add_argument("--out-extendible", default="artifacts/extendible_witnesses.jsonl")
    ap.add_argument("--out-obstructions", default="artifacts/obstruction_witnesses.jsonl")
    ap.add_argument("--max-witnesses-per-type", type=int, default=1)
    ap.add_argument("--no-facial-root-check", action="store_true",
                    help="Do not require the rooted 4-cycle to be facial in completions.")
    args = ap.parse_args()

    for fpath in [args.out_types, args.out_extendible, args.out_obstructions]:
        d = os.path.dirname(fpath)
        if d and not os.path.exists(d):
            os.makedirs(d)

    locals_ = generate_radius3_local_types(n_cap=args.ncap)
    print(f"[enumerate] generated {len(locals_)} candidate local types (<= {args.ncap} vertices).")

    with open(args.out_types, "w", encoding="utf-8") as f:
        for lt in locals_:
            f.write(json.dumps(asdict(lt)) + "\n")

    extendible = 0
    obstructions = 0

    require_facial_root = not args.no_facial_root_check

    with open(args.out_extendible, "w", encoding="utf-8") as f_ext, \
         open(args.out_obstructions, "w", encoding="utf-8") as f_obs:

        for idx, lt in enumerate(locals_):
            if idx % 20 == 0:
                print(f"... processing type {idx+1}/{len(locals_)}", end="\r")

            witnesses = complete_to_Q(
                lt,
                n_max=args.nmax,
                max_witnesses=args.max_witnesses_per_type,
                require_facial_root=require_facial_root,
            )
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

            # Run your completeness / admissible-config finder on the witness.
            # Support either API: verify_completeness(nx.Graph) OR verify_completeness(EmbeddedGraph).
            failed = False
            note = ""
            try:
                try:
                    EG = nx_to_embedded(Gw)
                    _ = verify_completeness(EG)
                except TypeError:
                    _ = verify_completeness(Gw)
            except Exception as e:
                failed = True
                note = str(e)

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
