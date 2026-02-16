
import networkx as nx
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Set, Tuple, Optional, Iterable

def _root_cycle() -> nx.Graph:
    G0 = nx.Graph()
    G0.add_nodes_from([0, 1, 2, 3])
    G0.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
    return G0

def _all_stubs(G: nx.Graph, dist: Dict[int, int]) -> List[int]:
    stubs: List[int] = []
    for v in range(G.number_of_nodes()):
        if dist.get(v, 99) <= 2:
            need = 3 - G.degree(v)
            if need > 0:
                stubs.extend([v] * need)
    return stubs

def _is_planar_with_embedding(G: nx.Graph):
    ok, emb = nx.check_planarity(G, counterexample=False)
    return ok, emb

def _root_quad_is_face(G: nx.Graph, emb: nx.PlanarEmbedding) -> bool:
    try:
        face = emb.traverse_face(0, 1)
        # print(f"DEBUG: face from 0-1 is {face}")
        return face == [0, 1, 2, 3]
    except Exception as e:
        # print(f"DEBUG: traverse_face error: {e}")
        return False

def backtrack(G: nx.Graph, color: Dict[int, int], dist: Dict[int, int], next_vid: int, depth: int):
    stubs = _all_stubs(G, dist)
    if not stubs:
        ok, emb = _is_planar_with_embedding(G)
        if not ok or emb is None: return
        if not _root_quad_is_face(G, emb): return
        print(f"FOUND TYPE: n={next_vid}, edges={list(G.edges())}")
        return

    if depth > 12: return # limit depth for diag

    v = stubs[0]
    rest = stubs[1:]

    # Option 1
    seen_u = set()
    for u in rest:
        if u in seen_u: continue
        seen_u.add(u)
        if v == u or G.has_edge(v, u): continue
        if color[v] == color[u]: continue
        if G.degree(v) >= 3 or G.degree(u) >= 3: continue

        G.add_edge(v, u)
        ok, _ = _is_planar_with_embedding(G)
        if ok and nx.is_bipartite(G):
            backtrack(G, color, dist, next_vid, depth + 1)
        G.remove_edge(v, u)

    # Option 2
    if next_vid < 12: # limit ncap for diag
        w = next_vid
        G.add_node(w)
        color[w] = 1 - color[v]
        dist[w] = dist[v] + 1
        G.add_edge(v, w)
        if dist[w] <= 3:
            ok, _ = _is_planar_with_embedding(G)
            if ok and nx.is_bipartite(G):
                backtrack(G, color, dist, next_vid + 1, depth + 1)
        G.remove_edge(v, w)
        G.remove_node(w)
        del color[w]
        del dist[w]

if __name__ == "__main__":
    base = _root_cycle()
    color = {0: 0, 1: 1, 2: 0, 3: 1}
    dist = {0: 0, 1: 0, 2: 0, 3: 0}
    backtrack(base, color, dist, 4, 0)
