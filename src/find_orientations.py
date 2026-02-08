import barnette_proof as bp
from barnette_proof import EmbeddedGraph, OccC2, OccC4, OccPinch
import itertools
import json

def is_bipartite(G):
    color = {}
    for node in G.adj:
        if node not in color:
            stack = [(node, 0)]
            while stack:
                v, c = stack.pop()
                if v in color:
                    if color[v] != c: return False
                    continue
                color[v] = c
                for u in G.adj[v]: stack.append((u, 1 - c))
    return True

def find_reduction(G, kind, occ):
    if kind == "C2":
        a, b, c, d, e, f = occ.a, occ.b, occ.c, occ.d, occ.e, occ.f
        u1, u4, u5, u6 = occ.u1, occ.u4, occ.u5, occ.u6
        externals = [u1, u4, u5, u6]
        link_map = {u1: a, u4: d, u5: e, u6: f}
        patch_vs = (a, b, c, d, e, f)
        names = {u1: "u1", u4: "u4", u5: "u5", u6: "u6"}
    elif kind == "C4":
        v1, v2, v3, v4 = occ.v1, occ.v2, occ.v3, occ.v4
        u1, u2, u3, u4 = occ.u1, occ.u2, occ.u3, occ.u4
        externals = [u1, u2, u3, u4]
        link_map = {u1: v1, u2: v2, u3: v3, u4: v4}
        patch_vs = (v1, v2, v3, v4)
        names = {u1: "u1", u2: "u2", u3: "u3", u4: "u4"}
    elif kind == "PINCH":
        v1, v2, v3, v4 = occ.v1, occ.v2, occ.v3, occ.v4
        w, t, r, s = occ.w, occ.t, occ.r, occ.s
        u2, u4 = occ.u2, occ.u4
        externals = [r, s, u2, u4]
        link_map = {r: t, s: t, u2: v2, u4: v4}
        patch_vs = (v1, v2, v3, v4, w, t)
        names = {r: "r", s: "s", u2: "u2", u4: "u4"}

    for x_indices in itertools.combinations(range(len(externals)), 2):
        y_indices = [i for i in range(len(externals)) if i not in x_indices]
        x_ext = [externals[i] for i in x_indices]
        y_ext = [externals[i] for i in y_indices]
        
        for rx_p in itertools.permutations(list(x_ext) + ["y"]):
            for ry_p in itertools.permutations(list(y_ext) + ["x"]):
                try:
                    H = G.copy()
                    x, y = H.next_id, H.next_id + 1
                    H.next_id += 2
                    H.create_empty_vertex(x)
                    H.create_empty_vertex(y)
                    for u in x_ext: H.replace_neighbor(u, link_map[u], x)
                    for u in y_ext: H.replace_neighbor(u, link_map[u], y)
                    for vv in patch_vs:
                        if vv in H.adj: del H.adj[vv], H.rot[vv], H.pos[vv]
                    
                    H.set_vertex_rotation(x, [v if v != "y" else y for v in rx_p])
                    H.set_vertex_rotation(y, [v if v != "x" else x for v in ry_p])
                    H.validate_rotation_embedding()
                    if is_bipartite(H):
                        return {
                            "x_ext": [names[u] for u in x_ext],
                            "y_ext": [names[u] for u in y_ext],
                            "x_rot": [names.get(v, v) for v in rx_p],
                            "y_rot": [names.get(v, v) for v in ry_p]
                        }
                except: pass
    return None

if __name__ == "__main__":
    final_results = {}
    
    # C2
    G8 = bp.make_prism(8)
    occ2 = bp.detect_C2(G8)
    if occ2: final_results["C2"] = find_reduction(G8, "C2", occ2)
    
    # C4
    G_base = bp.make_prism(6)
    G_c4 = bp.expand_refined_C4_from_edge(G_base, 0, 1)
    occ4 = bp.detect_refined_C4(G_c4)
    if occ4: final_results["C4"] = find_reduction(G_c4, "C4", occ4)
    
    # PINCH
    GP = bp.make_custom_pinch_example()
    occp = bp.detect_C_pinch_ii(GP)
    if occp: final_results["PINCH"] = find_reduction(GP, "PINCH", occp)
    
    with open("reduction_data.json", "w") as f:
        json.dump(final_results, f, indent=2)
