import barnette_proof as bp
from barnette_proof import EmbeddedGraph, OccC2, OccC4, OccPinch
import itertools

def find_working_reduction(G, kind, occ):
    if kind == "C2":
        a, b, c, d, e, f = occ.a, occ.b, occ.c, occ.d, occ.e, occ.f
        u1, u4, u5, u6 = occ.u1, occ.u4, occ.u5, occ.u6
        externals = [u1, u4, u5, u6]
        link_map = {u1: a, u4: d, u5: e, u6: f}
        patch_vs = (a, b, c, d, e, f)
        ext_names = ["u1", "u4", "u5", "u6"]
        ext_values = [u1, u4, u5, u6]
    elif kind == "C4":
        v1, v2, v3, v4 = occ.v1, occ.v2, occ.v3, occ.v4
        u1, u2, u3, u4 = occ.u1, occ.u2, occ.u3, occ.u4
        externals = [u1, u2, u3, u4]
        link_map = {u1: v1, u2: v2, u3: v3, u4: v4}
        patch_vs = (v1, v2, v3, v4)
        ext_names = ["u1", "u2", "u3", "u4"]
        ext_values = [u1, u2, u3, u4]
    elif kind == "PINCH":
        v1, v2, v3, v4 = occ.v1, occ.v2, occ.v3, occ.v4
        w, t, r, s = occ.w, occ.t, occ.r, occ.s
        u2, u4 = occ.u2, occ.u4
        externals = [r, s, u2, u4]
        link_map = {r: t, s: t, u2: v2, u4: v4}
        patch_vs = (v1, v2, v3, v4, w, t)
        ext_names = ["r", "s", "u2", "u4"]
        ext_values = [r, s, u2, u4]

    id_to_name = {v: k for k, v in zip(ext_names, ext_values)}

    for x_ext_indices in itertools.combinations(range(4), 2):
        y_ext_indices = [i for i in range(4) if i not in x_ext_indices]
        x_ext = [externals[i] for i in x_ext_indices]
        y_ext = [externals[i] for i in y_ext_indices]
        for rx_permutation in itertools.permutations(x_ext + ["y"]):
            for ry_permutation in itertools.permutations(y_ext + ["x"]):
                try:
                    H = G.copy()
                    x, y = H.next_id, H.next_id + 1
                    H.next_id += 2
                    H.create_empty_vertex(x); H.create_empty_vertex(y)
                    for u in x_ext: H.replace_neighbor(u, link_map[u], x)
                    for u in y_ext: H.replace_neighbor(u, link_map[u], y)
                    for vv in patch_vs:
                        if vv in H.adj: del H.adj[vv]; del H.rot[vv]; del H.pos[vv]
                    rx = [v if v != "y" else y for v in rx_permutation]
                    ry = [v if v != "x" else x for v in ry_permutation]
                    H.set_vertex_rotation(x, rx)
                    H.set_vertex_rotation(y, ry)
                    H.validate_rotation_embedding()
                    res = {
                        "kind": kind,
                        "x_externals": [ext_names[i] for i in x_ext_indices],
                        "y_externals": [ext_names[i] for i in y_ext_indices],
                        "x_rot": [id_to_name.get(v, v) for v in rx_permutation],
                        "y_rot": [id_to_name.get(v, v) for v in ry_permutation]
                    }
                    return res
                except Exception: pass
    return None

if __name__ == "__main__":
    results = []
    # C2
    res = find_working_reduction(bp.make_prism(8), "C2", bp.detect_C2(bp.make_prism(8)))
    if res: results.append(res)
    # C4
    G4 = bp.expand_refined_C4_from_edge(bp.make_cube(), 0, 1)
    res = find_working_reduction(G4, "C4", bp.detect_refined_C4(G4))
    if res: results.append(res)
    # PINCH
    GP = bp.make_custom_pinch_example()
    res = find_working_reduction(GP, "PINCH", bp.detect_C_pinch_ii(GP))
    if res: results.append(res)
    
    with open("working_orientations.txt", "w") as f:
        for r in results:
            f.write(str(r) + "\n")
