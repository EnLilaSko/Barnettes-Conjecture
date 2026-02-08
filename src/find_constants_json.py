import barnette_proof as bp
from barnette_proof import EmbeddedGraph, OccC2, OccC4, OccPinch
import itertools
import json

def get_success(G, kind, occ):
    if kind == "C2":
        a, b, c, d, e, f = occ.a, occ.b, occ.c, occ.d, occ.e, occ.f
        u1, u4, u5, u6 = occ.u1, occ.u4, occ.u5, occ.u6
        externals = [u1, u4, u5, u6]
        link_map = {u1: a, u4: d, u5: e, u6: f}
        patch_vs = (a, b, c, d, e, f)
        ext_names = ["u1", "u4", "u5", "u6"]
    elif kind == "C4":
        v1, v2, v3, v4 = occ.v1, occ.v2, occ.v3, occ.v4
        u1, u2, u3, u4 = occ.u1, occ.u2, occ.u3, occ.u4
        externals = [u1, u2, u3, u4]
        link_map = {u1: v1, u2: v2, u3: v3, u4: v4}
        patch_vs = (v1, v2, v3, v4)
        ext_names = ["u1", "u2", "u3", "u4"]
    else: # PINCH
        v1, v2, v3, v4 = occ.v1, occ.v2, occ.v3, occ.v4
        w, t, r, s = occ.w, occ.t, occ.r, occ.s
        u2, u4 = occ.u2, occ.u4
        externals = [r, s, u2, u4]
        link_map = {r: t, s: t, u2: v2, u4: v4}
        patch_vs = (v1, v2, v3, v4, w, t)
        ext_names = ["r", "s", "u2", "u4"]

    for x_idx in itertools.combinations(range(4), 2):
        y_idx = [i for i in range(4) if i not in x_idx]
        for rx_p in itertools.permutations(list(x_idx) + ["y"]):
            for ry_p in itertools.permutations(list(y_idx) + ["x"]):
                try:
                    H = G.copy()
                    x, y = H.next_id, H.next_id + 1
                    H.next_id += 2
                    H.create_empty_vertex(x); H.create_empty_vertex(y)
                    for i in x_idx: H.replace_neighbor(externals[i], link_map[externals[i]], x)
                    for i in y_idx: H.replace_neighbor(externals[i], link_map[externals[i]], y)
                    for vv in patch_vs:
                        if vv in H.adj: del H.adj[vv]; del H.rot[vv]; del H.pos[vv]
                    H.set_vertex_rotation(x, [externals[v] if v != "y" else y for v in rx_p])
                    H.set_vertex_rotation(y, [externals[v] if v != "x" else x for v in ry_p])
                    H.validate_rotation_embedding()
                    return {
                        "kind": kind,
                        "x_ext": [ext_names[i] for i in x_idx],
                        "y_ext": [ext_names[i] for i in y_idx],
                        "x_rot": [ext_names[i] if i != "y" else "y" for i in rx_p],
                        "y_rot": [ext_names[j] if j != "x" else "x" for j in ry_p]
                    }
                except Exception: pass
    return None

if __name__ == "__main__":
    results = {}
    try:
        # C2
        G = bp.make_prism(8)
        results["C2"] = get_success(G, "C2", bp.detect_C2(G))
        # PINCH
        GP = bp.make_custom_pinch_example()
        results["PINCH"] = get_success(GP, "PINCH", bp.detect_C_pinch_ii(GP))
    except Exception as e:
        results["error"] = str(e)
    
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, "reduction_params.json")
    
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote results to {out_path}")
