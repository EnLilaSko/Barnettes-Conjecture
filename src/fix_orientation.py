import barnette_proof as bp
from barnette_proof import EmbeddedGraph, OccC2, OccC4, OccPinch
import itertools

def is_bipartite(G):
    if not G.adj: return True
    color = {}
    nodes = list(G.adj.keys())
    while nodes:
        start_node = nodes.pop()
        if start_node in color: continue
        stack = [(start_node, 0)]
        while stack:
            v, c = stack.pop()
            if v in color:
                if color[v] != c: return False
                continue
            color[v] = c
            for u in G.adj[v]:
                stack.append((u, 1 - c))
    return True

def test_bipartite_orientations(G, kind, occ):
    print(f"Testing {kind} bipartite orientations...")
    if kind == "C2":
        a, b, c, d, e, f = occ.a, occ.b, occ.c, occ.d, occ.e, occ.f
        u1, u4, u5, u6 = occ.u1, occ.u4, occ.u5, occ.u6
        x_ext, y_ext = [u1, u6], [u4, u5]
        link_map = {u1: a, u6: f, u4: d, u5: e}
        patch_vs = (a, b, c, d, e, f)
        id_to_name = {u1: "u1", u4: "u4", u5: "u5", u6: "u6"}
    elif kind == "C4":
        v1, v2, v3, v4 = occ.v1, occ.v2, occ.v3, occ.v4
        u1, u2, u3, u4 = occ.u1, occ.u2, occ.u3, occ.u4
        x_ext, y_ext = [u1, u3], [u2, u4]
        link_map = {u1: v1, u3: v3, u2: v2, u4: v4}
        patch_vs = (v1, v2, v3, v4)
        id_to_name = {u1: "u1", u2: "u2", u3: "u3", u4: "u4"}
    else: # PINCH
        v1, v2, v3, v4 = occ.v1, occ.v2, occ.v3, occ.v4
        w, t, r, s = occ.w, occ.t, occ.r, occ.s
        u2, u4 = occ.u2, occ.u4
        x_ext, y_ext = [r, s], [u2, u4]
        link_map = {r: t, s: t, u2: v2, u4: v4}
        patch_vs = (v1, v2, v3, v4, w, t)
        id_to_name = {r: "r", s: "s", u2: "u2", u4: "u4"}

    for rx_p in itertools.permutations(x_ext + ["y"]):
        for ry_p in itertools.permutations(y_ext + ["x"]):
            try:
                H = G.copy()
                x, y = H.next_id, H.next_id + 1
                H.next_id += 2
                H.create_empty_vertex(x); H.create_empty_vertex(y)
                for u in x_ext: H.replace_neighbor(u, link_map[u], x)
                for u in y_ext: H.replace_neighbor(u, link_map[u], y)
                for vv in patch_vs:
                    if vv in H.adj: del H.adj[vv]; del H.rot[vv]; del H.pos[vv]
                H.set_vertex_rotation(x, [v if v != "y" else y for v in rx_p])
                H.set_vertex_rotation(y, [v if v != "x" else x for v in ry_p])
                H.validate_rotation_embedding()
                if is_bipartite(H):
                    print(f"  SUCCESS for {kind}!")
                    print(f"  x_rot: {[id_to_name.get(v, v) for v in rx_p]}")
                    print(f"  y_rot: {[id_to_name.get(v, v) for v in ry_p]}")
                    return True
            except Exception: pass
    print(f"  FAILED for {kind}")
    return False

if __name__ == "__main__":
    G8 = bp.make_prism(8)
    test_bipartite_orientations(G8, "C2", bp.detect_C2(G8))
    
    G_cube = bp.make_cube()
    # Cube: v1=0, v2=1, v3=3, v4=2. u1=4, u2=5, u3=7, u4=6.
    # We force detection or hardcode OccC4
    occ4 = bp.OccC4(v1=0, v2=1, v3=3, v4=2, u1=4, u2=5, u3=7, u4=6)
    test_bipartite_orientations(G_cube, "C4", occ4)

    GP = bp.make_custom_pinch_example()
    occp = bp.detect_C_pinch_ii(GP)
    test_bipartite_orientations(GP, "PINCH", occp)
