"""Produce admissibility witnesses for each rooted local type.

Goal
----
For each rooted radius-3 local type around a facial 4-cycle (the `root_face`),
choose a certified configuration (C2 / pinch(ii) / refined C4) and test
admissibility using a bounded-radius separator scan inspired by Lemma 52.

This script is intentionally *rotation-system free*: it only uses the abstract
graph + the distinguished rooted 4-cycle. This matches the lightweight JSON
local-type artifacts currently stored in `artifacts/local_types.jsonl`.

Outputs
-------
Writes two JSONL files:

  - admissibility_witnesses.jsonl: one record per local type where an admissible
    configuration is found.
  - admissibility_obstructions.jsonl: one record per local type where all
    candidate configurations fail the local separator scan.

Notes
-----
This is a *mechanical* helper for the proof pipeline. It is not intended to be
the final, referee-facing completeness argument on its own.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


# ----------------------------
# Minimal undirected graph API
# ----------------------------


class UG:
    """Simple undirected graph on vertices 0..n-1 with adjacency sets."""

    def __init__(self, n: int, edges: Iterable[Tuple[int, int]]):
        self.n = int(n)
        self.adj: List[Set[int]] = [set() for _ in range(self.n)]
        for u, v in edges:
            self.add_edge(u, v)

    def copy(self) -> "UG":
        g = UG(self.n, [])
        g.adj = [set(nei) for nei in self.adj]
        return g

    def add_vertex(self) -> int:
        self.adj.append(set())
        self.n += 1
        return self.n - 1

    def add_edge(self, u: int, v: int) -> None:
        if u == v:
            raise ValueError("loop edge")
        if not (0 <= u < self.n and 0 <= v < self.n):
            raise ValueError("vertex out of range")
        self.adj[u].add(v)
        self.adj[v].add(u)

    def remove_edge(self, u: int, v: int) -> None:
        self.adj[u].discard(v)
        self.adj[v].discard(u)

    def remove_vertex(self, v: int) -> None:
        # Remove incident edges; keep vertex id present (tombstone).
        for u in list(self.adj[v]):
            self.adj[u].discard(v)
        self.adj[v].clear()

    def neighbors(self, v: int) -> Set[int]:
        return self.adj[v]

    def has_edge(self, u: int, v: int) -> bool:
        return v in self.adj[u]

    def vertices(self) -> List[int]:
        return list(range(self.n))

    def degree(self, v: int) -> int:
        return len(self.adj[v])


# ----------------------------
# Root-face based configuration detection
# ----------------------------


@dataclass(frozen=True)
class OccC4Plain:
    v: Tuple[int, int, int, int]   # root 4-cycle vertices
    u: Tuple[int, int, int, int]   # outside neighbors (third neighbors of v[i])


@dataclass(frozen=True)
class OccPinchPlain:
    v: Tuple[int, int, int, int]
    w: int
    u2: int
    u4: int
    t: int
    r: int
    s: int


@dataclass(frozen=True)
class OccC2Plain:
    # Naming follows barnette_proof.reduce_C2(): remove a,b,c,d,e,f; terminals u1,u4,u5,u6.
    a: int
    b: int
    c: int
    d: int
    e: int
    f: int
    u1: int
    u4: int
    u5: int
    u6: int


def _cycle_neighbors(root: List[int], i: int) -> Tuple[int, int]:
    n = len(root)
    return root[(i - 1) % n], root[(i + 1) % n]


def outside_neighbor(g: UG, root: List[int], i: int) -> int:
    v = root[i]
    p, q = _cycle_neighbors(root, i)
    outside = [x for x in g.neighbors(v) if x not in (p, q)]
    if len(outside) != 1:
        raise ValueError(f"root vertex {v} does not have a unique outside neighbor")
    return outside[0]


def find_adjacent_quad_occurrences(g: UG, root: List[int]) -> List[OccC2Plain]:
    """Detect a C2 occurrence using only adjacency around the rooted 4-cycle.

    In a cubic embedding, the face across edge (v_i, v_{i+1}) is a 4-face iff
    the outside neighbors u_i and u_{i+1} are adjacent.

    We then canonicalize to match the (a,b,c,d,e,f) naming used by reduce_C2.
    """
    v = list(root)
    u = [outside_neighbor(g, v, i) for i in range(4)]

    # Try each root edge as the shared edge.
    out: List[OccC2Plain] = []
    for i in range(4):
        b = v[i]
        c = v[(i + 1) % 4]
        f = u[i]
        e = u[(i + 1) % 4]
        if not g.has_edge(e, f):
            continue

        # Left quad L is root oriented so that it is (a,b,c,d) with shared edge b-c.
        a = v[(i - 1) % 4]
        d = v[(i + 2) % 4]

        # Determine terminals u1,u4,u5,u6 as in barnette_proof._canonicalize_adjacent_quads.
        # u1 is the neighbor of a not in {b,d}.
        u1 = [x for x in g.neighbors(a) if x not in {b, d}]
        if len(u1) != 1:
            continue
        u1 = u1[0]

        u4 = [x for x in g.neighbors(d) if x not in {a, c}]
        if len(u4) != 1:
            continue
        u4 = u4[0]

        u5 = [x for x in g.neighbors(e) if x not in {c, f}]
        if len(u5) != 1:
            continue
        u5 = u5[0]

        u6 = [x for x in g.neighbors(f) if x not in {b, e}]
        if len(u6) != 1:
            continue
        u6 = u6[0]

        # Basic sanity checks: ensure the two quads are distinct except for the shared edge.
        removed = {a, b, c, d, e, f}
        if len(removed) < 6:
            continue
        out.append(OccC2Plain(a=a, b=b, c=c, d=d, e=e, f=f, u1=u1, u4=u4, u5=u5, u6=u6))

    return out


def find_pinch_occurrence(g: UG, root: List[int]) -> Optional[OccPinchPlain]:
    v = list(root)
    u = [outside_neighbor(g, v, i) for i in range(4)]

    # pinch(ii): isolated 4-face (no adjacent quads) + one opposite identification.
    if any(g.has_edge(u[i], u[(i + 1) % 4]) for i in range(4)):
        return None

    # Identify which opposite pair merges.
    if u[0] == u[2]:
        w = u[0]
        u2, u4 = u[1], u[3]
        v1, v3 = v[0], v[2]
    elif u[1] == u[3]:
        w = u[1]
        u2, u4 = u[2], u[0]
        v1, v3 = v[1], v[3]
    else:
        return None

    # t is the third neighbor of w distinct from the two root vertices it touches.
    tw = [x for x in g.neighbors(w) if x not in {v1, v3}]
    if len(tw) != 1:
        return None
    t = tw[0]

    rs = [x for x in g.neighbors(t) if x != w]
    if len(rs) != 2:
        return None
    r, s = rs
    return OccPinchPlain(v=tuple(v), w=w, u2=u2, u4=u4, t=t, r=r, s=s)


def find_refined_c4_occurrence(g: UG, root: List[int]) -> Optional[OccC4Plain]:
    v = list(root)
    u = [outside_neighbor(g, v, i) for i in range(4)]

    # refined C4 requires isolated 4-face and no opposite identification.
    if any(g.has_edge(u[i], u[(i + 1) % 4]) for i in range(4)):
        return None
    if u[0] == u[2] or u[1] == u[3]:
        return None
    if len(set(u)) != 4:
        return None

    return OccC4Plain(v=tuple(v), u=tuple(u))


# ----------------------------
# Reductions (adjacency-only)
# ----------------------------


def reduce_refined_c4_plain(g0: UG, occ: OccC4Plain) -> Tuple[UG, Set[int], Set[int], Set[int]]:
    """Adjacency-only version of reduce_C4() from barnette_proof.

    Returns (reduced_graph, boundary_terminals).
    """
    g = g0.copy()
    v1, v2, v3, v4 = occ.v
    u1, u2, u3, u4 = occ.u

    # Create gadget vertices x-y.
    x = g.add_vertex()
    y = g.add_vertex()
    added = {x, y}
    g.add_edge(x, y)
    g.add_edge(u1, x)
    g.add_edge(u3, x)
    g.add_edge(u2, y)
    g.add_edge(u4, y)

    # Remove root 4-cycle vertices.
    removed = {v1, v2, v3, v4}
    for vv in removed:
        g.remove_vertex(vv)
    return g, {u1, u2, u3, u4}, removed, added


def reduce_pinch_plain(g0: UG, occ: OccPinchPlain) -> Tuple[UG, Set[int], Set[int], Set[int]]:
    """Adjacency-only version of reduce_pinchii() from barnette_proof."""
    g = g0.copy()
    v1, v2, v3, v4 = occ.v
    w, u2, u4, t, r, s = occ.w, occ.u2, occ.u4, occ.t, occ.r, occ.s

    x = g.add_vertex()
    y = g.add_vertex()
    added = {x, y}
    g.add_edge(x, y)
    g.add_edge(u2, x)
    g.add_edge(u4, x)
    g.add_edge(r, y)
    g.add_edge(s, y)

    removed = {v1, v2, v3, v4, w, t}
    for vv in removed:
        g.remove_vertex(vv)
    return g, {u2, u4, r, s}, removed, added


def reduce_c2_plain(g0: UG, occ: OccC2Plain) -> Tuple[UG, Set[int], Set[int], Set[int]]:
    """Adjacency-only version of reduce_C2() from barnette_proof."""
    g = g0.copy()
    a, b, c, d, e, f = occ.a, occ.b, occ.c, occ.d, occ.e, occ.f
    u1, u4, u5, u6 = occ.u1, occ.u4, occ.u5, occ.u6

    x = g.add_vertex()
    y = g.add_vertex()
    added = {x, y}
    g.add_edge(x, y)
    g.add_edge(u1, x)
    g.add_edge(u6, x)
    g.add_edge(u4, y)
    g.add_edge(u5, y)

    removed = {a, b, c, d, e, f}
    for vv in removed:
        g.remove_vertex(vv)
    return g, {u1, u4, u5, u6}, removed, added


# ----------------------------
# Lemma-52-style bounded-radius separator scan
# ----------------------------


def closure_2(g: UG, B: Set[int], *, forbid: Optional[Set[int]] = None) -> Set[int]:
    """Return S = B ∪ N(B) ∪ N^2(B) in g.

    If `forbid` is provided, those vertices are not added and not expanded.
    """
    forbid = forbid or set()
    S = set(B) - forbid
    for b in list(S):
        for y in g.neighbors(b):
            if y not in forbid:
                S.add(y)
    # distance-2
    for x in list(S):
        for y in g.neighbors(x):
            if y not in forbid:
                S.add(y)
    return S


def _alive_vertices(g: UG) -> List[int]:
    # Tombstoned vertices have degree 0 in our model.
    return [v for v in range(g.n) if g.degree(v) > 0]


def is_connected_after_removal(g: UG, removed: Set[int]) -> bool:
    alive = [v for v in _alive_vertices(g) if v not in removed]
    if not alive:
        return True
    start = alive[0]
    seen = {start}
    dq = deque([start])
    while dq:
        v = dq.popleft()
        for w in g.neighbors(v):
            if w in removed or g.degree(w) == 0:
                continue
            if w not in seen:
                seen.add(w)
                dq.append(w)
    return len(seen) == len(alive)


def find_local_separators(g: UG, scan_nodes: Set[int]) -> Dict[str, List[List[int]]]:
    """Search for 1- and 2-vertex separators within the provided scan set."""
    S = sorted(scan_nodes)
    # Only consider alive vertices.
    S = [v for v in S if g.degree(v) > 0]

    cut1: List[List[int]] = []
    for v in S:
        if not is_connected_after_removal(g, {v}):
            cut1.append([v])

    cut2: List[List[int]] = []
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            a, b = S[i], S[j]
            if not is_connected_after_removal(g, {a, b}):
                cut2.append([a, b])

    return {"cut1": cut1, "cut2": cut2, "S": S}


# ----------------------------
# Main driver
# ----------------------------


def choose_candidates(g: UG, root: List[int]) -> List[Tuple[str, object]]:
    """Return a list of candidate (kind, occurrence) in priority order."""
    cands: List[Tuple[str, object]] = []
    # Try all C2 occurrences (across any of the four root edges) first.
    for occ_c2 in find_adjacent_quad_occurrences(g, root):
        cands.append(("C2", occ_c2))

    occ_p = find_pinch_occurrence(g, root)
    if occ_p is not None:
        cands.append(("pinch(ii)", occ_p))

    occ_c4 = find_refined_c4_occurrence(g, root)
    if occ_c4 is not None:
        cands.append(("refined_C4", occ_c4))
    else:
        # If neither C2 nor pinch, we still allow refined C4 as a fallback if it is
        # structurally meaningful: treat it as C4 with distinct outside neighbors.
        try:
            v = list(root)
            u = [outside_neighbor(g, v, i) for i in range(4)]
            if len(set(u)) == 4:
                cands.append(("refined_C4_fallback", OccC4Plain(v=tuple(v), u=tuple(u))))
        except Exception:
            pass

    return cands


def apply_reduction(g: UG, kind: str, occ: object) -> Tuple[UG, Set[int], Set[int], Set[int]]:
    if kind == "C2":
        return reduce_c2_plain(g, occ)  # type: ignore[arg-type]
    if kind == "pinch(ii)":
        return reduce_pinch_plain(g, occ)  # type: ignore[arg-type]
    if kind in ("refined_C4", "refined_C4_fallback"):
        return reduce_refined_c4_plain(g, occ)  # type: ignore[arg-type]
    raise ValueError(f"unknown kind {kind}")


def terminals_for(kind: str, occ: object) -> Set[int]:
    """Return the terminal set B for the reduction instance.

    This matches the boundary vertices whose rotations/attachments are rewritten
    by the reduction in the manuscript.
    """

    if kind == "C2":
        o = occ  # type: ignore[assignment]
        return {o.u1, o.u4, o.u5, o.u6}
    if kind == "pinch(ii)":
        o = occ  # type: ignore[assignment]
        return {o.u2, o.u4, o.r, o.s}
    if kind in ("refined_C4", "refined_C4_fallback"):
        o = occ  # type: ignore[assignment]
        return {o.u[0], o.u[1], o.u[2], o.u[3]}
    raise ValueError(f"unknown kind {kind}")


def add_out_supernode_for_stubs(g0: UG, stubs: List[List[int]]) -> Tuple[UG, Optional[int]]:
    """Return a graph where all stub endpoints are connected to a single OUT node.

    The local-type records represent edges leaving the radius-3 ball via `stubs`.
    When checking separator existence, we need *some* representation of the outside.
    Collapsing the unknown outside to a single highly-connected vertex is a simple,
    connectivity-maximizing completion.
    """

    if not stubs:
        return g0, None
    g = g0.copy()
    out = g.add_vertex()
    for s in stubs:
        # Robustly handle different stub formats (int, [int], or [int, ...])
        # produced by different versions of the generator/inference.
        if isinstance(s, list) and len(s) >= 1:
            v = int(s[0])
        elif isinstance(s, int):
            v = s
        else:
            continue
        if 0 <= v < out:
            g.add_edge(v, out)
    return g, out


def occ_to_json(kind: str, occ: object) -> Dict:
    if kind in ("refined_C4", "refined_C4_fallback"):
        o: OccC4Plain = occ  # type: ignore[assignment]
        return {"v": list(o.v), "u": list(o.u)}
    if kind == "pinch(ii)":
        o: OccPinchPlain = occ  # type: ignore[assignment]
        return {
            "v": list(o.v),
            "w": o.w,
            "u2": o.u2,
            "u4": o.u4,
            "t": o.t,
            "r": o.r,
            "s": o.s,
        }
    if kind == "C2":
        o: OccC2Plain = occ  # type: ignore[assignment]
        return {
            "a": o.a,
            "b": o.b,
            "c": o.c,
            "d": o.d,
            "e": o.e,
            "f": o.f,
            "u1": o.u1,
            "u4": o.u4,
            "u5": o.u5,
            "u6": o.u6,
        }
    return {"raw": str(occ)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in-witnesses",
        type=Path,
        default=None,
        help=(
            "Optional JSONL of extendible witnesses (as produced by enumerate_local_types.py). "
            "If set, admissibility is checked on each witness graph instead of on the raw local type."
        ),
    )
    ap.add_argument(
        "--in-types",
        type=Path,
        default=Path("artifacts/local_types.jsonl"),
        help="Input JSONL local types file.",
    )
    ap.add_argument(
        "--out-witnesses",
        type=Path,
        default=Path("artifacts/admissibility_witnesses.jsonl"),
        help="Output JSONL witnesses file.",
    )
    ap.add_argument(
        "--out-obstructions",
        type=Path,
        default=Path("artifacts/admissibility_obstructions.jsonl"),
        help="Output JSONL obstructions file.",
    )
    ap.add_argument("--max", type=int, default=None, help="Process at most N types.")
    args = ap.parse_args()

    in_path_types: Path = args.in_types
    in_path_witnesses: Optional[Path] = args.in_witnesses
    out_w: Path = args.out_witnesses
    out_o: Path = args.out_obstructions
    out_w.parent.mkdir(parents=True, exist_ok=True)
    out_o.parent.mkdir(parents=True, exist_ok=True)

    witnesses = 0
    obstructions = 0

    # Choose input stream.
    # - If --in-witnesses is provided, we read witness graphs and check admissibility
    #   *on the witness graph*
    # - Otherwise, we read raw local types (optionally augmented with a stub OUT node).
    if in_path_witnesses is not None:
        f_in = in_path_witnesses.open("r", encoding="utf-8")
        input_mode = "witnesses"
    else:
        f_in = in_path_types.open("r", encoding="utf-8")
        input_mode = "types"

    with f_in, out_w.open(
        "w", encoding="utf-8"
    ) as f_w, out_o.open("w", encoding="utf-8") as f_o:
        for idx, line in enumerate(f_in):
            if args.max is not None and idx >= args.max:
                break
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            # Build the graph instance and the root 4-face.
            out_id: Optional[int] = None
            if input_mode == "witnesses":
                # Each witness record has a local_type_key that encodes the rooted type.
                key = json.loads(rec["local_type_key"])
                root = list(key["root"])
                g = UG(rec["n_witness"], (tuple(e) for e in rec["edges_witness"]))
            else:
                g_base = UG(rec["n"], (tuple(e) for e in rec["edges"]))
                # In local_types.jsonl, the root face is implicitly [0, 1, 2, 3].
                root = list(rec.get("root_face", [0, 1, 2, 3]))
                stubs = rec.get("stubs")
                if stubs is None:
                    # Infer stubs: any vertex with degree < 3.
                    stubs = []
                    for v in range(rec["n"]):
                        deg = sum(1 for e in rec["edges"] if v in e)
                        for _ in range(3 - deg):
                            stubs.append([v])
                g, out_id = add_out_supernode_for_stubs(g_base, stubs)

            candidates = choose_candidates(g, root)
            if not candidates:
                f_o.write(
                    json.dumps(
                        {
                            "type_index": idx,
                            "reason": "no_candidate_configuration_detected",
                            "root_face": root,
                        }
                    )
                    + "\n"
                )
                obstructions += 1
                continue

            best = None
            fail_details: List[Dict] = []
            for kind, occ in candidates:
                try:
                    # Lemma 52 locality: if a reduction creates a 1- or 2-vertex
                    # separator, then there is one whose vertices lie in a bounded-radius
                    # closure around the boundary terminals *in the pre-reduction graph*.
                    #
                    # In our stubs model, the OUT supernode is a convenience artifact.
                    # We must forbid OUT from entering the closure, otherwise the closure
                    # would immediately explode to include all stubs.
                    B_pre = terminals_for(kind, occ)
                    closure_pre = closure_2(g, B_pre, forbid=set([out_id]) if out_id is not None else None)

                    g_red, B, removed, added = apply_reduction(g, kind, occ)
                    scan_nodes = (closure_pre - removed) | added
                    if out_id is not None:
                        scan_nodes.discard(out_id)
                except Exception as ex:
                    fail_details.append(
                        {
                            "kind": kind,
                            "occ": occ_to_json(kind, occ),
                            "error": str(ex),
                        }
                    )
                    continue

                seps = find_local_separators(g_red, scan_nodes)
                if len(seps["cut1"]) == 0 and len(seps["cut2"]) == 0:
                    best = (kind, occ, B, seps)
                    break
                fail_details.append(
                    {
                        "kind": kind,
                        "occ": occ_to_json(kind, occ),
                        "boundary": sorted(B),
                        "cut1": seps["cut1"][:50],
                        "cut2": seps["cut2"][:50],
                        "S_size": len(seps["S"]),
                    }
                )

            if best is None:
                f_o.write(
                    json.dumps(
                        {
                            "type_index": idx,
                            "root_face": root,
                            "failures": fail_details,
                        }
                    )
                    + "\n"
                )
                obstructions += 1
                continue

            kind, occ, B, seps = best
            f_w.write(
                json.dumps(
                    {
                        "type_index": idx,
                        "kind": kind,
                        "occ": occ_to_json(kind, occ),
                        "boundary": sorted(B),
                        "S_size": len(seps["S"]),
                    }
                )
                + "\n"
            )
            witnesses += 1

    print(f"wrote {witnesses} witnesses to {out_w}")
    print(f"wrote {obstructions} obstructions to {out_o}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
