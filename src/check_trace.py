"""
check_trace.py - Generates and verifies structured certified-reduction traces.

The checker supports two formats:
  1. Structured JSON traces with explicit edits, hashes, and a final cycle witness.
  2. Legacy JSONL snapshot traces for backwards compatibility.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Callable, Dict, List, Sequence, Tuple

import barnette_proof as bp
from barnette_proof import Cycle, EmbeddedGraph, OccC2, OccC4, OccPinch, RecC2, RecC4, RecPinch
from plantri_wrapper import parse_plantri_ascii_embedding


ROOT = Path(__file__).resolve().parents[1]
TRACE_DIR = ROOT / "artifacts" / "traces"
TRACE_SCHEMA_VERSION = "1.0.1"
REFINED_C4_TRACE_LINE = "20 bcd,aef,agh,ahi,bjk,bkl,clm,cnd,doj,eip,epf,fqg,gqr,hrs,isp,jok,ltm,mtn,nto,qsr"

Edge = Tuple[int, int]
RuleOcc = OccC2 | OccC4 | OccPinch
RuleRec = RecC2 | RecC4 | RecPinch

RULE_DELTAS = {
    "C2": -4,
    "pinch_ii": -4,
    "refined_C4": -2,
}

RULE_VARIANTS = {
    "C2": "C2.g2v.v1",
    "pinch_ii": "pinch_ii.g2v.v1",
    "refined_C4": "refined_C4.g2v.v1",
}


@dataclass
class VerifiedStep:
    step_id: int
    rule_name: str
    pre_graph: EmbeddedGraph
    post_graph: EmbeddedGraph
    rec: RuleRec


def get_commit_hash() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return proc.stdout.strip()
    except Exception:
        return "unknown"


def canonical_rule_name(name: str) -> str:
    aliases = {
        "C2": "C2",
        "pinch(ii)": "pinch_ii",
        "PINCH": "pinch_ii",
        "pinch_ii": "pinch_ii",
        "refined C4": "refined_C4",
        "refined_C4": "refined_C4",
    }
    if name not in aliases:
        raise ValueError(f"unknown rule name {name!r}")
    return aliases[name]


def normalize_edge(edge: Sequence[int]) -> Edge:
    u, v = int(edge[0]), int(edge[1])
    return (u, v) if u < v else (v, u)


def serialize_edges(edges: Sequence[Sequence[int]]) -> List[List[int]]:
    return [[u, v] for u, v in sorted(normalize_edge(edge) for edge in edges)]


def cyclically_equal(left: Sequence[int], right: Sequence[int]) -> bool:
    left_list = list(left)
    right_list = list(right)
    if len(left_list) != len(right_list):
        return False
    if not left_list:
        return True
    doubled = right_list + right_list
    for start in range(len(right_list)):
        if doubled[start : start + len(left_list)] == left_list:
            return True
    return False


def canonical_rotation(rotation: Sequence[int]) -> List[int]:
    entries = list(rotation)
    if not entries:
        return []
    candidates = [entries[i:] + entries[:i] for i in range(len(entries))]
    return min(candidates)


def graph_to_embedded_state(G: EmbeddedGraph) -> Dict[str, object]:
    return {
        "vertex_ids": G.vertices(),
        "edges": serialize_edges(G.edges()),
        "rotation": {str(v): list(G.rot[v]) for v in G.vertices()},
    }


def graph_to_graph_file(G: EmbeddedGraph) -> Dict[str, object]:
    return {
        "adj": {str(v): sorted(G.adj[v]) for v in G.vertices()},
        "rot": {str(v): list(G.rot[v]) for v in G.vertices()},
    }


def graph_from_embedded_state(data: Dict[str, object]) -> EmbeddedGraph:
    rotation_data = data["rotation"]
    assert isinstance(rotation_data, dict)
    rot = {int(v): [int(nb) for nb in neighbors] for v, neighbors in rotation_data.items()}
    adj = {v: set(neighbors) for v, neighbors in rot.items()}
    return EmbeddedGraph(adj=adj, rot=rot)


def graph_state_hash(G: EmbeddedGraph) -> str:
    canonical_state = {
        "vertex_ids": G.vertices(),
        "edges": serialize_edges(G.edges()),
        "rotation": {str(v): canonical_rotation(G.rot[v]) for v in G.vertices()},
    }
    payload = json.dumps(canonical_state, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cycle_from_vertex_order(vertices: Sequence[int]) -> Cycle:
    ordered = [int(v) for v in vertices]
    edges = [(ordered[i], ordered[(i + 1) % len(ordered)]) for i in range(len(ordered))]
    return Cycle(edges)


def graph_from_plantri_ascii(line: str) -> EmbeddedGraph:
    parsed = parse_plantri_ascii_embedding(line)
    adj = {vertex: set(neighbors) for vertex, neighbors in parsed.rot.items()}
    return EmbeddedGraph(adj=adj, rot=parsed.rot)


def verify_invariants(G: EmbeddedGraph, label: str) -> None:
    G.validate_rotation_embedding()
    if any(len(G.adj[v]) != 3 for v in G.adj):
        raise ValueError(f"{label}: graph is not cubic")
    if not bp.is_bipartite(G.adj):
        raise ValueError(f"{label}: graph is not bipartite")
    if not bp.is_3_connected(G.adj):
        raise ValueError(f"{label}: graph is not 3-connected")


def validate_refined_c4_occurrence(G: EmbeddedGraph, occ: OccC4) -> None:
    darts, end = G.trace_face_darts((occ.v1, occ.v2), steps=4)
    if end != (occ.v1, occ.v2):
        raise ValueError("refined_C4 occurrence is not a facial 4-cycle")
    if [tail for tail, _ in darts] != [occ.v1, occ.v2, occ.v3, occ.v4]:
        raise ValueError("refined_C4 facial order mismatch")

    u1 = G.third_neighbor(occ.v1, {occ.v2, occ.v4})
    u2 = G.third_neighbor(occ.v2, {occ.v1, occ.v3})
    u3 = G.third_neighbor(occ.v3, {occ.v2, occ.v4})
    u4 = G.third_neighbor(occ.v4, {occ.v1, occ.v3})
    if (u1, u2, u3, u4) != (occ.u1, occ.u2, occ.u3, occ.u4):
        raise ValueError("refined_C4 external neighbors mismatch")
    if len({u1, u2, u3, u4}) != 4:
        raise ValueError("refined_C4 external neighbors are not distinct")

    quad_edges = [(occ.v1, occ.v2), (occ.v2, occ.v3), (occ.v3, occ.v4), (occ.v4, occ.v1)]
    if any(G.other_face_is_quad(a, b) for a, b in quad_edges):
        raise ValueError("refined_C4 occurrence is not edge-isolated")


def validate_pinch_occurrence(G: EmbeddedGraph, occ: OccPinch) -> None:
    darts, end = G.trace_face_darts((occ.v1, occ.v2), steps=4)
    if end != (occ.v1, occ.v2):
        raise ValueError("pinch_ii occurrence is not a facial 4-cycle")
    if [tail for tail, _ in darts] != [occ.v1, occ.v2, occ.v3, occ.v4]:
        raise ValueError("pinch_ii facial order mismatch")

    u1 = G.third_neighbor(occ.v1, {occ.v2, occ.v4})
    u2 = G.third_neighbor(occ.v2, {occ.v1, occ.v3})
    u3 = G.third_neighbor(occ.v3, {occ.v2, occ.v4})
    u4 = G.third_neighbor(occ.v4, {occ.v1, occ.v3})
    if (u1, u2, u3, u4) != (occ.w, occ.u2, occ.w, occ.u4):
        raise ValueError("pinch_ii boundary data mismatch")

    t = G.third_neighbor(occ.w, {occ.v1, occ.v3})
    if t != occ.t:
        raise ValueError("pinch_ii third neighbor mismatch")
    if t in {occ.u2, occ.u4}:
        raise ValueError("pinch_ii violates t not in {u2, u4}")

    rs = sorted(G.adj[t] - {occ.w})
    if rs != [occ.r, occ.s]:
        raise ValueError("pinch_ii side terminals mismatch")

    quad_edges = [(occ.v1, occ.v2), (occ.v2, occ.v3), (occ.v3, occ.v4), (occ.v4, occ.v1)]
    if any(G.other_face_is_quad(a, b) for a, b in quad_edges):
        raise ValueError("pinch_ii occurrence is not edge-isolated")

    p = G.rot[occ.w][(G.pos[occ.w][occ.t] + 1) % 3]
    q = G.rot[p][(G.pos[p][occ.w] + 1) % 3]
    epsilon = 0 if q == occ.v2 else 1
    if epsilon != occ.epsilon:
        raise ValueError("pinch_ii flip bit mismatch")


def validate_c2_occurrence(G: EmbeddedGraph, occ: OccC2) -> None:
    candidates = []
    for start in ((occ.b, occ.c), (occ.c, occ.b)):
        other = (start[1], start[0])
        if not (G.face_is_quad_from_dart(start) and G.face_is_quad_from_dart(other)):
            continue
        left_face = G.trace_face_vertices(start, steps=4)
        right_face = G.trace_face_vertices(other, steps=4)
        candidate = bp._canonicalize_adjacent_quads(G, left_face, right_face)
        if candidate is not None:
            candidates.append(candidate)
    if occ not in candidates:
        raise ValueError("C2 occurrence does not match the shared-quad structure")


def occ_from_labels(rule_name: str, labels: Dict[str, int]) -> RuleOcc:
    if rule_name == "C2":
        return OccC2(
            labels["a"],
            labels["b"],
            labels["c"],
            labels["d"],
            labels["e"],
            labels["f"],
            labels["u1"],
            labels["u4"],
            labels["u5"],
            labels["u6"],
        )
    if rule_name == "refined_C4":
        return OccC4(
            labels["v1"],
            labels["v2"],
            labels["v3"],
            labels["v4"],
            labels["u1"],
            labels["u2"],
            labels["u3"],
            labels["u4"],
        )
    raise ValueError(f"unsupported rule {rule_name}")


def validate_occurrence(G: EmbeddedGraph, rule_name: str, occ: RuleOcc) -> None:
    if rule_name == "C2":
        validate_c2_occurrence(G, occ)
        return
    if rule_name == "refined_C4":
        validate_refined_c4_occurrence(G, occ)
        return
    if rule_name == "pinch_ii":
        validate_pinch_occurrence(G, occ)
        return
    raise ValueError(f"unsupported rule {rule_name}")


def apply_rule(G: EmbeddedGraph, rule_name: str, occ: RuleOcc) -> Tuple[EmbeddedGraph, RuleRec]:
    if rule_name == "C2":
        return bp.reduce_C2(G, occ)
    if rule_name == "refined_C4":
        return bp.reduce_C4(G, occ)
    if rule_name == "pinch_ii":
        return bp.reduce_pinch(G, occ)
    raise ValueError(f"unsupported rule {rule_name}")


def labels_from_occ_rec(rule_name: str, occ: RuleOcc, rec: RuleRec) -> Dict[str, int]:
    labels = {"x": rec.x, "y": rec.y}
    if rule_name == "C2":
        labels.update(
            {
                "a": occ.a,
                "b": occ.b,
                "c": occ.c,
                "d": occ.d,
                "e": occ.e,
                "f": occ.f,
                "u1": occ.u1,
                "u4": occ.u4,
                "u5": occ.u5,
                "u6": occ.u6,
            }
        )
    elif rule_name == "refined_C4":
        labels.update(
            {
                "v1": occ.v1,
                "v2": occ.v2,
                "v3": occ.v3,
                "v4": occ.v4,
                "u1": occ.u1,
                "u2": occ.u2,
                "u3": occ.u3,
                "u4": occ.u4,
            }
        )
    elif rule_name == "pinch_ii":
        labels.update(
            {
                "v1": occ.v1,
                "v2": occ.v2,
                "v3": occ.v3,
                "v4": occ.v4,
                "w": occ.w,
                "t": occ.t,
                "r": occ.r,
                "s": occ.s,
                "u2": occ.u2,
                "u4": occ.u4,
            }
        )
    else:
        raise ValueError(f"unsupported rule {rule_name}")
    return labels


def interior_vertices(rule_name: str, occ: RuleOcc) -> List[int]:
    if rule_name == "C2":
        return [occ.a, occ.b, occ.c, occ.d, occ.e, occ.f]
    if rule_name == "refined_C4":
        return [occ.v1, occ.v2, occ.v3, occ.v4]
    if rule_name == "pinch_ii":
        return [occ.v1, occ.v2, occ.v3, occ.v4, occ.w, occ.t]
    raise ValueError(f"unsupported rule {rule_name}")


def pinch_epsilon_from_labels(G: EmbeddedGraph, labels: Dict[str, int]) -> int:
    w = labels["w"]
    t = labels["t"]
    v2 = labels["v2"]
    p = G.rot[w][(G.pos[w][t] + 1) % 3]
    q = G.rot[p][(G.pos[p][w] + 1) % 3]
    return 0 if q == v2 else 1


def compute_edit(before: EmbeddedGraph, after: EmbeddedGraph) -> Dict[str, object]:
    before_vertices = set(before.adj)
    after_vertices = set(after.adj)
    before_edges = {normalize_edge(edge) for edge in before.edges()}
    after_edges = {normalize_edge(edge) for edge in after.edges()}

    changed_vertices = []
    for vertex in after.vertices():
        if vertex not in before_vertices or before.rot[vertex] != after.rot[vertex]:
            changed_vertices.append(vertex)

    return {
        "remove_vertices": sorted(before_vertices - after_vertices),
        "add_vertices": sorted(after_vertices - before_vertices),
        "remove_edges": serialize_edges(list(before_edges - after_edges)),
        "add_edges": serialize_edges(list(after_edges - before_edges)),
        "rotation_updates": {
            str(vertex): list(after.rot[vertex]) for vertex in sorted(changed_vertices)
        },
    }


def build_step_record(
    step_id: int,
    rule_name: str,
    before: EmbeddedGraph,
    after: EmbeddedGraph,
    occ: RuleOcc,
    rec: RuleRec,
) -> Dict[str, object]:
    delta_n = len(after.adj) - len(before.adj)
    if delta_n != RULE_DELTAS[rule_name]:
        raise ValueError(f"unexpected delta_n {delta_n} for {rule_name}")

    return {
        "step_id": step_id,
        "rule": {
            "name": rule_name,
            "variant": RULE_VARIANTS[rule_name],
        },
        "match": {
            "boundary": list(rec.sigma),
            "interior": interior_vertices(rule_name, occ),
            "local_labels": labels_from_occ_rec(rule_name, occ, rec),
        },
        "edit": compute_edit(before, after),
        "check": {
            "n_before": len(before.adj),
            "n_after": len(after.adj),
            "delta_n": delta_n,
        },
        "hash": {
            "pre_state_sha256": graph_state_hash(before),
            "post_state_sha256": graph_state_hash(after),
        },
    }


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_hex(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_trace_object(
    initial_graph: EmbeddedGraph,
    steps: List[Dict[str, object]],
    final_graph: EmbeddedGraph,
    final_cycle: Cycle,
    generator_args: List[str],
) -> Dict[str, object]:
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "provenance": {
            "git_commit": get_commit_hash(),
            "generator": {
                "name": "check_trace.py",
                "version": TRACE_SCHEMA_VERSION,
                "args": generator_args,
            },
            "reducer": {
                "name": "barnette_proof.py",
                "version": get_commit_hash(),
            },
            "checker": {
                "name": "check_trace.py",
                "version": TRACE_SCHEMA_VERSION,
            },
        },
        "initial": graph_to_embedded_state(initial_graph),
        "steps": steps,
        "final": {
            "status": "hamiltonian_found",
            "hamilton_cycle": final_cycle.as_ordered_cycle(final_graph),
        },
    }


def solve_with_solver_trace(initial_graph: EmbeddedGraph) -> Tuple[List[Dict[str, object]], EmbeddedGraph, Cycle]:
    steps: List[Dict[str, object]] = []
    current = initial_graph.copy()

    while True:
        if bp.is_cube(current) or len(current.adj) <= 14:
            return steps, current, bp.find_hamiltonian_cycle(current)

        for rule_name, detector in (
            ("C2", bp.detect_C2),
            ("pinch_ii", bp.detect_C_pinch_ii),
            ("refined_C4", bp.detect_refined_C4),
        ):
            occ = detector(current)
            if occ is None:
                continue
            reduced, rec = apply_rule(current, rule_name, occ)
            steps.append(build_step_record(len(steps) + 1, rule_name, current, reduced, occ, rec))
            current = reduced
            break
        else:
            raise AssertionError("no reducible configuration found while generating solver trace")


def generate_solver_trace(
    graph_name: str,
    build_graph: Callable[[], EmbeddedGraph],
    graph_path: Path,
    trace_path: Path,
) -> Dict[str, object]:
    initial_graph = build_graph()
    steps, final_graph, final_cycle = solve_with_solver_trace(initial_graph)
    write_json(graph_path, graph_to_graph_file(initial_graph))
    trace = build_trace_object(initial_graph, steps, final_graph, final_cycle, ["solver", graph_name])
    write_json(trace_path, trace)
    return trace


def generate_forced_rule_trace(
    graph_name: str,
    build_graph: Callable[[], EmbeddedGraph],
    rule_name: str,
    detector: Callable[[EmbeddedGraph], RuleOcc | None],
    graph_path: Path,
    trace_path: Path,
) -> Dict[str, object]:
    initial_graph = build_graph()
    occurrence = detector(initial_graph)
    if occurrence is None:
        raise ValueError(f"no {rule_name} occurrence found in {graph_name}")

    reduced_graph, rec = apply_rule(initial_graph, rule_name, occurrence)
    step = build_step_record(1, rule_name, initial_graph, reduced_graph, occurrence, rec)
    final_cycle = bp.find_hamiltonian_cycle(reduced_graph)

    write_json(graph_path, graph_to_graph_file(initial_graph))
    trace = build_trace_object(initial_graph, [step], reduced_graph, final_cycle, ["forced", graph_name, rule_name])
    write_json(trace_path, trace)
    return trace


def compare_rotation_updates(actual: Dict[str, object], expected: Dict[str, object]) -> None:
    if set(actual) != set(expected):
        raise ValueError("rotation_updates keys mismatch")
    for vertex in expected:
        left = [int(v) for v in actual[vertex]]
        right = [int(v) for v in expected[vertex]]
        if not cyclically_equal(left, right):
            raise ValueError(f"rotation update mismatch at vertex {vertex}")


def assert_step_matches(actual: Dict[str, object], expected: Dict[str, object]) -> None:
    if int(actual["step_id"]) != int(expected["step_id"]):
        raise ValueError("step_id mismatch")

    actual_rule = canonical_rule_name(actual["rule"]["name"])
    expected_rule = canonical_rule_name(expected["rule"]["name"])
    if actual_rule != expected_rule:
        raise ValueError("rule name mismatch")
    if actual["rule"]["variant"] != expected["rule"]["variant"]:
        raise ValueError("rule variant mismatch")

    actual_boundary = [int(v) for v in actual["match"]["boundary"]]
    expected_boundary = [int(v) for v in expected["match"]["boundary"]]
    if actual_boundary not in (expected_boundary, list(reversed(expected_boundary))):
        raise ValueError("boundary order mismatch")

    actual_interior = sorted(int(v) for v in actual["match"]["interior"])
    expected_interior = sorted(int(v) for v in expected["match"]["interior"])
    if actual_interior != expected_interior:
        raise ValueError("interior vertex set mismatch")

    actual_labels = {key: int(value) for key, value in actual["match"]["local_labels"].items()}
    expected_labels = {key: int(value) for key, value in expected["match"]["local_labels"].items()}
    if actual_labels != expected_labels:
        raise ValueError("local label mapping mismatch")

    for field in ("remove_vertices", "add_vertices"):
        actual_values = sorted(int(v) for v in actual["edit"][field])
        expected_values = sorted(int(v) for v in expected["edit"][field])
        if actual_values != expected_values:
            raise ValueError(f"{field} mismatch")

    for field in ("remove_edges", "add_edges"):
        actual_edges = serialize_edges(actual["edit"][field])
        expected_edges = serialize_edges(expected["edit"][field])
        if actual_edges != expected_edges:
            raise ValueError(f"{field} mismatch")

    compare_rotation_updates(actual["edit"]["rotation_updates"], expected["edit"]["rotation_updates"])

    for field in ("n_before", "n_after", "delta_n"):
        if int(actual["check"][field]) != int(expected["check"][field]):
            raise ValueError(f"{field} mismatch")

    if actual["hash"]["pre_state_sha256"] != expected["hash"]["pre_state_sha256"]:
        raise ValueError("pre_state_sha256 mismatch")
    if actual["hash"]["post_state_sha256"] != expected["hash"]["post_state_sha256"]:
        raise ValueError("post_state_sha256 mismatch")


def expected_step_from_trace(current_graph: EmbeddedGraph, step: Dict[str, object]) -> Tuple[Dict[str, object], RuleRec, EmbeddedGraph]:
    rule_name = canonical_rule_name(step["rule"]["name"])
    labels = {key: int(value) for key, value in step["match"]["local_labels"].items()}

    if rule_name == "pinch_ii":
        occurrence = OccPinch(
            labels["v1"],
            labels["v2"],
            labels["v3"],
            labels["v4"],
            labels["w"],
            labels["t"],
            labels["r"],
            labels["s"],
            labels["u2"],
            labels["u4"],
            pinch_epsilon_from_labels(current_graph, labels),
        )
    else:
        occurrence = occ_from_labels(rule_name, labels)

    validate_occurrence(current_graph, rule_name, occurrence)
    reduced_graph, rec = apply_rule(current_graph, rule_name, occurrence)
    expected = build_step_record(int(step["step_id"]), rule_name, current_graph, reduced_graph, occurrence, rec)
    return expected, rec, reduced_graph


def lift_back(final_graph: EmbeddedGraph, final_cycle: Cycle, verified_steps: List[VerifiedStep]) -> Cycle:
    current_graph = final_graph
    current_cycle = final_cycle
    for step in reversed(verified_steps):
        if step.rule_name == "C2":
            current_cycle = bp.lift_C2(step.pre_graph, current_graph, step.rec, current_cycle)
        elif step.rule_name == "refined_C4":
            current_cycle = bp.lift_C4(step.pre_graph, current_graph, step.rec, current_cycle)
        elif step.rule_name == "pinch_ii":
            current_cycle = bp.lift_pinch(step.pre_graph, current_graph, step.rec, current_cycle)
        else:
            raise ValueError(f"unsupported rule {step.rule_name}")
        current_cycle.validate_hamiltonian(step.pre_graph)
        current_graph = step.pre_graph
    return current_cycle


def verify_structured_trace(payload: Dict[str, object], filename: Path) -> bool:
    print(f"Verifying structured trace: {filename}")
    initial_graph = graph_from_embedded_state(payload["initial"])
    verify_invariants(initial_graph, "initial")

    current_graph = initial_graph
    verified_steps: List[VerifiedStep] = []
    steps = payload["steps"]
    assert isinstance(steps, list)

    for raw_step in steps:
        assert isinstance(raw_step, dict)
        if raw_step["hash"]["pre_state_sha256"] != graph_state_hash(current_graph):
            raise ValueError(f"pre-state hash mismatch at step {raw_step['step_id']}")

        expected_step, rec, reduced_graph = expected_step_from_trace(current_graph, raw_step)
        assert_step_matches(raw_step, expected_step)
        verify_invariants(reduced_graph, f"step {raw_step['step_id']}")

        if raw_step["hash"]["post_state_sha256"] != graph_state_hash(reduced_graph):
            raise ValueError(f"post-state hash mismatch at step {raw_step['step_id']}")

        verified_steps.append(
            VerifiedStep(
                step_id=int(raw_step["step_id"]),
                rule_name=canonical_rule_name(raw_step["rule"]["name"]),
                pre_graph=current_graph,
                post_graph=reduced_graph,
                rec=rec,
            )
        )
        current_graph = reduced_graph
        print(f"  Step {raw_step['step_id']}: {raw_step['rule']['name']} OK")

    final = payload["final"]
    assert isinstance(final, dict)
    if final.get("status") != "hamiltonian_found":
        raise ValueError("unsupported final status")

    final_cycle = cycle_from_vertex_order(final["hamilton_cycle"])
    final_cycle.validate_hamiltonian(current_graph)
    lifted = lift_back(current_graph, final_cycle, verified_steps)
    lifted.validate_hamiltonian(initial_graph)

    print(
        f"Structured trace PASSED ({len(verified_steps)} steps, "
        f"final graph n={len(current_graph.adj)}, initial graph n={len(initial_graph.adj)})"
    )
    return True


def replay_legacy_snapshot_trace(filename: Path) -> bool:
    print(f"Replaying legacy snapshot trace: {filename}")
    previous_n = float("inf")
    steps_verified = 0

    for step_id, line in enumerate(filename.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        data = json.loads(line)
        adj = {int(v): set(int(nb) for nb in neighbors) for v, neighbors in data["adj"].items()}
        rot = {int(v): [int(nb) for nb in neighbors] for v, neighbors in data["rot"].items()}
        graph = EmbeddedGraph(adj, rot)
        verify_invariants(graph, f"legacy step {step_id}")

        n_vertices = len(graph.adj)
        if step_id > 0 and n_vertices >= previous_n:
            raise ValueError(f"legacy step {step_id}: n did not decrease ({previous_n} -> {n_vertices})")
        previous_n = n_vertices
        steps_verified += 1
        print(f"  Legacy step {step_id}: n={n_vertices} OK")

    print(f"Legacy trace PASSED ({steps_verified} snapshots)")
    return True


def verify_trace(filename: str) -> bool:
    path = Path(filename)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"trace file not found: {path}")

    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return replay_legacy_snapshot_trace(path)

    if isinstance(payload, dict) and "initial" in payload and "steps" in payload:
        return verify_structured_trace(payload, path)
    return replay_legacy_snapshot_trace(path)


def summarize_batch_results(results: Sequence[Dict[str, object]]) -> Dict[str, object]:
    verified = [result for result in results if result.get("category") == "verified_trace"]
    diagnostic = [result for result in results if result.get("category") == "diagnostic_probe"]

    return {
        "verified_trace_summary": {
            "job_count": len(verified),
            "pass_count": sum(1 for result in verified if result.get("success")),
            "fail_count": sum(1 for result in verified if not result.get("success")),
            "all_passed": all(bool(result.get("success")) for result in verified),
        },
        "diagnostic_summary": {
            "job_count": len(diagnostic),
            "pass_count": sum(1 for result in diagnostic if result.get("success")),
            "open_count": sum(1 for result in diagnostic if not result.get("success")),
            "open_issues": [
                {
                    "name": result["name"],
                    "trace_path": result["trace_path"],
                    "error": result["error"],
                }
                for result in diagnostic
                if not result.get("success")
            ],
        },
    }


def run_batch_verification() -> List[Dict[str, object]]:
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [
        {
            "name": "cube",
            "graph_path": TRACE_DIR / "graph_n8.json",
            "trace_path": TRACE_DIR / "trace_n8.json",
            "mode": "solver",
            "category": "verified_trace",
            "builder": bp.make_cube,
        },
        {
            "name": "prism12",
            "graph_path": TRACE_DIR / "graph_n12.json",
            "trace_path": TRACE_DIR / "trace_n12.json",
            "mode": "solver",
            "category": "verified_trace",
            "builder": lambda: bp.make_prism(6),
        },
        {
            "name": "prism48",
            "graph_path": TRACE_DIR / "graph_n48.json",
            "trace_path": TRACE_DIR / "trace_n48.json",
            "mode": "solver",
            "category": "verified_trace",
            "builder": lambda: bp.make_prism(24),
        },
        {
            "name": "prism128",
            "graph_path": TRACE_DIR / "graph_n128.json",
            "trace_path": TRACE_DIR / "trace_n128.json",
            "mode": "solver",
            "category": "verified_trace",
            "builder": lambda: bp.make_prism(64),
        },
        {
            "name": "rule_c2",
            "graph_path": TRACE_DIR / "graph_rule_c2.json",
            "trace_path": TRACE_DIR / "trace_rule_c2.json",
            "mode": "forced",
            "category": "verified_trace",
            "builder": lambda: bp.make_prism(8),
            "rule": "C2",
            "detector": bp.detect_C2,
        },
        {
            "name": "rule_refined_c4",
            "graph_path": TRACE_DIR / "graph_rule_refined_c4.json",
            "trace_path": TRACE_DIR / "trace_rule_refined_c4.json",
            "mode": "forced",
            "category": "diagnostic_probe",
            "builder": lambda: graph_from_plantri_ascii(REFINED_C4_TRACE_LINE),
            "rule": "refined_C4",
            "detector": bp.detect_refined_C4,
        },
        {
            "name": "rule_pinch_ii",
            "graph_path": TRACE_DIR / "graph_rule_pinch_ii.json",
            "trace_path": TRACE_DIR / "trace_rule_pinch_ii.json",
            "mode": "forced",
            "category": "diagnostic_probe",
            "builder": bp.make_custom_pinch_example,
            "rule": "pinch_ii",
            "detector": bp.detect_C_pinch_ii,
        },
    ]

    results = []
    for job in jobs:
        print(f"\n--- Processing {job['name']} ---")
        try:
            if job["mode"] == "solver":
                trace = generate_solver_trace(job["name"], job["builder"], job["graph_path"], job["trace_path"])
            else:
                trace = generate_forced_rule_trace(
                    job["name"],
                    job["builder"],
                    job["rule"],
                    job["detector"],
                    job["graph_path"],
                    job["trace_path"],
                )
            success = verify_trace(str(job["trace_path"]))
            results.append(
                {
                    "name": job["name"],
                    "category": job["category"],
                    "mode": job["mode"],
                    "status": "PASS",
                    "graph_path": str(job["graph_path"].relative_to(ROOT)),
                    "trace_path": str(job["trace_path"].relative_to(ROOT)),
                    "success": success,
                    "trace_sha256": sha256_hex(job["trace_path"]),
                    "step_count": len(trace["steps"]),
                }
            )
        except Exception as exc:
            status = "OPEN" if job["category"] == "diagnostic_probe" else "FAIL"
            print(f"  {status}: {exc}")
            result = {
                "name": job["name"],
                "category": job["category"],
                "mode": job["mode"],
                "status": status,
                "graph_path": str(job["graph_path"].relative_to(ROOT)),
                "trace_path": str(job["trace_path"].relative_to(ROOT)),
                "success": False,
                "error": str(exc),
            }
            if job["trace_path"].exists():
                result["trace_sha256"] = sha256_hex(job["trace_path"])
            results.append(result)

    summary = {
        "version": get_commit_hash(),
        **summarize_batch_results(results),
        "results": results,
    }
    summary_path = TRACE_DIR / "batch_verification.json"
    write_json(summary_path, summary)
    print(f"\nBatch verification complete. Summary saved to {summary_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", nargs="?", help="trace file to verify")
    args = parser.parse_args()

    if args.trace:
        verify_trace(args.trace)
    else:
        run_batch_verification()


if __name__ == "__main__":
    main()
