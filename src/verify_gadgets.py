"""
verify_gadgets.py - Exhaustive verification of gadget interface types and lifts.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
from typing import Dict, Iterable, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]

NamedEdge = Tuple[str, str]
Pairing = Tuple[Tuple[str, str], ...]


@dataclass(frozen=True)
class GadgetSpec:
    name: str
    terminals: Tuple[str, str, str, str]
    reduced_internal_vertices: Tuple[str, str]
    reduced_edges: Tuple[NamedEdge, ...]
    patch_vertices: Tuple[str, ...]
    patch_edges: Tuple[NamedEdge, ...]
    terminal_attachments: Dict[str, str]
    excluded_types: Tuple[str, ...] = ()


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


def normalize_edge(edge: Sequence[str]) -> NamedEdge:
    u, v = edge
    return (u, v) if u < v else (v, u)


def serialize_edges(edges: Iterable[NamedEdge]) -> List[List[str]]:
    return [[u, v] for u, v in sorted(normalize_edge(edge) for edge in edges)]


def canonical_pairing(pairs: Iterable[Sequence[str]]) -> Pairing:
    normalized = [normalize_edge(pair) for pair in pairs]
    return tuple(sorted(normalized))


def build_degrees(vertices: Sequence[str], edges: Iterable[NamedEdge]) -> Dict[str, int]:
    degrees = {vertex: 0 for vertex in vertices}
    for u, v in edges:
        degrees[u] += 1
        degrees[v] += 1
    return degrees


def build_adjacency(vertices: Sequence[str], edges: Iterable[NamedEdge]) -> Dict[str, List[str]]:
    adj = {vertex: [] for vertex in vertices}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def connected_components(vertices: Sequence[str], edges: Iterable[NamedEdge]) -> List[Tuple[List[str], List[NamedEdge]]]:
    adj = build_adjacency(vertices, edges)
    used_vertices = {vertex for vertex in vertices if adj[vertex]}
    seen = set()
    components = []

    for start in sorted(used_vertices):
        if start in seen:
            continue
        stack = [start]
        comp_vertices = []
        comp_seen = set()
        while stack:
            vertex = stack.pop()
            if vertex in comp_seen:
                continue
            comp_seen.add(vertex)
            seen.add(vertex)
            comp_vertices.append(vertex)
            stack.extend(adj[vertex])

        comp_edge_set = set()
        for vertex in comp_vertices:
            for neighbor in adj[vertex]:
                comp_edge_set.add(normalize_edge((vertex, neighbor)))
        components.append((sorted(comp_vertices), sorted(comp_edge_set)))

    return components


def extract_interface_pairing(
    vertices: Sequence[str],
    internal_vertices: Sequence[str],
    terminals: Sequence[str],
    edges: Iterable[NamedEdge],
    require_terminal_activity: Dict[str, int] | None = None,
) -> Pairing | None:
    edge_list = [normalize_edge(edge) for edge in edges]
    degrees = build_degrees(vertices, edge_list)

    for vertex in internal_vertices:
        if degrees[vertex] != 2:
            return None

    for terminal in terminals:
        target = require_terminal_activity.get(terminal) if require_terminal_activity else None
        degree = degrees[terminal]
        if degree not in (0, 1):
            return None
        if target is not None and degree != target:
            return None

    components = connected_components(vertices, edge_list)
    if not components:
        return None

    pairings = []
    for comp_vertices, comp_edges in components:
        comp_terminals = sorted(
            vertex for vertex in comp_vertices if vertex in terminals and degrees[vertex] == 1
        )
        if len(comp_terminals) != 2:
            return None
        if len(comp_edges) != len(comp_vertices) - 1:
            return None
        pairings.append(tuple(comp_terminals))

    return canonical_pairing(pairings)


def expected_type_names(spec: GadgetSpec) -> Dict[str, Pairing]:
    a, b, c, d = spec.terminals
    return {
        "parallel": canonical_pairing(((a, b), (c, d))),
        f"cross:{a}-{c}": canonical_pairing(((a, c),)),
        f"cross:{a}-{d}": canonical_pairing(((a, d),)),
        f"cross:{b}-{c}": canonical_pairing(((b, c),)),
        f"cross:{b}-{d}": canonical_pairing(((b, d),)),
    }


def enumerate_reduced_interfaces(spec: GadgetSpec) -> Tuple[Dict[str, List[NamedEdge]], Dict[str, int]]:
    vertices = list(spec.reduced_internal_vertices) + list(spec.terminals)
    expected = expected_type_names(spec)
    pairing_to_name = {pairing: name for name, pairing in expected.items()}

    witnesses: Dict[str, List[NamedEdge]] = {}
    counts = {name: 0 for name in expected}

    edge_list = list(spec.reduced_edges)
    for mask in range(1 << len(edge_list)):
        selected = [edge_list[i] for i in range(len(edge_list)) if (mask >> i) & 1]
        pairing = extract_interface_pairing(vertices, spec.reduced_internal_vertices, spec.terminals, selected)
        if pairing is None or pairing not in pairing_to_name:
            continue
        name = pairing_to_name[pairing]
        counts[name] += 1
        witnesses.setdefault(name, [normalize_edge(edge) for edge in selected])

    return witnesses, counts


def find_patch_witness(spec: GadgetSpec, pairing: Pairing) -> List[NamedEdge] | None:
    active_terminals = {terminal for pair in pairing for terminal in pair}
    attachment_edges = [normalize_edge((terminal, spec.terminal_attachments[terminal])) for terminal in active_terminals]
    required_terminal_activity = {terminal: int(terminal in active_terminals) for terminal in spec.terminals}

    vertices = list(spec.patch_vertices) + list(spec.terminals)
    patch_edges = list(spec.patch_edges)
    for mask in range(1 << len(patch_edges)):
        chosen_patch_edges = [patch_edges[i] for i in range(len(patch_edges)) if (mask >> i) & 1]
        selected = attachment_edges + [normalize_edge(edge) for edge in chosen_patch_edges]
        found = extract_interface_pairing(
            vertices,
            spec.patch_vertices,
            spec.terminals,
            selected,
            require_terminal_activity=required_terminal_activity,
        )
        if found == pairing:
            return selected
    return None


def verify_spec(spec: GadgetSpec) -> Dict[str, object]:
    expected = expected_type_names(spec)
    reduced_witnesses, realized_counts = enumerate_reduced_interfaces(spec)
    admissible = {name: pairing for name, pairing in expected.items() if name not in spec.excluded_types}

    missing_types = sorted(name for name in admissible if name not in reduced_witnesses)
    unexpected_types = sorted(name for name in reduced_witnesses if name not in expected)

    lift_witnesses = {}
    missing_lifts = []
    excluded_type_violations = []
    for name, pairing in expected.items():
        witness = find_patch_witness(spec, pairing)
        if name in spec.excluded_types:
            if witness is not None:
                excluded_type_violations.append(name)
            continue
        if witness is None:
            missing_lifts.append(name)
            continue
        lift_witnesses[name] = {
            "pairing": [list(pair) for pair in pairing],
            "edges": serialize_edges(witness),
        }

    interface_details = []
    for name in sorted(expected):
        interface_details.append(
            {
                "type": name,
                "pairing": [list(pair) for pair in expected[name]],
                "reduced_instance_count": realized_counts.get(name, 0),
                "reduced_witness_edges": serialize_edges(reduced_witnesses.get(name, [])),
                "patch_witness_edges": lift_witnesses.get(name, {}).get("edges", []),
                "status": (
                    "EXCLUDED"
                    if name in spec.excluded_types and name not in excluded_type_violations
                    else "PASS"
                    if name in reduced_witnesses and name not in missing_lifts
                    else "FAIL"
                ),
            }
        )

    status = "PASS"
    if missing_types or unexpected_types or missing_lifts or excluded_type_violations:
        status = "FAIL"

    return {
        "name": spec.name,
        "status": status,
        "terminals": list(spec.terminals),
        "expected_interface_count": len(expected),
        "admissible_interface_count": len(admissible),
        "realized_interface_count": len(reduced_witnesses),
        "missing_interface_types": missing_types,
        "unexpected_interface_types": unexpected_types,
        "missing_lift_witnesses": missing_lifts,
        "excluded_interface_types": list(spec.excluded_types),
        "excluded_type_violations": excluded_type_violations,
        "interfaces": interface_details,
    }


def build_specs() -> List[GadgetSpec]:
    return [
        GadgetSpec(
            name="Refined C4",
            terminals=("u1", "u3", "u2", "u4"),
            reduced_internal_vertices=("x", "y"),
            reduced_edges=(
                ("x", "y"),
                ("x", "u1"),
                ("x", "u3"),
                ("y", "u2"),
                ("y", "u4"),
            ),
            patch_vertices=("v1", "v2", "v3", "v4"),
            patch_edges=(
                ("v1", "v2"),
                ("v2", "v3"),
                ("v3", "v4"),
                ("v4", "v1"),
            ),
            terminal_attachments={
                "u1": "v1",
                "u3": "v3",
                "u2": "v2",
                "u4": "v4",
            },
            excluded_types=("parallel",),
        ),
        GadgetSpec(
            name="C2",
            terminals=("u1", "u6", "u4", "u5"),
            reduced_internal_vertices=("x", "y"),
            reduced_edges=(
                ("x", "y"),
                ("x", "u1"),
                ("x", "u6"),
                ("y", "u4"),
                ("y", "u5"),
            ),
            patch_vertices=("a", "b", "c", "d", "e", "f"),
            patch_edges=(
                ("a", "b"),
                ("b", "c"),
                ("c", "d"),
                ("d", "a"),
                ("b", "f"),
                ("f", "e"),
                ("e", "c"),
            ),
            terminal_attachments={
                "u1": "a",
                "u6": "f",
                "u4": "d",
                "u5": "e",
            },
        ),
        GadgetSpec(
            name="Pinch(ii)",
            terminals=("r", "s", "u2", "u4"),
            reduced_internal_vertices=("x", "y"),
            reduced_edges=(
                ("x", "y"),
                ("x", "r"),
                ("x", "s"),
                ("y", "u2"),
                ("y", "u4"),
            ),
            patch_vertices=("v1", "v2", "v3", "v4", "w", "t"),
            patch_edges=(
                ("v1", "v2"),
                ("v2", "v3"),
                ("v3", "v4"),
                ("v4", "v1"),
                ("w", "v1"),
                ("w", "v3"),
                ("w", "t"),
            ),
            terminal_attachments={
                "r": "t",
                "s": "t",
                "u2": "v2",
                "u4": "v4",
            },
        ),
    ]


def main() -> None:
    commit = get_commit_hash()
    specs = build_specs()
    gadget_results = [verify_spec(spec) for spec in specs]

    summary = []
    for result in gadget_results:
        covered_types = len(
            [entry for entry in result["interfaces"] if entry["status"] == "PASS"]
        )
        summary.append(
            {
                "name": result["name"],
                "status": result["status"],
                "realized_type_count": result["realized_interface_count"],
                "admissible_type_count": result["admissible_interface_count"],
                "covered_type_count": covered_types,
            }
        )
        print(
            f"{result['name']}: {result['status']} "
            f"({covered_types}/{result['admissible_interface_count']} admissible interface types covered)"
        )

    output = {
        "version": commit,
        "summary": summary,
        "gadgets": gadget_results,
    }

    out_path = ROOT / "data" / "logic_verification.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
