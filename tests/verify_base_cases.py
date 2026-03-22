"""
verify_base_cases.py - Exhaustive verification of small graphs in Q.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import barnette_proof as bp
from barnette_proof import EmbeddedGraph
from plantri_wrapper import PlantriGraph, iter_barnette_graph_rotations_via_plantri


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


def default_plantri_path() -> Path:
    local_plantri = ROOT / "plantri.exe"
    return local_plantri if local_plantri.exists() else Path("plantri")


def embedded_from_plantri(graph: PlantriGraph) -> EmbeddedGraph:
    adj = {v: set(neighbors) for v, neighbors in graph.rot.items()}
    return EmbeddedGraph(adj=adj, rot=graph.rot)


def serialize_rotation(rot: Dict[int, List[int]]) -> Dict[str, List[int]]:
    return {str(v): list(neighbors) for v, neighbors in sorted(rot.items())}


def serialize_adjacency(adj: Dict[int, set[int]]) -> Dict[str, List[int]]:
    return {str(v): sorted(neighbors) for v, neighbors in sorted(adj.items())}


def classify_name(n: int, index: int) -> str:
    if n == 8:
        return "Cube"
    if n == 12:
        return "Prism-6"
    return f"graph_n{n}_{index}"


def sha256_hex(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_base_case_verification(n_limit: int, plantri_path: Path) -> Dict[str, object]:
    if n_limit % 2 != 0 or n_limit < 8:
        raise ValueError("n_limit must be an even integer at least 8")

    commit_hash = get_commit_hash()
    counts_by_n: Dict[str, int] = {}
    instances = []

    for n in range(8, n_limit + 1, 2):
        seen_raw_lines = set()
        count = 0
        for index, plantri_graph in enumerate(
            iter_barnette_graph_rotations_via_plantri(str(plantri_path), n),
            start=1,
        ):
            if plantri_graph.raw_line in seen_raw_lines:
                continue
            seen_raw_lines.add(plantri_graph.raw_line)

            G = embedded_from_plantri(plantri_graph)
            bp.validate_in_Q(G)
            cycle = bp.find_hamiltonian_cycle(G)
            cycle.validate_hamiltonian(G)

            instances.append(
                {
                    "name": classify_name(n, index),
                    "n": n,
                    "index": index,
                    "graph_sha256": hashlib.sha256(
                        plantri_graph.raw_line.encode("utf-8")
                    ).hexdigest(),
                    "plantri_ascii": plantri_graph.raw_line,
                    "rotation": serialize_rotation(plantri_graph.rot),
                    "adj": serialize_adjacency(G.adj),
                    "witness": cycle.as_ordered_cycle(G),
                }
            )
            count += 1

        counts_by_n[str(n)] = count

    return {
        "version": commit_hash,
        "n_base": n_limit,
        "plantri_command": ["-b", "-c3", "-d", "-a"],
        "counts_by_n": counts_by_n,
        "total_instances": len(instances),
        "instances": instances,
    }


def write_log(log_path: Path, json_path: Path, output: Dict[str, object], started_at: float) -> None:
    lines = [
        f"Verification Log - Base Case n={output['n_base']}",
        f"Timestamp: {time.ctime(started_at)}",
        f"Commit: {output['version']}",
        f"Total instances checked: {output['total_instances']}",
        "Counts by n:",
    ]
    counts_by_n = output["counts_by_n"]
    assert isinstance(counts_by_n, dict)
    for n_key in sorted(counts_by_n, key=int):
        lines.append(f"  n={n_key}: {counts_by_n[n_key]}")

    lines.append("")
    lines.append("Solved instances:")
    instances = output["instances"]
    assert isinstance(instances, list)
    for instance in instances:
        assert isinstance(instance, dict)
        lines.append(
            f"  {instance['name']}: n={instance['n']} sha256={instance['graph_sha256']}"
        )
        lines.append(f"    plantri: {instance['plantri_ascii']}")
        lines.append(f"    witness: {instance['witness']}")

    lines.append("")
    lines.append(f"Manifest SHA256: {sha256_hex(json_path)}")
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-limit", type=int, default=14, help="maximum even order to enumerate")
    parser.add_argument(
        "--plantri",
        default=str(default_plantri_path()),
        help="path to the plantri executable",
    )
    parser.add_argument(
        "--json-out",
        default=str(ROOT / "artifacts" / "base_cases_n14.json"),
        help="output JSON artifact",
    )
    parser.add_argument(
        "--log-out",
        default=str(ROOT / "artifacts" / "base_cases_n14.log"),
        help="output log artifact",
    )
    args = parser.parse_args()

    started_at = time.time()
    json_path = Path(args.json_out)
    log_path = Path(args.log_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    output = run_base_case_verification(args.n_limit, Path(args.plantri))

    json_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_log(log_path, json_path, output, started_at)

    print(
        "Base case verification complete. "
        f"{output['total_instances']} instances checked up to n={output['n_base']}."
    )


if __name__ == "__main__":
    main()
