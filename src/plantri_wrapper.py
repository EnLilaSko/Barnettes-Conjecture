from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Dict, Iterator, List


@dataclass(frozen=True)
class PlantriGraph:
    rot: Dict[int, List[int]]
    raw_line: str


def _label_to_vertex(label: str) -> int:
    if len(label) == 1 and ord(label) >= ord("a"):
        # plantri's -a output uses successive byte values starting at 'a'
        # once it runs past 'z', so latin-1 decoding preserves the labels.
        return ord(label) - ord("a")
    raise ValueError(f"unsupported plantri vertex label {label!r}")


def parse_plantri_ascii_embedding(line: str) -> PlantriGraph:
    """
    Parse one `plantri -a` line into a rotation system.

    Example:
      8 bcd,aef,afg,age,bdh,bhc,chd,egf
    """
    line = line.rstrip("\r\n")
    if not line:
        raise ValueError("empty plantri output line")

    head, sep, tail = line.partition(" ")
    if not sep:
        raise ValueError(f"unexpected plantri line format: {line!r}")

    n = int(head)
    chunks = tail.split(",")
    if len(chunks) != n:
        raise ValueError(f"expected {n} adjacency chunks, got {len(chunks)}")

    rot: Dict[int, List[int]] = {}
    for v, chunk in enumerate(chunks):
        rot[v] = [_label_to_vertex(label) for label in chunk]
    return PlantriGraph(rot=rot, raw_line=line)


def iter_barnette_graph_rotations_via_plantri(
    plantri_path: str,
    n_vertices: int,
    connectivity: int = 3,
) -> Iterator[PlantriGraph]:
    """
    Yield one embedded representative for each graph in Q on `n_vertices`.

    `plantri -b -c# -d -a t` outputs bipartite cubic plane graphs that are dual
    to Eulerian triangulations on `t = (n_vertices + 4) / 2` vertices.
    """
    if n_vertices % 2 != 0:
        raise ValueError("Barnette graphs must have an even number of vertices")

    tri_vertices = (n_vertices + 4) // 2
    if 2 * tri_vertices - 4 != n_vertices:
        raise ValueError(f"invalid Barnette graph size {n_vertices}")

    plantri = Path(plantri_path)
    if not plantri.is_absolute():
        plantri = Path.cwd() / plantri

    cmd = [
        str(plantri),
        "-b",
        f"-c{connectivity}",
        "-d",
        "-a",
        str(tri_vertices),
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
        cwd=str(plantri.parent),
    )
    assert proc.stdout is not None

    try:
        for raw_line in proc.stdout:
            line = raw_line.decode("latin-1").rstrip("\r\n")
            if not line or "," not in line:
                continue
            yield parse_plantri_ascii_embedding(line)
    finally:
        proc.stdout.close()

    stderr = ""
    if proc.stderr is not None:
        stderr = proc.stderr.read()
        proc.stderr.close()

    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"plantri failed (rc={rc}) with stderr:\n{stderr}")
