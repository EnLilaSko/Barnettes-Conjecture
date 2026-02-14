from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Dict, List, Iterator, Tuple

@dataclass(frozen=True)
class PlantriGraph:
    # rotation system: neighbors in clockwise order at each vertex
    rot: Dict[int, List[int]]

def _parse_plantri_ascii_line(line: str) -> PlantriGraph:
    """
    plantri -a outputs one graph per line, like:
      7 bcdefg,agfdc,abd,acbfe,adf,aedbg,afb
    where vertices are a,b,c,... and each comma-separated chunk is the neighbor
    list (clockwise) for that vertex.
    """
    line = line.strip()
    if not line:
        raise ValueError("empty plantri line")

    parts = line.split()
    if len(parts) != 2:
        raise ValueError(f"unexpected plantri -a line format: {line!r}")

    n_str, adj_str = parts
    n = int(n_str)
    chunks = adj_str.split(",")
    if len(chunks) != n:
        raise ValueError(f"expected {n} adjacency chunks, got {len(chunks)}")

    # plantri labels vertices with 'a','b',...
    # chunk i corresponds to vertex chr(ord('a') + i)
    def v_of(ch: str) -> int:
        return ord(ch) - ord("a")

    rot: Dict[int, List[int]] = {}
    for i, chunk in enumerate(chunks):
        v = i
        rot[v] = [v_of(c) for c in chunk]
    return PlantriGraph(rot=rot)

def iter_barnette_graph_rotations_via_plantri(
    plantri_path: str,
    N_vertices_dual: int,
    connectivity: int = 3,
) -> Iterator[PlantriGraph]:
    """
    Generate Barnette graphs (3-connected bipartite cubic planar) of size N
    by generating Eulerian triangulations and taking duals.

    Requires plantri installed. Uses:
      plantri -b -c{connectivity} -d -a t
    where t = (N + 4)/2
    """
    if N_vertices_dual % 2 != 0:
        raise ValueError("Barnette graphs have even number of vertices; got odd N")

    t = (N_vertices_dual + 4) // 2
    if 2 * t - 4 != N_vertices_dual:
        raise ValueError("N not compatible with dual-of-triangulation sizing")

    cmd = [
        plantri_path,
        "-b",
        f"-c{connectivity}",
        "-d",
        "-a",
        str(t),
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        universal_newlines=True,
    )
    assert proc.stdout is not None

    try:
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            yield _parse_plantri_ascii_line(line)
    finally:
        proc.stdout.close()

    rc = proc.wait()
    if rc != 0:
        err = ""
        if proc.stderr is not None:
            err = proc.stderr.read()
        raise RuntimeError(f"plantri failed (rc={rc}). stderr:\n{err}")
