# CRF Verification Logic and Semantics

## 1. Core Semantic Choice: Combinatorial Embedding

The graph state is defined as `(V, E, rot)`, where `rot[v]` is the cyclic order of neighbors of `v` (length 3 in cubic graphs).

### Invariant Checking

1. **Half-edges**: Build half-edges $h=(u \to v)$ for each undirected edge $\{u,v\}$.
2. **Permutations**:
   - $\alpha(h) = (v \to u)$ (twin)
   - $\sigma(h) = (u \to w)$ where $w$ is the successor of $v$ in `rot[u]`
   - **Face permutation**: $\varphi = \sigma \circ \alpha$
3. **Planarity Check**: Faces are the orbits of $\varphi$. Recompute $F = \#\text{orbits}(\varphi)$ and verify Euler characteristic $|V| - |E| + F = 2$.

## 2. Strict Verification Rules

### Rule A: Rotation Authority

The checker MUST reject the trace if:

- `rotation[v]` contains a neighbor `x` but edge $\{v,x\}$ is missing in the edge set.
- Degree of any vertex $\neq 3$.
- Rotations are not mutually consistent across incident edges.

### Rule B: Edit Completeness

The checker MUST reject the trace if:

- Any edge incident to `remove_vertices` survives after the edit.
- Any edge incident to `add_vertices` is missing from `add_edges`.
- Any entry in `rotation_updates` does not list exactly 3 neighbors.

### Rule C: Global Topology

The checker MUST reject the trace if:

- The graph is not 3-vertex-connected at any step.
- The graph is not bipartite at any step.

## 3. Schema Definitions

- `trace.min.schema.json`: Minimal schema for deterministic replay and audit.
- `trace.full.schema.json`: Comprehensive schema including hashes, provenance, and solver artifacts.
