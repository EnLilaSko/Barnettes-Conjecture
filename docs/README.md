# README.md
## barnette_solution — certified reductions + completeness proof for Q

This project is a combined **mathematical exposition** and **reference implementation** for a certified-reduction
approach to Barnette’s conjecture in the class:

> **Q** = {cubic, bipartite, 3-connected planar graphs (with a fixed embedding via rotation system)}

It integrates:

1) A **completeness / unavoidability proof** (discharging + local classification) showing every graph in Q contains at least one of three certified configurations.

2) A **deterministic recursive solver** that:
- detects a certified configuration with a bounded-radius certificate,
- applies the corresponding certified reduction,
- recurses on the smaller graph,
- deterministically lifts a Hamilton cycle back to the original graph.

---

## Files

- `barnette_proof.py`  
  Main implementation. Includes:
  - `EmbeddedGraph` (adjacency + rotation system + face tracing)
  - configuration detectors (`detect_C2`, `detect_refined_C4`, `detect_C_pinch_ii`)
  - discharging computations (`total_initial_charge`, `apply_discharging_R`)
  - completeness witness (`verify_completeness`)
  - solver (`find_hamiltonian_cycle`)
  - example generators (`make_cube`, `make_prism`, etc.)
  - unit-style validations (planarity via Euler/face orbits; Hamilton cycle validation)

- `proof_summary.md`  
  Mathematical proof of:
  - total charge `-8` (Euler),
  - discharging rule forcing existence of a 4-face,
  - classification of any 4-face into one of the three configurations:
    - **C₂** (adjacent 4-faces),
    - **refined C₄** (edge-isolated 4-face with distinct external neighbors),
    - **C_pinch(ii)** (edge-isolated pinched 4-face with `t∉{u2,u4}`).

- `framework_compliance.md`  
  Checklist mapping framework requirements to proof/code components.

- `examples.ipynb`  
  Example gallery:
  - cube (base case),
  - octagonal prism (C₂ expected),
  - custom pinch(ii) example,
  - truncated octahedron (squares + hexagons).

- `test_completeness.py`  
  Basic tests for:
  - charge identity `-8`,
  - 4-face existence from discharging,
  - existence of a completeness witness,
  - solver returns a valid Hamiltonian cycle on selected examples.

- `benchmark_results.md`  
  Simple benchmark harness notes.

---

## The three certified configurations

### 1) C₂ (adjacent 4-faces)
An edge whose two incident faces are both 4-faces.

### 2) refined C₄ (isolated 4-face, distinct externals)
A facial 4-cycle `v1v2v3v4` such that:
- each `vi` has external neighbor `ui` not in the quad,
- `u1,u2,u3,u4` are all distinct,
- no quad edge is shared with another 4-face.

### 3) C_pinch(ii) (isolated pinched quad)
A facial 4-cycle `v1v2v3v4` with external neighbors satisfying:
- `u1=u3=w` (pinch vertex),
- let `t` be the third neighbor of `w` (besides `v1,v3`),
- require `t∉{u2,u4}`,
- and the quad is edge-isolated (no adjacent 4-face across any quad edge).

---

## Quickstart

### Run examples
```bash
python barnette_proof.py
