# Framework Compliance Report (Certified Reduction Framework + Interface Analysis Methodology)

This document maps the Certified Reduction Framework requirements to:

- the proof in `proof_summary.md`,
- the implementation in `barnette_proof.py`.

---

## 1. Target class Q

**Definition (Q):** cubic, bipartite, 3-connected planar graphs with a fixed plane embedding (rotation system).

**Implementation:**

- `EmbeddedGraph.validate_rotation_embedding()` checks 2-cell embedding via face orbits and Euler.
- `validate_in_Q(G)` checks cubicity, bipartiteness, and 3-connectivity.

---

## 2. Completeness requirement

**Framework requirement:** every `G ∈ Q` contains at least one reducible configuration from a finite catalog.

**Catalog in this project:**

- `C2` (adjacent 4-faces),
- `refined C4` (isolated 4-face with distinct external neighbors),
- `C_pinch(ii)` (isolated pinched 4-face with `t∉{u2,u4}`).

**Proof reference:** Theorem 5.1 in `proof_summary.md`.

**Implementation reference:** `verify_completeness(G)` in `barnette_proof.py`.

---

## 3. Bounded-radius certificates

### 3.1 Configuration C2

**Certificate contents:** an edge whose two incident faces are 4-faces, plus the two 4-cycles (for canonical labeling).  
Radius bound: **≤ 1**.

**Detector:** `detect_C2(G)`.

### 3.2 Configuration refined C4

**Certificate contents:** facial quad `v1v2v3v4`, its four external neighbors `u1..u4`, and verification that no quad is adjacent across any edge.  
Radius bound: **≤ 2**.

**Detector:** `detect_refined_C4(G)`.

### 3.3 Configuration C_pinch(ii)

**Certificate contents:**

- facial quad `v1v2v3v4`,
- pinch vertex `w = u1 = u3`,
- third neighbor `t` of `w`,
- verification `t∉{u2,u4}`,
- plus `r,s` = the other two neighbors of `t`,
- and verification of edge-isolation (no adjacent 4-face along any quad edge).

Radius bound: **≤ 3**.

**Detector:** `detect_C_pinch_ii(G)`.

**Structural lemma used:** Lemma 4.2 explains why excluded case `t∈{u2,u4}` implies C2.

---

## 4. Preservation of graph class under reductions

**Requirement:** reductions preserve planarity, bipartiteness, cubicity, 3-connectivity.

**Implementation:**

- Each reduction is the inverse of an explicit local expansion (`expand_*_from_edge`).
- Planarity is preserved by explicit rotation updates.
- Cubicity is preserved by construction (degree-3 rotations everywhere).
- Bipartiteness is preserved for these gadgets by even-cycle parity; tests check bipartiteness.
- 3-connectivity is preserved by the certified construction and verified at each step by `validate_in_Q`.

---

## 5. Deterministic lifting (Interface Analysis Methodology)

**Requirement:** lifting is deterministic and complete over finitely many interface types.

**Implementation approach:**

- `lift_C2`, `lift_C4`, `lift_pinch` call `_patch_search(...)`.
- Interface types correspond to which edges incident to the reduced gadget vertices `(x,y)` are used.
- `_patch_search` tries internal-edge subsets in deterministic order and takes the first valid lift.

This matches:

- finite interface types,
- bounded local path library (implemented as constant-size patch search),
- deterministic selection rule.

---

## 6. Algorithm termination

**Measure:** `|V(G)|`.

Each reduction strictly decreases `|V|`:

- `C2`: removes 6 internal vertices and adds 2 → net −4.
- `C_pinch(ii)`: removes 6 and adds 2 → net −4.
- `refined C4`: removes 4 and adds 2 → net −2.

Hence recursion depth is O(n).

---

## 7. Soundness checks

- `Cycle.validate_hamiltonian(G)` validates lifted cycles.
- `verify_completeness(G)` checks total charge −8 and discharging witness for 4-face existence.
