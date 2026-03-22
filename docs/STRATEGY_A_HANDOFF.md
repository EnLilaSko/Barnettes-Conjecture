# Strategy-A Handoff

This file is the shortest useful bootstrap for the next Codex session working on Barnette's conjecture in `Strategy-A`.

## What Strategy-A is

`Strategy-A` is the computer-assisted branch. The current proof direction is:

1. Find unavoidable local configurations in Barnette graphs.
2. Replace those local patches by certified smaller gadgets.
3. Revalidate the reduced graph in `Q` after every step.
4. Use finite computation and artifacts to back every nontrivial admissibility claim.

In practice, the branch is now moving away from "three tiny universal rules" and toward a finite interface-family program, especially for refined `C4`.

## Current status

As of commit `3879f0b24ca926cefe18757669b1b65deb0bba51` on branch `Strategy-A`:

- Base cases are real, not placeholders.
- The proof-side completeness scan through 30 vertices is partially improved by a curated refined-`C4` family library.
- `pinch(ii)` is not the immediate blocker.
- `C2` still has a recurring exception family.
- The main open frontier is a small residual set of refined-`C4` families.

## Most important results already established

### 1. Base cases

- `tests/verify_base_cases.py` now performs a real exhaustive `plantri`-based check.
- Main artifact: `artifacts/base_cases_n14.json`

### 2. Proof-side completeness frontier through 30 vertices

Main artifacts:

- `artifacts/completeness_frontier_scan_n30.json`
- `artifacts/completeness_unresolved_analysis_n30.json`
- `artifacts/completeness_frontier_scan_n30_after_refined_c4_library.json`
- `artifacts/completeness_unresolved_analysis_n30_after_refined_c4_library.json`

Important numbers:

- Before the refined-`C4` proof-side library: `42` unresolved graphs through 30 vertices.
- After the library: `35` unresolved graphs through 30 vertices.
- After the library:
  - `7` graphs are certified via refined `C4`
  - `23` unresolved graphs still contain refined `C4`
  - `12` unresolved graphs are `C2`-only failures

### 3. Refined-C4 classification and certified positive families

Key source files:

- `src/refined_c4_local.py`
- `src/refined_c4_gadget_search.py`
- `src/refined_c4_verified_library.py`
- `src/refined_c4_unresolved_family_search.py`
- `src/refined_c4_universal_family_search.py`

Key artifacts:

- `artifacts/refined_c4_scan.json`
- `artifacts/refined_c4_recurrent_skeleton_classes.json`
- `artifacts/refined_c4_nonrecurrent_class_analysis.json`
- `artifacts/refined_c4_unresolved_family_search_top20.json`
- `artifacts/unresolved_refined_c4_family_census_n30_after_refined_c4_library.json`

What is true now:

- The old two-vertex refined-`C4` reducer is not a universal certified rule.
- The 8-vertex obstruction layer was classified computationally.
- There is now a curated proof-side library of certified refined-`C4` gadget families in:
  - `src/refined_c4_verified_library.py`
- That library is used by the proof-side certified reduction path in:
  - `src/barnette_proof.py`

Important limitation:

- The recursive solver does **not** yet use the larger refined-`C4` gadgets for recursion.
- Reason: there is not yet a generalized lift-back library for those larger gadgets.
- So the proof-side checker is ahead of the recursive solver.

### 4. Residual refined-C4 frontier

Main artifact:

- `artifacts/unresolved_refined_c4_family_census_n30_after_refined_c4_library.json`

Residual state after adding the current library:

- `23` unresolved refined-`C4` graphs
- `52` unresolved refined-`C4` occurrences

The most important residual families are still led by:

- `L=6,6,6,6|B=4|I=0,0,1,1,2,2,3,3|A=2,2,2,2|E=1,1,1,1`
- several two-occurrence families at outer-face types:
  - `(6,6,6,10)`
  - `(6,6,6,6)`
  - `(6,8,6,8)`
  - `(6,8,8,10)`

Three stubborn residual two-occurrence families were strengthened from "candidate-only" to "stronger negatives":

- `artifacts/refined_c4_universal_b5_66610_allcols_limit200.json`
- `artifacts/refined_c4_universal_b6_6666_allcols_limit200.json`
- `artifacts/refined_c4_universal_b6_68810_allcols_limit200.json`

What those say:

- `(6,6,6,10) / B=5`: exactly `1` viable seed candidate in the searched 8-vertex layer, and it fails on the sibling occurrence.
- `(6,6,6,6) / B=6`: only `3` viable seed candidates in the searched 8-vertex layer, none universalize.
- `(6,8,8,10) / B=6`: only `2` viable seed candidates in the searched 8-vertex layer, none universalize.

This is the clearest signal that the next step is probably a larger gadget or a new structural lemma, not just re-running the same 8-vertex search.

### 5. Pinch(ii)

Key files:

- `src/pinch_ii_scan.py`
- `src/pinch_ii_local_census.py`

Key artifacts:

- `artifacts/pinch_ii_scan.json`
- `artifacts/pinch_ii_local_census.json`

Current result:

- Up to 40 vertices, no genuine `pinch(ii)` occurrence was found.
- More strongly: no opposite-pinched facial 4-cycle was found at all up to 40 vertices.

Conclusion:

- `pinch(ii)` is not the nearest blocker right now.

### 6. C2

Key files:

- `src/c2_scan.py`
- `src/c2_exception_analysis.py`

Key artifacts:

- `artifacts/c2_scan_n24.json`
- `artifacts/c2_exception_analysis_n30.json`

Current result:

- `C2` works often, but it is not universal.
- There is a recurring distinct-terminal exception family failing by non-3-connectivity.

## Most important code paths

- `src/barnette_proof.py`
  - main reduction logic
  - proof-side certified reduction path
  - solver still limited to liftable rules

- `src/refined_c4_verified_library.py`
  - curated proof-side refined-`C4` gadget family library

- `src/refined_c4_gadget_search.py`
  - core candidate search and replay logic

- `src/refined_c4_universal_family_search.py`
  - search for a gadget that works across all occurrences of one coarse family

- `src/completeness_frontier_scan.py`
  - global proof-side frontier scan

- `src/completeness_unresolved_analysis.py`
  - analysis of graphs with no currently certified step

## Recommended next steps

### Best next move

Attack the residual refined-`C4` families, starting with the smallest stubborn ones:

1. Try targeted 10-vertex work on the residual two-occurrence families.
2. If that stalls, look for a structural invariant explaining why the sibling occurrence fails when the seed occurrence passes.

### After that

1. Build a generalized lift-back library for the larger refined-`C4` gadgets.
   Right now this is the biggest gap between proof-side certification and actual recursive solving.

2. Return to the recurring `C2` exception family.
   That is the second major completeness obstruction after refined `C4`.

3. Only after those two are in better shape, revisit `pinch(ii)`.

## Environment notes

- The repo now ignores:
  - `.python313/`
  - `.venv/`
  - `__pycache__/`
  - `*.pyc`
  - `plantri.exe`

- Those are intentionally local-only.
- On a new machine, the usual setup is:
  - create a local Python environment
  - provide a local `plantri` binary

This means the next Codex should not assume Python or `plantri` are present just because the branch has all the search code.

## Reading order for the next session

If a future session needs the shortest useful warm start, read in this order:

1. `docs/STRATEGY_A_HANDOFF.md`
2. `src/barnette_proof.py`
3. `src/refined_c4_verified_library.py`
4. `artifacts/completeness_frontier_scan_n30_after_refined_c4_library.json`
5. `artifacts/unresolved_refined_c4_family_census_n30_after_refined_c4_library.json`
6. `artifacts/refined_c4_universal_b5_66610_allcols_limit200.json`
7. `artifacts/refined_c4_universal_b6_6666_allcols_limit200.json`
8. `artifacts/refined_c4_universal_b6_68810_allcols_limit200.json`
