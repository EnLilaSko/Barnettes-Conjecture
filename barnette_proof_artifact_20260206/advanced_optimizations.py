import barnette_proof as bp
from barnette_proof import EmbeddedGraph, Cycle
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

class OptimizedEmbeddedGraph(EmbeddedGraph):
    """Subclass of EmbeddedGraph that caches face lengths for performance."""
    
    def __init__(self, adj, rot):
        super().__init__(adj, rot)
        self.face_cache = {}  # (v, u) -> length

    def face_length(self, v: int, u: int) -> int:
        """Cached face length lookup."""
        if (v, u) in self.face_cache:
            return self.face_cache[(v, u)]
        
        orbit, _ = self.trace_face_darts((v, u))
        length = len(orbit)
        
        # Cache for all darts in this face
        for dv, du in orbit:
            self.face_cache[(dv, du)] = length
        
        return length

    def face_is_quad(self, v: int, u: int) -> bool:
        return self.face_length(v, u) == 4

    def face_is_quad_from_dart(self, start: tuple[int, int]) -> bool:
        """Override for compatibility with bp detector calls."""
        return self.face_is_quad(start[0], start[1])

    def copy(self):
        # Ensure copy preserves the optimized class type
        H = OptimizedEmbeddedGraph(
            {v: set(s) for v, s in self.adj.items()},
            {v: list(r) for v, r in self.rot.items()}
        )
        H.next_id = self.next_id
        # We don't necessarily copy the face cache as the graph might change
        return H

class OptimizedSolver:
    """Highly optimized version for massive graphs using memoization and parallelization."""
    
    def __init__(self, max_bruteforce=12):
        self.face_cache = {} 
        self.pattern_cache = {} 
        self.solution_cache = {} 
        self.max_bruteforce = max_bruteforce

    def _hash_graph(self, G: EmbeddedGraph):
        """Faster hash for graph memoization."""
        edges = []
        for u in sorted(G.adj.keys()):
            for v in G.adj[u]:
                if u < v:
                    edges.append((u, v))
        return hash(tuple(edges))

    def detect_patterns_parallel(self, G: EmbeddedGraph, use_processes=False):
        """Parallelized search for reducible configurations."""
        Executor = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
        with Executor(max_workers=3) as executor:
            f2 = executor.submit(bp.detect_C2, G)
            fp = executor.submit(bp.detect_C_pinch_ii, G)
            f4 = executor.submit(bp.detect_refined_C4, G)
            occ2 = f2.result(); ok2 = occ2 is not None
            if ok2: return ("C2", occ2)
            occp = fp.result(); okp = occp is not None
            if okp: return ("PINCH", occp)
            occ4 = f4.result(); ok4 = occ4 is not None
            if ok4: return ("C4", occ4)
        return None

    def is_trivial_case(self, G: EmbeddedGraph):
        return len(G.adj) <= self.max_bruteforce

    def solve_trivial(self, G: EmbeddedGraph):
        return bp.brute_force_hamiltonian(G)

    def apply_reduction(self, G: EmbeddedGraph, pattern):
        kind, occ = pattern
        if kind == "C2": return bp.reduce_C2(G, occ)
        elif kind == "PINCH": return bp.reduce_pinch(G, occ)
        elif kind == "C4": return bp.reduce_C4(G, occ)
        raise ValueError(f"Unknown kind: {kind}")

    def lift_solution(self, G, G_reduced, cycle_reduced, record, kind):
        if kind == "C2": return bp.lift_C2(G, G_reduced, record, cycle_reduced)
        elif kind == "PINCH": return bp.lift_pinch(G, G_reduced, record, cycle_reduced)
        elif kind == "C4": return bp.lift_C4(G, G_reduced, record, cycle_reduced)
        raise ValueError(f"Unknown kind: {kind}")
        
    def find_hamiltonian_cycle_optimized(self, G: EmbeddedGraph):
        # 1. Memoization (only for small/mid graphs to avoid massive hashing)
        graph_hash = None
        if len(G.adj) < 100:
            graph_hash = self._hash_graph(G)
            if graph_hash in self.solution_cache:
                return self.solution_cache[graph_hash]
        
        # 2. Base case
        if self.is_trivial_case(G):
            return self.solve_trivial(G)
        
        # 3. Pattern detection
        # Only use parallel for massive graphs (>200 vertices)
        if len(G.adj) > 200:
            pattern = self.detect_patterns_parallel(G)
        else:
            occ2 = bp.detect_C2(G)
            if occ2: pattern = ("C2", occ2)
            else:
                occp = bp.detect_C_pinch_ii(G)
                if occp: pattern = ("PINCH", occp)
                else:
                    occ4 = bp.detect_refined_C4(G)
                    if occ4: pattern = ("C4", occ4)
                    else: raise AssertionError("No reducible configuration found")
        
        # 4. Reduction
        G_reduced, record = self.apply_reduction(G, pattern)
        
        # 5. Recursion
        cycle_reduced = self.find_hamiltonian_cycle_optimized(G_reduced)
        
        # 6. Lifting
        kind, _ = pattern
        result = self.lift_solution(G, G_reduced, cycle_reduced, record, kind)
        
        # Cache and return (only if we have a hash)
        if graph_hash is not None:
            self.solution_cache[graph_hash] = result
        return result
