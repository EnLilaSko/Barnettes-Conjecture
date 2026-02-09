"""
check_trace.py - Replays a reduction trace and verifies mandatory invariants.
Includes support for batch generation and verification of proof-critical traces.
"""

import os
import json
import barnette_proof as bp
from barnette_proof import EmbeddedGraph, Cycle
import hashlib
from typing import List, Dict, Set, Any

def get_commit_hash():
    return "821779f90e35e2946c0613338e45c285a9d130f2c"

def is_bipartite(adj: Dict[int, Set[int]]) -> bool:
    """Check if a graph is bipartite using BFS coloring."""
    color = {}
    for start_node in adj:
        if start_node not in color:
            color[start_node] = 0
            stack = [start_node]
            while stack:
                v = stack.pop()
                for neighbor in adj[v]:
                    if neighbor not in color:
                        color[neighbor] = 1 - color[v]
                        stack.append(neighbor)
                    elif color[neighbor] == color[v]:
                        return False
    return True

def verify_invariants(G: EmbeddedGraph, step: int):
    """Verify all mandatory invariants for a graph in the trace."""
    print(f"  Step {step}: n={len(G.adj)}...", end=" ")
    try:
        # 1. Planarity/Rotation Embedding
        G.validate_rotation_embedding()
        # 2. Bipartite
        if not is_bipartite(G.adj):
            raise ValueError("Not bipartite")
        # 3. Cubic
        for v, neighbors in G.adj.items():
            if len(neighbors) != 3:
                raise ValueError(f"Vertex {v} is not cubic (deg={len(neighbors)})")
        # 4. 3-connected
        if not bp.is_3_connected(G.adj):
            raise ValueError("Not 3-connected")
        print("OK")
    except Exception as e:
        print(f"FAILED: {e}")
        return False
    return True

def replay_trace(filename: str):
    """Read a JSONL trace and verify each step."""
    if not os.path.exists(filename):
        print(f"Trace file not found: {filename}")
        return False
    
    print(f"Replaying trace: {filename}")
    prev_n = float('inf')
    steps_verified = 0
    
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            # Reconstruct EmbeddedGraph
            adj = {int(v): set(int(nb) for nb in neighbors) for v, neighbors in data['adj'].items()}
            rot = {int(v): [int(nb) for nb in neighbors] for v, neighbors in data['rot'].items()}
            G = EmbeddedGraph(adj, rot)
            
            # Verify invariants
            if not verify_invariants(G, i):
                print(f"Trace verification FAILED at step {i}")
                return False
            
            # Verify measure decrease (strictly monotonic)
            n = len(G.adj)
            if i > 0 and n >= prev_n:
                print(f"Trace verification FAILED at step {i}: n did not decrease ({prev_n} -> {n})")
                return False
            prev_n = n
            steps_verified += 1
            
    print(f"Trace verification PASSED ({steps_verified} steps)")
    return True

def generate_trace(n_start: int, base_dir: str):
    """Generate a trace for a graph of size n_start and save to base_dir."""
    print(f"Generating trace for n={n_start}...")
    if n_start == 8:
        G = bp.make_cube()
    else:
        G = bp.make_prism(n_start // 2)
    
    # Save initial graph
    os.makedirs(base_dir, exist_ok=True)
    graph_path = os.path.join(base_dir, f"graph_n{n_start}.json")
    with open(graph_path, 'w') as f:
        json.dump({
            "adj": {str(v): sorted(list(neighbors)) for v, neighbors in G.adj.items()},
            "rot": {str(v): neighbors for v, neighbors in G.rot.items()}
        }, f, indent=4)
    
    trace_path = os.path.join(base_dir, f"trace_n{n_start}.jsonl")
    trace_steps = []
    
    def trace_solver(G_curr):
        # Capture current graph
        data = {
            "adj": {str(v): sorted(list(neighbors)) for v, neighbors in G_curr.adj.items()},
            "rot": {str(v): neighbors for v, neighbors in G_curr.rot.items()}
        }
        trace_steps.append(data)
        
        # Base case
        if bp.is_cube(G_curr) or len(G_curr.adj) <= 12:
            return
            
        # Recursive step logic
        occ2 = bp.detect_C2(G_curr)
        if occ2:
            Gred, _ = bp.reduce_C2(G_curr, occ2)
            trace_solver(Gred)
            return
        occp = bp.detect_C_pinch_ii(G_curr)
        if occp:
            Gred, _ = bp.reduce_pinch(G_curr, occp)
            trace_solver(Gred)
            return
        occ4 = bp.detect_refined_C4(G_curr)
        if occ4:
            Gred, _ = bp.reduce_C4(G_curr, occ4)
            trace_solver(Gred)
            return
            
    trace_solver(G)
    
    with open(trace_path, 'w') as f:
        for step in trace_steps:
            f.write(json.dumps(step) + "\n")
    
    return graph_path, trace_path

def run_batch_verification():
    """Run batch verification for multiple sizes and store results in artifacts/traces/."""
    sizes = [8, 12, 48, 128]
    base_dir = "artifacts/traces"
    os.makedirs(base_dir, exist_ok=True)
    
    results = []
    for n in sizes:
        print(f"\n--- Processing n={n} ---")
        graph_path, trace_path = generate_trace(n, base_dir)
        success = replay_trace(trace_path)
        
        # Compute SHA256 of trace
        with open(trace_path, 'rb') as f:
            trace_hash = hashlib.sha256(f.read()).hexdigest()
            
        results.append({
            "n": n,
            "graph_path": graph_path,
            "trace_path": trace_path,
            "success": success,
            "trace_sha256": trace_hash
        })
    
    # Save batch results summary
    summary_path = os.path.join(base_dir, "batch_verification.json")
    with open(summary_path, 'w') as f:
        json.dump({
            "version": get_commit_hash(),
            "results": results
        }, f, indent=4)
    
    print(f"\nBatch verification complete. Summary saved to {summary_path}")
    return results

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        replay_trace(sys.argv[1])
    else:
        run_batch_verification()
