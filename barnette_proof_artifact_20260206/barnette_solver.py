"""
Barnette's Conjecture Solver - Production CLI
Usage: python barnette_solver.py --graph graph.json --output solution.json
"""

import argparse
import time
import json
import sys
import os
import barnette_proof as bp
from barnette_proof import EmbeddedGraph, Cycle

def load_graph_from_file(input_file: str) -> EmbeddedGraph:
    """
    Load an EmbeddedGraph from a JSON file.
    Expected format:
    {
        "adj": { "0": [1, 2, 3], ... },
        "rot": { "0": [1, 2, 3], ... }
    }
    """
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # JSON keys are always strings, convert to ints
        adj = {int(v): set(neighbors) for v, neighbors in data['adj'].items()}
        rot = {int(v): list(neighbors) for v, neighbors in data['rot'].items()}
        
        G = EmbeddedGraph(adj, rot)
        G.validate_rotation_embedding() # Basic check
        return G
    except Exception as e:
        print(f"Error loading graph: {e}")
        sys.exit(1)

def save_solution_to_file(G: EmbeddedGraph, cycle: Cycle, output_file: str):
    """Save the Hamiltonian cycle to a JSON file."""
    try:
        # Save as both edge list and ordered vertex list
        edge_list = [list(e) for e in cycle.E]
        vertex_order = cycle.as_ordered_cycle(G)
        
        data = {
            "vertices": len(G.adj),
            "edges_count": len(cycle.E),
            "cycle_edges": edge_list,
            "vertex_order": vertex_order
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"Error saving solution: {e}")

def run_benchmarks():
    """Run a small performance suite on standard instances."""
    print("=== Performance Benchmark ===")
    for n in [12, 48, 128]:
        G = bp.make_prism(n)
        print(f"Solving Prism-{n} ({2*n} vertices)...")
        start = time.time()
        cycle = bp.find_hamiltonian_cycle(G)
        elapsed = time.time() - start
        
        if cycle:
            print(f"  ✓ Found in {elapsed:.4f}s")
        else:
            print(f"  ✗ Failed")

def solve_graph_file(input_file, output_file=None, validate=False):
    """Solve a graph from file and save solution"""
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return False

    G = load_graph_from_file(input_file)
    
    print(f"Graph loaded: {len(G.adj)} vertices.")
    start = time.time()
    
    try:
        cycle = bp.find_hamiltonian_cycle(G)
    except Exception as e:
        print(f"Solver error: {e}")
        return False
        
    elapsed = time.time() - start
    
    if cycle is None:
        print(f"No Hamiltonian cycle found (took {elapsed:.3f}s)")
        return False
    
    print(f"✓ Found Hamiltonian cycle in {elapsed:.3f} seconds")
    
    if validate:
        print("Validating solution...")
        try:
            cycle.validate_hamiltonian(G)
            print("  ✓ Solution verified.")
        except Exception as e:
            print(f"  ✗ Solution validation failed: {e}")
    
    if output_file:
        save_solution_to_file(G, cycle, output_file)
        print(f"Solution saved to {output_file}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Barnette's Conjecture Solver - Certified Reduction")
    parser.add_argument("--graph", help="Input graph file (JSON)")
    parser.add_argument("--output", help="Output solution file (JSON)")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--validate", action="store_true", help="Validate solution after solving")
    
    args = parser.parse_args()
    
    if args.benchmark:
        run_benchmarks()
    elif args.graph:
        solve_graph_file(args.graph, args.output, args.validate)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
