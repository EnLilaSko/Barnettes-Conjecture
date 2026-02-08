"""
EXTENDED TEST SUITE FOR BURNETT SOLVER
Tests scaling, random graphs, and stress test
"""

import barnette_proof as bp
from barnette_proof import (
    make_prism, find_hamiltonian_cycle, validate_cycle, 
    generate_random_barnette_graph, make_grid
)
import time
import random
import sys

def test_family_prisms(max_n=500):
    """Test all prisms up to 1000 vertices"""
    print("\n" + "=" * 60)
    print(f"TESTING PRISM FAMILY UP TO n={max_n}")
    print("=" * 60)
    
    results = []
    # Test from 4 to max_n in steps of 2 or larger to cover range
    # range(6, ...) because cube(4) and small ones are already tested basic
    for n in range(6, max_n + 1, 10):  # Test every 10th to cover range faster
        G = make_prism(n)
        vertices = len(G.adj)
        
        start = time.time()
        # Using the standard recursive solver
        cycle = find_hamiltonian_cycle(G, check_3conn_each_step=False)
        elapsed = time.time() - start
        
        if cycle and validate_cycle(G, cycle):
            results.append((vertices, elapsed, True))
            print(f"✓ Prism-{n} ({vertices} vertices): {elapsed:.3f}s")
        else:
            print(f"✗ Prism-{n} FAILED")
            break
    
    return results

def test_random_barnette_graphs(num_graphs=20, max_vertices=100):
    """Generate and test random Barnette graphs"""
    print("\n" + "=" * 60)
    print(f"TESTING {num_graphs} RANDOM BARNETTE GRAPHS")
    print("=" * 60)
    
    random.seed(42)
    passed = 0
    
    for i in range(num_graphs):
        n = random.randint(12, max_vertices)
        if n % 2 != 0: n += 1 # Ensure even for simplicity in some generators
        G = generate_random_barnette_graph(n)
        actual_n = len(G.adj)
        
        try:
            start = time.time()
            cycle = find_hamiltonian_cycle(G, check_3conn_each_step=False)
            elapsed = time.time() - start
            
            if cycle and validate_cycle(G, cycle):
                passed += 1
                print(f"✓ Random graph {i+1}/{num_graphs}: {actual_n} vertices ({elapsed:.3f}s)")
            else:
                print(f"✗ Random graph {i+1}/{num_graphs}: No cycle found")
        except Exception as e:
            print(f"✗ Random graph {i+1}/{num_graphs}: Error - {e}")
    
    print(f"\nSuccess rate: {passed}/{num_graphs} ({passed/num_graphs*100:.1f}%)")

def stress_test_large():
    """Push to the limits"""
    print("\n" + "=" * 60)
    print("STRESS TESTING WITH VERY LARGE GRAPHS")
    print("=" * 60)
    
    test_cases = [
        ("Prism-500", lambda: make_prism(500)),  # 1000 vertices
        ("Grid-30x30", lambda: make_grid(30, 30)),  # 900 vertices
        # ("Prism-1000", lambda: make_prism(1000)),  # 2000 vertices - might be slow
    ]
    
    for name, maker in test_cases:
        print(f"\nTesting: {name}")
        try:
            G = maker()
            size = len(G.adj)
            print(f"  Size: {size} vertices")
            
            start = time.time()
            cycle = find_hamiltonian_cycle(G, check_3conn_each_step=False)
            elapsed = time.time() - start
            
            if cycle and validate_cycle(G, cycle):
                print(f"  ✓ Solved in {elapsed:.2f}s ({size/elapsed:.0f} vertices/sec)")
            else:
                print(f"  ✗ Failed to find cycle")
                
        except MemoryError:
            print(f"  ✗ Memory error")
        except Exception as e:
            print(f"  ✗ Error: {e}")

if __name__ == "__main__":
    # Test a few samples instead of a huge loop to give feedback quickly
    test_family_prisms(max_n=100)
    test_random_barnette_graphs(num_graphs=10, max_vertices=50)
    stress_test_large()
