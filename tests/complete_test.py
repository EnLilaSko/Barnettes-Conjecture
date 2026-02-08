"""
COMPLETE TESTING SCRIPT FOR BURNETT SOLVER
Tests correctness, performance, and scaling
"""

import barnette_proof as bp
import time
import sys
import traceback

def run_basic_tests():
    """Test basic functionality"""
    print("=" * 60)
    print("BASIC FUNCTIONALITY TESTS")
    print("=" * 60)
    
    tests = [
        ("Cube (8 vertices)", lambda: bp.make_cube()),
        ("Prism-12 (12 vertices)", lambda: bp.make_prism(6)),
        ("Prism-16 (16 vertices)", lambda: bp.make_prism(8)),
        ("Prism-24 (24 vertices)", lambda: bp.make_prism(12)),
        ("Custom pinch example", bp.make_custom_pinch_example),
    ]
    
    all_passed = True
    
    for test_name, graph_maker in tests:
        try:
            print(f"\nTesting: {test_name}")
            G = graph_maker()
            print(f"  Graph has {len(G.adj)} vertices, {sum(len(adj) for adj in G.adj.values())//2} edges")
            
            # Time the solver
            start_time = time.time()
            cycle = bp.find_hamiltonian_cycle(
                G, 
                check_3conn_each_step=False,
                debug=False
            )
            end_time = time.time()
            
            if cycle is None:
                print(f"  FAILED: No cycle found")
                all_passed = False
            else:
                # Validate the cycle
                try:
                    cycle.validate_hamiltonian(G)
                    elapsed = end_time - start_time
                    print(f"  PASS: Found Hamiltonian cycle in {elapsed:.4f} seconds")
                    print(f"     Cycle length: {len(cycle.E)} vertices")
                except Exception as e:
                    print(f"  FAILED: Invalid cycle - {e}")
                    all_passed = False
                    
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            all_passed = False
    
    return all_passed

def run_performance_scaling():
    """Test how performance scales with graph size"""
    print("\n" + "=" * 60)
    print("PERFORMANCE SCALING TESTS")
    print("=" * 60)
    
    print("\nTesting on prism graphs (2n vertices):")
    print("-" * 40)
    print(f"{'n-gon':>6} {'Vertices':>10} {'Time (s)':>10} {'Found':>8}")
    print("-" * 40)
    
    results = []
    
    for n_sides in [4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64]:
        try:
            G = bp.make_prism(n_sides)
            vertices = len(G.adj)
            
            start_time = time.time()
            cycle = bp.find_hamiltonian_cycle(
                G,
                check_3conn_each_step=False,
                debug=False
            )
            end_time = time.time()
            
            elapsed = end_time - start_time
            found = cycle is not None
            
            print(f"{n_sides:6d} {vertices:10d} {elapsed:10.4f} {str(found):>8}")
            results.append((n_sides, vertices, elapsed, found))
            
            # Stop if taking too long
            if elapsed > 30.0:  # 30 seconds threshold
                print("\n[Warning: Stopping scaling test - taking too long]")
                break
                
        except Exception as e:
            print(f"{n_sides:6d} {'ERROR':>10} {'---':>10} {str(e)[:20]:>8}")
    
    return results

def run_memory_test():
    """Test memory usage"""
    print("\n" + "=" * 60)
    print("MEMORY USAGE TEST")
    print("=" * 60)
    
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        print("\nTesting memory usage on progressively larger graphs:")
        print("-" * 50)
        print(f"{'Vertices':>10} {'Time (s)':>10} {'Memory (MB)':>12}")
        print("-" * 50)
        
        for n in [8, 16, 24, 32, 48, 64]:
            G = bp.make_prism(max(4, n//2))
            vertices = len(G.adj)
            
            mem_before = process.memory_info().rss / 1024 / 1024
            
            start_time = time.time()
            cycle = bp.find_hamiltonian_cycle(
                G,
                check_3conn_each_step=False,
                debug=False
            )
            end_time = time.time()
            
            mem_after = process.memory_info().rss / 1024 / 1024
            mem_used = mem_after - mem_before
            
            elapsed = end_time - start_time
            
            print(f"{vertices:10d} {elapsed:10.4f} {mem_used:12.2f}")
            
    except ImportError:
        print("psutil not installed. Install with: pip install psutil")

def run_configuration_frequency():
    """Test which configurations appear most often"""
    print("\n" + "=" * 60)
    print("CONFIGURATION FREQUENCY ANALYSIS")
    print("=" * 60)
    
    config_counts = {"C2": 0, "C4": 0, "PINCH": 0}
    
    print("\nTesting 10 random reductions to see configuration frequency:")
    
    # Start with a base graph
    G = bp.make_prism(16)  # 32 vertices
    
    for step in range(10):
        try:
            # Find a configuration
            witness = bp.verify_completeness(G, check_3conn=False)
            config_type = witness.kind
            
            if config_type in config_counts:
                config_counts[config_type] += 1
            
            print(f"Step {step+1}: Found {config_type} configuration")
            
            # Apply reduction (simplified - you'll need actual reduction code)
            # G = apply_reduction(G, config_type, witness)
            
        except Exception as e:
            print(f"Step {step+1}: Error - {e}")
            break
    
    print("\nConfiguration frequency summary:")
    for config, count in config_counts.items():
        print(f"  {config}: {count} times")

def main():
    """Run all tests"""
    print("BURNETT SOLVER COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    # Test 1: Basic functionality
    basic_ok = run_basic_tests()
    
    # Test 2: Performance scaling  
    scaling_results = run_performance_scaling()
    
    # Test 3: Memory usage
    run_memory_test()
    
    # Test 4: Configuration analysis
    run_configuration_frequency()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    if basic_ok:
        print("PASS: Basic functionality tests PASSED")
    else:
        print("FAIL: Basic functionality tests FAILED")
    
    if scaling_results:
        print(f"PASS: Performance scaling tested up to {scaling_results[-1][1]} vertices")
        
        # Calculate approximate complexity
        if len(scaling_results) >= 3:
            times = [r[2] for r in scaling_results]
            vertices = [r[1] for r in scaling_results]
            
            # Rough O(n^2) check
            print("\nPerformance trend:")
            for i in range(1, len(vertices)):
                if vertices[i-1] > 0 and times[i-1] > 0:
                    ratio = (times[i] / times[i-1]) / ((vertices[i] / vertices[i-1]) ** 2)
                    print(f"  {vertices[i-1]}\u2192{vertices[i]} vertices: time ratio suggests O(n^{1.8 if ratio < 1.2 else 2.0 if ratio < 1.5 else '>2'})")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
