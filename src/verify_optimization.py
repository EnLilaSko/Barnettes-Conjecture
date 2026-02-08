import barnette_proof as bp
import advanced_optimizations as opt
import time

def benchmark(solver_name, solver_func, G):
    start = time.time()
    try:
        cycle = solver_func(G)
        elapsed = time.time() - start
        if cycle:
            cycle.validate_hamiltonian(G)
            return elapsed, True
        return elapsed, False
    except Exception as e:
        print(f"Error in {solver_name}: {e}")
        return time.time() - start, False

def run_comparison():
    solver_opt = opt.OptimizedSolver()
    
    print(f"{'n-gon':>6} {'Vertices':>10} {'Original (s)':>15} {'Optimized (s)':>15} {'Speedup':>10}")
    print("-" * 65)
    
    for n in [12, 24, 48, 96, 128, 256, 512]:
        G = bp.make_prism(n)
        # For optimized solver, we wrap the graph to enable face caching
        G_opt = opt.OptimizedEmbeddedGraph(
            {v: set(s) for v, s in G.adj.items()},
            {v: list(r) for v, r in G.rot.items()}
        )
        G_opt.next_id = G.next_id
        
        t_orig, ok_orig = benchmark("Original", bp.find_hamiltonian_cycle, G)
        t_opt, ok_opt = benchmark("Optimized", solver_opt.find_hamiltonian_cycle_optimized, G_opt)
        
        speedup = t_orig / t_opt if t_opt > 0 else 0
        print(f"{n:6d} {2*n:10d} {t_orig:15.4f} {t_opt:15.4f} {speedup:10.2f}x")

if __name__ == "__main__":
    run_comparison()
