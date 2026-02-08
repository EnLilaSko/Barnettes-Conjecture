import barnette_proof as bp
import time

print("=== Basic Functionality Test ===")

# Test 1: Cube (8 vertices)
print("\n1. Testing cube (8 vertices):")
G1 = bp.make_cube()
print(f"   Graph has {len(G1.adj)} vertices")
t0 = time.time()
C1 = bp.find_hamiltonian_cycle(G1, check_3conn_each_step=False, debug=False)
t1 = time.time()
print(f"   Found Hamiltonian cycle: {C1 is not None}")
print(f"   Time: {t1-t0:.4f} seconds")
if C1:
    C1.validate_hamiltonian(G1)
    print("   ✓ Cycle validation passed!")

# Test 2: Prism (16 vertices)
print("\n2. Testing prism (16 vertices):")
G2 = bp.make_prism(8)  # 8-gonal prism = 16 vertices
print(f"   Graph has {len(G2.adj)} vertices")
t0 = time.time()
C2 = bp.find_hamiltonian_cycle(G2, check_3conn_each_step=False, debug=False)
t1 = time.time()
print(f"   Found Hamiltonian cycle: {C2 is not None}")
print(f"   Time: {t1-t0:.4f} seconds")

# Test 3: Performance scaling
print("\n3. Performance scaling test:")
for n_vertices in [8, 12, 16, 20, 24]:
    G = bp.make_prism(max(4, n_vertices//2))
    t0 = time.time()
    C = bp.find_hamiltonian_cycle(G, check_3conn_each_step=False, debug=False)
    t1 = time.time()
    print(f"   {len(G.adj):2d} vertices: {t1-t0:.4f} seconds")
