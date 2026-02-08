import traceback
import sys
import barnette_proof as bp

try:
    G = bp.make_prism(8)
    # We suspect KeyError: 0 in find_hamiltonian_cycle -> reduce_C2
    # Let's print vertices and OccC2 if possible
    bp.find_hamiltonian_cycle(G, debug=True)
except Exception:
    traceback.print_exc(file=sys.stdout)
