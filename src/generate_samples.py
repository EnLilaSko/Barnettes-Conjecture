import barnette_proof as bp
import json

def export_graph(G, filename):
    data = {
        "adj": {str(v): list(neighbors) for v, neighbors in G.adj.items()},
        "rot": {str(v): neighbors for v, neighbors in G.rot.items()}
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    # Create a few sample graphs
    export_graph(bp.make_prism(12), "prism12.json") # 24 vertices
    export_graph(bp.make_prism(48), "prism48.json") # 96 vertices
    print("Exported prism12.json and prism48.json")
