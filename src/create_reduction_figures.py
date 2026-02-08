"""
Create publication-quality figures for the three reduction configurations
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def create_c2_reduction_figure():
    """Create figure showing C2 reduction (adjacent 4-faces)"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Original graph (two adjacent 4-faces)
    G1 = nx.Graph()
    
    # Two adjacent squares: vertices 0-3 (square 1) and 4-7 (square 2)
    # They share edge (1,4) and (2,5)
    square1 = [(0,1), (1,2), (2,3), (3,0)]
    square2 = [(4,5), (5,6), (6,7), (7,4)]
    shared_edges = [(1,4), (2,5)]
    
    G1.add_edges_from(square1 + square2 + shared_edges)
    
    # External neighbors
    G1.add_edges_from([(0,8), (3,9), (6,10), (7,11)])
    
    # Layout
    pos = {
        0: (0, 2), 1: (1, 2), 2: (1, 1), 3: (0, 1),
        4: (2, 2), 5: (3, 2), 6: (3, 1), 7: (2, 1),
        8: (-1, 1.5), 9: (-1, 0.5), 10: (4, 1.5), 11: (4, 0.5)
    }
    
    nx.draw(G1, pos, ax=ax1, with_labels=True, node_color='lightblue', 
            node_size=500, font_size=10, font_weight='bold')
    
    # Highlight the two 4-faces
    nx.draw_networkx_edges(G1, pos, edgelist=square1, ax=ax1, 
                          width=3, alpha=0.5, edge_color='red')
    nx.draw_networkx_edges(G1, pos, edgelist=square2, ax=ax1, 
                          width=3, alpha=0.5, edge_color='blue')
    
    ax1.set_title('Original: $C_2$ Configuration\n(Adjacent 4-faces)', fontsize=12)
    # Corrected label: removed the manual text call that might overlap or be misplaced
    
    # Reduced graph
    G2 = nx.Graph()
    G2.add_edges_from([(8, 'x'), ('x', 9), (10, 'y'), ('y', 11), ('x', 'y')])
    
    pos2 = {8: (-1, 1.5), 'x': (0.5, 1.5), 9: (-1, 0.5),
            10: (2, 1.5), 'y': (1, 1.5), 11: (2, 0.5)}
    
    nx.draw(G2, pos2, ax=ax2, with_labels=True, node_color='lightgreen',
            node_size=500, font_size=10, font_weight='bold')
    
    ax2.set_title('Reduced: 2-vertex gadget\n$x:[u_1, y, u_6],\ y:[u_4, u_5, x]$', fontsize=12)
    
    plt.suptitle('$C_2$ Reduction: Adjacent 4-faces → 2-vertex gadget', fontsize=14)
    plt.tight_layout()
    plt.savefig('c2_reduction.pdf', dpi=300)
    plt.savefig('c2_reduction.png', dpi=300)
    print("Created c2_reduction.pdf and .png")

def create_c4_reduction_figure():
    """Create figure showing refined C4 reduction"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Original: isolated 4-face
    G1 = nx.Graph()
    
    # Square vertices
    square = [(0,1), (1,2), (2,3), (3,0)]
    G1.add_edges_from(square)
    
    # External neighbors
    G1.add_edges_from([(0,4), (1,5), (2,6), (3,7)])
    
    pos = {
        0: (0, 1), 1: (1, 1), 2: (1, 0), 3: (0, 0),
        4: (-1, 0.5), 5: (0.5, 2), 6: (2, 0.5), 7: (0.5, -1)
    }
    
    nx.draw(G1, pos, ax=ax1, with_labels=True, node_color='lightblue',
            node_size=500, font_size=10, font_weight='bold')
    
    # Highlight the 4-face
    nx.draw_networkx_edges(G1, pos, edgelist=square, ax=ax1,
                          width=3, alpha=0.5, edge_color='red')
    
    ax1.set_title('Original: Refined $C_4$\n(Isolated 4-face, distinct neighbors)', fontsize=12)
    
    # Reduced: single edge
    G2 = nx.Graph()
    G2.add_edges_from([(4, 'x'), ('x', 'y'), ('y', 6), (5, 'x'), (7, 'y')])
    
    pos2 = {
        4: (-1, 0.5), 'x': (0, 0.5), 'y': (1, 0.5), 6: (2, 0.5),
        5: (0, 1.5), 7: (0, -0.5)
    }
    
    nx.draw(G2, pos2, ax=ax2, with_labels=True, node_color='lightgreen',
            node_size=500, font_size=10, font_weight='bold')
    
    ax2.set_title('Reduced: Edge $xy$\n$x:[u_1, y, u_3],\ y:[u_2, u_4, x]$', fontsize=12)
    
    plt.suptitle('Refined $C_4$ Reduction: Isolated 4-face → edge', fontsize=14)
    plt.tight_layout()
    plt.savefig('c4_reduction.pdf', dpi=300)
    plt.savefig('c4_reduction.png', dpi=300)
    print("Created c4_reduction.pdf and .png")

def create_pinch_reduction_figure():
    """Create figure showing C_pinch(ii) reduction"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Original: pinched configuration
    G1 = nx.Graph()
    
    # Pinched square: v1-w-v3-v4 where w is pinch vertex
    # Actually: v1-v2-v3-v4 square with u1=u3=w
    square = [(0,1), (1,2), (2,3), (3,0)]
    G1.add_edges_from(square)
    
    # External connections: u1=u3=w, plus u2, u4
    G1.add_edges_from([(0,'w'), (2,'w'), (1,5), (3,6)])
    
    # Third neighbor t of w
    G1.add_edges_from([('w', 't')])
    # Neighbors r,s of t
    G1.add_edges_from([('t', 'r'), ('t', 's')])
    
    pos = {
        0: (0, 1), 1: (1, 1), 2: (1, 0), 3: (0, 0),
        'w': (0.5, 0.5), 5: (1.5, 1), 6: (1.5, 0),
        't': (0.5, -0.5), 'r': (-0.5, -0.5), 's': (1.5, -0.5)
    }
    
    nx.draw(G1, pos, ax=ax1, with_labels=True, node_color='lightblue',
            node_size=500, font_size=10, font_weight='bold')
    
    # Highlight the square
    nx.draw_networkx_edges(G1, pos, edgelist=square, ax=ax1,
                          width=3, alpha=0.5, edge_color='red')
    
    ax1.set_title('Original: $C_{pinch}(ii)$\n(Pinched 4-face, $t\\notin\\{u_2,u_4\\}$)', fontsize=12)
    
    # Reduced graph
    G2 = nx.Graph()
    G2.add_edges_from([('r', 'x'), ('s', 'x'), ('x', 'y'), ('y', 5), ('y', 6)])
    
    pos2 = {
        'r': (-1, 0), 'x': (0, 0), 's': (1, 0),
        'y': (0, 1), 5: (1, 1), 6: (-1, 1)
    }
    
    nx.draw(G2, pos2, ax=ax2, with_labels=True, node_color='lightgreen',
            node_size=500, font_size=10, font_weight='bold')
    
    ax2.set_title('Reduced: Gadget with $x,y$\n$x:[r, s, y],\ y:[u_2, u_4, x]$', fontsize=12)
    
    plt.suptitle('$C_{pinch}(ii)$ Reduction: Pinched configuration → gadget', fontsize=14)
    plt.tight_layout()
    plt.savefig('pinch_reduction.pdf', dpi=300)
    plt.savefig('pinch_reduction.png', dpi=300)
    print("Created pinch_reduction.pdf and .png")

def create_all_figures():
    """Create all reduction figures"""
    print("Creating reduction figures for publication...")
    import os
    # Ensure fonts are compatible
    plt.rcParams.update({'font.size': 10})
    
    create_c2_reduction_figure()
    create_c4_reduction_figure()
    create_pinch_reduction_figure()
    print("\nAll figures created successfully!")
    print("Files created:")
    print("  - c2_reduction.pdf/.png")
    print("  - c4_reduction.pdf/.png")
    print("  - pinch_reduction.pdf/.png")

if __name__ == "__main__":
    create_all_figures()
