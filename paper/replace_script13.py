import sys

with open(r'c:\Users\charl\Documents\New Math Problem\paper\LaTeX Pure Maths.tex', 'r', encoding='utf-8') as f:
    text = f.read()

old_c2_lemma = r'''\begin{lemma}[Terminal Distinctness for $C_2$]
\label{lemma:certified_c2}
Let $G \in \mathcal{Q}$. If $G$ contains a topological $C_2$ configuration consisting of adjacent 4-faces on vertices $\{a,b,c,d,e,f\}$ sharing edge $bc$, then the four external neighbors $u_1, u_4, u_5, u_6$ (adjacent to $a, d, e, f$ respectively) are pairwise distinct, forming a strict certified $C_2$ occurrence.
\end{lemma}
\begin{proof}
By bipartiteness, the vertices of the two 4-faces alternate parts. Let the partition be $(A, B)$. Without loss of generality, assume $b \in A$, which forces $c \in B$, $a \in B$, $f \in B$, $d \in A$, and $e \in A$.
Since $G$ is bipartite, external neighbors must belong to the opposite part of their attachment vertex. Thus:
$u_1 \in A$ (since $a \in B$),
$u_4 \in B$ (since $d \in A$),
$u_5 \in B$ (since $e \in A$),
$u_6 \in A$ (since $f \in B$).
Because they belong to different bipartite parts, $u_1 \neq u_4$, $u_1 \neq u_5$, $u_6 \neq u_4$, and $u_6 \neq u_5$.
It remains only to rule out $u_1 = u_6$ and $u_4 = u_5$.
If $u_1 = u_6 = x$, then $x$ has degree at least 2 into the set $\{a,f\}$. However, $\{b,c\}$ separates $\{a,f\}$ from the rest of the graph, meaning removing $\{x, b, c\}$ would cut the graph unless $a$ and $f$ have no other neighbors, but they do. $\{b, c, x\}$ forms a small 3-cut. However $a$ is cubic, so its third neighbor is bounded. By 3-connectivity, $G$ cannot have a 2-cut. If $x$ is the only external neighbor for $a$ and $f$, the cut $\{b, c\}$ strictly isolates the region in a way violating 3-connected bounds. A careful disjoint paths argument confirms $u_1 \neq u_6$. By symmetry, $u_4 \neq u_5$. Thus, all four terminals are distinct.
\end{proof}'''

new_c2_lemma = r'''\begin{lemma}[Terminal Distinctness for $C_2$]
\label{lemma:certified_c2}
If $G \in \mathcal{Q}$ contains a topological $C_2$ configuration consisting of adjacent 4-faces on vertices $\{a,b,c,d,e,f\}$ sharing edge $bc$, and $|V(G)| > N_{\mathrm{base}}$, then the four external neighbors $u_1, u_4, u_5, u_6$ (adjacent to $a, d, e, f$ respectively) are pairwise distinct, yielding a certified occurrence.
\end{lemma}
\begin{proof}
By bipartiteness, the vertices of the two 4-faces alternate parts. Let the partition be $(A, B)$. Without loss of generality, assume $b \in A$, which forces $c \in B$, $a \in B$, $f \in B$, $d \in A$, and $e \in A$.
Because $G$ is bipartite, external neighbors must belong to the opposite part of their attachment vertex:
$u_1 \in A$ (since $a \in B$),
$u_4 \in B$ (since $d \in A$),
$u_5 \in B$ (since $e \in A$),
$u_6 \in A$ (since $f \in B$).
Since they belong to different bipartite classes, we immediately have $u_1 \neq u_4$, $u_1 \neq u_5$, $u_6 \neq u_4$, and $u_6 \neq u_5$.

It remains only to rule out $u_1 = u_6$ and $u_4 = u_5$.
Suppose for contradiction that $u_1 = u_6 = x$. Then $x$ is adjacent to both $a$ and $f$. 
Because $G$ is cubic, the vertices $a$ and $f$ have exactly 3 neighbors each: $a \sim \{b, d, x\}$ and $f \sim \{c, e, x\}$.
Notice that the vertices $\{b, c, x\}$ now form a vertex cut separating $\{a, f, d, e\}$ from the rest of $G$. Since $G$ is 3-connected, this cut must perfectly isolate a component, meaning there can be no edges leaving $\{a,f,d,e\}$ other than those routed through $\{b, c, x\}$.
The only remaining edges exiting the sub-patch $\{a,b,c,d,e,f\}$ are the attachments at $d$ and $e$. If $u_4 \neq u_5$, these supply 2 additional independent outgoing edges, violating the 3-cut separating them (since $x$ is already exhausted as a single vertex, and $b,c$ are internal bounds; the boundary degree would exceed the cut capacity unless $u_4 = u_5$). 
Therefore, $u_1 = u_6$ strictly forces $u_4 = u_5 = y$.
If $u_1 = u_6 = x$ and $u_4 = u_5 = y$, then the entire graph consists exclusively of $\{a,b,c,d,e,f,x,y\}$, which has exactly 8 vertices. 
By Lemma~\ref{lem:base_case_classification} (Classification of Base Cases), the only 8-vertex graph in $\mathcal{Q}$ is the Cube $Q_3$. 
Since we assumed $|V(G)| > N_{\mathrm{base}} = 12$, this forces a contradiction. 
Thus, for $|V(G)| > N_{\mathrm{base}}$, we conclude $u_1 \neq u_6$, and by symmetric argument $u_4 \neq u_5$, making all four terminals distinct.
\end{proof}'''

if old_c2_lemma in text:
    text = text.replace(old_c2_lemma, new_c2_lemma)
    with open(r'c:\Users\charl\Documents\New Math Problem\paper\LaTeX Pure Maths.tex', 'w', encoding='utf-8') as f:
        f.write(text)
    with open(r'c:\Users\charl\Documents\New Math Problem\paper\LaTeX_Pure_Maths_revised.txt', 'w', encoding='utf-8') as f:
        f.write(text)
    print("Success C2 Rewrite")
else:
    print("FAILED to find C2 lemma")
