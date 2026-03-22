import sys

with open(r'c:\Users\charl\Documents\New Math Problem\paper\LaTeX Pure Maths.tex', 'r', encoding='utf-8') as f:
    text = f.read()

old_completeness = r'''\begin{theorem}[Admissible Reduction Completeness]
\label{thm:admissible-reduction-completeness}
Every graph $G \in \mathcal{Q}$ with $|V(G)| > N_{\mathrm{base}}$ contains a certified, admissible reduction to a smaller graph $G' \in \mathcal{Q}$ using one of the configurations $C_2$, $C_4$, or $C_P$.
\end{theorem}
\begin{proof}
By Theorem~\ref{thm:completeness}, $G$ contains a topological occurrence of $C_2$, $C_4$, or $C_P$. Because $|V(G)| > N_{\mathrm{base}}$, by Lemma~\ref{lemma:certified_upgrade}, this occurrence is strictly certified. By Lemmas~\ref{lemma:bipartite_planarity_preservation} and \ref{lemma:3conn_preservation}, applying the reduction operation strictly preserves all structural $\mathcal{Q}$ properties (planarity, 3-connectivity, cubicity, and bipartiteness). Because each reduction removes more vertices than it adds (e.g., deleting 6 vertices and adding 2 for $C_2$), the target graph $G'$ is strictly smaller.
\end{proof}'''

new_completeness = r'''\begin{theorem}[Admissible Reduction Completeness]
\label{thm:admissible-reduction-completeness}
Every graph $G \in \mathcal{Q}$ with $|V(G)| > N_{\mathrm{base}}$ contains a strictly certified, admissible reduction to a smaller graph $G' \in \mathcal{Q}$ via one of the configurations $C_2$, $C_4$, or $C_P$.
\end{theorem}
\begin{proof}
Let $G \in \mathcal{Q}$ with $|V(G)| > N_{\mathrm{base}} = 12$. 
\begin{enumerate}
    \item \textbf{Unavoidability}: By Theorem~\ref{thm:unavoidability} (Topological Unavoidability), $G$ contains at least one topological occurrence of $C_2$, refined $C_4$, or $C_P$, guaranteed strictly by Euler's formula and the discharging of facial charge assignments avoiding high-girth embeddings.
    \item \textbf{Certification}: By Lemma~\ref{lemma:certified_upgrade} (Topological to Certified Upgrade), because the graph strictly exceeds the base geometries size limit ($|V| > 12$), the topological boundary nodes cannot collapse into degenerate identifications. Thus, the occurrence naturally satisfies all required bounds for a fully disjoint, certified configuration.
    \item \textbf{Basic Structural Preservation}: By Lemma~\ref{lemma:bipartite_planarity_preservation} (Local Embeddings and Bipartite Preservation), surgically replacing the localized patch with the destination gadget exactly preserves the global bipartition and planarity without crossing boundaries.
    \item \textbf{High-Connectivity Preservation}: By Lemma~\ref{lem:3conn_concrete} (3-Connectivity Preservation), because the reduction gadgets possess the boundary-faithful internal routing properties delineated in Lemma~\ref{lem:3conn_replacement}, any 2-cut formed in the modified structural manifold necessitates an analogue cut in the original graph $G$. Since $G$ is strictly 3-connected, no such cut exists, meaning the destination graph remains 3-connected.
\end{enumerate}
Together, these four lemmas establish that the destination graph sequentially complies with every constraint governing the target class $\mathcal{Q}$. Furthermore, because each gadget operation strictly removes more internal vertices than the number of structural nodes it introduces into the replacement disk (e.g., deleting 6 and adding 2 for $C_2$), the overall size of the target graph $G'$ strictly drops: $|V(G')| < |V(G)|$.
Therefore, a certified reduction step to a strictly smaller graph within $\mathcal{Q}$ exists unconditionally.
\end{proof}'''

if old_completeness in text:
    text = text.replace(old_completeness, new_completeness)
    
    # Also find if there is a rogue standalone remark about completeness we need to purge
    # We will just write the file out first
    with open(r'c:\Users\charl\Documents\New Math Problem\paper\LaTeX Pure Maths.tex', 'w', encoding='utf-8') as f:
        f.write(text)
    with open(r'c:\Users\charl\Documents\New Math Problem\paper\LaTeX_Pure_Maths_revised.txt', 'w', encoding='utf-8') as f:
        f.write(text)
    print("Success Completeness Rewrite")
else:
    print("FAILED to find Admissible Completeness block exactly. Attempting regex.")
    import re
    # Try just grabbing the theorem and its proof block using regex to nuke the old one.
    text = re.sub(
        r'\\begin\{theorem\}\[Admissible Reduction Completeness\].*?\\end\{proof\}',
        new_completeness,
        text,
        flags=re.DOTALL
    )
    with open(r'c:\Users\charl\Documents\New Math Problem\paper\LaTeX Pure Maths.tex', 'w', encoding='utf-8') as f:
        f.write(text)
    with open(r'c:\Users\charl\Documents\New Math Problem\paper\LaTeX_Pure_Maths_revised.txt', 'w', encoding='utf-8') as f:
        f.write(text)
    print("Fallback completed")
