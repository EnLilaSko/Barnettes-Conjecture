import sys

with open(r'c:\Users\charl\Documents\New Math Problem\paper\LaTeX Pure Maths.tex', 'r', encoding='utf-8') as f:
    text = f.read()

old_upgrade = r'''\begin{lemma}[Topological to Certified Upgrade]
\label{lemma:certified_upgrade}
For any $G \in \mathcal{Q}$ with $|V(G)| > N_{\mathrm{base}}$, every topological occurrence of $C_2$, $C_P$, or refined $C_4$ forms a strict certified occurrence (satisfying all side-conditions of Section~\ref{sec:catalog}).
\end{lemma}
\begin{proof}
For $C_2$ and $C_P$, Lemmas~\ref{lemma:certified_c2} and \ref{lemma:certified_cp} establish the necessary terminal distinctness (where the $C_2$ distinctness firmly relies on the bound $|V(G)| > N_{\mathrm{base}}$ to break structural degeneracies like the Cube). For refined $C_4$, the topological definition explicitly requires the four external neighbors to be distinct, which inherently satisfies the catalog certificate conditions (Definition~\ref{def:refinedC4-occ}).
\end{proof}'''

new_upgrade = r'''\begin{lemma}[Topological to Certified Upgrade]
\label{lemma:certified_upgrade}
For every $G \in \mathcal{Q}$ with $|V(G)| > N_{\mathrm{base}}$, the topological occurrences of $C_2$, $C_P$, or $C_4$ guaranteed by unavoidability can be refined to strictly satisfy all terminal distinctness side-conditions required for a certified occurrence.
\end{lemma}
\begin{proof}
For topological $C_2$ and $C_P$, Lemmas~\ref{lemma:certified_c2} and \ref{lemma:certified_cp} establish the necessary terminal distinctness. In particular, the $C_2$ terminal distinctness relies explicitly on the bound $|V(G)| > N_{\mathrm{base}} = 12$ to rule out the degenerate mapping onto the Cube base-case where distinctness fails. For refined $C_4$, the topological definition inherently bounds the four external neighbors to be distinct, satisfying the catalog certificate conditions natively (Definition~\ref{def:refinedC4-occ}).
\end{proof}'''

if old_upgrade in text:
    text = text.replace(old_upgrade, new_upgrade)
    with open(r'c:\Users\charl\Documents\New Math Problem\paper\LaTeX Pure Maths.tex', 'w', encoding='utf-8') as f:
        f.write(text)
    with open(r'c:\Users\charl\Documents\New Math Problem\paper\LaTeX_Pure_Maths_revised.txt', 'w', encoding='utf-8') as f:
        f.write(text)
    print('Replaced Upgrade Lemma')
else:
    print('Upgrade block not found exactly as queried.')
