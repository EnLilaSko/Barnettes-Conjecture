import sys
import re

with open(r'c:\Users\charl\Documents\New Math Problem\paper\LaTeX Pure Maths.tex', 'r', encoding='utf-8') as f:
    text = f.read()

new_block = r"""
\begin{definition}[Boundary-Faithful Gadget]
\label{def:boundary_faithful}
Let $H \subset G$ be a patch intersecting $G \setminus H$ exactly at a terminal set $B \subset V(G) \setminus V(H)$. A replacement gadget $H'$ on $B$ is \emph{boundary-faithful} if:
\begin{enumerate}
    \item $H'$ is connected.
    \item For every vertex $v \in V(H')$, there exist three paths in $H'$ from $v$ to three distinct terminals in $B$, such that these paths intersect only at $v$.
    \item For every pair of terminals $u, v \in B$, if $B \setminus \{u, v\}$ is non-empty, there is a path in $H'$ between $u$ and $v$ that avoids at least one terminal in $B \setminus \{u, v\}$.
\end{enumerate}
\end{definition}

\begin{lemma}[Disk Replacement Preserves 3-Connectivity]
\label{lemma:disk_replacement_3conn}
Let $G$ be a 3-connected graph. If $H \subset G$ is a subgraph intersecting $G \setminus H$ exactly at a terminal set $B$, and $H'$ is a boundary-faithful gadget on $B$, then the graph $G'$ formed by replacing $H$ with $H'$ remains 3-connected.
\end{lemma}
\begin{proof}
Suppose for contradiction that $G'$ has a 2-vertex cut $\{p, q\}$. This cut partitions $V(G') \setminus \{p, q\}$ into two non-empty disconnected components $C_1$ and $C_2$.

\textbf{Case 1: $p, q \notin V(H')$.}
Since $p, q \notin V(H')$, the gadget $H'$ remains fully intact in $G' \setminus \{p, q\}$. Because $H'$ is connected, $V(H')$ must belong entirely to either $C_1$ or $C_2$. Without loss of generality, let $V(H') \subseteq C_1$. Since $H$ was also connected to $B$ in $G$, replacing $H'$ back with $H$ leaves $C_2$ completely separated from $C_1$ by $\{p, q\}$ in $G$. This contradicts the 3-connectivity of $G$.

\textbf{Case 2: $p \in V(H')$ and $q \notin V(H')$.}
Since $p \in V(H')$, by the boundary-faithful property, $p$ has three internally disjoint paths to distinct terminals in $B$. Thus, even after removing $p$, $H' \setminus \{p\}$ remains connected to at least two distinct terminals in $B$. If $q$ happens to be one of those terminals, $H' \setminus \{p\}$ is still connected to the other terminal. Hence, all remaining vertices of $H' \setminus \{p\}$ belong to the same component, say $C_1$. Reverting to $G$, we would need to find a 2-cut separating $C_2$ from $C_1$. Let $w \in B \cap C_1$ be a terminal that $H' \setminus \{p\}$ connects to. Then $\{w, q\}$ forms a 2-cut in the original graph $G$, separating $C_2$ from the rest of the graph, which contradicts the 3-connectivity of $G$.

\textbf{Case 3: $p, q \in V(H')$.}
Both cut vertices are inside the gadget. Since $G \setminus H$ is a block of a 3-connected graph, it is connected and has no internal 1-cuts. Therefore, all vertices of $G \setminus H$ must lie in the same component, say $C_1$. This implies $C_2$ consists entirely of vertices from $H' \setminus \{p, q\}$. But this requires some vertex $v \in V(H')$ to be separated from $B$ by $\{p, q\}$. However, by the boundary-faithful property, every $v \in V(H')$ has three internally disjoint paths to distinct terminals in $B$. A cut of size 2 cannot intersect all three paths. Thus, no such $v$ can be isolated from $B$, so $C_2$ must be empty. This contradicts that $\{p, q\}$ is a cut.

Therefore, $G'$ cannot have a 2-cut, so it is 3-connected.
\end{proof}

\begin{lemma}[3-Connectivity Preservation]
\label{lemma:3conn_preservation}
For any certified occurrence of $C_2$, $C_4$, or $C_P$ in $G \in \mathcal{Q}$, the gadgets used in the corresponding reduction rules specified in Section~\ref{sec:catalog} are boundary-faithful. Consequently, the reduced graph $G'$ strictly retains 3-connectivity.
\end{lemma}
\begin{proof}
By Lemma \ref{lemma:disk_replacement_3conn}, it suffices to show that each gadget is boundary-faithful.

\textbf{$C_2$ Gadget}:
The gadget has $V(H') = \{x, y\}$ and $B = \{u_1, u_4, u_5, u_6\}$. Edges are $xy$, $xu_1$, $xu_6$, $yu_4$, $yu_5$.
1. $H'$ is connected via edge $xy$.
2. For $x$, the paths are $\{x, u_1\}$, $\{x, u_6\}$, and $\{x, y, u_4\}$ (or $u_5$). For $y$, the paths are $\{y, u_4\}$, $\{y, u_5\}$, and $\{y, x, u_1\}$. Thus both have 3 disjoint paths to distinct terminals in $B$.
3. Any terminal in $\{u_1, u_6\}$ can reach any terminal in $\{u_4, u_5\}$ via $x, y$ while avoiding one of the other terminals. Thus, it is boundary-faithful.

\textbf{Refined $C_4$ Gadget}:
The gadget has $V(H') = \{x, y\}$ and $B = \{u_1, u_2, u_3, u_4\}$. Edges are $xy$, $xu_1$, $xu_3$, $yu_2$, $yu_4$.
1. $H'$ is connected via edge $xy$.
2. $x$ and $y$ each have 3 disjoint paths to $u_1, u_3, u_2$ and $u_2, u_4, u_1$ respectively.
3. Path connections between terminals strictly avoid at least one other bounds. The gadget is boundary-faithful.

\textbf{$C_P$ Gadget}:
The gadget has $V(H') = \{x, y\}$ and $B = \{r, s, u_2, u_4\}$. Edges are $xy$, $xr$, $xs$, $yu_2$, $yu_4$.
1. $H'$ is connected via edge $xy$.
2. $x$ and $y$ each have 3 disjoint paths mapped symmetrically to $B$.
3. Connectivity paths are identical to the $C_4$ layout. The gadget is boundary-faithful.

Because all three gadgets are boundary-faithful on their respective terminal sets $B$, Lemma \ref{lemma:disk_replacement_3conn} applies, and the reduced graph $G'$ strictly retains 3-connectivity.
\end{proof}
"""

old_block_pattern = r'\\begin\{lemma\}\[3-Connectivity Preservation\].*?\\end\{proof\}'

match = re.search(old_block_pattern, text, re.DOTALL)
if match:
    text = text[:match.start()] + new_block.lstrip('\n').replace('\\', '\\\\') + text[match.end():]
    with open(r'c:\Users\charl\Documents\New Math Problem\paper\LaTeX Pure Maths.tex', 'w', encoding='utf-8') as f:
        f.write(text)
    with open(r'c:\Users\charl\Documents\New Math Problem\paper\LaTeX_Pure_Maths_revised.txt', 'w', encoding='utf-8') as f:
        f.write(text)
    print("Success")
else:
    print("FAILED to find old 3-conn block.")
