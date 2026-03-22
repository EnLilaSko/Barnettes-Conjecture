import sys

with open(r'c:\Users\charl\Documents\New Math Problem\paper\LaTeX Pure Maths.tex', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Remove \usepackage{algorithm} and \usepackage{algorithmic} if they exist
text = text.replace(r'\usepackage{algorithm}', '')
text = text.replace(r'\usepackage{algorithmic}', '')

# 2. Fix TikZ node
text = text.replace(r'\secref{subsec:algorithm}\\Algorithm \&\\Correctness Proof', r'\secref{sec:main-theorem}\\Main Theorem\\Proof')

# 3. Line 394: "inductive step of Algorithm~\ref{alg:main} operates" -> "inductive framework operates"
text = text.replace(r'inductive step of Algorithm~\ref{alg:main} operates', r'inductive framework operates')

# 4. Line 1010: "verify planarity by enumerating $\varphi'$-cycles and checking Euler's formula"
text = text.replace(r'verify planarity by enumerating $\varphi''$-cycles and checking Euler''s formula',
                    r'prove planarity by enumerating $\varphi''$-cycles and applying Euler''s formula')

# 5. Line 1217: "Specifically, we verify:"
text = text.replace(r'Lemma~\ref{lem:splicing}. Specifically, we verify:', r'Lemma~\ref{lem:splicing}. Specifically, we require:')

# 6. Delete old base cases block and old remark text.
old_remark_text = r'''In the computer-assisted manuscript, this bridge was discharged by finite enumeration of rooted local types.
In the present pure-mathematics manuscript, it is replaced by a fully analytic admissibility argument:
one proves that every topological occurrence can be refined'''

new_remark_text = r'''one proves that every topological occurrence can be refined'''
text = text.replace(old_remark_text, new_remark_text)

# 7. Delete old Base cases subsection (lines 1559-1569 equivalent)
old_base_cases_sec = r'''\subsection{Base cases}\label{subsec:basecases}

\begin{lemma}[Base cases up to $N_{\mathrm{base}}$]\label{lem:base-cases}
Every graph $G\in\mathcal{Q}$ with $|V(G)|\le N_{\mathrm{base}}$ is Hamiltonian.
Moreover, up to isomorphism the only graphs in $\mathcal{Q}$ with $|V|\le 14$ are the cube graph $Q_3$ (on $8$ vertices) and the hexagonal prism $C_6\square K_2$ (on $12$ vertices), and each admits an explicit Hamilton cycle.
\end{lemma}

\begin{proof}
For $Q_3$ and $C_6\square K_2$, explicit Hamilton cycles are given in \S\ref{subsec:small-graphs-hamilton}.
The classification of $\mathcal{Q}$ on at most $14$ vertices is a finite argument; one may present it as a direct combinatorial classification of $3$-regular bipartite polyhedra with $|V|\le 14$ (equivalently, Eulerian triangulations with at most $9$ faces in the dual), and then verify Hamiltonicity for each type.
\end{proof}'''
text = text.replace(old_base_cases_sec, '')

# 8. Delete Algorithm and Correctness Section + explicit cycles (lines 1605-1655ish)
old_algo_block = r'''\subsection{Algorithm and correctness proof}

\label{subsec:algorithm}

\begin{algorithm}[ht]
\caption{Hamiltonian Cycle Finder for $\mathcal Q$ (proof skeleton)}
\label{alg:main}
\begin{algorithmic}[1]
\Require $(G,\pi) \in \mathcal{Q}$
\Ensure Hamiltonian cycle $H$ in $G$
\If{$|V(G)| \le N_{\mathrm{base}}$}
    \State \Return a Hamiltonian cycle of $G$ (Lemma~\ref{lem:base-cases})
\EndIf
\State choose an admissible certified occurrence $C$ (Theorem~\ref{thm:certified-completeness})
\State $(G',\pi') \gets \textsc{Reduce}(G,\pi;C)$
\State $H' \gets \textsc{HamiltonianCycle}(G',\pi')$
\State \Return $\textsc{LiftCycle}(H', G, C)$
\end{algorithmic}
\end{algorithm}

\begin{theorem}[Correctness of Algorithm~\ref{alg:main} and Barnette's Conjecture for $\mathcal{Q}$]
\label{thm:algorithm-correctness}
For every $(G,\pi)\in\mathcal{Q}$, Algorithm~\ref{alg:main} terminates and returns a Hamiltonian cycle of $G$.
Consequently, every $3$-connected cubic bipartite plane graph is Hamiltonian.
\end{theorem}

\begin{proof}
We argue by well-founded induction on the measure $\mu(G)$ (Section~\ref{sec:measure}), using Lemma~\ref{lem:well-founded}.
If $|V(G)|\le N_{\mathrm{base}}$, Lemma~\ref{lem:base-cases} supplies a Hamiltonian cycle.

Otherwise, $|V(G)|>N_{\mathrm{base}}$.
By Theorem~\ref{thm:certified-completeness}, there exists an admissible certified occurrence $C$ and a reduction step
$(G,\pi)\to (G',\pi')$.
Property preservation implies $(G',\pi')\in\mathcal{Q}$ and $\mu(G')<_{\mathrm{lex}}\mu(G)$ (Lemma~\ref{lem:measure-decrease}).
By the induction hypothesis, the recursive call returns a Hamiltonian cycle in $G'$.
Finally, the bidirectional Hamiltonicity theorem (Theorem~\ref{thm:bidirectional-hamiltonicity}) and the lifting library (\S\ref{sec:lifting-library}--\S\ref{sec:lifting-theorem}) lift $H'$ to a Hamiltonian cycle in $G$.
\end{proof}

\subsection{Explicit Hamilton cycles for the small graphs}
\label{subsec:small-graphs-hamilton}

\paragraph{The cube $Q_3$.}
One Hamilton cycle is
\[
000\to 001\to 011\to 010\to 110\to 111\to 101\to 100\to 000,
\]
which clearly visits all $8$ vertices.

\paragraph{The hexagonal prism $C_6\square K_2$.}
Let the prism consist of two $6$-cycles $(u_0,\dots,u_5)$ and $(v_0,\dots,v_5)$ joined by rungs $u_iv_i$.
A Hamilton cycle is
\[
u_0\to u_1\to u_2\to u_3\to u_4\to v_4\to v_3\to v_2\to v_1\to v_0\to v_5\to u_5\to u_0,
\]
which correctly alternates between the two layers and uses all $12$ vertices.
'''
text = text.replace(old_algo_block, '')

with open(r'c:\Users\charl\Documents\New Math Problem\paper\LaTeX Pure Maths.tex', 'w', encoding='utf-8') as f:
    f.write(text)
with open(r'c:\Users\charl\Documents\New Math Problem\paper\LaTeX_Pure_Maths_revised.txt', 'w', encoding='utf-8') as f:
    f.write(text)
print("Success")
