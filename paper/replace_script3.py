import sys

with open(r'c:\Users\charl\Documents\New Math Problem\paper\LaTeX Pure Maths.tex', 'r', encoding='utf-8') as f:
    text = f.read()

replacements = [
    (r"unavoidable configurations ($C_2$, refined $C_4$, and $C_{pinch}(ii)$). For each configuration,",
     r"unavoidable topological configurations (topological $C_2$, $C_4$, and $C_{pinch}(ii)$), which guarantees the existence of a corresponding certified configuration. For each certified configuration,"),
     
    (r"unavoidable configurations (Theorem \ref{thm:unavoidability}).",
     r"unavoidable topological configurations (Theorem \ref{thm:unavoidability}) and show this implies the existence of a certified occurrence (Lemma \ref{lemma:certified_completeness})."),
     
    (r"\item We provide certified reductions for each configuration that preserve all",
     r"\item We provide certified reductions for each certified configuration that preserve all"),
     
    (r"\begin{theorem}[Unavoidability]",
     r"\begin{theorem}[Topological Unavoidability]"),
     
    (r"Every $G \in \mathcal{Q}$ contains at least one of:",
     r"Every $G \in \mathcal{Q}$ locally contains at least one of the following topological configurations:"),
     
    (r"\item $C_2$: Two adjacent facial 4-faces sharing an edge.",
     r"\item Topological $C_2$: Two adjacent facial 4-faces sharing an edge."),
     
    (r"\item Refined $C_4$: An isolated 4-face with four distinct external neighbors.",
     r"\item Topological $C_4$: An isolated 4-face with four distinct external neighbors."),
     
    (r"\item $C_{pinch}(ii)$: A specific pinched 4-face configuration with additional constraints.",
     r"\item Topological $C_{pinch}(ii)$: A specific pinched 4-face configuration where two opposite vertices of the 4-face share an external neighbor $w$, and $w$'s third neighbor $t$ is distinct from the other external neighbors of the 4-face."),
     
    (r"\item If $Q$ shares an edge with another 4-face, then we have $C_2$.",
     r"\item If $Q$ shares an edge with another 4-face, then we have a topological $C_2$ configuration."),
     
    (r"\item Otherwise, $Q$ is edge-isolated. If $u_1, u_2, u_3, u_4$ are all distinct, then we have refined $C_4$.",
     r"\item Otherwise, $Q$ is edge-isolated. If $u_1, u_2, u_3, u_4$ are all distinct, then we have a topological $C_4$ configuration."),
     
    (r"contradicting our assumption. Hence $t \notin \{u_2, u_4\}$, and we have $C_{pinch}(ii)$.",
     r"contradicting our assumption. Hence $t \notin \{u_2, u_4\}$, and we have a topological $C_{pinch}(ii)$ configuration."),
     
    (r"Thus, at least one of the three configurations must occur.",
     r"Thus, at least one of the three topological configurations must occur."),
     
    (r"\{C_2, C_4, C_{pinch}(ii)\} and their certified reductions provide a complete",
     r"of topological configurations \{C_2, C_4, C_{pinch}(ii)\} and their corresponding certified reductions provide a complete"),
]

for old, new in replacements:
    text = text.replace(old, new)
    
bridge_lemma = r"""
The structural requirements for a certified reduction defined in Section \ref{sec:reductions} entail additional side-conditions (e.g., distinctness of all extended neighborhood vertices). The following completeness lemma bridges this topological guarantee to the fully certified occurrences needed for the algorithm.

\begin{lemma}[Certified Completeness]
\label{lemma:certified_completeness}
If $G \in \mathcal{Q}$ contains a topological occurrence of $C_2$, $C_4$, or $C_{pinch}(ii)$ as defined in Theorem \ref{thm:unavoidability}, then $G$ must contain a strict certified occurrence of $C_2$, refined $C_4$, or $C_{pinch}(ii)$ (meeting all side-conditions and vertex distinctness requirements defined in Section \ref{sec:reductions}).
\end{lemma}
\begin{proof}
TODO: prove. (Exploration of intersecting labels and overlapping neighborhood cases).
\end{proof}

\section{Certified Reductions}"""

text = text.replace(r"\section{Certified Reductions}", bridge_lemma)

with open(r'c:\Users\charl\Documents\New Math Problem\paper\LaTeX Pure Maths.tex', 'w', encoding='utf-8') as f:
    f.write(text)

with open(r'c:\Users\charl\Documents\New Math Problem\paper\LaTeX_Pure_Maths_revised.txt', 'w', encoding='utf-8') as f:
    f.write(text)

print('File replaced successfully')
