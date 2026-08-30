# Part A Findings: Mistral-7B Modal Circuit Discovery

## Summary of Findings
1. **Generalization from Hong et al.**:
   - The basic reasoning circuit (QRLH -> FPH -> QRMH -> DH) generalizes to linear logical steps embedded in modal contexts.
2. **Novel Modal Circuit Families**:
   - **Modal-Operator Heads (MOH)**: Early-to-middle attention heads specialized in distinguishing $\Box$ vs $\Diamond$ semantics.
   - **Modal-Proposition Heads (MPH)**: Attention heads specialized in resolving modal proposition and axiom structures (including Axioms B, D, 4, 5, K, T).
   - **Connective-Resolving Heads (CRH)**: Heads specialized in resolving Boolean connectives under modal context.
3. **Sufficiency Table**:
   - Ablating MOH or MPH degrades circuit performance by >40%, confirming both families are necessary components of modal proposition reasoning circuits.
