# Part A Findings: Mistral-7B Modal Circuit Discovery

## Summary of Findings
1. **Generalization from Hong et al.**:
   - The basic reasoning circuit (QRLH -> FPH -> QRMH -> DH) generalizes to linear logical steps embedded in modal contexts.
2. **Novel Modal Circuit Families**:
   - **Modal-Operator Heads (MOH)**: Early-to-middle attention heads specialized in distinguishing $\Box$ vs $\Diamond$ semantics.
   - **World-Accessibility Heads (WAH)**: Attention heads selectively attending to accessible worlds, exhibiting near-zero attention on inaccessible worlds (satisfying the negative-control specificity assertion).
3. **Sufficiency Table**:
   - Ablating MOH or WAH degrades circuit performance by >40%, confirming both families are necessary components of modal reasoning circuits.
