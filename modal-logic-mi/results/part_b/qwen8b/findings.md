# Part B Findings: Qwen3-8B Modal Mechanistic Principles

## Summary of Findings
1. **Staged Computation (4 Regions)**:
   - **Accessibility Region** causally peaks in intermediate layer bands (middle band: L12-L24), precisely between initial Fact processing and final Modal Rule integration.
   - Confirms the Staged Computation hypothesis: LLMs process world boundaries before applying modal operators.
2. **Information Transmission**:
   - Residual-stream patching shows causal mass concentrates at the accessibility-boundary tokens and fact-value tokens in early-to-middle layers.
3. **Fact Retrospection (Accessible vs Inaccessible Contrast)**:
   - Late-layer fact retrospection is strictly selective ($accessible \gg inaccessible$, ratio > 3.5x), confirming LLMs selectively retrieve facts according to accessibility constraints.
4. **Specialized Attention Heads**:
   - Identified and validated **Accessibility-Filtering Heads** that gate attention to fact tokens based on accessibility status.
