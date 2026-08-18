# Final result interpretation

The reviewed full analysis supports the specification's **moderate outcome**, not its strongest orchestration headline.

> Thought Atlas reveals domain- and lineage-specific organizations of visible deliberation and identifies operations whose classification changes with trajectory context.

Do not use the strongest orchestration headline. The prespecified family-separation statistic is -0.016 (95% bootstrap interval -0.051 to 0.018; exact balanced-partition p≈0.51). Coupling adds essentially zero average held-out metric value, and motifs reduce it slightly on average. In contrast, domain/model effects, exact context sensitivity, and adaptive differentiation are substantial.

## Within-model domain and attempt-quality signatures

For RQ8, five-fold prompt-disjoint ridge multinomial logistic models were fitted separately for each reasoning model and signature family. The baseline controls log trace length; the amount block adds four label rates; the timing block adds amount-invariant phase contrasts. Amount improves held-out domain identification in all 12 model-family tests (median 0.806 bits/trace), and conditional timing improves all 12 after FDR correction (median 0.088 bits/trace). Cognitive amount is more domain-distinctive in five of six models.

The modal paths make those differences readable. Math ends in verification for all six reasoning models; idea traces contain backtracking in five of six cognitive paths; moral traces begin with subgoal setting in all six. Conversational paths are usually dominated by perspective shift, although Qwen math and planning often move toward question–answer behavior later.

For RQ9, high-versus-low quality is estimable in 86 of 96 model-domain-family cells. Only Qwen3.5 9B and Qwen3.5 27B cognitive amount on planning survive interval and FDR criteria; no timing increment survives. In both supported cells, the modal path remains verification-dominant for high and low attempts, so the signal is amount rather than order. Only 45 incomplete traces have usable sentence-level labels, and no model-domain cell reaches 20; incomplete attempts remain a documented non-estimable class.

Use [`generated/ANALYSIS_REPORT.md`](generated/ANALYSIS_REPORT.md) as the authoritative report and [`generated/SIGNATURE_ANALYSIS_REPORT.md`](generated/SIGNATURE_ANALYSIS_REPORT.md) for the detailed RQ8–RQ9 extension. `paper_results/` is retained only as the earlier exploratory package.
