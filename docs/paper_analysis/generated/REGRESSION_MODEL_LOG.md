# Regression and model run log

This file records the exact predictive estimands used before and during the signature-localization extension. It is intended as a durable Methods-writing source, not as a causal-analysis claim.

## Shared fitting contract

All signature models use the frozen five-fold `prompt_fold` registry grouped by `instance_id`. Every reported prediction is out of fold. Numeric predictors are mean-imputed and standardized using the training fold only. Categorical targets use L2-penalized multinomial logistic regression; continuous targets use L2-penalized linear regression. In both cases the slope penalty is 1.0 and the intercept is unpenalized.

The length baseline contains log token count and log valid-segment count. Amount features are behavior-positive segments per valid segment. Timing features are middle-minus-early and late-minus-early shares conditional on the behavior's total occurrence, and therefore enter only after amount.

## Exact regression mathematics

Let `i` index traces, `j` predictors, and `c` target classes. Inside each training fold, every numeric predictor is transformed as `z_ij = (x_ij - mean_train,j) / sd_train,j`; missing values receive the training mean first, and a zero standard deviation is replaced by 1. The same training-fold quantities transform the held-out fold. Fixed prompt-stratum dummy columns are created from prompt metadata before cross-validation and use no outcomes; their imputation, scaling, and coefficients still come from the training fold.

For a categorical target with `K` classes, class 0 is the reference: `eta_i0 = 0` and `eta_ic = alpha_c + z_i^T beta_c` for `c = 1,...,K-1`. Probabilities are softmax values, `p_ic = exp(eta_ic) / sum_r exp(eta_ir)`. The fitted parameters minimize `-sum_i log p_i,y_i + (lambda/2) sum_c ||beta_c||_2^2`, with `lambda = 1`. Only slopes are penalized; intercepts are not. Pairwise domain and binary-outcome models are the `K = 2` special case.

For a continuous native outcome, `yhat_i = alpha + z_i^T beta`. The fitted parameters minimize `sum_i (y_i - alpha - z_i^T beta)^2 + lambda ||beta||_2^2`, again with `lambda = 1` and no intercept penalty. Equivalently, `theta_hat = (X^T X + lambda P)^+ X^T y`, where `P_00 = 0`, all slope diagonals of `P` are 1, and `+` denotes the Moore–Penrose inverse.

### Exact feature definitions

For behavior `b`, amount is `a_ib = m_ib / n_i`, where `m_ib` is the number of valid segments labeled positive and `n_i` is the number of valid segments. For phase `p` in early/middle/late, phase prevalence is `r_ibp = m_ibp / n_ip`. We normalize within behavior as `s_ibp = r_ibp / sum_q r_ibq` when the sum is positive, otherwise all three shares are zero. The two regression timing features are `t_ib1 = s_ib,middle - s_ib,early` and `t_ib2 = s_ib,late - s_ib,early`. Their sum is not an amount measure: timing enters only after all four family amounts are already in the base model.

The color direction in the timing figure uses the descriptive centroid `c_ib = 0.5 s_ib,middle + s_ib,late`, conditional on the behavior occurring. It is not an additional regression predictor. The amount color is the one-domain-versus-rest standardized mean difference; the timing color is the corresponding unstandardized centroid difference.

### Held-out predictive gain

For nested categorical models `A` and `B`, the trace-level gain is `g_i(B|A) = log2[p_B(y_i|x_i) / p_A(y_i|x_i)]`. The reported increment is `Delta(B|A) = mean_i g_i(B|A)` in bits per held-out trace. A value of `Delta` means the richer model assigns the true class `2^Delta` times as much probability on average on the geometric-mean scale. Amount uses the length model as `A`; timing uses the length-plus-all-amounts model as `A`.

For continuous outcomes, `g_i(B|A) = [(y_i-yhat_A,i)^2 - (y_i-yhat_B,i)^2] / Var(y)`. Averaging this expression equals `R2_B - R2_A`, so categorical and continuous analyses both measure paired held-out predictive improvement, on their appropriate scoring scales.

## Exact grouped Shapley calculation

Within one family there are `m = 4` behavior players. An amount player contributes one rate; a timing player contributes its two timing contrasts together. We fit every `2^4 = 16` coalition on the same outer folds. For trace `i`, define `v_i(S) = log2 p_S(y_i|x_i)` for categorical targets, or `v_i(S) = -(y_i-yhat_S,i)^2 / Var(y)` for continuous targets.

The exact contribution of behavior `b` is `phi_ib = sum over S not containing b of [|S|! (m-|S|-1)! / m!] * [v_i(S union {b}) - v_i(S)]`. The factorial weight is the fraction of all behavior-addition orders in which coalition `S` appears immediately before `b`. Thus each behavior receives its average marginal held-out contribution over every possible order, instead of credit from one arbitrary coefficient or one arbitrary feature order.

For a domain–behavior dot, we average `phi_ib` over held-out traces whose true class is that domain. Exact Shapley efficiency gives `sum_b phi_ib = v_i(all four) - v_i(no behaviors)` for every trace; the implementation's maximum numerical reconstruction error is below `4e-16`. Dot area encodes positive predictive credit. The blue/orange overlay comes from the separate direction contrast because Shapley importance alone cannot say whether a behavior is more or less prevalent, or earlier or later.

### Why this specification

Fixed-model fits remove model identity as an easy shortcut. Prompt-disjoint folds test generalization to unseen prompts. Length controls stop verbosity from masquerading as process structure. Ridge regularization stabilizes correlated behavior features and sparse prompt-stratum indicators. Proper log score rewards calibrated probability assigned to the actual class, not only whether the top class wins. Exact grouped Shapley is preferable to reading regression coefficients because coefficients depend on reference coding, scaling, and correlations; Shapley instead allocates the actual held-out score improvement while sharing credit across correlated behaviors. Timing is tested after amount so that ‘later’ cannot simply mean ‘more.’

## Models that existed before localization

The RQ3 staged outcome ladder pools models within domain and compares metadata, dialectical amount, executive amount, their union, timing, coupling, and motifs. Its exact M0–M5 column lists remain in `04_outcome_models/model_specifications.json`.

The original RQ8 omnibus model predicts all eight domains separately within each model. The original RQ9 sensitivity model predicts high versus low quality separately within model and domain, with continuous scores median-split only for that common-scale sensitivity analysis.

## New localization runs

### RQ8a: particular domain pairs

All 28 unordered domain pairs are fit separately for six models and two signature families. Amount is evaluated beyond the two length controls; timing is evaluated beyond length and all four family amounts. The 336 tests in each step form one BH correction family.

### RQ8b: behavior attribution

For each four-behavior family, all 16 feature subsets are fit on the same held-out folds. Exact Shapley values average a behavior's marginal held-out log-score contribution over every addition order. Contributions are summarized by the true domain. A separate one-domain-versus-rest contrast supplies direction; importance alone does not imply more or less behavior.

### RQ9a: native outcomes

Models are fit inside each model–domain cell. Binary outcomes remain binary; moral, idea, and safety remain continuous. The baseline contains both length controls and available prompt-stratum indicators. Cells require at least 40 scored traces, at least 20 per binary class, or at least three distinct continuous scores. Behavior attribution again uses all 16 feature subsets.

## Uncertainty, multiplicity, and language

Intervals use 1,000 paired bootstrap resamples of held-out trace scores. Categorical targets are class-stratified; continuous targets are outcome-quantile-stratified. One-sided paired sign-flip tests ask whether the mean held-out increment is positive. A result is called supported only when the 95% interval excludes zero and BH q < .05.

For gain values `g_i`, the randomization p-value is `(1 + number of r with mean_i(s_ri g_i) >= mean_i(g_i)) / (R + 1)`, where every `s_ri` is independently `-1` or `+1` and `R = 1,000`. For sorted p-values, Benjamini–Hochberg reports `q_(k) = min_{j >= k} [m p_(j) / j]`, capped at 1 and restored to the original test order. Requiring both a positive bootstrap lower bound and `q < .05` makes a supported ring deliberately stricter than either criterion alone.

Shapley values allocate predictive information under correlated features; they do not identify a causal mechanism. Outcome analyses support wording such as ‘predicts,’ ‘is associated with,’ or ‘carries signal,’ never ‘causes,’ ‘produces,’ or ‘improves.’

## Reproducibility

```bash
.venv/bin/python scripts/run_signature_analysis.py --config configs/paper_analysis.yaml --force
```
