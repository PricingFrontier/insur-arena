# allstate_sev

- **Task:** sev  |  **Source:** OpenML  |  **Access:** `openml_id:42571`
- **License:** comp. data on OpenML (fetch-only)  |  **Redistributable:** False (benchmark is fetch-only — data is never re-hosted)
- **Target:** `loss`  |  no exposure (full-unit)
- **Dropped (id / leakage / other targets):** id
- **Capping:** —
- **Split mode:** IID random; no temporal column in this version.
- **IID audit:** random split appropriate (published IID use).
- **Non-trivial check:** the canonical task GLM must beat an intercept-only model (skill-vs-GLM has a non-degenerate spread); trivial/perfect-fit datasets are excluded.
- **Notes:** Allstate Claims Severity, 188k, 130 anon features.
- **tri-boost characterization (2026-08-04): diffuse-representation-limit dataset.** The
  anonymized signal is spread across ~10,000 weak, correlated fANOVA supports (130 mains +
  thousands of order-2/3 tables). tri-boost's *unpruned* deploy ensemble BEATS constrained
  catboost on every split tested (s0 0.28238 vs 0.28580; s7 0.28063 vs 0.28390, gamma
  deviance), and the table-bank conversion is lossless (keep=ALL = ensemble to 2e-5) — but no
  product-sized table set can hold the signal: the K-curve from one fitted model reads
  CV-selected 2,394 tables = 0.29482, best-ranked 5,000 = 0.29265, 7,500 = 0.28770, ALL
  10,085 (3,880 deployed post-purify) = 0.28236. Per-table leave-one-out gains sum to ~0 for
  the dropped 7,691 (their joint value is +4.4% — marginals don't add on correlated tables),
  so no honest selector reaches ccb at interpretable K. ACCEPTED as a representation-limit
  loss (freclaimdam precedent — but note the headline differs: here the MODEL wins and the
  explainability contract is what pays). The av33 set-level prune guard bounds the loss
  (blow-up splits like s7's +23% selection detonations are caught and relaxed); the residual
  base gap vs ccb at default tol is the accepted cost. Full forensics:
  t-boost-vs-ccb-gap-2026-08-04 memory + tri-boost/speed_accuracy_work.md.
