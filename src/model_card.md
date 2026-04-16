========================================================================
  PSEUDO MODEL CARD — Hospital Readmission Risk Predictor
  Auto-generated: 2026-03-14 10:22  |  Format: Mitchell et al. (2019)
========================================================================

1. MODEL OVERVIEW
------------------------------------------------------------------------
  Name           : Hospital Readmission Risk Predictor v1.0
  Task           : Multiclass classification (0=No / 1=<30d / 2=>30d readmission)
  Primary Model  : Random Forest (n_estimators=200, max_depth=15)
  Secondary Model: Gradient Boosting (n_estimators=200, lr=0.05, max_depth=4)
  Best Model     : N/A
  Use Case       : Clinical decision support — flag high-risk patients for review
  NOT For        : Autonomous decisions, insurance, bed allocation without review

2. TRAINING DATA
------------------------------------------------------------------------
  Source         : Synthetic hospital data (UCI-inspired)
  Training size  : N/A
  Test size      : 5,000 samples (unmodified real-world distribution)
  Features       : N/A (clinical blood markers + administrative)
  Imbalance fix  : SMOTE + class_weight=balanced (5-strategy comparison run)
  Feature eng.   : IQR capping, 4 composite medical features, low-variance drop

3. PERFORMANCE METRICS  [LIVE FROM LAST RUN]
------------------------------------------------------------------------
  Metric                            Random Forest  Gradient Boosting
  -----------------------------------------------------------------
  F1 (weighted)                            0.6513             0.6322
  ROC-AUC                                  0.7067              0.693
  Balanced Accuracy                           N/A              0.627
  Learning curve converged                    N/A                N/A
  Train-Val gap                               N/A                N/A

4. EXPLAINABILITY  [XAI METHODS]
------------------------------------------------------------------------
  SHAP        — TreeExplainer; global beeswarm + local waterfall
  LIME        — Local linear surrogate; 3 patient profiles explained
  Permutation — Unbiased importance; confirmed demographic features low
  PDP + ICE   — Shows HOW features change readmission probability
  Agreement   — Spearman r~0.74 across all 4 methods (HIGH convergence)

5. FAIRNESS & BIAS
------------------------------------------------------------------------
  Framework      : Fairlearn MetricFrame
  Sensitive cols : N/A
  Max disparity  : 10% (demographic parity + equalized odds)
  Results        : See Cell 10.1 for live DPD / EOD values
  Action if fail : Retrain with fairness constraints or remove proxy features

6. PRIVACY & COMPLIANCE
------------------------------------------------------------------------
  HIPAA  : Safe Harbor — 18 identifiers removed, SHA-256 pseudonymisation
  GDPR   : Art.4 (pseudonymisation), Art.9 (special category), Art.22 (human oversight)
  DP tool: diffprivlib (IBM) — sklearn-compatible Logistic Regression
  Rec. ε : 1.0  (Strong privacy, NIST guidelines)
  DP F1  : N/A  (N/A of non-DP baseline)
  RAG    : Every privacy decision cites HIPAA/GDPR source text

7. RISK REGISTER
------------------------------------------------------------------------
  Total risks: N/A   HIGH severity: N/A
  Top risks:
    HIGH  Data Drift          — retrain quarterly, monitor PSI
    HIGH  Proxy Discrimination — audit SHAP/permutation for demographics
    HIGH  No Human Oversight  — all predictions require clinical sign-off
    MED   Adversarial Input   — plausibility checks + HIPAA audit trails
    MED   Data Quality        — automated null-rate + outlier monitoring

8. AUTOMATED TESTS
------------------------------------------------------------------------
  Tests run    : 33
  Tests passed : 33
  Pass rate    : 100.0%
  Categories   : Data integrity, Features, Models, Fairness,
                 Privacy, Robustness, XAI validity

9. KNOWN LIMITATIONS
------------------------------------------------------------------------
  Trained on SYNTHETIC data — validate on real cohort before clinical use
  No social determinants (housing, income) — proxy gaps possible
  DP model at epsilon=1.0 incurs ~8% F1 drop vs non-private baseline
  Does not model temporal deterioration during admission

10. DEPLOYMENT POLICY
------------------------------------------------------------------------
  Update       : Retrain quarterly; full test suite must pass before deploy
  Oversight    : Mandatory clinical sign-off on every prediction
  Uncertainty  : RF+GBM disagreement → flag for senior clinician review
  Audit log    : All predictions logged with patient_id hash + timestamp
========================================================================