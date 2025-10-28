Audit Summary — COMPAS Recidivism Risk Scores

We audited the COMPAS recidivism dataset for racial bias using IBM’s AIF360 toolkit. The audit examined group-level disparities in outcomes, focusing on false positive rates (FPR) and statistical parity differences between privileged (white) and unprivileged (non-white) groups. A baseline logistic regression trained on the original data produced an FPR that was substantially higher for the unprivileged group than for the privileged group (e.g., FPR_unprivileged − FPR_privileged > 0). This indicates that non-white defendants were more likely to be incorrectly labeled high risk compared to white defendants, which could lead to disproportionate adverse consequences such as harsher sentencing or denial of parole.

To remediate this disparity, we applied a pre-processing reweighing algorithm that adjusts instance weights to remove dependence between the protected attribute and labels. After reweighing and retraining, the disparity in FPR narrowed substantially, demonstrating improved parity. While reweighing reduces group-level bias, it is not a silver bullet: there is often a tradeoff between fairness and overall predictive performance, and some fairness definitions may conflict. Therefore, we recommend a governance stack that includes (1) selection of fairness metrics aligned with policy goals (e.g., minimize FPR gap for criminal justice contexts), (2) iterative auditing across multiple metrics (FPR gap, statistical parity, predictive parity), (3) human oversight for decisions flagged as high risk, and (4) continuous monitoring after deployment.

Finally, mitigating bias should be supplemented by structural interventions: improving data collection, adding contextual features (e.g., socio-economic indicators), and ensuring models are used only as one input in human decision processes. A documented audit trail and regular independent reviews must be part of any deployment plan to ensure responsible use.

How to ensure ethical adherence:

Data governance & consent: Use only consented data and apply strong de-identification.

Bias audits: Test model performance across age, gender, race, and socio-economic groups; reweight or augment data as needed.

Human-in-the-loop: Ensure clinicians make final decisions; model gives risk scores + explanations.

Transparency & logging: Keep model cards, data lineage, and inference logs for auditing.

Security & compliance: Encrypt data at rest/in transit and follow relevant healthcare regulations.
