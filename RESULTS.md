# Results: Tawseel failed-delivery baseline

<!-- Read by your Track Mentor, not graded by tests. Replace the comments. -->

# Baseline Results

| Model | ROC-AUC | PR-AUC | Brier Score |
|---|---|---|---|
| Dummy Classifier (Prior) | 0.500 | 0.220 | 0.1717 |
| Logistic Regression Pipeline | 0.6247 | 0.3264 | 0.1660 |

### Notes for Sherif
Our failure-risk baseline achieves a PR-AUC lift of +0.106 over the dummy baseline on unseen temporal data without using post-event attributes. Point-in-time customer history metrics (prior parcel counts and failure rates) provide realistic signal while strictly preventing temporal data leakage. This modest model serves as a reliable, honest benchmark that will not collapse in production.
