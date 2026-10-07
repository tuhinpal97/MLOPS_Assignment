# Model Card - Heart Disease Risk Classifier

## Intended use
Educational demonstration of an MLOps lifecycle for binary classification. It predicts whether the historical UCI Cleveland record belongs to the disease-present class. It is **not** approved for clinical use.

## Dataset
303 Cleveland records using the commonly studied 13 predictor subset plus the `num` outcome. `num=0` is mapped to class 0 and `num in {1,2,3,4}` to class 1. The source contains missing values in `ca` and `thal`; the preprocessing pipeline imputes them.

## Champion selection
Logistic Regression was selected using mean 5-fold stratified cross-validation ROC-AUC on the 80% training split (0.9065), ahead of Random Forest (0.8961). The untouched 20% holdout was used only for final evaluation.

## Holdout performance
- Accuracy: 0.8689
- Precision: 0.8125
- Recall: 0.9286
- ROC-AUC: 0.9578

The production artifact is refit on all 303 cleaned records after model selection.

## Limitations
The dataset is small, historical, geographically limited, and not representative of modern clinical populations. Performance estimates have substantial uncertainty. The feature meanings and coding reflect an older research dataset, not a modern EHR schema. The model should not be used for diagnosis, triage, or treatment decisions.

## Monitoring
The API exposes request count, latency and predicted-class counters to Prometheus. Raw request features are not logged. A simple optional batch KS-test drift checker is included for numeric features.
