# compas_audit.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# AIF360 imports
from aif360.datasets import CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.postprocessing import EqOddsPostprocessing

def load_compas():
    # AIF360 provides a loader for COMPAS
    dataset = CompasDataset()
    return dataset

def print_base_stats(dataset, protected_attribute='race', privileged_groups=[{'race': 1}], unprivileged_groups=[{'race': 0}]):
    metric = BinaryLabelDatasetMetric(dataset, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    print("Dataset shape:", dataset.features.shape)
    print("Mean outcome (positive):", metric.mean_difference())  # statistical parity difference
    print("Privileged mean:", metric.mean(privileged=True))
    print("Unprivileged mean:", metric.mean(privileged=False))

def to_sklearn(dataset):
    X = dataset.features
    y = dataset.labels.ravel()
    return X, y

def train_model(X_train, y_train):
    model = LogisticRegression(solver='liblinear')
    model.fit(X_train, y_train)
    return model

def compute_classification_metric(orig_dataset, dataset_true, dataset_pred, privileged_groups, unprivileged_groups):
    classified_metric = ClassificationMetric(dataset_true, dataset_pred,
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
    fpr_priv = classified_metric.false_positive_rate(privileged=True)
    fpr_unpriv = classified_metric.false_positive_rate(privileged=False)
    fnr_priv = classified_metric.false_negative_rate(privileged=True)
    fnr_unpriv = classified_metric.false_negative_rate(privileged=False)
    print("FPR privileged:", fpr_priv)
    print("FPR unprivileged:", fpr_unpriv)
    print("FPR difference (unpriv - priv):", fpr_unpriv - fpr_priv)
    print("FNR privileged:", fnr_priv)
    print("FNR unprivileged:", fnr_unpriv)
    print("FNR difference (unpriv - priv):", fnr_unpriv - fnr_priv)

def main():
    # Load dataset
    dataset = load_compas()
    privileged_groups = [{'race': 1}]   # in AIF360 compas: race=1 often denotes white (check docs)
    unprivileged_groups = [{'race': 0}] # non-white

    print("=== Base data stats ===")
    print_base_stats(dataset, protected_attribute='race', privileged_groups=privileged_groups, unprivileged_groups=unprivileged_groups)

    # Convert to sklearn arrays
    X, y = to_sklearn(dataset)

    # Split
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(X, y, np.arange(X.shape[0]), test_size=0.3, random_state=42, stratify=y)

    # Train logistic regression
    clf = train_model(X_train, y_train)

    # Predictions on test
    y_pred = clf.predict(X_test)

    # Build AIF360 datasets for true and predicted
    test_bld = dataset.subset(idx_test)  # true test dataset in AIF360 format
    # create a copy for predictions
    pred_dataset = test_bld.copy()
    pred_dataset.labels = y_pred.reshape((-1,1))

    # Compute fairness-aware classification metrics
    print("\n=== Classification metrics (before mitigation) ===")
    compute_classification_metric(dataset, test_bld, pred_dataset, privileged_groups, unprivileged_groups)

    # Confusion matrix (overall)
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix (overall):")
    print(cm)

    # Visualization: false positive rates by race
    # Using AIF360 helper: compute FPR for groups
    classified_metric = ClassificationMetric(test_bld, pred_dataset, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    fpr_priv = classified_metric.false_positive_rate(privileged=True)
    fpr_unpriv = classified_metric.false_positive_rate(privileged=False)

    labels = ['Privileged', 'Unprivileged']
    fprs = [fpr_priv, fpr_unpriv]

    plt.figure(figsize=(6,4))
    sns.barplot(x=labels, y=fprs)
    plt.title('False Positive Rates by Group (Before Mitigation)')
    plt.ylabel('False Positive Rate')
    plt.ylim(0,1)
    plt.savefig('fpr_by_group_before.png')
    print("\nSaved visualization: fpr_by_group_before.png")

    # Mitigation example: Reweighing pre-processing
    RW = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    dataset_transf = RW.fit_transform(dataset)

    # Convert transformed dataset to sklearn arrays
    X_tr = dataset_transf.features
    y_tr = dataset_transf.labels.ravel()
    sample_weight = dataset_transf.instance_weights

    # Retrain with sample weights
    clf2 = LogisticRegression(solver='liblinear')
    clf2.fit(X_tr, y_tr, sample_weight=sample_weight)

    # Predictions on original test set
    y_pred2 = clf2.predict(X_test)
    pred_dataset2 = test_bld.copy()
    pred_dataset2.labels = y_pred2.reshape((-1,1))

    print("\n=== Classification metrics (after Reweighing) ===")
    compute_classification_metric(dataset, test_bld, pred_dataset2, privileged_groups, unprivileged_groups)

    # Save second visualization
    classified_metric2 = ClassificationMetric(test_bld, pred_dataset2, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    fpr_priv2 = classified_metric2.false_positive_rate(privileged=True)
    fpr_unpriv2 = classified_metric2.false_positive_rate(privileged=False)

    fprs2 = [fpr_priv2, fpr_unpriv2]
    plt.figure(figsize=(6,4))
    sns.barplot(x=labels, y=fprs2)
    plt.title('False Positive Rates by Group (After Reweighing)')
    plt.ylabel('False Positive Rate')
    plt.ylim(0,1)
    plt.savefig('fpr_by_group_after.png')
    print("Saved visualization: fpr_by_group_after.png")

if __name__ == "__main__":
    main()
