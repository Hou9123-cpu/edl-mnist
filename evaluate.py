"""Evaluation metrics for the EDL experiment.

Computes standard classification metrics (accuracy, macro and per-class
precision/recall/F1, confusion matrix) plus Dirichlet-based uncertainty
estimates, and a Gaussian Naive Bayes baseline for reference.
"""
import time

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

from config import N_CLASSES


def uncertainty_from_alpha(alpha):
    """Decompose predictive uncertainty into aleatoric and epistemic parts.

    ``alpha`` is an (N, K) array of Dirichlet concentration parameters.

    * Total uncertainty = entropy of the expected categorical p = alpha / S.
    * Aleatoric uncertainty = expected entropy of p drawn from Dir(alpha).
    * Epistemic uncertainty = mutual information = total - aleatoric.
    """
    alpha = torch.as_tensor(alpha, dtype=torch.float32)
    S = alpha.sum(dim=-1)
    proba = alpha / S.unsqueeze(-1)

    total = -(proba * torch.log(proba + 1e-12)).sum(dim=-1)
    aleatoric = -(proba * (torch.special.digamma(alpha + 1.0)
                           - torch.special.digamma(S + 1.0).unsqueeze(-1))).sum(dim=-1)
    epistemic = total - aleatoric
    return {"total": total.numpy(),
            "aleatoric": aleatoric.numpy(),
            "epistemic": epistemic.numpy()}


def compute_metrics(y_true, y_pred, alpha):
    """Return a metrics dict for the given true/predicted labels and alpha."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    per_class = {}
    for c in range(N_CLASSES):
        per_class[str(c)] = {
            "precision": float(precision_score(y_true, y_pred, labels=[c], average=None, zero_division=0)[0]),
            "recall": float(recall_score(y_true, y_pred, labels=[c], average=None, zero_division=0)[0]),
            "f1": float(f1_score(y_true, y_pred, labels=[c], average=None, zero_division=0)[0]),
        }
    metrics["per_class"] = per_class
    metrics["confusion_matrix"] = cm.tolist()

    unc = uncertainty_from_alpha(alpha)
    correct = (np.asarray(y_pred) == np.asarray(y_true))
    metrics["uncertainty"] = {
        "mean_total": float(unc["total"].mean()),
        "mean_aleatoric": float(unc["aleatoric"].mean()),
        "mean_epistemic": float(unc["epistemic"].mean()),
        "mean_epistemic_correct": float(unc["epistemic"][correct].mean()),
        "mean_epistemic_incorrect": float(unc["epistemic"][~correct].mean()),
    }
    return metrics


def gnb_baseline(X_train, y_train, X_test, y_test):
    """Gaussian Naive Bayes baseline on flattened pixel features (reference)."""
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    t0 = time.time()
    clf.fit(X_train, y_train)
    fit_time = time.time() - t0
    y_pred = clf.predict(X_test)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "fit_time_s": float(fit_time),
    }
