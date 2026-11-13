"""Evaluation helpers shared by every notebook.

PR-AUC (average precision) is the primary metric: with about 3% positives,
accuracy is meaningless and ROC-AUC is optimistic. A random ranking scores a
PR-AUC roughly equal to the positive rate.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from . import config


def pr_auc(y_true, scores) -> float:
    return float(average_precision_score(y_true, scores))


def evaluate(y_true, scores) -> dict[str, float]:
    """PR-AUC and ROC-AUC of a score vector (any monotone score works)."""
    return {"pr_auc": pr_auc(y_true, scores), "roc_auc": float(roc_auc_score(y_true, scores))}


def precision_recall_at_k(y_true, scores, k: int = config.SHORTLIST_K) -> tuple[float, float]:
    """Precision and recall of the top-k shortlist.

    Ties at the cut-off are broken by the original row order (stable sort), so
    the result is deterministic.
    """
    y_true = np.asarray(y_true)
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")[:k]
    hits = int(y_true[order].sum())
    return hits / k, hits / max(int(y_true.sum()), 1)


def bootstrap_ci(y_true, scores, metric=pr_auc, n_resamples: int = config.BOOTSTRAP_RESAMPLES,
                 seed: int = config.SEED, level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap interval: resample rows with replacement.

    Resamples without any positive are redrawn, because PR-AUC is undefined there.
    """
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values = []
    while len(values) < n_resamples:
        idx = rng.integers(0, n, n)
        if y_true[idx].sum() == 0 or y_true[idx].sum() == n:
            continue
        values.append(metric(y_true[idx], scores[idx]))
    lo, hi = np.percentile(values, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return float(lo), float(hi)


def paired_bootstrap_diff(y_true, scores_a, scores_b, metric=pr_auc,
                          n_resamples: int = config.BOOTSTRAP_RESAMPLES, seed: int = config.SEED,
                          level: float = 0.95) -> tuple[float, float, float]:
    """Difference metric(a) - metric(b) with a paired percentile interval.

    Both models are scored on the same resampled rows, so the interval reflects
    the difference between models rather than the noise of the test sample.
    """
    y_true = np.asarray(y_true)
    scores_a, scores_b = np.asarray(scores_a), np.asarray(scores_b)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    while len(diffs) < n_resamples:
        idx = rng.integers(0, n, n)
        if y_true[idx].sum() == 0 or y_true[idx].sum() == n:
            continue
        diffs.append(metric(y_true[idx], scores_a[idx]) - metric(y_true[idx], scores_b[idx]))
    lo, hi = np.percentile(diffs, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    point = metric(y_true, scores_a) - metric(y_true, scores_b)
    return float(point), float(lo), float(hi)


def brier(y_true, probabilities) -> float:
    """Mean squared error of predicted probabilities (lower is better)."""
    return float(brier_score_loss(y_true, np.clip(probabilities, 0.0, 1.0)))


def full_report(y_true, scores, probabilities=None, k: int = config.SHORTLIST_K) -> dict[str, float]:
    """Every metric reported in the final test table for one model."""
    out = evaluate(y_true, scores)
    lo, hi = bootstrap_ci(y_true, scores)
    out["pr_auc_lo"], out["pr_auc_hi"] = lo, hi
    out[f"precision_at_{k}"], out[f"recall_at_{k}"] = precision_recall_at_k(y_true, scores, k)
    if probabilities is not None:
        out["brier"] = brier(y_true, probabilities)
    return out
