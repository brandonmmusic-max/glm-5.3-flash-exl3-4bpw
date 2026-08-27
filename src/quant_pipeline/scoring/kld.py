from __future__ import annotations

import numpy as np


def log_softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if not values.size or not np.all(np.isfinite(values)):
        raise ValueError("logits must be non-empty and finite")
    maximum = np.max(values, axis=-1, keepdims=True)
    shifted = values - maximum
    return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))


def token_kld(teacher_logits: np.ndarray, student_logits: np.ndarray) -> np.ndarray:
    if np.shape(teacher_logits) != np.shape(student_logits):
        raise ValueError("teacher and student logits must have identical shapes")
    teacher = log_softmax(teacher_logits)
    student = log_softmax(student_logits)
    probabilities = np.exp(teacher)
    result = np.sum(probabilities * (teacher - student), axis=-1)
    if not np.all(np.isfinite(result)):
        raise ValueError("KLD result is non-finite")
    return result


def summarize(values: np.ndarray) -> dict[str, float | int]:
    x = np.asarray(values, dtype=np.float64).ravel()
    if not x.size:
        raise ValueError("cannot summarize empty values")
    if not np.all(np.isfinite(x)):
        raise ValueError("cannot summarize non-finite values")
    sorted_x = np.sort(x)
    tail = sorted_x[max(0, int(np.floor(0.95 * x.size))) :]
    return {
        "count": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "p50": float(np.quantile(x, 0.50)),
        "p95": float(np.quantile(x, 0.95)),
        "p99": float(np.quantile(x, 0.99)),
        "cvar95": float(np.mean(tail)),
        "max": float(sorted_x[-1]),
    }


def paired_bootstrap(a: np.ndarray, b: np.ndarray, samples: int = 2000, seed: int = 0) -> dict[str, float]:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError("paired samples must have equal shape")
    if not a.size or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("paired samples must be non-empty and finite")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("bootstrap samples must be a positive integer")
    differences = a - b
    generator = np.random.default_rng(seed)
    means = np.empty(samples)
    for index in range(samples):
        means[index] = np.mean(generator.choice(differences, size=differences.size, replace=True))
    return {
        "mean_delta": float(np.mean(differences)),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "p_directional": float(min(np.mean(means <= 0), np.mean(means >= 0)) * 2),
    }
