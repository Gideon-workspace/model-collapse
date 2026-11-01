"""
Pure-NumPy statistics for measuring distributional collapse. No torch
dependency, so these are fully unit-tested in this environment.
"""

import numpy as np


def empirical_distribution(samples: np.ndarray, n_categories: int) -> np.ndarray:
    """samples: 1-D int array of category ids in [0, n_categories).
    Returns a probability vector of length n_categories (Laplace-smoothed-free,
    zero-mass categories stay at exactly 0)."""
    counts = np.bincount(samples, minlength=n_categories).astype(np.float64)
    total = counts.sum()
    if total == 0:
        return np.zeros(n_categories)
    return counts / total


def shannon_entropy_bits(probs: np.ndarray) -> float:
    p = probs[probs > 0]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))


def kl_from_uniform_bits(probs: np.ndarray) -> float:
    """KL(empirical || uniform) in bits. 0 = perfectly uniform (no collapse),
    larger = more concentrated on fewer categories."""
    n = len(probs)
    p = probs[probs > 0]
    # KL(p||u) = sum p*log2(p / (1/n)) = sum p*log2(p) + log2(n)
    return float(np.sum(p * np.log2(p)) + np.log2(n))


def effective_support_size(probs: np.ndarray) -> float:
    """2 ** entropy_bits: the 'effective number of categories' under the
    distribution (e.g. 1.0 if all mass on one category, n if uniform)."""
    return float(2 ** shannon_entropy_bits(probs))


def distinct_count(samples: np.ndarray, threshold_frac: float = 0.0) -> int:
    """Number of categories that appear at all (threshold_frac=0) or with at
    least threshold_frac of the total mass."""
    uniq, counts = np.unique(samples, return_counts=True)
    if threshold_frac <= 0:
        return len(uniq)
    frac = counts / counts.sum()
    return int((frac >= threshold_frac).sum())


def distribution_summary(samples: np.ndarray, n_categories: int) -> dict:
    probs = empirical_distribution(samples, n_categories)
    return dict(
        entropy_bits=shannon_entropy_bits(probs),
        kl_from_uniform_bits=kl_from_uniform_bits(probs),
        effective_support=effective_support_size(probs),
        distinct_count=distinct_count(samples),
        n_categories=n_categories,
        n_samples=int(len(samples)),
    )


def pca_2d(X: np.ndarray):
    """Simple PCA via SVD, no sklearn dependency.
    X: [N, D]. Returns (X_2d: [N,2], explained_var_ratio: [2], components: [2,D])."""
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    total_var = (S ** 2).sum()
    explained = (S[:2] ** 2) / total_var if total_var > 0 else np.zeros(2)
    X2d = Xc @ Vt[:2].T
    return X2d, explained, Vt[:2]


def participation_ratio(X: np.ndarray) -> float:
    """Effective dimensionality of a set of vectors via the participation ratio
    of the covariance eigenspectrum: (sum lambda_i)^2 / sum(lambda_i^2).
    1.0 = all variance in one direction (maximally collapsed representation),
    D = variance spread equally across all D dimensions."""
    Xc = X - X.mean(axis=0, keepdims=True)
    cov = np.cov(Xc, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 0, None)
    s1 = eigvals.sum()
    s2 = (eigvals ** 2).sum()
    if s2 == 0:
        return 0.0
    return float((s1 ** 2) / s2)


def mean_pairwise_cosine_similarity(X: np.ndarray, max_n: int = 2000, seed: int = 0) -> float:
    """Average cosine similarity between all pairs of rows (subsampled if large).
    High values indicate anisotropic / collapsed embeddings."""
    rng = np.random.default_rng(seed)
    if X.shape[0] > max_n:
        idx = rng.choice(X.shape[0], size=max_n, replace=False)
        X = X[idx]
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    Xn = X / norms
    sims = Xn @ Xn.T
    n = sims.shape[0]
    off_diag_sum = sims.sum() - np.trace(sims)
    return float(off_diag_sum / (n * (n - 1)))
