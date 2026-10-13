"""Auto-tuning spectral clustering of voiceprints (NME-SC, Park et al. 2019, as used in NVIDIA NeMo).

No fixed similarity threshold: for each recording it picks how strongly to prune the affinity graph and how many
speakers there are from the eigengap of the graph Laplacian. That matters because the same-vs-different
threshold of a voiceprint model moves a lot between recordings (expressive talk, similar voices, mixed gender).
"""
from __future__ import annotations

import numpy as np


def _laplacian_eigs(A: np.ndarray):
    d = A.sum(1)
    L = np.diag(d) - A
    return np.linalg.eigh(L)


def _kmeans(X: np.ndarray, k: int, iters: int = 50, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    best, best_inertia = None, np.inf
    for _ in range(8):
        # k-means++ init
        c = [X[rng.integers(len(X))]]
        for _ in range(1, k):
            d2 = np.min(((X[:, None, :] - np.array(c)[None]) ** 2).sum(-1), 1)
            c.append(X[rng.choice(len(X), p=d2 / d2.sum()) if d2.sum() > 0 else rng.integers(len(X))])
        C = np.array(c)
        for _ in range(iters):
            lab = ((X[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
            newC = np.array([X[lab == j].mean(0) if np.any(lab == j) else C[j] for j in range(k)])
            if np.allclose(newC, C):
                break
            C = newC
        inertia = ((X - C[lab]) ** 2).sum()
        if inertia < best_inertia:
            best, best_inertia = lab, inertia
    return best


def nme_sc(E: np.ndarray, max_speakers: int = 8, min_speakers: int = 1) -> np.ndarray:
    """Cluster L2-normalised embeddings E (n, d). Returns labels 0..k-1."""
    n = len(E)
    if n < 3:
        return np.zeros(n, dtype=int)
    S = np.clip(E @ E.T, -1, 1)
    S = (S + 1) / 2  # affinities in [0, 1]
    np.fill_diagonal(S, 1.0)
    max_k = min(max_speakers, n - 1)
    best = None
    p_max = max(2, min(n - 1, int(n * 0.25)))
    for p in sorted(set(np.linspace(2, p_max, num=min(12, p_max - 1), dtype=int))):
        # keep each row's p strongest links, symmetrise
        A = np.zeros_like(S)
        idx = np.argpartition(-S, p, axis=1)[:, :p]
        rows = np.repeat(np.arange(n), p)
        A[rows, idx.ravel()] = 1.0
        A = 0.5 * (A + A.T)
        w, _ = _laplacian_eigs(A)
        w = np.sort(w)[: max_k + 1]
        gaps = np.diff(w)
        k = int(np.argmax(gaps[min_speakers - 1:])) + min_speakers
        g = gaps[k - 1] / (w[-1] + 1e-10)
        ratio = (p / n) / (g + 1e-10)
        if best is None or ratio < best[0]:
            best = (ratio, p, k)
    _, p, k = best
    A = np.zeros_like(S)
    idx = np.argpartition(-S, p, axis=1)[:, :p]
    A[np.repeat(np.arange(n), p), idx.ravel()] = 1.0
    A = 0.5 * (A + A.T)
    _, v = _laplacian_eigs(A)
    X = v[:, :k]
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
    return _kmeans(X, k) if k > 1 else np.zeros(n, dtype=int)


def merge_with_constraints(E: np.ndarray, durs: np.ndarray, labels: np.ndarray, windows: np.ndarray,
                           same: float = 0.80, together_max: float = 0.3, small_sec: float = 8.0,
                           small_share: float = 0.10) -> np.ndarray:
    """Merge over-split clusters into people.

    E: voiceprints of local voices (n, d); durs: their clean speech seconds; labels: initial clusters;
    windows: id of the analysis window each local voice came from.
    Averaged over a whole cluster, voiceprints are stable: on real recordings two clusters of the same person
    score ~0.86-0.97, different people <= ~0.7. So clusters are merged (most similar first) while the
    similarity is at the same-person level, unless the segmentation often heard them as two voices at the
    same time. Finally clusters with very little speech are folded into the most similar one.
    """
    lab = labels.copy()

    def stats():
        ids = sorted(set(lab))
        cent = {c: E[lab == c].T @ durs[lab == c] for c in ids}
        cent = {c: v / (np.linalg.norm(v) + 1e-9) for c, v in cent.items()}
        present = {c: len(set(windows[lab == c])) for c in ids}
        together: dict[tuple, int] = {}
        for w in set(windows):
            cs = sorted(set(lab[windows == w]))
            for i in range(len(cs)):
                for j in range(i + 1, len(cs)):
                    together[(cs[i], cs[j])] = together.get((cs[i], cs[j]), 0) + 1
        return ids, cent, present, together

    while True:
        ids, cent, present, together = stats()
        best = None
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                t = together.get((a, b), 0)
                if t > together_max * min(present[a], present[b]):
                    continue
                sim = float(cent[a] @ cent[b])
                if sim >= same and (best is None or sim > best[0]):
                    best = (sim, a, b)
        if best is None:
            break
        lab[lab == best[2]] = best[1]
    ids, cent, present, together = stats()
    talk = {c: float(durs[lab == c].sum()) for c in ids}
    total = sum(talk.values())
    for c in sorted(ids, key=lambda x: talk[x]):
        # small = little speech in absolute terms AND as a share (in a short meeting everyone has little)
        if len(set(lab)) < 2 or talk[c] >= small_sec or talk[c] >= small_share * total:
            continue
        others = [o for o in set(lab) if o != c]
        dst = max(others, key=lambda o: float(cent[c] @ cent[o]))
        lab[lab == c] = dst
        talk[dst] = talk.get(dst, 0) + talk[c]
    # relabel 0..k-1 by talk time
    order = sorted(set(lab), key=lambda c: -durs[lab == c].sum())
    return np.array([order.index(c) for c in lab])
