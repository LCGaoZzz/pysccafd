"""edgeR TMM normalisation port (calcNormFactors method="TMM" + cpm).

Ported from edgeR 4.0.16 ``.calcFactorTMM``/``.calcFactorQuantile``/
``calcNormFactors.default`` and the ``cpm`` default (non-log) transform,
matching R's operation order (``t(t(y)/lib.size)*1e6``).
"""

from __future__ import annotations

import numpy as np


def _calc_factor_tmm(obs: np.ndarray, ref: np.ndarray, libsize_obs: float, libsize_ref: float,
                     logratio_trim: float = 0.3, sum_trim: float = 0.05,
                     do_weighting: bool = True, acutoff: float = -1e10) -> float:
    obs = np.asarray(obs, dtype=float)
    ref = np.asarray(ref, dtype=float)
    n_o = libsize_obs
    n_r = libsize_ref
    with np.errstate(divide="ignore", invalid="ignore"):
        log_r = np.log2((obs / n_o) / (ref / n_r))
        abs_e = (np.log2(obs / n_o) + np.log2(ref / n_r)) / 2.0
        v = (n_o - obs) / n_o / obs + (n_r - ref) / n_r / ref
    fin = np.isfinite(log_r) & np.isfinite(abs_e) & (abs_e > acutoff)
    log_r = log_r[fin]
    abs_e = abs_e[fin]
    v = v[fin]
    if len(log_r) == 0 or np.max(np.abs(log_r)) < 1e-6:
        return 1.0
    n = len(log_r)
    lo_l = int(np.floor(n * logratio_trim)) + 1
    hi_l = n + 1 - lo_l
    lo_s = int(np.floor(n * sum_trim)) + 1
    hi_s = n + 1 - lo_s
    # R rank(): average ranks for ties.
    rank_log = _r_rank(log_r)
    rank_abs = _r_rank(abs_e)
    keep = (rank_log >= lo_l) & (rank_log <= hi_l) & (rank_abs >= lo_s) & (rank_abs <= hi_s)
    if do_weighting:
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.sum(log_r[keep] / v[keep]) / np.sum(1.0 / v[keep])
    else:
        f = float(np.mean(log_r[keep]))
    if not np.isfinite(f):
        f = 0.0
    return float(2.0 ** f)


def _r_rank(x: np.ndarray) -> np.ndarray:
    """R rank() default (ties.method="average")."""
    order = np.argsort(x, kind="heapsort")
    ranks = np.empty(len(x), dtype=float)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def calc_norm_factors_tmm(matrix: np.ndarray) -> np.ndarray:
    """edgeR calcNormFactors(method="TMM") for a dense genes x samples matrix."""
    x = np.asarray(matrix, dtype=float)
    nsamples = x.shape[1]
    lib_size = x.sum(axis=0)
    allzero = (x > 0).sum(axis=1) == 0
    if allzero.any():
        x = x[~allzero, :]
    if x.shape[0] == 0 or nsamples == 1:
        return np.ones(nsamples)

    # .calcFactorQuantile returns quantile / lib.size; the reference column is
    # chosen on that ratio (closest to its mean).
    f75 = np.array([np.quantile(x[:, j], 0.75) for j in range(nsamples)]) / lib_size
    if np.median(f75) < 1e-20:
        ref_column = int(np.argmax(np.sqrt(x).sum(axis=0)))
    else:
        ref_column = int(np.argmin(np.abs(f75 - f75.mean())))
    f = np.empty(nsamples)
    for i in range(nsamples):
        f[i] = _calc_factor_tmm(x[:, i], x[:, ref_column], lib_size[i], lib_size[ref_column])
    f = f / np.exp(np.mean(np.log(f)))
    return f


def scaling_tmm(matrix):
    """Benchmark1.R Scaling(matrix, "TMM") semantics; returns a DataFrame."""
    import pandas as pd
    values = matrix.to_numpy(dtype=float)
    row_sums = values.sum(axis=1)
    matrix = matrix.loc[row_sums != 0]
    values = matrix.to_numpy(dtype=float)
    row_var = values.var(axis=1)
    matrix = matrix.loc[row_var != 0]
    values = matrix.to_numpy(dtype=float)
    col_sums = values.sum(axis=0)
    matrix = matrix.loc[:, col_sums != 0]
    values = matrix.to_numpy(dtype=float)
    lib_size = values.sum(axis=0)
    factors = calc_norm_factors_tmm(values)
    normalized = values / (lib_size * factors)[None, :] * 1e6
    return pd.DataFrame(normalized, index=matrix.index, columns=matrix.columns)


def cpm(counts: np.ndarray, lib_size: np.ndarray, norm_factors: np.ndarray) -> np.ndarray:
    return counts / (lib_size * norm_factors)[None, :] * 1e6
