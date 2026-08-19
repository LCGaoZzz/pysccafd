"""Vectorised batched bayesglm hurdle tests (see mast_de for the reference).

The official MAST stage fits, per gene and per cell type, four tiny bayesglm
models (discrete/continuous x full/null) whose designs have at most two
columns. Because every solve is a 2x2 (or 1x1) prior-augmented normal-equation
system with closed-form solution, all genes of one cell type are fitted
simultaneously with array operations instead of a Python loop. The update rules
follow the reference implementation in ``mast_de._bayesglm_fit`` (IRLS with
prior rows, adaptive prior.sd, gaussian dispersion).
"""

from __future__ import annotations

import numpy as np
from scipy import special


def _binomial_bayesglm_batch(g: np.ndarray, y: np.ndarray, prior_sd0: np.ndarray,
                             prior_sd1: np.ndarray, maxit: int = 100):
    """Batched binomial bayesglm with intercept + single covariate g.

    y: (m, n) binary; g: (n,) 0/1. prior_sd*: (m,) per-gene prior scales for
    intercept / slope. Returns (beta0, beta1, mu).
    """
    m, n = y.shape
    beta0 = np.zeros(m)
    beta1 = np.zeros(m)
    mu = (y + 0.5) / 2.0
    eta = np.log(mu / (1.0 - mu))
    p0 = 1.0 / prior_sd0 ** 2
    p1 = 1.0 / prior_sd1 ** 2
    psd0 = prior_sd0.copy()
    psd1 = prior_sd1.copy()
    dev_old = -2.0 * (y * np.log(mu) + (1 - y) * np.log(1 - mu)).sum(axis=1)
    active = np.ones(m, dtype=bool)
    for _ in range(maxit):
        mu_eta = mu * (1.0 - mu)
        good = np.abs(mu_eta) > 1e-10
        w2 = np.where(good, mu_eta, 0.0)                    # w = sqrt(mu_eta); w^2 = mu_eta
        z = np.where(good, eta + (y - mu) / np.maximum(mu_eta, 1e-12), 0.0)
        w2g = w2 * g[None, :]
        s11 = w2.sum(axis=1) + p0
        s12 = w2g.sum(axis=1)
        s22 = (w2g * g[None, :]).sum(axis=1) + p1
        b1 = (w2 * z).sum(axis=1)
        b2 = (w2g * z).sum(axis=1)
        det = s11 * s22 - s12 * s12
        det = np.where(np.abs(det) < 1e-300, 1e-300, det)
        beta0_new = (s22 * b1 - s12 * b2) / det
        beta1_new = (s11 * b2 - s12 * b1) / det

        # adaptive prior.sd (df=1): needs V=(X'W^2X+P)^-1 diag and centered coefs
        v00 = s22 / det
        v01 = -s12 / det
        v11 = s11 / det
        gbar = g.mean()
        centered0 = beta0_new + beta1_new * gbar             # intercept centered at colMeans
        centered_var0 = v00 + 2.0 * gbar * v01 + gbar ** 2 * v11
        sd0 = np.sqrt((centered0 ** 2 + centered_var0 + psd0 ** 2) / 2.0)
        sd1 = np.sqrt((beta1_new ** 2 + v11 + psd1 ** 2) / 2.0)

        eta_new = beta0_new[:, None] + beta1_new[:, None] * g[None, :]
        mu_new = np.clip(1.0 / (1.0 + np.exp(-eta_new)), 1e-10, 1 - 1e-10)
        dev = -2.0 * (y * np.log(mu_new) + (1 - y) * np.log(1 - mu_new)).sum(axis=1)
        converged = np.abs(dev - dev_old) / (0.1 + np.abs(dev)) < 1e-8
        beta0[active] = beta0_new[active]
        beta1[active] = beta1_new[active]
        psd0[active] = sd0[active]
        psd1[active] = sd1[active]
        eta[active] = eta_new[active]
        mu[active] = mu_new[active]
        dev_old[active] = dev[active]
        active &= ~converged
        p0 = 1.0 / psd0 ** 2
        p1 = 1.0 / psd1 ** 2
        if not active.any():
            break
    return beta0, beta1, mu


def _binomial_bayesglm_null_batch(y: np.ndarray, prior_sd0: np.ndarray, maxit: int = 100):
    m, n = y.shape
    beta0 = np.zeros(m)
    mu = (y + 0.5) / 2.0
    p0 = 1.0 / prior_sd0 ** 2
    psd0 = prior_sd0.copy()
    dev_old = -2.0 * (y * np.log(mu) + (1 - y) * np.log(1 - mu)).sum(axis=1)
    active = np.ones(m, dtype=bool)
    for _ in range(maxit):
        mu_eta = mu * (1.0 - mu)
        good = np.abs(mu_eta) > 1e-10
        w2 = np.where(good, mu_eta, 0.0)
        z = np.where(good, np.log(np.maximum(mu, 1e-12) / np.maximum(1 - mu, 1e-12)) + (y - mu) / np.maximum(mu_eta, 1e-12), 0.0)
        beta0_new = (w2 * z).sum(axis=1) / (w2.sum(axis=1) + p0)
        psd0_new = np.sqrt((beta0_new ** 2 + 1.0 / (w2.sum(axis=1) + p0) + psd0 ** 2) / 2.0)
        mu_new = np.broadcast_to(
            np.clip(1.0 / (1.0 + np.exp(-beta0_new))[:, None], 1e-10, 1 - 1e-10), y.shape
        ).copy()
        dev = -2.0 * (y * np.log(mu_new) + (1 - y) * np.log(1 - mu_new)).sum(axis=1)
        converged = np.abs(dev - dev_old) / (0.1 + np.abs(dev)) < 1e-8
        beta0[active] = beta0_new[active]
        psd0[active] = psd0_new[active]
        mu[active] = mu_new[active]
        dev_old[active] = dev[active]
        active &= ~converged
        p0 = 1.0 / psd0 ** 2
        if not active.any():
            break
    return beta0, mu


def _gaussian_bayesglm_batch(g: np.ndarray, y_masked: np.ndarray, mask: np.ndarray,
                             prior_sd0: np.ndarray, prior_sd1: np.ndarray, maxit: int = 100):
    """Batched gaussian bayesglm on per-gene row subsets (mask selects rows).

    y_masked: (m, n) values (zeros where masked out); mask: (m, n) boolean.
    """
    m, n = y_masked.shape
    n_pos = mask.sum(axis=1).astype(float)
    n_pos = np.maximum(n_pos, 1.0)
    # dispersion starts at var(y)/1e4 over the gene's positive subset only
    mean_pos = np.where(mask, y_masked, 0.0).sum(axis=1) / n_pos
    var_pos = np.where(mask, (y_masked - mean_pos[:, None]) ** 2, 0.0).sum(axis=1) / n_pos
    disp = var_pos / 10000.0
    psd0 = prior_sd0.copy()
    psd1 = prior_sd1.copy()
    beta0 = np.zeros(m)
    beta1 = np.zeros(m)
    fitted = np.zeros_like(y_masked)
    dev_old = np.where(mask, (y_masked - mean_pos[:, None]) ** 2, 0.0).sum(axis=1)
    active = np.ones(m, dtype=bool)
    for _ in range(maxit):
        p0 = disp / psd0 ** 2
        p1 = disp / psd1 ** 2
        s11 = n_pos + p0
        s12 = (mask * g[None, :]).sum(axis=1)
        s22 = (mask * g[None, :] * g[None, :]).sum(axis=1) + p1
        b1 = y_masked.sum(axis=1)
        b2 = (y_masked * g[None, :]).sum(axis=1)
        det = s11 * s22 - s12 * s12
        det = np.where(np.abs(det) < 1e-300, 1e-300, det)
        beta0_new = (s22 * b1 - s12 * b2) / det
        beta1_new = (s11 * b2 - s12 * b1) / det

        v00 = s22 / det
        v01 = -s12 / det
        v11 = s11 / det
        gbar = (mask * g[None, :]).sum(axis=1) / n_pos
        centered0 = beta0_new + beta1_new * gbar
        centered_var0 = v00 + 2.0 * gbar * v01 + gbar ** 2 * v11
        sd0 = np.sqrt((centered0 ** 2 + centered_var0 * disp + psd0 ** 2) / 2.0)
        sd1 = np.sqrt((beta1_new ** 2 + v11 * disp + psd1 ** 2) / 2.0)

        pred = beta0_new[:, None] + beta1_new[:, None] * g[None, :]
        resid = np.where(mask, y_masked - pred, 0.0)
        ss = (resid ** 2).sum(axis=1)
        mse_resid = ss / n_pos
        # mse.uncertainty = mean over the gene's SUBSET rows of x'Vx * dispersion
        gbar_sub = (mask * g[None, :]).sum(axis=1) / n_pos
        g2bar_sub = (mask * (g ** 2)[None, :]).sum(axis=1) / n_pos
        mse_uncertainty = np.maximum(0.0, (v00 + 2 * gbar_sub * v01 + g2bar_sub * v11) * disp)
        disp_new = mse_resid + mse_uncertainty
        converged = np.abs(ss - dev_old) / (0.1 + np.abs(ss)) < 1e-8
        beta0[active] = beta0_new[active]
        beta1[active] = beta1_new[active]
        psd0[active] = sd0[active]
        psd1[active] = sd1[active]
        disp[active] = disp_new[active]
        fitted[active] = pred[active]
        dev_old[active] = ss[active]
        active &= ~converged
        if not active.any():
            break
    return beta0, beta1, fitted, disp


def hurdle_lr_pvalues_batch(et_log2: np.ndarray, group: np.ndarray,
                            prior_var: float, prior_dof: float) -> np.ndarray:
    """Batched version of mast_de._hurdle_lr_pvalues (same semantics)."""
    genes_n = et_log2.shape[0]
    disc = et_log2 > 0
    xg = group.astype(float)
    p_values = np.ones(genes_n)

    # ---- discrete ----
    y_d = disc.astype(float)
    separable = (y_d.sum(axis=1) == 0) | (y_d.sum(axis=1) == y_d.shape[1])
    lr_d = np.zeros(genes_n)
    if (~separable).any():
        idx = np.where(~separable)[0]
        bf0, bf1, mu_f = _binomial_bayesglm_batch(xg, y_d[idx], np.full(len(idx), 10.0), np.full(len(idx), 2.5))
        bn0, mu_n = _binomial_bayesglm_null_batch(y_d[idx], np.full(len(idx), 10.0))
        yb = y_d[idx]
        ll_f = (yb * np.log(np.clip(mu_f, 1e-300, 1)) + (1 - yb) * np.log(np.clip(1 - mu_f, 1e-300, 1))).sum(axis=1)
        ll_n = (yb * np.log(np.clip(mu_n, 1e-300, 1)) + (1 - yb) * np.log(np.clip(1 - mu_n, 1e-300, 1))).sum(axis=1)
        lr_d[idx] = 2.0 * (ll_f - ll_n)

    # ---- continuous ----
    lr_c = np.zeros(genes_n)
    y_masked = np.where(disc, et_log2, 0.0)
    n_pos = disc.sum(axis=1)
    sy = np.where(n_pos > 1,
                  np.array([et_log2[i, disc[i]].std(ddof=1) if n_pos[i] > 1 else 1.0 for i in range(genes_n)]),
                  1.0)
    eligible = (n_pos >= 5) & (np.array([group[disc[i]].var() > 0 and len(np.unique(group[disc[i]])) == 2
                                         if n_pos[i] > 0 else False for i in range(genes_n)]))
    if eligible.any():
        idx = np.where(eligible)[0]
        bf0, bf1, fitted_f, _ = _gaussian_bayesglm_batch(
            xg, y_masked[idx], disc[idx], 10.0 * 2.0 * sy[idx], 2.5 * 2.0 * sy[idx])
        bn0, fitted_n, _ = _gaussian_bayesglm_batch_null(
            y_masked[idx], disc[idx], 10.0 * 2.0 * sy[idx])
        n_i = n_pos[idx].astype(float)
        ss_f = (np.where(disc[idx], y_masked[idx] - fitted_f, 0.0) ** 2).sum(axis=1)
        ss_n = (np.where(disc[idx], y_masked[idx] - fitted_n, 0.0) ** 2).sum(axis=1)
        sigma2_f = (ss_f + prior_dof * prior_var) / (n_i + prior_dof)
        sigma2_n = (ss_n + prior_dof * prior_var) / (n_i + prior_dof)
        y_i = y_masked[idx]
        ll_f = -0.5 * n_i * np.log(2 * np.pi * sigma2_f) - ss_f / (2 * sigma2_f)
        ll_n = -0.5 * n_i * np.log(2 * np.pi * sigma2_n) - ss_n / (2 * sigma2_n)
        lr_c[idx] = 2.0 * (ll_f - ll_n)

    stat = lr_d + lr_c
    p = np.where(stat > 0, special.chdtrc(2, np.maximum(stat, 0)), 1.0)
    return np.clip(p, 0.0, 1.0)


def _gaussian_bayesglm_batch_null(y_masked: np.ndarray, mask: np.ndarray,
                                  prior_sd0: np.ndarray, maxit: int = 100):
    m, n = y_masked.shape
    n_pos = np.maximum(mask.sum(axis=1).astype(float), 1.0)
    mean_pos = np.where(mask, y_masked, 0.0).sum(axis=1) / n_pos
    var_pos = np.where(mask, (y_masked - mean_pos[:, None]) ** 2, 0.0).sum(axis=1) / n_pos
    disp = var_pos / 10000.0
    psd0 = prior_sd0.copy()
    beta0 = np.zeros(m)
    fitted = np.zeros_like(y_masked)
    dev_old = np.where(mask, (y_masked - mean_pos[:, None]) ** 2, 0.0).sum(axis=1)
    active = np.ones(m, dtype=bool)
    for _ in range(maxit):
        p0 = disp / psd0 ** 2
        beta0_new = y_masked.sum(axis=1) / (n_pos + p0)
        v00 = 1.0 / (n_pos + p0)
        psd0_new = np.sqrt((beta0_new ** 2 + v00 * disp + psd0 ** 2) / 2.0)
        resid = np.where(mask, y_masked - beta0_new[:, None], 0.0)
        ss = (resid ** 2).sum(axis=1)
        disp_new = ss / n_pos + np.maximum(0.0, v00 * disp)
        fitted_new = np.broadcast_to(beta0_new[:, None], y_masked.shape)
        converged = np.abs(ss - dev_old) / (0.1 + np.abs(ss)) < 1e-8
        beta0[active] = beta0_new[active]
        psd0[active] = psd0_new[active]
        disp[active] = disp_new[active]
        fitted[active] = fitted_new[active]
        dev_old[active] = ss[active]
        active &= ~converged
        if not active.any():
            break
    return beta0, fitted, disp
