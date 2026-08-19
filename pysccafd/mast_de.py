"""MAST-style hurdle differential expression, native Python port.

Re-implements the statistical core of the official SCCAF-D signature builder
(``DEAnalysisMAST`` + ``buildSignatureMatrixMAST`` from SCCAF-D's DWLS.R):

* deterministic parts are exact ports: log2(x+0.1) group means, ROCR-equivalent
  AUC (average-rank Mann-Whitney), the fc>0.5 prefilter, BH FDR, the
  G-in-50..200 kappa-selection loop (using the LINPACK kappa port) and the
  mean-expression signature;
* the GLM core is a faithful re-implementation of MAST 1.28.0's vendored
  ``arm::bayesglm`` IRLS (prior-augmented weighted least squares with adaptive
  Student-t prior scales, gaussian dispersion updates) plus the ``ebayes``
  inverse-gamma variance prior (exact analytic marginal-likelihood MLE), and
  hurdle likelihood-ratio tests (chisq, df = 2).

Cross-language GLM fits agree with R to optimizer-level (not bit-level)
accuracy; SCCAF-D's own stochastic cell-selection stage dominates run-to-run
variation, which is the acceptance basis for this method.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import optimize, special

from pydwls import r_kappa

_EPS = 1e-8


# ---------------------------------------------------------------------------
# bayesglm IRLS core (arm / MAST vendored version)
# ---------------------------------------------------------------------------

def _bayesglm_fit(x: np.ndarray, y: np.ndarray, family: str, prior_scale: np.ndarray,
                  prior_mean: np.ndarray | None = None, prior_df: np.ndarray | None = None,
                  weights: np.ndarray | None = None, maxit: int = 100):
    """Weighted bayesglm IRLS with prior-augmented rows.

    x: (n, k) design WITHOUT separate handling — must include intercept as
    column 0 when an intercept is used. family: "binomial" or "gaussian".
    prior_scale: length k, already scaled (arm semantics: 2.5*s_y for gaussian
    slopes divided by x scales, 2.5 for binomial; 10*s_y for intercept).
    """
    n, k = x.shape
    if prior_mean is None:
        prior_mean = np.zeros(k)
    if prior_df is None:
        prior_df = np.ones(k)
    if weights is None:
        weights = np.ones(n)

    if family == "binomial":
        mu = (y + 0.5) / 2.0
        eta = np.log(mu / (1.0 - mu))
        dispersion = 1.0
    else:
        mu = np.full(n, y.mean())
        eta = mu.copy()
        dispersion = float(y.var()) / 10000.0
    var_y = float(y.var()) if family == "gaussian" else 1.0

    prior_sd = prior_scale.copy()
    dev_old = _deviance(y, mu, weights, family)
    coef = None
    for _ in range(maxit):
        if family == "binomial":
            mu_eta = mu * (1.0 - mu)
            good = np.abs(mu_eta) > 1e-10
            varmu = mu * (1.0 - mu)
        else:
            mu_eta = np.ones(n)
            good = np.ones(n, dtype=bool)
            varmu = np.ones(n)
        z = eta.copy()
        z[good] = (eta[good] + (y[good] - mu[good]) / mu_eta[good])
        w = np.zeros(n)
        w[good] = np.sqrt(weights[good] * mu_eta[good] ** 2 / varmu[good])

        z_star = np.concatenate([z, prior_mean])
        w_star = np.concatenate([w, np.sqrt(dispersion) / prior_sd])
        x_star = np.vstack([x, np.eye(k)])

        fit_coef = _lstsq_pivot(x_star * w_star[:, None], z_star * w_star)
        coef = fit_coef

        if not np.all(np.isfinite(coef)):
            break

        # adaptive prior.sd update (prior.df == 1 -> Cauchy-like)
        if not np.all(np.isinf(prior_df)):
            xw = x_star * w_star[:, None]
            q, r = np.linalg.qr(xw)
            v_coefs = np.linalg.inv(r) @ np.linalg.inv(r).T if k > 1 else np.array([[1.0 / (r[0, 0] ** 2)]])
            diag_v = np.diag(v_coefs)
            centered = coef.copy()
            col_means = x.mean(axis=0)
            centered[0] = float(coef @ col_means)
            sampling_var = diag_v.copy()
            sampling_var[0] = float((v_coefs @ col_means) @ col_means)
            sd_tmp = ((centered - prior_mean) ** 2 + sampling_var * dispersion
                      + prior_df * prior_sd ** 2) / (1.0 + prior_df)
            mask = ~np.isinf(prior_df)
            prior_sd[mask] = np.sqrt(sd_tmp[mask])

        predictions = x @ coef
        eta = predictions
        if family == "binomial":
            mu = 1.0 / (1.0 + np.exp(-eta))
            mu = np.clip(mu, 1e-10, 1 - 1e-10)
        else:
            mu = eta.copy()
            mse_resid = float(np.mean((z_star[:n] - w * predictions) ** 2))
            xw = x_star * w_star[:, None]
            q, r = np.linalg.qr(xw)
            v_coefs = np.linalg.inv(r) @ np.linalg.inv(r).T if k > 1 else np.array([[1.0 / (r[0, 0] ** 2)]])
            mse_uncertainty = max(0.0, float(np.mean(np.sum((x @ v_coefs) * x, axis=1)) * dispersion))
            dispersion = mse_resid + mse_uncertainty

        dev = _deviance(y, mu, weights, family)
        if not np.isfinite(dev) or not np.isfinite(dispersion):
            break
        if abs(dev - dev_old) / (0.1 + abs(dev)) < _EPS:
            dev_old = dev
            break
        dev_old = dev

    return coef, mu, dispersion, dev_old


def _lstsq_pivot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(a, b, rcond=None)
    return coef


def _deviance(y, mu, weights, family):
    if family == "binomial":
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(y > 0, y * np.log(np.where(mu > 0, mu, 1.0)), 0.0)
            term2 = np.where(y < 1, (1 - y) * np.log(np.where(mu < 1, 1 - mu, 1.0)), 0.0)
        return float(2 * np.sum(weights * (y * np.log(1e-300 + mu) + (1 - y) * np.log(1e-300 + 1 - mu))))
    resid = (y - mu) * np.sqrt(weights)
    return float(np.sum(resid ** 2))


def _loglik_binomial(y, mu):
    mu = np.clip(mu, 1e-300, 1 - 1e-300)
    return float(np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu)))


def _loglik_gaussian(y, mu, sigma2):
    n = len(y)
    return float(-0.5 * n * np.log(2 * np.pi * sigma2) - np.sum((y - mu) ** 2) / (2 * sigma2))


# ---------------------------------------------------------------------------
# ebayes: exact analytic marginal MLE (MAST ebayes, H0 model)
# ---------------------------------------------------------------------------

def ebayes(assay_t: np.ndarray) -> tuple[float, float]:
    """MAST ebayes on (genes x cells) log-scale values; zeros treated missing.

    Returns (priorVar, priorDOF) = (b0/a0, 2*a0).
    """
    a = assay_t.astype(float).copy()
    a[a == 0] = np.nan
    means = np.nanmean(a, axis=1)
    centered = a - means[:, None]
    r_ng = np.sum(~np.isnan(centered), axis=1) - 1
    ss_g = np.nansum(centered ** 2, axis=1)
    valid = (r_ng > 0) & np.isfinite(ss_g)
    r_ng = r_ng[valid].astype(float)
    ss_g = ss_g[valid]

    def neg_ll(theta):
        a0, b0 = theta
        if a0 <= 0 or b0 <= 0:
            return 1e300
        lbeta = special.betaln(r_ng / 2.0, a0)
        li = -lbeta - r_ng / 2.0 * np.log(b0) - np.log1p(ss_g / (2.0 * b0)) * (r_ng / 2.0 + a0)
        return -float(np.sum(li))

    def grad(theta):
        a0, b0 = theta
        if a0 <= 0 or b0 <= 0:
            return np.array([1e300, 1e300])
        s_a0 = np.sum(special.digamma(r_ng / 2.0 + a0) - special.digamma(a0) - np.log1p(ss_g / (2.0 * b0)))
        s_b0 = np.sum((a0 * ss_g - r_ng * b0) / (ss_g * b0 + 2.0 * b0 ** 2))
        return np.array([-s_a0, -s_b0])

    result = optimize.minimize(neg_ll, x0=[1.0, 1.0], jac=grad, method="L-BFGS-B",
                               bounds=[(1e-3, None), (1e-3, None)])
    a0, b0 = result.x
    v = max(b0 / a0, 0.0)
    df = max(2.0 * a0, 0.0)
    return v, df


# ---------------------------------------------------------------------------
# AUC (ROCR-equivalent, ties averaged)
# ---------------------------------------------------------------------------

def auc_rocr(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = labels == 1
    n1 = int(pos.sum())
    n0 = int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(scores, kind="heapsort")
    ranks = np.empty(len(scores), dtype=float)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    rank_sum_pos = ranks[pos].sum()
    return float((rank_sum_pos - n1 * (n1 + 1) / 2.0) / (n1 * n0))


# ---------------------------------------------------------------------------
# Hurdle DE per cell type
# ---------------------------------------------------------------------------

def _hurdle_lr_pvalues(et_log2: np.ndarray, group: np.ndarray,
                       prior_var: float, prior_dof: float) -> np.ndarray:
    """Per-gene hurdle LR p-values (chisq df=2) for Population effect.

    et_log2: (genes, cells) log2-scale values; group: 0/1 per cell.
    """
    genes_n = et_log2.shape[0]
    p_values = np.ones(genes_n)
    disc = et_log2 > 0
    x_full = np.column_stack([np.ones(len(group)), group.astype(float)])
    x_null = np.ones((len(group), 1))

    for g in range(genes_n):
        y_d = disc[g].astype(float)
        # discrete component (binomial bayesglm, arm-default priors)
        prior_scale_full = np.array([10.0, 2.5])      # binary column: x.scale = 1
        prior_scale_null = np.array([10.0])
        if y_d.sum() in (0, len(y_d)):
            lr_d = 0.0
        else:
            coef_f, mu_f, _, _ = _bayesglm_fit(x_full, y_d, "binomial", prior_scale_full)
            coef_n, mu_n, _, _ = _bayesglm_fit(x_null, y_d, "binomial", prior_scale_null)
            ll_f = _loglik_binomial(y_d, mu_f)
            ll_n = _loglik_binomial(y_d, mu_n)
            lr_d = 2.0 * (ll_f - ll_n)

        # continuous component on positive cells
        pos = disc[g]
        y_c = et_log2[g, pos]
        lr_c = 0.0
        if pos.sum() >= 5 and group[pos].var() > 0 and len(np.unique(group[pos])) == 2:
            xg = x_full[pos, :]
            xn = x_null[pos, :]
            n_pos = len(y_c)
            sy = float(y_c.std(ddof=1)) if n_pos > 1 else 1.0
            prior_full = np.array([10.0 * 2.0 * sy, 2.5 * 2.0 * sy])
            prior_null = np.array([10.0 * 2.0 * sy])
            coef_f, mu_f, disp_f, _ = _bayesglm_fit(xg, y_c, "gaussian", prior_full)
            coef_n, mu_n, disp_n, _ = _bayesglm_fit(xn, y_c, "gaussian", prior_null)
            ss_f = float(np.sum((y_c - mu_f) ** 2))
            ss_n = float(np.sum((y_c - mu_n) ** 2))
            sigma2_f = (ss_f + prior_dof * prior_var) / (n_pos + prior_dof)
            sigma2_n = (ss_n + prior_dof * prior_var) / (n_pos + prior_dof)
            ll_f = _loglik_gaussian(y_c, mu_f, sigma2_f)
            ll_n = _loglik_gaussian(y_c, mu_n, sigma2_n)
            lr_c = 2.0 * (ll_f - ll_n)

        stat = lr_d + lr_c
        if stat <= 0:
            p_values[g] = 1.0
        else:
            p_values[g] = float(special.chdtrc(2, stat))  # P(X > stat) for chisq df=2
    return p_values


def _bh_fdr(p: np.ndarray) -> np.ndarray:
    n = len(p)
    order = np.argsort(p, kind="heapsort")
    ranked = p[order]
    adjusted = ranked * n / (np.arange(n) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)
    out = np.empty(n)
    out[order] = adjusted
    return out


def build_signature_matrix_mast(scdata: pd.DataFrame, cell_types: list[str],
                                diff_cutoff: float = 0.5, pval_cutoff: float = 0.01):
    """SCCAF-D buildSignatureMatrixMAST port; returns the signature DataFrame."""
    values = scdata.to_numpy(dtype=float)
    genes = list(scdata.index)
    ct = np.asarray([str(t) for t in cell_types])
    data_log2 = np.log2(values + 0.1)

    type_order = list(dict.fromkeys(ct.tolist()))
    tables: dict[str, pd.DataFrame] = {}
    numberof_genes: dict[str, int] = {}
    for cell_type in type_order:
        in_group = ct == cell_type
        if in_group.sum() == 0 or in_group.sum() == len(ct):
            continue
        # stat.log2: group means in log2 space
        with np.errstate(divide="ignore", invalid="ignore"):
            m1 = np.log2(np.mean(2.0 ** data_log2[:, in_group] - 0.1, axis=1) + 0.1)
            m0 = np.log2(np.mean(2.0 ** data_log2[:, ~in_group] - 0.1, axis=1) + 0.1)
        log2_fc = m1 - m0
        aucs = np.array([auc_rocr(data_log2[g], in_group.astype(int)) for g in range(data_log2.shape[0])])
        aucs = np.where(np.isnan(aucs), 0.5, aucs)

        de_mask = log2_fc > diff_cutoff
        if de_mask.sum() <= 1:
            continue
        de_idx = np.where(de_mask)[0]

        # MAST: others first, then the group's cells
        ordered = np.concatenate([np.where(~in_group)[0], np.where(in_group)[0]])
        group01 = np.concatenate([np.zeros((~in_group).sum()), np.ones(in_group.sum())])
        prior_var, prior_dof = ebayes(data_log2[de_idx, :])
        from .mast_fast import hurdle_lr_pvalues_batch
        p_vals = hurdle_lr_pvalues_batch(data_log2[de_idx, :][:, ordered], group01,
                                         prior_var, prior_dof)

        table = pd.DataFrame({
            "Gene": [genes[i] for i in de_idx],
            "test.type": "hurdle",
            "p_value": p_vals,
            "log2.mean.Cluster_Other": m0[de_idx],
            f"log2.mean.{cell_type}": m1[de_idx],
            "log2fold_change": log2_fc[de_idx],
            "Auc": aucs[de_idx],
        })
        table = table.sort_values("Auc", ascending=False, kind="stable")  # rev(order(Auc))
        tables[cell_type] = table.reset_index(drop=True)

        padj = _bh_fdr(table["p_value"].to_numpy())
        keep = (padj < pval_cutoff) & (table["log2fold_change"].to_numpy() > diff_cutoff)
        gene_names = table["Gene"].to_numpy()
        non_mir = np.array([("MIR" not in g and "Mir" not in g) for g in gene_names])
        selected_rows = table[keep & non_mir]
        # R's G-loop re-orders the selected genes by log2fold_change desc.
        selected_rows = selected_rows.sort_values("log2fold_change", ascending=False, kind="stable")
        tables[cell_type] = table.assign(pvalue_adjusted=padj)
        numberof_genes[cell_type] = len(selected_rows)
        tables[cell_type].attrs["selected"] = selected_rows["Gene"].tolist()

    # G selection loop by kappa
    def top_genes(G: int) -> list[str]:
        out: list[str] = []
        for t in type_order:
            if t not in tables:
                continue
            sel = tables[t].attrs["selected"]
            if len(sel) > 0:
                out.extend(sel[: min(G, len(sel))])
        seen: set[str] = set()
        unique: list[str] = []
        for g in out:
            if g not in seen:
                seen.add(g)
                unique.append(g)
        return unique

    def mean_signature(gene_list: list[str]) -> pd.DataFrame:
        idx = [genes.index(g) for g in gene_list]
        sub = values[idx, :]
        cols = {}
        for t in type_order:
            mask = ct == t
            cols[t] = sub[:, mask].mean(axis=1)
        return pd.DataFrame(cols, index=gene_list)[type_order]

    counts = [numberof_genes.get(t, 0) for t in type_order]
    if all(c == 0 for c in counts):
        raise ValueError("MAST signature selection retained no genes")
    margin = min([49] + [c - 1 for c in counts])
    condition_numbers: list[float] = []
    for G in range(50, 201):
        gene_list = top_genes(G)
        if not gene_list:
            condition_numbers.append(math.inf)
            continue
        sig = mean_signature(gene_list)
        condition_numbers.append(r_kappa(sig.to_numpy(dtype=float)))

    arr = np.asarray(condition_numbers, dtype=float)
    g_star = int(np.nanargmin(arr)) + 1 + margin
    return mean_signature(top_genes(g_star))
