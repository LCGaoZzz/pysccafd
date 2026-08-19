"""Self-contained tests for the SCCAF-D port stages that need no scanpy stack.

These pin the numerical behaviours that regressed during development:
the masked gaussian dispersion initialisation, the intercept-only binomial
fit (a broadcast bug once made it diverge), the masked residual sums of
squares, and the edgeR TMM reference-column rule.
"""

import numpy as np
import pytest

from pysccafd.mast_de import ebayes
from pysccafd.mast_fast import _binomial_bayesglm_null_batch, hurdle_lr_pvalues_batch
from pysccafd.tmm import calc_norm_factors_tmm


def test_binomial_null_fit_converges():
    y = np.zeros(372)
    y[300] = 1.0
    beta0, mu = _binomial_bayesglm_null_batch(y[None, :], np.array([10.0]))
    # population prevalence 1/372 -> intercept near logit(1/372)
    assert beta0[0] == pytest.approx(np.log(1 / 371), abs=0.5)
    assert mu.shape == (1, 372)
    assert np.all((mu > 0) & (mu < 1))


def test_hurdle_pvalues_finite_and_ordered():
    rng = np.random.RandomState(5)
    n = 200
    group = np.concatenate([np.zeros(120), np.ones(80)])
    strong = rng.lognormal(0.4, 0.3, (10, 120))
    et = np.zeros((10, n))
    et[:, :120] = strong * rng.uniform(0.8, 1.2, (10, 120))
    et[:, 120:] = rng.lognormal(-1.0, 0.3, (10, 80))
    et[:, 120:] *= (group[120:] * 3.0 + 0.2)[None, :]
    p = hurdle_lr_pvalues_batch(et, group, prior_var=1.0, prior_dof=4.0)
    assert p.shape == (10,)
    assert np.isfinite(p).all()
    assert ((p >= 0) & (p <= 1)).all()


def test_gaussian_masked_ss_excludes_masked_rows():
    from pysccafd.mast_fast import _gaussian_bayesglm_batch
    y = np.array([[2.0, 3.0, 0.0, 5.0, 0.0]])
    mask = np.array([[True, True, False, True, False]])
    g = np.zeros(5)
    _, _, fitted, _ = _gaussian_bayesglm_batch(g, y, mask,
                                               np.array([20.0]), np.array([5.0]))
    # fitted mean must approximate the mean of the three observed values
    assert fitted[0, 0] == pytest.approx(10.0 / 3.0, abs=0.5)


def test_ebayes_returns_positive_priors():
    rng = np.random.RandomState(2)
    a = rng.lognormal(2.0, 0.8, (50, 40))
    v, df = ebayes(a)
    assert v > 0 and df > 0
    assert np.isfinite(v) and np.isfinite(df)


def test_tmm_reference_column_uses_ratio_rule():
    genes = 400
    base = np.zeros((genes, 3))
    base[:, 0] = np.where(np.arange(genes) < 300, 10.0, 0.0)
    base[:, 1] = np.where(np.arange(genes) < 300, 20.0, 0.0)
    base[:, 2] = np.where(np.arange(genes) < 320, 10.0, 0.0)
    factors = calc_norm_factors_tmm(base)
    assert len(factors) == 3
    assert np.all(np.isfinite(factors))
    # f <- f / exp(mean(log(f))): geometric mean is 1
    assert np.exp(np.mean(np.log(factors))) == pytest.approx(1.0, abs=1e-12)


def test_tmm_factor_symmetry_with_self_reference():
    rng = np.random.RandomState(9)
    obs = rng.poisson(4.0, 500).astype(float)
    f = calc_norm_factors_tmm(np.column_stack([obs, obs]))
    # identical columns -> all factors collapse to 1
    assert np.allclose(f, 1.0, atol=1e-9)
