"""Statistical tests on standardised residuals.

A well-specified fit produces whitened residuals that are an
independent, identically distributed standard-normal sample. The four
checks here probe that claim from complementary angles:

* :func:`residual_moments` — mean, variance, skewness and kurtosis
  (pooled and per spatial component) plus the per-component covariance,
  which should equal the identity when the diffusion is correct;
* :func:`autocorrelation_tests` — Ljung--Box test on the residuals and
  on their squares, measured strictly along time;
* :func:`normality_test` — Kolmogorov--Smirnov test against ``N(0, 1)``
  with Q--Q data for plotting;
* :func:`mse_consistency` — predicted vs realised normalised mean
  squared error of the inferred force.

References
----------
Ljung, G. M., & Box, G. E. P. (1978). On a measure of lack of fit in
time series models. Biometrika, 65(2), 297--303.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from .residuals import ResidualBundle


# --------------------------------------------------------------------- #
# Moments
# --------------------------------------------------------------------- #
def residual_moments(bundle: ResidualBundle) -> dict:
    """Pooled and per-component moments of the standardised residuals.

    The per-component covariance should be the identity if the inferred
    diffusion correctly describes the noise; deviations are summarised by
    the Frobenius distance from ``I`` and the worst off-diagonal entry.
    """
    z = np.asarray(bundle.z, dtype=np.float64)
    z_c = np.asarray(bundle.z_components, dtype=np.float64)
    n = int(z.size)
    if n == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "skew": float("nan"),
            "excess_kurt": float("nan"),
        }

    mean = float(np.mean(z))
    std = float(np.std(z, ddof=1)) if n > 1 else float("nan")
    if std and std > 0:
        z0 = (z - mean) / std
        skew = float(np.mean(z0**3))
        excess_kurt = float(np.mean(z0**4) - 3.0)
    else:
        skew = float("nan")
        excess_kurt = float("nan")

    # Per-component
    if z_c.shape[0] > 1:
        m_c = z_c.mean(axis=0)
        s_c = z_c.std(axis=0, ddof=1)
        scale = np.where(s_c > 0, s_c, 1.0)  # guard against zero std
        zc0 = (z_c - m_c) / scale
        skew_c = np.mean(zc0**3, axis=0)
        kurt_c = np.mean(zc0**4, axis=0) - 3.0
        cov = np.cov(z_c, rowvar=False)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]])
        d = cov.shape[0]
        diff = cov - np.eye(d)
        cov_frob = float(np.linalg.norm(diff, ord="fro"))
        offdiag = diff - np.diag(np.diag(diff))
        offdiag_max = float(np.max(np.abs(offdiag))) if d > 1 else 0.0
    else:
        d = z_c.shape[1]
        m_c = np.full(d, np.nan)
        s_c = np.full(d, np.nan)
        skew_c = np.full(d, np.nan)
        kurt_c = np.full(d, np.nan)
        cov = np.full((d, d), np.nan)
        cov_frob = float("nan")
        offdiag_max = float("nan")

    return {
        "n": n,
        "mean": mean,
        "std": std,
        "skew": skew,
        "excess_kurt": excess_kurt,
        "per_component": {
            "mean": m_c,
            "std": s_c,
            "skew": skew_c,
            "excess_kurt": kurt_c,
            "covariance": cov,
            "cov_minus_I_frob": cov_frob,
            "cov_offdiag_max": offdiag_max,
        },
    }


# --------------------------------------------------------------------- #
# Autocorrelation (Ljung--Box)
# --------------------------------------------------------------------- #
def _unit_series(bundle: ResidualBundle) -> list[np.ndarray]:
    """Time-ordered residual series, one per (dataset, particle, component).

    Masked-out steps are dropped, so each series holds the valid samples
    of a single scalar channel in time order. Keeping autocorrelation
    within a single channel avoids mixing particles or spatial
    components at short lags — the pitfall of a flattened pooled stream.
    """
    series: list[np.ndarray] = []
    for z_full, mask in bundle.whitened:
        z_full = np.asarray(z_full)  # (K, N, d)
        mask = np.asarray(mask)  # (K, N)
        _, N, d = z_full.shape
        for n in range(N):
            m = mask[:, n]
            if int(m.sum()) < 3:
                continue
            zc = z_full[m, n, :]  # (Kv, d), time-ordered
            for c in range(d):
                series.append(zc[:, c])
    return series


def _pooled_acf(series: list[np.ndarray], n_lags: int) -> tuple[np.ndarray, int]:
    """Autocorrelation at lags ``1..n_lags`` pooled across many series.

    Centres by the global mean and normalises by the global variance,
    accumulating lagged products *within* each series only. For a single
    gap-free series this reduces to the textbook biased ACF estimator.
    Returns ``(acf, n_total)``.
    """
    if not series:
        return np.full(n_lags, np.nan), 0
    allv = np.concatenate(series)
    n_total = int(allv.size)
    if n_total < 2:
        return np.full(n_lags, np.nan), n_total
    mu = float(allv.mean())
    var = float(np.mean((allv - mu) ** 2))
    if var <= 0.0:
        return np.full(n_lags, np.nan), n_total

    num = np.zeros(n_lags, dtype=np.float64)
    for s in series:
        sc = np.asarray(s, dtype=np.float64) - mu
        m = sc.size
        kmax = min(n_lags, m - 1)
        for k in range(1, kmax + 1):
            num[k - 1] += float(np.dot(sc[: m - k], sc[k:]))
    acf = num / (n_total * var)
    return acf, n_total


def _ljung_box(acf: np.ndarray, n: int) -> dict:
    """Ljung--Box statistic from a pooled ACF and effective sample size."""
    n_lags = acf.size
    ks = np.arange(1, n_lags + 1)
    finite = np.isfinite(acf) & (n - ks > 0)
    if not finite.any():
        return {"statistic": float("nan"), "pvalue": float("nan"), "n_lags": 0}
    ks_f = ks[finite]
    rk = acf[finite]
    Q = float(n * (n + 2) * np.sum(rk**2 / (n - ks_f)))
    df = int(finite.sum())
    p = float(stats.chi2.sf(Q, df=df))
    return {"statistic": Q, "pvalue": p, "n_lags": df}


def autocorrelation_tests(bundle: ResidualBundle, n_lags: int | None = None) -> dict:
    """Ljung--Box tests on the residuals and their squares.

    Autocorrelation of ``z`` flags missing dynamics or a too-small
    basis; autocorrelation of ``z**2`` flags state-dependent noise that
    a constant diffusion misses. Both are measured strictly along time
    (see :func:`_unit_series`).

    Returns keys ``"acf"``, ``"acf_squared"``, ``"ljung_box"``,
    ``"ljung_box_squared"`` and ``"n_eff"``. The default lag count is
    ``min(20, n_eff // 5)``.
    """
    series = _unit_series(bundle)
    n_eff = int(sum(int(s.size) for s in series))
    if n_eff < 6 or not series:
        nan = {"statistic": float("nan"), "pvalue": float("nan"), "n_lags": 0}
        return {
            "acf": np.array([]),
            "acf_squared": np.array([]),
            "ljung_box": nan,
            "ljung_box_squared": nan,
            "n_eff": n_eff,
        }

    if n_lags is None:
        n_lags = int(min(20, max(1, n_eff // 5)))

    acf, _ = _pooled_acf(series, n_lags)
    acf2, _ = _pooled_acf([s**2 for s in series], n_lags)

    return {
        "acf": acf,
        "acf_squared": acf2,
        "ljung_box": _ljung_box(acf, n_eff),
        "ljung_box_squared": _ljung_box(acf2, n_eff),
        "n_eff": n_eff,
    }


# --------------------------------------------------------------------- #
# Normality
# --------------------------------------------------------------------- #
def normality_test(bundle: ResidualBundle) -> dict:
    """Kolmogorov--Smirnov test of the pooled residuals against ``N(0, 1)``.

    Returns ``{"ks": {statistic, pvalue}, "qq": {sample_quantiles,
    theoretical_quantiles}}``. The residuals are whitened to known mean
    0 and variance 1 by construction, so the fully-specified ``N(0, 1)``
    reference is appropriate (no parameters are estimated from ``z``).
    """
    z = np.asarray(bundle.z, dtype=np.float64)
    n = z.size
    if n < 8:
        nan = {"statistic": float("nan"), "pvalue": float("nan")}
        return {"ks": nan, "qq": {"sample_quantiles": z, "theoretical_quantiles": z * 0}}

    ks_stat, ks_p = stats.kstest(z, "norm")
    z_sorted = np.sort(z)
    probs = (np.arange(1, n + 1) - 0.5) / n
    theo = stats.norm.ppf(probs)
    return {
        "ks": {"statistic": float(ks_stat), "pvalue": float(ks_p)},
        "qq": {"sample_quantiles": z_sorted, "theoretical_quantiles": theo},
    }


# --------------------------------------------------------------------- #
# Predicted vs realised MSE
# --------------------------------------------------------------------- #
def mse_consistency(inferer, bundle: ResidualBundle) -> dict:
    """Compare predicted vs realised Itô NMSE for the inferred force.

    The whitened residuals satisfy ``r^T A^{-1} r / dt = ||z||^2``. Under
    a correctly specified model each ``||z_t||^2`` follows a
    :math:`\\chi^2_d` law (mean ``d``, variance ``2d``), so the sample
    mean of the squared norms has standard error ``sqrt(2d / n)``. We
    report a z-score for the excess ``(mean - d)``; ``|z| > 5`` signals
    bias or a misspecified diffusion. The realised NMSE divides the
    systematic part of that excess by the mean force magnitude
    ``<F^T A^{-1} F>`` and is compared to ``force_predicted_MSE``.
    """
    sqn = np.asarray(bundle.z_squared_norms, dtype=np.float64)
    F2 = np.asarray(
        getattr(bundle, "force_quadratic_form", np.zeros(0)),
        dtype=np.float64,
    )
    n_obs = int(bundle.n_obs)
    d = int(bundle.d)

    out: dict = {}
    pred = getattr(inferer, "force_predicted_MSE", None)
    out["predicted_NMSE"] = float(pred) if isinstance(pred, (int, float)) and pred == pred else None

    if n_obs == 0:
        out["realised_NMSE"] = float("nan")
        out["ratio"] = float("nan")
        return out

    mean_mahal = float(np.mean(sqn))
    excess = mean_mahal - d
    excess_stderr = (2.0 * d / max(n_obs, 1)) ** 0.5
    excess_z = float(excess / excess_stderr) if excess_stderr > 0 else float("nan")

    mean_dt = float(getattr(bundle, "mean_dt", float("nan")))
    # Convert the chi-square excess to a mean-square force error. For the
    # overdamped increment residual the factor is 1; for the underdamped
    # acceleration residual it is KAPPA_UD (the residual carries a 2/3 noise
    # factor, so the same excess maps to a 2/3-smaller force error).
    factor = float(getattr(bundle, "nmse_excess_factor", 1.0))
    SSE = factor * max(excess, 0.0) / mean_dt if (mean_dt > 0 and np.isfinite(mean_dt)) else 0.0
    denom = float(np.mean(F2)) if F2.size > 0 else float("nan")
    realised = SSE / denom if (denom and denom > 0 and np.isfinite(denom)) else float("nan")

    out["realised_NMSE"] = float(realised)
    out["mean_mahalanobis_norm"] = float(mean_mahal)
    out["force_quadratic_form"] = float(denom)
    out["excess_z"] = excess_z

    pred_nmse = out.get("predicted_NMSE")
    if pred_nmse is not None and pred_nmse > 0 and realised == realised:
        out["ratio"] = float(realised / pred_nmse)
    else:
        out["ratio"] = float("nan")
    return out


# --------------------------------------------------------------------- #
# Four-point cross-covariance (parametric flow residuals)
# --------------------------------------------------------------------- #
def parametric_four_point_diagnostic(data, drift_fn, dt, n_substeps=4):
    r"""Four-point diagnostic: test that Cov[r_i, r_{i+2}] = 0.

    Under correct model specification and i.i.d. measurement noise,
    midpoint flow residuals separated by two steps share no measurement
    noise source, so their cross-covariance should vanish.  Deviations
    indicate model misspecification, correlated measurement noise,
    or higher-order effects.

    Parameters
    ----------
    data : TrajectoryCollection or TrajectoryDataset
        Observed trajectory data.
    drift_fn : callable (d,) → (d,)
        Drift function with parameters already closed over.
    dt : float
        Observation time step.
    n_substeps : int
        RK4 micro-steps per interval.

    Returns
    -------
    result : dict
        ``C_02`` : (d, d) empirical cross-covariance of r_0 and r_2.
        ``frobenius_norm`` : ||C_02||_F.
        ``n_quadruplets`` : number of quadruplets used.
    """
    import jax.numpy as jnp

    from SFI.integrate.rk4 import ode_flow
    from SFI.trajectory import TrajectoryCollection, TrajectoryDataset

    if isinstance(data, TrajectoryDataset):
        datasets = [data]
    elif isinstance(data, TrajectoryCollection):
        datasets = data.datasets
    else:
        raise TypeError(type(data))

    def disp_fn(z):
        return ode_flow(drift_fn, z, dt, n_substeps) - z

    C02_sum = None
    n_total = 0

    for ds in datasets:
        X = np.asarray(ds.X)
        if X.ndim == 2:
            X = X[:, None, :]
        T, N_part, d = X.shape

        for p in range(N_part):
            Y = X[:, p, :]  # (T, d)
            n_quads = T - 3
            if n_quads < 1:
                continue

            # Midpoint residuals for all steps
            residuals = []
            for t in range(T - 1):
                y_mid = 0.5 * (Y[t] + Y[t + 1])
                dy = Y[t + 1] - Y[t]
                phi = np.asarray(disp_fn(jnp.asarray(y_mid)))
                residuals.append(dy - phi)
            residuals = np.array(residuals)  # (T-1, d)

            # Cross-covariance of r_t and r_{t+2}
            r0 = residuals[:-2]  # (T-3, d)
            r2 = residuals[2:]  # (T-3, d)
            C02 = np.einsum("ni,nj->ij", r0, r2) / r0.shape[0]

            if C02_sum is None:
                C02_sum = C02
            else:
                C02_sum = C02_sum + C02
            n_total += 1

    if n_total == 0:
        raise ValueError("Not enough data for 4-point diagnostic.")

    assert C02_sum is not None
    C02_avg = C02_sum / n_total
    frob = float(np.linalg.norm(C02_avg))

    return {
        "C_02": C02_avg,
        "frobenius_norm": frob,
        "n_quadruplets": n_total,
    }
