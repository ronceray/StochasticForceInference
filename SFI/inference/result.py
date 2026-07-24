# SFI/inference/result.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import norm as _norm

from SFI.statefunc.psf import PSF

# Import the bound state-function + PSF (for vectorization helpers)
from SFI.statefunc.sf import SF

# =====================================================================
# Module-level utility: CI for 1D kernel profiles
# =====================================================================


def kernel_predict_ci(
    r_eval: Union[np.ndarray, jnp.ndarray],
    kernels: Sequence[Tuple[callable, str]],
    coeffs: Union[np.ndarray, jnp.ndarray],
    cov_block: Union[np.ndarray, jnp.ndarray],
    *,
    alpha: float = 0.95,
) -> Dict[str, np.ndarray]:
    r"""Confidence interval for a 1D kernel profile.

    For a reconstructed kernel profile

    .. math::

       k(r) = \sum_{\alpha} c_\alpha \, \phi_\alpha(r),

    the variance at each :math:`r` is

    .. math::

       \operatorname{Var}[k(r)]
           = \boldsymbol{\phi}(r)^\top \, \Sigma_c \, \boldsymbol{\phi}(r),

    where :math:`\Sigma_c` is the covariance sub-block for the
    coefficients of this basis group.

    Parameters
    ----------
    r_eval : array_like, shape ``(R,)``
        Radial grid on which to evaluate the kernel.
    kernels : list of ``(callable, label)``
        Kernel basis functions, e.g. from
        :func:`~SFI.bases.pairs.exp_poly_kernels`.
        Each callable maps ``r -> phi(r)``.
    coeffs : array_like, shape ``(K,)``
        Inferred coefficients for this basis block.
    cov_block : array_like, shape ``(K, K)``
        Covariance sub-block for these coefficients
        (from ``inf.force_coefficients_covariance[i0:i1, i0:i1]``
        after calling ``inf.compute_force_error()``).
    alpha : float
        Confidence level (default 0.95 for 95 % CI).

    Returns
    -------
    dict with keys:

    - **r** — the input radial grid (as numpy array).
    - **mean** — kernel profile ``coeffs @ phi(r)``.
    - **std** — pointwise standard deviation.
    - **lower**, **upper** — symmetric CI bounds.
    - **phi** — basis matrix ``(K, R)`` (useful for further analysis).
    """
    r_arr = np.asarray(r_eval)
    r_jax = jnp.asarray(r_arr)
    coeffs = np.asarray(coeffs)
    cov_block = np.asarray(cov_block)

    # Build basis matrix: phi[k, r_idx] = phi_k(r_idx)
    phi = np.array([np.asarray(fn(r_jax)) for fn, _ in kernels])  # (K, R)

    mean = coeffs @ phi  # (R,)
    # Var[k(r)] = phi(r)^T Sigma phi(r) — vectorised over r
    # phi_cov = Sigma @ phi  →  (K, R)
    phi_cov = cov_block @ phi  # (K, R)
    var = np.einsum("kr,kr->r", phi, phi_cov)  # (R,)
    std = np.sqrt(np.maximum(var, 0.0))

    z = float(_norm.ppf((1.0 + alpha) / 2.0))
    return dict(
        r=r_arr,
        mean=mean,
        std=std,
        lower=mean - z * std,
        upper=mean + z * std,
        phi=phi,
    )


class InferenceResultSF(SF):
    """
    A fitted, callable state function that *is* an SF and carries
    parameter covariance + metadata for downstream uncertainty handling.

    Notes
    -----
    - `param_cov` is the covariance of the *flattened* parameter vector
      defined by the underlying PSF template order (see PSF.flatten_params).
    - Covariance estimation is handled upstream (in the inferer).
    - Call :meth:`predict_var` / :meth:`predict_ci` for pointwise uncertainty.
    """

    # Extra fields beyond SF:
    _psf_ref: PSF  # reference PSF (from the parent SF)
    param_cov: Optional[jnp.ndarray]  # Σ_θ in the PSF template's vector order
    meta: Dict[str, Any]  # free-form: dt, A_hat, G/M, modes, sizes…

    def __init__(
        self,
        sf: SF,
        *,
        param_cov: Optional[jnp.ndarray] = None,
        meta: Optional[Dict[str, Any]] = None,
    ):
        # Initialize SF with the original psf + params; keep drop_features consistent
        super().__init__(sf._psf, sf.params, drop_features=sf.drop_features)
        object.__setattr__(self, "_psf_ref", sf._psf)
        object.__setattr__(self, "param_cov", param_cov)
        object.__setattr__(self, "meta", {} if meta is None else dict(meta))

    # ---------------------------------------------------------------------
    # Convenience: parameter vectorization consistent with the PSF template
    # ---------------------------------------------------------------------
    def flatten_params(self) -> jnp.ndarray:
        """Return θ̂ as a 1D vector using the PSF template order."""
        return self._psf_ref.flatten_params(self.params)

    def materialize_params(self, vec: jnp.ndarray) -> dict[str, jax.Array]:
        """Inverse of `flatten_params`: make a param dict from a vector."""
        return self._psf_ref.unflatten_params(vec)

    # ------------------------------------------------------------------
    #  Internal: Jacobian evaluation
    # ------------------------------------------------------------------
    def _jacobian(
        self,
        x: jnp.ndarray,
        *,
        extras: Optional[Dict[str, Any]] = None,
        mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        r"""Evaluate ∂F/∂θ at ``x`` using the underlying PSF's ``d_theta()``.

        Returns
        -------
        J : jnp.ndarray, shape ``(…, *rank_shape, n_params)``
            Jacobian of the model output w.r.t. the *full* flattened
            parameter vector.  For a rank-1 force ``F : (N, d)`` this is
            ``(N, d, p)``; for a rank-2 diffusion ``D : (N, d, d)`` it is
            ``(N, d, d, p)``.
        """
        J_psf = self._psf_ref.d_theta()  # a new PSF object
        J_raw = J_psf(x, params=self.params, extras=extras, mask=mask)
        # J_raw shape: (..., *rank_shape, n_features_fused)
        # n_features_fused = n_features_child × n_params
        # For CoeffNode n_features_child is 1, so fused == n_params.
        # In general, we know the total n_params from the template.
        n_params = int(self._psf_ref.template.size)
        # Unfuse last axis: (..., *rank_shape, n_features_child, n_params)
        # then sum over the feature axis (contraction already done for CoeffNode,
        # but for safety reshape → n_feat_child × n_params → sum over n_feat_child).
        n_fused = J_raw.shape[-1]
        n_feat_child = n_fused // n_params if n_params > 0 else 1
        if n_feat_child == 1:
            # Common case: CoeffNode — fused axis is already (n_params,)
            return J_raw
        # Rare case: multi-feature PSFs — reshape and sum
        batch_rank_shape = J_raw.shape[:-1]
        J_raw = J_raw.reshape(*batch_rank_shape, n_feat_child, n_params)
        return J_raw.sum(axis=-2)

    # ------------------------------------------------------------------
    #  Uncertainty interface
    # ------------------------------------------------------------------
    def _check_param_cov(self):
        """Raise if param_cov is not available."""
        if self.param_cov is None:
            raise RuntimeError(
                "Parameter covariance is not available on this result. "
                "Call compute_force_error() (or compute_diffusion_error()) "
                "on the inferer first, or use an optimizer that provides "
                "the Hessian (e.g. L-BFGS-B instead of Adam)."
            )

    def predict_var(
        self,
        x: jnp.ndarray,
        *,
        extras: Optional[Dict[str, Any]] = None,
        mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        r"""Pointwise predictive variance via the delta method.

        .. math::

           \operatorname{Var}\!\bigl[F_i(x)\bigr]
               \approx \bigl(J_\theta(x)\,\Sigma_\theta\,J_\theta(x)^\top\bigr)_{ii}

        For **linear** models (basis expansion) this is **exact**,
        not an approximation.

        Parameters
        ----------
        x : array, shape ``(N, dim)``
            Query points.
        extras : dict, optional
            Extra arguments forwarded to the underlying state function
            (e.g. ``{"box": box}`` for periodic boundary conditions).
        mask : array, optional
            Boolean mask forwarded to evaluation.

        Returns
        -------
        var : jnp.ndarray
            Per-component variance.  Shape matches the model output rank:
            ``(N, d)`` for a rank-1 (force) model, ``(N, d, d)`` for rank-2
            (diffusion tensor).
        """
        self._check_param_cov()
        J = self._jacobian(x, extras=extras, mask=mask)  # (..., *rank, p)
        S = jnp.asarray(self.param_cov)  # (p, p)
        # Var_i = J_i @ S @ J_i^T  (diagonal only)
        # Efficient: (J @ S) elementwise-* J, sum over last axis
        JS = jnp.einsum("...p,pq->...q", J, S)
        return jnp.einsum("...p,...p->...", JS, J)

    def predict_cov(
        self,
        x: jnp.ndarray,
        *,
        extras: Optional[Dict[str, Any]] = None,
        mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        r"""Full pointwise covariance matrix via the delta method.

        .. math::

           \Sigma_F(x) = J_\theta(x)\,\Sigma_\theta\,J_\theta(x)^\top

        Parameters
        ----------
        x : array, shape ``(N, dim)``
        extras, mask : forwarded to the underlying state function.

        Returns
        -------
        cov : jnp.ndarray
            For rank-1 models: shape ``(N, d, d)``.
            For rank-2 models: shape ``(N, d, d, d, d)`` (rarely needed).
        """
        self._check_param_cov()
        J = self._jacobian(x, extras=extras, mask=mask)  # (..., r, p)
        S = jnp.asarray(self.param_cov)  # (p, p)
        return jnp.einsum("...ip,pq,...jq->...ij", J, S, J)

    def predict_ci(
        self,
        x: jnp.ndarray,
        *,
        alpha: float = 0.95,
        extras: Optional[Dict[str, Any]] = None,
        mask: Optional[jnp.ndarray] = None,
    ) -> Dict[str, jnp.ndarray]:
        r"""Pointwise confidence intervals via the delta method.

        Parameters
        ----------
        x : array, shape ``(N, dim)``
        alpha : float
            Confidence level (default 0.95 for 95 % CI).
        extras, mask : forwarded to the underlying state function.

        Returns
        -------
        dict with keys:

        - **mean** — model prediction ``F̂(x)``.
        - **std** — pointwise standard deviation ``√Var[F(x)]``.
        - **lower**, **upper** — symmetric CI bounds
          ``F̂ ± z_{α/2} · std``.
        """
        mean = self(x, extras=extras, mask=mask)
        var = self.predict_var(x, extras=extras, mask=mask)
        std = jnp.sqrt(jnp.maximum(var, 0.0))
        z = float(_norm.ppf((1.0 + alpha) / 2.0))
        return dict(mean=mean, std=std, lower=mean - z * std, upper=mean + z * std)

    # ---------------------------------------------------------------------
    # Pretty representation
    # ---------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover
        kind = self.meta.get("kind", "unknown")
        n_params = int(self.flatten_params().size)
        has_cov = self.param_cov is not None
        labels = self.meta.get("basis_labels", None)
        n_basis = len(labels) if labels else self.meta.get("basis_features", "?")
        return f"InferenceResultSF({kind}, basis={n_basis}, params={n_params}, cov={'yes' if has_cov else 'no'})"

    def summary(self) -> str:
        """Formatted coefficient table (if labels and coefficients are available)."""
        import numpy as np

        from SFI.utils.formatting import model_summary

        labels = self.meta.get("basis_labels", None)
        theta = np.asarray(self.flatten_params())
        kind = self.meta.get("kind", "model")

        if labels is None:
            labels = [f"b{j}" for j in range(theta.size)]

        return model_summary(
            labels,
            theta,
            title=f"{kind.capitalize()} Coefficient Table",
        )

    # ---------------------------------------------------------------------
    # Persistence (equinox-based model serialization)
    # ---------------------------------------------------------------------
    def save(self, path) -> "Path":
        """Save this fitted model to ``<path>.eqx`` + ``<path>.meta.json``.

        The saved files can be reloaded with :meth:`load`, provided
        the user supplies a *template* built from the same PSF/Basis.

        See :func:`SFI.inference.serialization.save_model`.
        """
        from SFI.inference.serialization import save_model

        return save_model(self, path)

    @classmethod
    def load(cls, path, template: "InferenceResultSF") -> "InferenceResultSF":
        """Reload a model saved by :meth:`save`.

        Parameters
        ----------
        path : str or Path
            Base path (without extension).
        template : InferenceResultSF
            Skeleton with the same tree structure (same PSF + dummy params).

        See Also
        --------
        SFI.inference.serialization.load_model
        """
        from SFI.inference.serialization import load_model

        return load_model(path, template)
