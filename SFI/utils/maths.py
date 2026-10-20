import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_la
from jax import dtypes as _jdtypes
from jax import jit

# ---------------------------------------------------------------------
#  Mathematical utilities
# ---------------------------------------------------------------------


def default_float_dtype():
    """Return JAX's currently-active default float dtype.

    Returns ``float64`` when ``JAX_ENABLE_X64`` is set, ``float32`` otherwise.
    Use this everywhere user-supplied arrays enter the simulation pipeline
    so that all internal arithmetic shares a single dtype and ``lax.scan``
    carry-in / carry-out types always agree.
    """
    return _jdtypes.canonicalize_dtype(jnp.float64)


def as_default_float(x):
    """Return ``x`` cast to the JAX default float dtype.

    Accepts any array-like; passes through ``None`` unchanged so callers
    can use it on optional inputs (e.g. ``v0``).
    """
    if x is None:
        return None
    return jnp.asarray(x, dtype=default_float_dtype())


def fd_velocity(x, dt, *, scheme: str = "central"):
    """Finite-difference velocity along the leading (time) axis.

    Reconstructs ``v(t) ≈ dx/dt`` from a position array using a
    finite-difference stencil — the same secant-velocity convention the
    underdamped inference engine uses internally.  The output keeps the
    same leading time dimension as ``x`` (boundary frames fall back to a
    one-sided stencil).

    Parameters
    ----------
    x : array_like, shape ``(T, ...)``
        Positions sampled in time along axis 0 (e.g. ``(T, d)`` or
        ``(T, N, d)``).
    dt : float or array_like, shape ``(T - 1,)``
        Time step.  Scalar for uniform sampling, or per-interval spacings.
    scheme : {"central", "forward", "backward"}
        Interior stencil.  ``"central"`` is second-order accurate and the
        default; ``"forward"`` / ``"backward"`` are first-order.

    Returns
    -------
    v : jax.Array, shape ``(T, ...)``
        Velocity estimate, same shape as ``x``.
    """
    x = jnp.asarray(x)
    if x.shape[0] < 2:
        raise ValueError("fd_velocity needs at least 2 time points.")
    dt = jnp.asarray(dt)
    n_extra = x.ndim - 1

    def _bcast(a):
        # reshape a per-step (T-1,) array to broadcast over the trailing axes
        return a.reshape(a.shape + (1,) * n_extra)

    # forward differences between consecutive frames, shape (T-1, ...)
    fwd = (x[1:] - x[:-1]) / (dt if dt.ndim == 0 else _bcast(dt))

    if scheme == "forward":
        return jnp.concatenate([fwd, fwd[-1:]], axis=0)
    if scheme == "backward":
        return jnp.concatenate([fwd[:1], fwd], axis=0)
    if scheme == "central":
        denom = (2.0 * dt) if dt.ndim == 0 else _bcast(dt[:-1] + dt[1:])
        central = (x[2:] - x[:-2]) / denom
        return jnp.concatenate([fwd[:1], central, fwd[-1:]], axis=0)
    raise ValueError(f"Unknown scheme {scheme!r}; expected central/forward/backward.")


def stable_pinv(G):
    """Numerically-stable pseudo-inverse of a Gram matrix.

    Normalizes G by its diagonal before inversion so that all diagonal
    entries of the rescaled matrix are 1.  This avoids ill-conditioning
    when basis functions have very different scales.

    Algorithm: let d_i = sqrt(G_{ii}) (clamped to 1 when zero).  Compute
    pinv(G / outer(d, d)) and rescale by outer(1/d, 1/d) to recover the
    pseudo-inverse of the original G.
    """
    # Normalize the matrix before computing the pseudo-inverse, for numerical stability
    G_norm = jnp.sqrt(jnp.diag(G))  # Extract diagonal elements as normalization factors
    # Avoid division by zero: if G_norm[i] is zero, keep it zero in scaling
    safe_G_norm = jnp.where(G_norm > 0, G_norm, 1.0)  # Replace 0 with 1 to avoid NaN in division
    return jnp.linalg.pinv(G / jnp.outer(safe_G_norm, safe_G_norm)) * jnp.outer(1 / safe_G_norm, 1 / safe_G_norm)


@jit
def sqrtm_psd(A: jax.Array, eps: float = 0.0) -> jax.Array:
    """
    Symmetric PSD matrix square root, applied to the last two (matrix) axes.
    - Symmetrizes input for stability.
    - Clips negative eigenvalues to 0 (PSD) then takes sqrt.
    - Re-symmetrizes the result to kill small asymmetries from numerics.
    """
    A = 0.5 * (A + jnp.swapaxes(A, -1, -2))
    w, V = jnp.linalg.eigh(A)
    w = jnp.clip(w, min=0.0)  # PSD safeguard
    if eps:
        w = w + eps  # optional jitter if you like
    S = (V * jnp.sqrt(w)[..., None, :]) @ jnp.swapaxes(V, -1, -2)
    return 0.5 * (S + jnp.swapaxes(S, -1, -2))


def solve_or_pinv(A: jax.Array, b: jax.Array, tol: float = 1e-15) -> jax.Array:
    """
    Solve A ⋅ x = b for x, with a fallback to the Moore–Penrose pseudo-inverse
    if A is singular or not square.  To improve numerical stability, we first
    normalize A by its diagonal: A_norm = D^{-1} A D^{-1}, b_norm = D^{-1} b,
    solve A_norm ⋅ x_norm = b_norm, and then recover x = D^{-1} x_norm.

    This ensures that the diagonal entries of A_norm are 1 (assuming A has
    positive diagonal), which often makes the linear solve or pseudo-inverse
    more robust when A has widely varying scales on its diagonal.

    Parameters
    ----------
    A : jax.Array, shape (k, k)
        The matrix to solve against.  We assume that A has nonnegative diagonal
        entries; if any diagonal entry is zero, we clip it to a small floor
        to avoid division by zero.
    b : jax.Array, shape (k,)
        The right-hand side vector.
    tol : float, default=1e-15
        The tolerance for the pseudo-inverse.  If A_norm is effectively singular,
        we compute x_norm = pinv(A_norm, rcond=tol) @ b_norm.

    Returns
    -------
    x : jax.Array, shape (k,)
        The solution vector to A ⋅ x = b, computed as follows:
          1) d_i = sqrt(max(A_{ii}, tol))
             (we floor each diagonal entry to tol > 0 to avoid zero divides)
          2) A_norm = D_inv @ A @ D_inv  where D_inv = diag(1 / d_i)
             b_norm = b / d
          3) Solve A_norm ⋅ x_norm = b_norm:
             - if A_norm is non-singular, use a direct solver
             - otherwise, fall back to x_norm = pinv(A_norm) @ b_norm
          4) Recover x = x_norm / d
    """
    # 1) Extract and “floor” the diagonal of A to avoid zeros or negatives.
    #    If A_{ii} is very small or negative (due to numerical noise), we floor it.
    diag_A = jnp.diag(A)  # shape (k,)
    # Clip to tol to avoid sqrt of zero or negative
    diag_clipped = jnp.clip(diag_A, min=tol)  # shape (k,)
    d = jnp.sqrt(diag_clipped)  # shape (k,)

    # 2) Build the inverse scaling matrix D_inv = diag(1 / d_i)
    #    We do this by dividing each row of A by d_i and each column by d_j.
    #    More precisely: A_norm[i,j] = A[i,j] / (d[i] * d[j]).
    D_inv = 1.0 / d  # shape (k,)
    # Use broadcasting to normalize A: first divide each row by d,
    # then each column by d.
    A_norm = (A * D_inv[:, None]) * D_inv[None, :]

    # 3) Normalize the RHS vector b as well: b_norm = b / d.
    b_norm = b / d  # shape (k,)

    # 4) Attempt a direct solve of A_norm ⋅ x_norm = b_norm.
    #    If A_norm is non-singular, this will succeed.
    try:
        # We do not assume positive-definite here, so use 'gen' unless we detect
        # symmetry and positive definiteness.  For simplicity, assume 'gen'.
        x_norm = jsp_la.solve(A_norm, b_norm, assume_a="gen")
    except Exception:  # broad catch: JAX/XLA doesn't expose a stable LinAlgError type
        # 4a) Fallback: if A_norm is singular or not square, use pseudo-inverse.
        x_norm = jnp.linalg.pinv(A_norm, rtol=tol) @ b_norm

    # 5) Scale back to obtain the final solution: x_i = x_norm[i] / d[i].
    x = x_norm / d  # shape (k,)

    return x
