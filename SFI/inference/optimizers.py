# SFI/inference/optimizers.py
"""
Optimizer back-ends for parametric inference methods.

Provides L-BFGS-B (via SciPy) and Adam (via optax) wrappers with
logging, best-parameter tracking, and a unified result interface.
"""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)


def optimize_lbfgsb(loss, loss_grad, theta0_flat, *, maxiter, tol):
    """L-BFGS-B via SciPy (returns a SciPy OptimizeResult)."""
    import time

    from scipy.optimize import minimize

    def fun(theta_np):
        return float(loss(jnp.asarray(theta_np)))

    def jac(theta_np):
        return jnp.asarray(loss_grad(jnp.asarray(theta_np)), dtype=float)

    logger.info("[force_nonlinear] Starting L-BFGS-B optimization...")
    t0 = time.perf_counter()
    res = minimize(
        fun,
        jnp.asarray(theta0_flat),
        jac=jac,
        method="L-BFGS-B",
        tol=tol,
        options={"maxiter": maxiter},
    )
    t1 = time.perf_counter()
    logger.info(
        "[force_nonlinear] L-BFGS-B finished in %.2f s | nit=%d, nfev=%d, final f=%.6g, status=%d (%s)",
        t1 - t0,
        res.nit,
        res.nfev,
        res.fun,
        res.status,
        res.message,
    )
    return res


def optimize_adam(
    loss,
    loss_grad,
    theta0_flat,
    *,
    maxiter,
    learning_rate,
    lr_schedule,
    loss_grad_batch=None,
    batch_rng_seed=0,
    batch_schedule=None,
):
    """Adam via optax (returns a namespace mimicking SciPy OptimizeResult).

    When *loss_grad_batch* is provided, mini-batch stochastic gradients
    are used for parameter updates while the full-data *loss* is still
    used for tracking / best-parameter selection.

    Parameters
    ----------
    loss_grad_batch : callable, optional
        ``loss_grad_batch(theta, rng_key) -> grad``.  When given, each
        Adam step uses a stochastic gradient instead of the full-data
        gradient.  Ignored when *batch_schedule* is set.
    batch_rng_seed : int
        Seed for the mini-batch PRNG stream.
    batch_schedule : list of (float, callable), optional
        Batch-size annealing schedule.  Each entry is
        ``(step_fraction, grad_fn)`` where *grad_fn* has the signature
        ``grad_fn(theta, rng_key) -> grad``.  The list must be sorted
        by ascending *step_fraction*; the last entry should have
        fraction 1.0.  At each step the active gradient function is
        the first whose fraction exceeds ``step / maxiter``.
    """
    import time

    import optax

    # Build learning-rate schedule
    if lr_schedule == "cosine":
        schedule = optax.cosine_decay_schedule(
            init_value=float(learning_rate),
            decay_steps=maxiter,
        )
    elif lr_schedule == "constant" or lr_schedule is None:
        schedule = float(learning_rate)
    else:
        raise ValueError(f"Unknown lr_schedule {lr_schedule!r}; choose 'cosine' or 'constant'.")

    opt = optax.adam(schedule)
    theta = jnp.asarray(theta0_flat)
    opt_state = opt.init(theta)

    best_f = float("inf")
    best_theta = theta

    use_schedule = batch_schedule is not None
    use_minibatch = loss_grad_batch is not None or use_schedule

    if use_minibatch:
        rng_key = jax.random.PRNGKey(batch_rng_seed)
        if use_schedule:
            logger.info(
                "[force_nonlinear] Starting Adam with batch-size annealing (%d phases, maxiter=%d, seed=%d)...",
                len(batch_schedule),
                maxiter,
                batch_rng_seed,
            )
        else:
            logger.info(
                "[force_nonlinear] Starting Adam optimization with mini-batch (maxiter=%d, seed=%d)...",
                maxiter,
                batch_rng_seed,
            )
    else:
        logger.info("[force_nonlinear] Starting Adam optimization (maxiter=%d)...", maxiter)

    prev_phase = -1
    t0 = time.perf_counter()
    for step in range(maxiter):
        if use_schedule:
            frac = step / maxiter
            phase = 0
            active_grad = batch_schedule[-1][1]
            for i, (sf, gfn) in enumerate(batch_schedule):
                if frac < sf:
                    active_grad = gfn
                    phase = i
                    break
            if phase != prev_phase:
                prev_phase = phase
                logger.info(
                    "[force_nonlinear]   phase %d/%d starts at step %d",
                    phase,
                    len(batch_schedule),
                    step,
                )
            rng_key, subkey = jax.random.split(rng_key)
            g = active_grad(theta, subkey)
        elif use_minibatch:
            rng_key, subkey = jax.random.split(rng_key)
            assert loss_grad_batch is not None
            g = loss_grad_batch(theta, subkey)
        else:
            g = loss_grad(theta)
        updates, opt_state = opt.update(g, opt_state, theta)
        theta = optax.apply_updates(theta, updates)

        if step % max(1, maxiter // 20) == 0 or step == maxiter - 1:
            f_val = float(loss(theta))
            if f_val < best_f:
                best_f = f_val
                best_theta = theta
            logger.info(
                "[force_nonlinear]   step %5d / %d  loss=%.6g  |grad|=%.3e",
                step,
                maxiter,
                f_val,
                float(jnp.linalg.norm(g)),
            )

    t1 = time.perf_counter()
    # Final eval
    f_final = float(loss(theta))
    if f_final < best_f:
        best_f = f_final
        best_theta = theta
    logger.info(
        "[force_nonlinear] Adam finished in %.2f s | %d steps, final loss=%.6g, best loss=%.6g",
        t1 - t0,
        maxiter,
        f_final,
        best_f,
    )

    # Return an object with the same .x / .fun interface as SciPy
    from types import SimpleNamespace

    return SimpleNamespace(
        x=best_theta,
        fun=best_f,
        nit=maxiter,
        nfev=maxiter,
        success=True,
        message="Adam optimization completed.",
        status=0,
    )
