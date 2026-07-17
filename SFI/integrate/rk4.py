# RK4 ODE flow integrator for parametric inference.
"""
rk4.py
======
Classical fourth-order Runge-Kutta integrator in JAX, suitable for
differentiating through the ODE flow via ``jax.jacobian`` / ``jax.hessian``.
"""

import jax

__all__ = [
    "rk4_step",
    "euler_step",
    "euler_flow",
    "ode_flow",
    "select_flow",
]


def rk4_step(f, x, h):
    """One classical fourth-order Runge-Kutta step.

    Parameters
    ----------
    f : callable  (d,) → (d,)
        Autonomous vector field, JAX-traceable.
    x : array (d,)
        Current state.
    h : scalar
        Step size.

    Returns
    -------
    x_new : array (d,)
        State after one RK4 step.
    """
    k1 = f(x)
    k2 = f(x + 0.5 * h * k1)
    k3 = f(x + 0.5 * h * k2)
    k4 = f(x + h * k3)
    return x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def euler_step(f, x, h):
    """One forward-Euler step.

    Parameters
    ----------
    f : callable  (d,) → (d,)
        Autonomous vector field, JAX-traceable.
    x : array (d,)
        Current state.
    h : scalar
        Step size.

    Returns
    -------
    x_new : array (d,)
        State after one Euler step.
    """
    return x + h * f(x)


def _generic_flow(step_fn, f, x0, dt, n_substeps):
    """Integrate dx/dt = f(x) from *x0* over total time *dt* using *step_fn*.

    Uses ``jax.lax.scan`` so the full computation is JAX-traceable and
    differentiable.
    """
    if n_substeps < 1:
        raise ValueError(
            f"n_substeps must be >= 1, got {n_substeps}. "
            "Use integrator='euler' with n_substeps=1 for the "
            "single-step Euler mode."
        )
    h = dt / n_substeps

    def body(x, _):
        return step_fn(f, x, h), None

    x_final, _ = jax.lax.scan(body, x0, None, length=n_substeps)
    return x_final


def euler_flow(f, x0, dt, n_substeps):
    """Integrate dx/dt = f(x) from *x0* over total time *dt* using Euler.

    Uses *n_substeps* forward-Euler micro-steps each of size
    ``h = dt / n_substeps``.  The loop is implemented via
    ``jax.lax.scan`` so the full computation is JAX-traceable and
    differentiable.

    Parameters
    ----------
    f : callable (d,) → (d,)
        Drift vector field (parameters should already be closed over).
    x0 : array (d,)
        Initial state.
    dt : scalar
        Total integration interval.
    n_substeps : int  (**static** Python int, not a JAX tracer)
        Number of Euler micro-steps.  Must be a compile-time constant.

    Returns
    -------
    x_final : array (d,)
        State at time *dt*.
    """
    return _generic_flow(euler_step, f, x0, dt, n_substeps)


def ode_flow(f, x0, dt, n_substeps):
    """Integrate dx/dt = f(x) from *x0* over total time *dt*.

    Uses *n_substeps* RK4 micro-steps each of size ``h = dt / n_substeps``.
    The loop is implemented via ``jax.lax.scan`` so the full computation is
    JAX-traceable and differentiable.

    Parameters
    ----------
    f : callable (d,) → (d,)
        Drift vector field (parameters should already be closed over).
    x0 : array (d,)
        Initial state.
    dt : scalar
        Total integration interval.
    n_substeps : int  (**static** Python int, not a JAX tracer)
        Number of RK4 micro-steps.  Must be a compile-time constant.
        Must be >= 1; use ``integrator='euler'`` for the Euler path.

    Returns
    -------
    x_final : array (d,)
        State at time *dt*.
    """
    return _generic_flow(rk4_step, f, x0, dt, n_substeps)


def select_flow(integrator: str):
    """Return the ODE flow function for the given integrator name.

    Parameters
    ----------
    integrator : {'rk4', 'euler'}

    Returns
    -------
    flow : callable
        Either :func:`ode_flow` or :func:`euler_flow`.
    """
    if integrator == "rk4":
        return ode_flow
    elif integrator == "euler":
        return euler_flow
    else:
        raise ValueError(f"Unknown integrator {integrator!r}; expected 'rk4' or 'euler'")
