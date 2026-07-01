# underdamped.py
"""
Underdamped Langevin simulator (velocity-Verlet-like, generic F(x,v) and D(x[,v])).

This mirrors :mod:`overdamped` as closely as possible, but simulates the
phase-space SDE

    dx = v dt
    dv = F(x, v) dt + sqrt(2 D(x, v)) dW

where diffusion acts on *velocity* increments. The returned
:class:`~SFI.trajectory.collection.TrajectoryCollection` stores **positions
only** by design.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
from jax import jit, lax, random

from SFI.statefunc import PSF, SF
from SFI.utils.maths import as_default_float

from .base import Array, DiffusionKind, LangevinBase


@dataclass
class UnderdampedProcess(LangevinBase):
    """Underdamped Langevin simulator.

    Parameters
    ----------
    F : PSF | SF
        Force model with `rank=vector`, `needs_v=True`, and `pdepth∈{0,1}`.
        If a PSF is provided, bind parameters via :meth:`set_params` prior to
        simulation.
    D : float | Array | PSF | SF
        Diffusion model acting on velocities: scalar σ (interpreted as σ·I),
        constant (d×d) matrix, or a PSF/SF with `rank=matrix`.
        If provided as PSF/SF, it may depend on (x) or (x, v), controlled by
        its `needs_v` flag.

    Notes
    -----
    This class does **not** insert particle axes; it follows the `pdepth`
    convention of the statefunc objects, similarly to :class:`OverdampedProcess`.
    """

    # Whether the (state-dependent) diffusion SF requires v.
    _D_needs_v: bool = False
    # Bound diffusion callable (only used for eager structural-extras preparation).
    _D_sf: Optional[SF] = None

    # ------------------------------- API --------------------------------
    def initialize(self, x0: Array, v0: Optional[Array] = None) -> None:
        """Initialize the process state.

        Parameters
        ----------
        x0 : Array
            Initial position. Must satisfy:
              - If `F.pdepth == 0`: shape (d,)
              - If `F.pdepth == 1`: shape (P, d)
        v0 : Array, optional
            Initial velocity. Must have the same shape as `x0`. Defaults to 0.
        """
        self._normalize_basis_to_psf()
        self._assert_force_contract(self.F)

        # Canonicalize dtype (single normalization point; see
        # OverdampedProcess.initialize for rationale).
        x0 = as_default_float(x0)
        v0 = as_default_float(v0)

        # Deduce (P, d) from x0
        if x0.ndim == 1:
            d = int(x0.shape[0])
            P = None
        elif x0.ndim == 2:
            P = int(x0.shape[0])
            d = int(x0.shape[1])
        else:
            raise ValueError("x0 must have shape (d,) or (P, d).")

        f_pdepth = getattr(self.F, "pdepth", None)
        if f_pdepth not in (0, 1):
            raise ValueError("F.pdepth must be 0 or 1 for underdamped simulations.")
        if f_pdepth == 0 and x0.ndim != 1:
            if x0.shape[0] == 1 and x0.ndim == 2:
                x0 = x0[0]
                if v0 is not None and v0.ndim == 2 and v0.shape[0] == 1:
                    v0 = v0[0]
            else:
                raise ValueError("F expects no particle axis (pdepth=0): x0 must be (d,).")
        if f_pdepth == 1 and x0.ndim != 2:
            raise ValueError("F expects a particle axis (pdepth=1): x0 must be (P, d).")

        if v0 is None:
            v0 = jnp.zeros_like(x0)
        if v0.shape != x0.shape:
            raise ValueError(f"v0.shape must match x0.shape; got v0={v0.shape}, x0={x0.shape}.")

        # Bind models
        self._F_sf = self._bind_force()
        if isinstance(self.D, (PSF, SF)):
            if isinstance(self.D, SF):
                self._D_sf = self.D
            else:
                if self.theta_D is None:
                    raise ValueError("Diffusion PSF not bound: call set_params(theta_D=...).")
                self._D_sf = self.D.bind(self.theta_D)  # type: ignore[attr-defined]
            self._D_needs_v = bool(getattr(self.D, "needs_v", False))
        else:
            self._D_sf = None
            self._D_needs_v = False

        # Prepare structural extras once for all bound expressions.
        # (The base helper caches preparation globally; pass the full list here.)
        self._invalidate_prepared_extras()
        exprs = [self._F_sf] + ([self._D_sf] if self._D_sf is not None else [])
        extras = self._prepare_model_extras(x_probe=x0, v_probe=v0, exprs=exprs)

        # Validate force output shape
        f_out = self._F_sf(x0, v=v0, extras=extras)
        if f_out.shape != x0.shape:
            raise ValueError(f"Force output shape {f_out.shape} does not match input shape {x0.shape}.")

        # Diffusion (constant vs state-dependent)
        self._check_diffusion_contract(self.D, d=d, f_pdepth=int(f_pdepth))
        self._setup_diffusion(d=d, with_v=self._D_needs_v)

        # Persist runtime state/metadata basics
        self._x = x0
        self._v = v0
        num_particles = int(P) if P is not None else 1
        self.metadata.clear()
        self.metadata.update(
            dict(
                kind="underdamped",
                dimension=d,
                pdepth=int(f_pdepth),
                num_particles=num_particles,
                x0=x0,
                v0=v0,
            )
        )

    def simulate(
        self,
        dt: float,
        Nsteps: int,
        key: Array,
        *,
        oversampling: int = 4,
        prerun: int = 0,
        jit_compile: bool = True,
        compute_observables: bool = False,
    ):
        """Run the integrator and return a :class:`TrajectoryCollection` of positions.

        Parameters
        ----------
        dt
            Time step between recorded frames.
        Nsteps
            Number of recorded time steps.
        key
            PRNG key for the simulation.
        oversampling
            Number of velocity-Verlet substeps between recorded frames.
            The effective substep size is ``dt / oversampling``.
            Although all integrators have a consistent continuous limit, they
            introduce short-range, algorithm-specific temporal correlations at
            the scale of a single step.  Downsampling by recording only every
            ``oversampling``-th substep ensures these artefacts never reach
            the inference layer.  The default of 4 is a safe minimum for
            typical use; increase it when ``dt`` is large or the process
            varies rapidly.
        prerun
            Number of recorded steps to discard as burn-in.
        jit_compile
            If True, JIT-compile the single-step integrator before scanning.
        compute_observables
            Not yet implemented for the underdamped case.

        Returns
        -------
        TrajectoryCollection
            A collection with a single dataset containing the positions only
            (velocities are not stored by design).  The underlying dataset has:

            - ``X`` of shape ``(Nsteps, d)`` or ``(Nsteps, P, d)``,
            - metadata combining model info (kind, dimension, pdepth, etc.)
              and run info (dt, Nsteps, oversampling, prerun).
        """
        if self._F_sf is None:
            raise RuntimeError("Call initialize(x0, v0=...) before simulate().")
        if compute_observables:
            raise NotImplementedError(
                "Underdamped observables are not implemented in this variant, "
                "because velocities are intentionally not stored."
            )

        # Split extras into static values and per-frame schedules.
        static_extras, schedules, eg_out, el_out = self._materialize_step_extras(dt=dt, Nsteps=Nsteps)

        step = self._make_step(static_extras)
        if jit_compile:
            step = jit(step)

        # Carry is (x, v), but we record positions only.
        traj_x, final_state, info = self._scan_positions(
            (self._x, self._v),
            step_fn=step,
            dt=dt,
            Nsteps=Nsteps,
            oversampling=oversampling,
            prerun=prerun,
            key=key,
            schedules=schedules or None,
        )

        self._x, self._v = final_state

        run_meta: Dict[str, Any] = dict(self.metadata)
        run_meta.update(dict(dt=float(dt), **info))
        if schedules:
            run_meta["time_dependent_extras"] = sorted(schedules)

        return self._traj_to_collection(
            traj_x, dt=dt, meta=run_meta, extras_global_out=eg_out, extras_local_out=el_out
        )

    # ----------------------------- Internals -----------------------------
    def _make_step(self, extras=None):
        r"""Create the substep function (kick–drift–kick, velocity-Verlet-like).

        .. physics:: Velocity-Verlet integrator (underdamped)
           :label: velocity-verlet-underdamped
           :category: Simulation

           Stochastic splitting (kick–drift–kick):

           .. math::

              v_{1/2} &= v + \tfrac{1}{2}\mathrm{d}t\, F(x,v)
                + \sqrt{\mathrm{d}t/2}\; B(x,v)\,\xi_1 \\
              x' &= x + \mathrm{d}t\, v_{1/2} \\
              v' &= v_{1/2} + \tfrac{1}{2}\mathrm{d}t\, F(x', v_{1/2})
                + \sqrt{\mathrm{d}t/2}\; B(x', v_{1/2})\,\xi_2

           Preserves the symplectic structure of the deterministic part.
        """
        F = self._F_sf
        if extras is None:
            extras = self._model_extras()
        static_extras = extras
        kind = self._D_kind
        Bc = self._B_const
        Bf = self._B_fn
        D_needs_v = self._D_needs_v

        if F is None:
            raise RuntimeError("Force not bound. Did you call initialize() after set_params()?")

        def _eval_B(x: Array, v: Array, ex) -> Array:
            if kind is DiffusionKind.STATE_FUNC:
                if Bf is None:
                    raise RuntimeError("State-dependent diffusion not initialized.")
                return Bf(x, v, extras=ex) if D_needs_v else Bf(x, extras=ex)
            if Bc is None:
                raise RuntimeError("Constant diffusion not initialized.")
            return Bc

        def _apply(B: Array, xi: Array) -> Array:
            # Be permissive: if B happens to be (d,d), treat it as constant.
            return self._apply_B(B, xi, state_dependent=(B.ndim > 2))

        def step(state, ddt: float, key: Array, sched=None):
            x, v = state
            k1, k2 = random.split(key)
            # Both half-kicks use the frame's scheduled value (zeroth-order
            # hold: the protocol is piecewise constant by definition).
            ex = static_extras if sched is None else {**(static_extras or {}), **sched}

            # --- First half-kick
            a0 = F(x, v=v, extras=ex)
            B0 = _eval_B(x, v, ex)
            xi1 = self._noise(k1, v)
            dv1 = _apply(B0, xi1)
            v_half = v + 0.5 * ddt * a0 + jnp.sqrt(ddt / 2.0) * dv1

            # --- Drift
            x_new = x + ddt * v_half

            # --- Second half-kick
            a1 = F(x_new, v=v_half, extras=ex)
            B1 = _eval_B(x_new, v_half, ex)
            xi2 = self._noise(k2, v)
            dv2 = _apply(B1, xi2)
            v_new = v_half + 0.5 * ddt * a1 + jnp.sqrt(ddt / 2.0) * dv2

            return (x_new, v_new)

        return step

    @staticmethod
    def _scan_positions(
        initial_state,
        *,
        step_fn,
        dt: float,
        Nsteps: int,
        oversampling: int,
        prerun: int,
        key: Array,
        schedules=None,
    ):
        """Scan loop that records positions only (carry contains (x,v)).

        ``schedules`` follows the same per-frame contract as
        :meth:`LangevinBase._scan`.
        """
        if oversampling < 1:
            raise ValueError("oversampling must be >= 1")

        ddt = dt / float(oversampling)
        scheduled = bool(schedules)

        if scheduled:
            sched0 = {k: v[0] for k, v in schedules.items()}

            def one_substep_at(sched):
                def one_substep(carry, _):
                    st, k = carry
                    k, sub = random.split(k)
                    st = step_fn(st, ddt, sub, sched)
                    return (st, k), None

                return one_substep

            def one_recorded_step(carry, sched_t):
                st, k = carry
                (st, k), _ = lax.scan(one_substep_at(sched_t), (st, k), None, length=oversampling)
                return (st, k), st[0]

            if prerun > 0:
                (state, key), _ = lax.scan(
                    lambda c, _: one_recorded_step(c, sched0), (initial_state, key), None, length=prerun
                )
            else:
                state = initial_state

            if Nsteps > 0:
                from SFI.langevin.base import LangevinBase

                xs = LangevinBase._shift_schedules(schedules)
                (state, key), traj_x = lax.scan(one_recorded_step, (state, key), xs, length=Nsteps)
            else:
                x_example = initial_state[0]
                traj_x = jax.tree_util.tree_map(lambda a: a[None, ...][:0], x_example)

        else:

            def one_substep(carry, _):
                st, k = carry
                k, sub = random.split(k)
                st = step_fn(st, ddt, sub)
                return (st, k), None

            def one_recorded_step_plain(carry, _):
                st, k = carry
                (st, k), _ = lax.scan(one_substep, (st, k), None, length=oversampling)
                # record positions only
                return (st, k), st[0]

            if prerun > 0:
                (state, key), _ = lax.scan(one_recorded_step_plain, (initial_state, key), None, length=prerun)
            else:
                state = initial_state

            if Nsteps > 0:
                (state, key), traj_x = lax.scan(one_recorded_step_plain, (state, key), None, length=Nsteps)
            else:
                x_example = initial_state[0]
                traj_x = jax.tree_util.tree_map(lambda a: a[None, ...][:0], x_example)

        info = {
            "Nsteps": int(Nsteps),
            "oversampling": int(oversampling),
            "prerun": int(prerun),
        }
        return traj_x, state, info

    # --------------------------- Validations -----------------------------
    @staticmethod
    def _assert_force_contract(F: PSF | SF) -> None:
        """Validate force has the right rank/dim flags for underdamped use."""
        needs_v = getattr(F, "needs_v", False)
        if not needs_v:
            raise ValueError("Underdamped force must require velocity (needs_v=True).")
        rank = getattr(F, "rank", None)
        if rank != 1:
            raise ValueError("Force must have rank=vector.")
        dim = getattr(F, "dim", None)
        if dim is None or not isinstance(dim, int):
            raise ValueError("Force must declare an integer `dim` attribute.")
        pdepth = getattr(F, "pdepth", None)
        if pdepth not in (0, 1):
            raise ValueError("Force `pdepth` must be 0 or 1 for underdamped simulations.")

    @staticmethod
    def _check_diffusion_contract(D, *, d: int, f_pdepth: int) -> None:
        """Validate diffusion against the force contract, allowing benign broadcast."""
        from SFI.langevin.noise import NoiseModel

        if isinstance(D, NoiseModel):
            if D.dim != d:
                raise ValueError(f"NoiseModel n_fields={D.dim} must match force dim={d}.")
            return
        if isinstance(D, (int, float)):
            return
        if isinstance(D, jnp.ndarray) and D.ndim == 2:
            if D.shape != (d, d):
                raise ValueError(f"Constant diffusion must be shape (d,d)={(d, d)}, got {D.shape}.")
            return

        if not isinstance(D, (PSF, SF)):
            raise TypeError("Diffusion must be a float, (d×d) array, PSF/SF, or NoiseModel.")

        rank = getattr(D, "rank", None)
        if rank != 2:
            raise ValueError("Diffusion PSF/SF must have rank=matrix.")
        dim = getattr(D, "dim", None)
        if dim != d:
            raise ValueError(f"Diffusion dim={dim} must match force dim={d}.")
        d_pdepth = getattr(D, "pdepth", None)
        if d_pdepth not in (0, 1):
            raise ValueError("Diffusion `pdepth` must be 0 or 1.")
        if not (d_pdepth == f_pdepth or d_pdepth == 0):
            raise ValueError(
                f"Incompatible pdepth: force pdepth={f_pdepth}, diffusion pdepth={d_pdepth}. "
                "Only equal depths or diffusion depth=0 (broadcast) are allowed."
            )
