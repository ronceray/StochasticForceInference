"""
Overdamped Langevin simulator (Euler–Maruyama / Heun) with post-run observables.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import jax.numpy as jnp
from jax import jit, vmap

from SFI.statefunc import PSF, SF
from SFI.utils.maths import as_default_float

from .base import Array, DiffusionKind, LangevinBase


@dataclass
class OverdampedProcess(LangevinBase):
    """Overdamped Langevin simulator (Euler–Maruyama or stochastic Heun).

    Parameters
    ----------
    F : PSF | SF
        Force model with `rank=vector`, `needs_v=False`, and `pdepth∈{0,1}`.
        If a PSF is provided, bind parameters via :meth:`set_params` prior to
        simulation.
    D : float | Array | PSF | SF
        Diffusion model: scalar σ (interpreted as σ·I), constant (d×d) matrix,
        or a PSF/SF with `rank=matrix`, `pdepth∈{0,1}` compatible with `F`.
    theta_F, theta_D : Optional[Array], optional
        Parameter vectors for binding PSF → SF.
    extras_global, extras_local : Optional[dict], optional
        Frozen, time-independent extras passed to both ``F`` and ``D`` at
        every call. Users should classify extras explicitly as:

        - extras_global: system-wide objects (geometry, external field ...)
        - extras_local:  per-particle objects (species labels, radii, ...)

        At runtime these are merged into a single ``extras`` mapping, with
        local keys overriding global ones, and passed identically to both
        models.

    Notes
    -----
    This class does **not** insert particle axes. The shapes must match the
    model contract:

    - If ``F.pdepth == 0``, ``x0.shape == (d,)``.
    - If ``F.pdepth == 1``, ``x0.shape == (P, d)``.

    Observables
    -----------
    After the run (on recorded steps only), we compute:

    - information ``I`` approx ``0.25 * sum_t <dx_t, D_inv(x_t) . F(x_t)>``
    - entropy ``S`` approx ``sum_t <dx_t, D_inv(x_mid) . (F(x_t)+F(x_{t+dt}))/2>``

    where ``dx_t = x_{t+dt} - x_t`` and ``x_mid = (x_{t+dt}+x_t)/2``.
    We evaluate F(x) exactly once per recorded step and reuse it for both terms.
    """

    # cached diffusion inverse for constant-D case
    _Dinv_const: Optional[Array] = None
    # bound diffusion callable for state-dependent case (so we can evaluate D at x_t / x_mid)
    _D_sf: Optional[SF] = None

    # ------------------------------- API --------------------------------
    def initialize(self, x0: Array) -> None:
        """Initialize the process state.

        Parameters
        ----------
        x0 : Array
            Initial position. Must satisfy:

            - If ``F.pdepth == 0``: shape ``(d,)``
            - If ``F.pdepth == 1``: shape ``(P, d)``

        Side effects
        ------------
        Binds PSF parameters (if any), validates model contracts, and prepares
        diffusion shortcuts (constant vs state-dependent).
        """
        # Basic contract inspection from the statefunc objects
        self._normalize_basis_to_psf()
        self._assert_force_contract(self.F)

        # Canonicalize dtype: any user-provided array (float32, float64,
        # numpy int, plain list, …) is cast to JAX's currently-active
        # default float dtype.  This is the single normalization point
        # that makes lax.scan carry-in / carry-out dtypes always agree.
        x0 = as_default_float(x0)

        # Deduce (P, d) from x0
        if x0.ndim == 1:
            d = int(x0.shape[0])
            P = None
        elif x0.ndim == 2:
            P = int(x0.shape[0])
            d = int(x0.shape[1])
        else:
            raise ValueError("x0 must have shape (d,) or (P, d).")

        # Check against F.pdepth
        f_pdepth = getattr(self.F, "pdepth", None)
        if f_pdepth not in (0, 1):
            raise ValueError("F.pdepth must be 0 or 1 for overdamped simulations.")
        if f_pdepth == 0 and x0.ndim != 1:
            if x0.shape[0] == 1 and x0.ndim == 2:
                # Silently drop the first axis
                x0 = x0[0]
            else:
                raise ValueError("F expects no particle axis (pdepth=0): x0 must be (d,).")
        if f_pdepth == 1 and x0.ndim != 2:
            raise ValueError("F expects a particle axis (pdepth=1): x0 must be (P, d).")

        # Bind F now, keeping the unbound object intact
        self._F_sf = self._bind_force()

        # Bind D early *if* it's a PSF/SF, so structural extras can be prepared once for both.
        D_sf_for_extras: Optional[SF] = None
        if isinstance(self.D, SF):
            D_sf_for_extras = self.D
        elif isinstance(self.D, PSF):
            if self.theta_D is None:
                raise ValueError("Diffusion PSF not bound: call set_params(theta_D=...).")
            D_sf_for_extras = self.D.bind(self.theta_D)  # type: ignore[attr-defined]

        # Eagerly materialize any structural extras needed by bound expressions.
        # This MUST happen before the first JIT-triggering evaluation.
        self._invalidate_prepared_extras()
        exprs = [self._F_sf] + ([D_sf_for_extras] if D_sf_for_extras is not None else [])
        extras = self._prepare_model_extras(x_probe=x0, exprs=exprs)

        # Validate output dimension of F on a cheap probe (shape-only).
        # Use the merged process-level extras seen by both F and D.
        f_out = self._F_sf(x0, extras=extras)
        if f_out.shape != x0.shape:
            raise ValueError(f"Force output shape {f_out.shape} does not match input shape {x0.shape}.")

        # Prepare diffusion shortcuts (constant vs state-dependent) for the integrator
        self._check_diffusion_contract(self.D, d=d, f_pdepth=f_pdepth)
        self._setup_diffusion(d=d, with_v=False)

        # Also prepare D or D^{-1} for post-run observables
        from SFI.langevin.noise import NoiseModel

        if isinstance(self.D, NoiseModel):
            # Use the effective per-site D for observables approximation
            extras = self._model_extras()
            D_eff = self.D.effective_D_per_site(extras)
            self._Dinv_const = jnp.linalg.pinv(D_eff)
            self._D_sf = None
        elif isinstance(self.D, (int, float)):
            sigma = float(self.D)
            self._Dinv_const = (1.0 / sigma) * jnp.eye(d)  # pinv(σ I) = (1/σ) I
            self._D_sf = None
        elif isinstance(self.D, jnp.ndarray) and self.D.ndim == 2:
            self._Dinv_const = jnp.linalg.pinv(self.D)
            self._D_sf = None
        elif isinstance(self.D, (PSF, SF)):
            # Reuse the already-bound callable (if any)
            if D_sf_for_extras is None:
                # Defensive fallback; should not happen if the above binding logic ran
                if isinstance(self.D, SF):
                    D_sf_for_extras = self.D
                else:
                    if self.theta_D is None:
                        raise ValueError("Diffusion PSF not bound: call set_params(theta_D=...).")
                    D_sf_for_extras = self.D.bind(self.theta_D)  # type: ignore[attr-defined]

            self._D_sf = D_sf_for_extras
            self._Dinv_const = None
        else:
            raise TypeError("D must be a float, (d×d) array, or a PSF/SF.")

        # Persist runtime state/metadata basics
        self._x = x0
        num_particles = int(P) if P is not None else 1
        self.metadata.clear()
        self.metadata.update(
            dict(
                kind="overdamped",
                dimension=d,
                pdepth=int(f_pdepth),
                num_particles=num_particles,
                x0=x0,
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
        compute_observables: bool = True,
        method: str = "heun",
    ):
        r"""
        Integrate overdamped Langevin dynamics and return a
        :class:`TrajectoryCollection` of positions.

        Parameters
        ----------
        dt
            Time step between recorded frames.
        Nsteps
            Number of recorded time steps.
        key
            PRNG key for the simulation.
        oversampling
            Number of integration substeps between recorded frames.
            The effective substep size is ``dt / oversampling``.
            Although all integrators have a consistent continuous limit, they
            introduce short-range, algorithm-specific temporal correlations at
            the scale of a single step.  Downsampling by recording only every
            ``oversampling``-th substep ensures these artefacts never reach
            the inference layer.  The default of 4 is a safe minimum for
            typical use; increase it when ``dt`` is large or the process
            varies rapidly.
        prerun
            Number of recorded steps to discard as burn-in, using the same
            ``dt`` and ``oversampling``.
        jit_compile
            If True, JIT-compile the single-step integrator before scanning.
        method
            Integration scheme.  ``"heun"`` (default) selects the stochastic
            Heun predictor-corrector scheme, which achieves **weak order 2**
            for constant (additive) diffusion — the dominant use case — at
            the cost of two force evaluations per substep.  For
            state-dependent diffusion the Heun scheme still uses the
            Itô-correct left-point noise evaluation, giving weak order 1 but
            with better error constants than Euler–Maruyama.  ``"euler"``
            selects the classical Euler–Maruyama integrator (weak order 1).
        compute_observables
            If True, compute post-run information and entropy production
            estimates on the recorded trajectory and store them in the
            dataset metadata under the ``"observables"`` key.

        .. physics:: Information functional & entropy production (overdamped)
           :label: info-entropy-overdamped
           :category: Observable

           .. math::

              I \approx \tfrac{1}{4}\sum_t
                \mathrm{d}X_t^\top\, D^{-1}(x_t)\, F(x_t)

           .. math::

              S \approx \sum_t
                \mathrm{d}X_t^\top\, D^{-1}(x_{\text{mid}})\,
                \tfrac{1}{2}\bigl[F(x_t)+F(x_{t+1})\bigr]

           :math:`I` estimates the information content; :math:`S` the
           entropy production (time-reversal asymmetry).

        Returns
        -------
        TrajectoryCollection
            A collection with a single dataset containing the positions.
            The underlying dataset has:

            - ``X`` of shape ``(Nsteps, d)`` or ``(Nsteps, P, d)``,
            - metadata combining model info (kind, dimension, pdepth, etc.),
              run info (dt, Nsteps, oversampling, prerun), and optional
              observables.
        """
        if self._F_sf is None:
            raise RuntimeError("Call initialize(x0) before simulate().")

        _valid_methods = ("euler", "heun")
        if method not in _valid_methods:
            raise ValueError(f"Unknown method {method!r}; expected one of {_valid_methods}.")
        self._method = method

        # Split extras into static values and per-frame schedules (the
        # time-dependent ones: TimeSeriesExtra of length Nsteps, or f(t)
        # callables materialized at the frame times).
        static_extras, schedules, eg_out, el_out = self._materialize_step_extras(dt=dt, Nsteps=Nsteps)

        step = self._make_step(static_extras)
        if jit_compile:
            step = jit(step)

        traj, info = self._scan(
            self._x,
            step_fn=step,
            dt=dt,
            Nsteps=Nsteps,
            oversampling=oversampling,
            prerun=prerun,
            key=key,
            schedules=schedules or None,
        )

        # Update final state for continuation
        self._x = traj[-1]

        # Run-level metadata snapshot
        run_meta: Dict[str, Any] = dict(self.metadata)
        run_meta.update(dict(dt=float(dt), integrator=method, **info))
        if schedules:
            run_meta["time_dependent_extras"] = sorted(schedules)

        # Optional post-run observables (information/entropy)
        if compute_observables:
            X = traj
            dX = X[1:] - X[:-1]
            X_mid = 0.5 * (X[1:] + X[:-1])

            extras = static_extras

            # Frame-aligned extras: a[k] governs [X[k] -> X[k+1]] (zeroth-
            # order hold), so the evaluation at X[k] pairs with the frame-k
            # schedule slice.
            if schedules:

                def _ex_at(s):
                    return {**(extras or {}), **s}

                F_all = vmap(
                    lambda x, s: self._F_sf(x, extras=_ex_at(s))  # type: ignore[misc]
                )(X, schedules)
            else:
                F_all = vmap(
                    lambda x, _extras=extras: self._F_sf(x, extras=_extras)  # type: ignore[misc]
                )(X)
            F_t, F_tp = F_all[:-1], F_all[1:]
            F_avg = 0.5 * (F_t + F_tp)

            if self._Dinv_const is not None:
                I_terms = jnp.einsum("...m,...n,mn->", dX, F_t, self._Dinv_const)
                S_terms = jnp.einsum("...m,...n,mn->", dX, F_avg, self._Dinv_const)
            else:
                if self._D_sf is None:
                    raise RuntimeError("State-dependent diffusion SF not initialized.")

                if schedules:
                    sched_t = {k: v[:-1] for k, v in schedules.items()}
                    Dinv_t = vmap(
                        lambda x, s: jnp.linalg.pinv(self._D_sf(x, extras=_ex_at(s)))  # type: ignore[misc]
                    )(X[:-1], sched_t)
                    Dinv_mid = vmap(
                        lambda x, s: jnp.linalg.pinv(self._D_sf(x, extras=_ex_at(s)))  # type: ignore[misc]
                    )(X_mid, sched_t)
                else:
                    Dinv_t = vmap(
                        lambda x, _extras=extras: jnp.linalg.pinv(self._D_sf(x, extras=_extras))  # type: ignore[misc]
                    )(X[:-1])
                    Dinv_mid = vmap(
                        lambda x, _extras=extras: jnp.linalg.pinv(self._D_sf(x, extras=_extras))  # type: ignore[misc]
                    )(X_mid)

                I_terms = jnp.einsum("...m,...n,...mn->", dX, F_t, Dinv_t)
                S_terms = jnp.einsum("...m,...n,...mn->", dX, F_avg, Dinv_mid)

            observables = {
                "information": float(0.25 * I_terms),
                "entropy": float(S_terms),
            }
            run_meta["observables"] = observables

        # Hand off to the base helper: positions → TrajectoryCollection
        coll = self._traj_to_collection(
            traj, dt=dt, meta=run_meta, extras_global_out=eg_out, extras_local_out=el_out
        )
        return coll

    # ----------------------------- Internals -----------------------------
    def _make_step(self, extras=None):
        """Dispatch to the selected integration scheme.

        ``extras`` is the merged *static* extras dict; per-frame scheduled
        values arrive through the step's optional ``sched`` argument and
        override it.
        """
        method = getattr(self, "_method", "euler")
        if extras is None:
            extras = self._model_extras()
        if method == "heun":
            return self._make_heun_step(extras)
        return self._make_euler_step(extras)

    def _make_euler_step(self, extras=None):
        r"""Create the Euler–Maruyama substep function (no observables inside).

        .. physics:: Euler–Maruyama integrator (overdamped)
           :label: euler-maruyama-overdamped
           :category: Simulation

           .. math::

              x_{t+\mathrm{d}t}
              = x_t + \mathrm{d}t\, F(x_t)
                + \sqrt{\mathrm{d}t}\, B(x_t)\,\xi_t

           where :math:`B = \sqrt{2D}` and
           :math:`\xi_t \sim \mathcal{N}(0, I)`.
        """
        F = self._F_sf
        kind = self._D_kind
        Bc = self._B_const
        Bf = self._B_fn
        noise_model = self._noise_model
        static_extras = extras

        if F is None:
            raise RuntimeError("Force not bound. Did you call initialize() after set_params()?")

        def step(x: Array, ddt: float, key: Array, sched=None) -> Array:
            """Single Euler–Maruyama substep."""
            ex = static_extras if sched is None else {**(static_extras or {}), **sched}

            # Drift term
            drift = F(x, extras=ex)

            # Noise increment
            if kind is DiffusionKind.NOISE_MODEL:
                assert noise_model is not None
                inc = noise_model.sample(key, x, ex)
            elif kind is DiffusionKind.STATE_FUNC:
                assert Bf is not None
                xi = self._noise(key, x)
                Bx = Bf(x, extras=ex)
                inc = self._apply_B(Bx, xi, state_dependent=True)
            else:
                xi = self._noise(key, x)
                inc = self._apply_B(Bc, xi, state_dependent=False)

            return x + ddt * drift + jnp.sqrt(ddt) * inc

        return step

    def _make_heun_step(self, extras=None):
        r"""Create the stochastic Heun (predictor-corrector) substep function.

        .. physics:: Stochastic Heun integrator (overdamped)
           :label: heun-overdamped
           :category: Simulation

           Predictor (Euler):

           .. math::

              \hat x = x_t + \mathrm{d}t\, F(x_t)
                       + \sqrt{\mathrm{d}t}\, B(x_t)\,\xi_t

           Corrector (trapezoidal drift):

           .. math::

              x_{t+\mathrm{d}t}
              = x_t + \tfrac{1}{2}\mathrm{d}t\,[F(x_t) + F(\hat x)]
                + \sqrt{\mathrm{d}t}\, B(x_t)\,\xi_t

           For constant (additive) diffusion this achieves
           **weak order 2**.  For state-dependent diffusion it uses the
           left-point noise evaluation to preserve the Itô convention,
           giving weak order 1 but with better error constants than
           Euler–Maruyama.  Costs two force evaluations per substep.
        """
        F = self._F_sf
        kind = self._D_kind
        Bc = self._B_const
        Bf = self._B_fn
        noise_model = self._noise_model
        static_extras = extras

        if F is None:
            raise RuntimeError("Force not bound. Did you call initialize() after set_params()?")

        def step(x: Array, ddt: float, key: Array, sched=None) -> Array:
            """Single stochastic Heun substep."""
            ex = static_extras if sched is None else {**(static_extras or {}), **sched}

            drift = F(x, extras=ex)

            # Noise increment (evaluated at left point for Itô correctness)
            if kind is DiffusionKind.NOISE_MODEL:
                assert noise_model is not None
                noise_inc = jnp.sqrt(ddt) * noise_model.sample(key, x, ex)
            elif kind is DiffusionKind.STATE_FUNC:
                assert Bf is not None
                xi = self._noise(key, x)
                Bx = Bf(x, extras=ex)
                noise_inc = jnp.sqrt(ddt) * self._apply_B(Bx, xi, state_dependent=True)
            else:
                xi = self._noise(key, x)
                noise_inc = jnp.sqrt(ddt) * self._apply_B(Bc, xi, state_dependent=False)

            # Euler predictor
            x_pred = x + ddt * drift + noise_inc

            # Corrector: trapezoidal average of drift, same noise realization
            # (predictor and corrector share the frame's scheduled extras —
            # the protocol is piecewise constant by definition).
            drift_pred = F(x_pred, extras=ex)
            return x + 0.5 * ddt * (drift + drift_pred) + noise_inc

        return step

    # --------------------------- Validations -----------------------------
    @staticmethod
    def _assert_force_contract(F: PSF | SF) -> None:
        """Validate force has the right rank/dim flags for overdamped use."""
        needs_v = getattr(F, "needs_v", False)
        if needs_v:
            raise ValueError("Overdamped force must not require velocity (needs_v=False).")
        rank = getattr(F, "rank", None)
        if rank != 1:
            raise ValueError("Force must have rank=vector.")
        dim = getattr(F, "dim", None)
        if dim is None or not isinstance(dim, int):
            raise ValueError("Force must declare an integer `dim` attribute.")
        pdepth = getattr(F, "pdepth", None)
        if pdepth not in (0, 1):
            raise ValueError("Force `pdepth` must be 0 or 1 for overdamped simulations.")

    @staticmethod
    def _check_diffusion_contract(D, *, d: int, f_pdepth: int) -> None:
        """Validate diffusion against the force contract, allowing benign broadcast."""
        # NoiseModel instances
        from SFI.langevin.noise import NoiseModel

        if isinstance(D, NoiseModel):
            if D.dim != d:
                raise ValueError(f"NoiseModel n_fields={D.dim} must match force dim={d}.")
            return

        # Scalars and constant matrices are fine
        if isinstance(D, (int, float)):
            return
        if isinstance(D, jnp.ndarray) and D.ndim == 2:
            if D.shape != (d, d):
                raise ValueError(f"Constant diffusion must be shape (d,d)={(d, d)}, got {D.shape}.")
            return

        # PSF/SF path
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
