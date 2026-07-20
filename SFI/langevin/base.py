"""
Langevin base utilities (shape-agnostic, PSF/SF-centric).

This module provides:

- A light base class with common plumbing shared by overdamped/underdamped
  simulators (parameter binding, diffusion handling, scan loop, metadata).
- A small integrator protocol type used by concrete simulators.

Design
------
We *do not* insert or manage particle axes here. Particle axes are owned by the
PSF/SF contract via ``pdepth`` and ``particles_input``. Hence this module uses
ellipsis-friendly einsums and lets models decide whether an input has a particle
axis (pdepth=1) or not (pdepth=0). In particular:

- ``F(x)`` must return an array with the same leading shape as x.
- ``D(x)`` must return either a constant (d x d) matrix or an array with shape
  ``(..., d, d)``, where the leading axes ``...`` match those of x.

Einsum index convention
-----------------------
- Spatial (coordinate) indices: m, n (and following letters if needed).
- Particle indices: i, j.

We therefore use einsums of the form::

    Constant diffusion:   'mn,...n->...m'
    State-dependent diff: '...mn,...n->...m'

which transparently covers both pdepth=0 (``...`` empty) and pdepth=1 (``...`` = i).

Extras
------
PSF/SF may accept an ``extras`` keyword (e.g., neighbor lists for SPDE grids).
The base class lets you *freeze* time-independent extras via
``extras_global`` / ``extras_local`` and update them later via
:meth:`set_extras`. At runtime we merge these into a single dictionary
that is passed as ``extras=...`` to both F and D.

.. note::

   Extras may be **time-dependent**: pass a
   :class:`~SFI.trajectory.TimeSeriesExtra` whose leading axis matches the
   ``Nsteps`` of the upcoming ``simulate`` call (one value per recorded
   frame, held constant across oversampling substeps), or a plain callable
   ``f(t)`` of physical time, materialized host-side at the frame times
   ``t = k·dt`` before the scan.  Schedules enter the scan as consumed
   inputs and are attached to the output collection so that inference can
   consume them per frame.  Static extras behave exactly as before.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax import jit, lax, random

from SFI.bases.constants import constant_array
from SFI.trajectory.reserved_extras import RESERVED_NAMES, ExtrasContext, resolve_reserved

# Public API types
from SFI.statefunc import PSF, SF, Basis  # noqa: F401
from SFI.statefunc.nodes.interactions.prepare import (
    prepare_structural_extras_for_expr,
    purge_cache_extras,
)
from SFI.utils.maths import sqrtm_psd  # PSD matrix square-root

# Defer import to avoid circular dependency; the isinstance check in
# _setup_diffusion uses a lazy import guard.
_NoiseModel = None  # populated on first use

Array = jnp.ndarray


def _get_noise_model_class():
    """Lazy import of NoiseModel to avoid circular dependency."""
    global _NoiseModel
    if _NoiseModel is None:
        from SFI.langevin.noise import NoiseModel as _NM

        _NoiseModel = _NM
    return _NoiseModel


class DiffusionKind(Enum):
    """Kinds of diffusion representations accepted by the simulators."""

    CONST_SCALAR = auto()  # σ: float, interpreted as σ·I
    CONST_MATRIX = auto()  # D ∈ ℝ^{d×d}
    STATE_FUNC = auto()  # PSF/SF mapping x (or x,v) → (..., d, d)
    NOISE_MODEL = auto()  # NoiseModel instance (conserved noise, etc.)


@dataclass
class LangevinBase:
    """Common plumbing for Langevin simulators.

    This class is *not* a simulator itself. Overdamped/underdamped simulators
    should subclass it and (i) bind models during `initialize()` by calling
    :meth:`_bind_force` and :meth:`_setup_diffusion`, (ii) prepare a suitable
    integrator, and (iii) call :meth:`_scan`.

    Parameters
    ----------
    F : Basis | PSF | SF
        Force model. A :class:`~SFI.statefunc.Basis` is auto-promoted to a
        linear PSF via :meth:`Basis.to_psf`; pass ``theta_F`` as the
        coefficient vector (or as ``{"coeff": array}``).
    D : float | Array | Basis | PSF | SF
        Diffusion model: scalar, constant (d×d) matrix, or a Basis/PSF/SF.
    theta_F, theta_D : Optional[Array], optional
        Parameters to bind PSF → SF. If `F`/`D` are already `SF` or constants, ignored.

    Attributes
    ----------
    metadata : dict
        Filled by simulators (algorithm name, dt, number of steps, etc.).
    """

    # unbound/original objects as provided by the user
    F: Union[Basis, PSF, SF]
    D: Union[float, Array, Basis, PSF, SF]

    # parameter vectors to bind PSF → SF (kept separate; we don't overwrite F/D)
    theta_F: Optional[Array] = None
    theta_D: Optional[Array] = None

    # extras dictionaries (frozen at init unless changed via set_extras)
    # Users classify extras explicitly as global vs local. At runtime we
    # merge them into a single dict passed as `extras=...` to both F and D.
    extras_global: Optional[Dict[str, Any]] = None
    extras_local: Optional[Dict[str, Any]] = None

    # runtime fields (derived, not part of the public state)
    _F_sf: Optional[SF] = field(init=False, default=None, repr=False)  # specialized force
    _D_kind: DiffusionKind = field(init=False, default=DiffusionKind.CONST_SCALAR, repr=False)
    _B_const: Optional[Array] = field(init=False, default=None, repr=False)  # sqrt(2D) if constant
    _B_fn: Optional[Callable[..., Array]] = field(init=False, default=None, repr=False)  # sqrt(2D(x[,v]))
    _noise_model: Optional[Any] = field(
        init=False, default=None, repr=False
    )  # NoiseModel instance (conserved noise, etc.)

    metadata: dict = field(default_factory=dict, init=False)

    # Track whether we've already materialized structural extras for the current model+extras.
    _structural_extras_prepared: bool = False

    # Build-once store for dispatcher-owned structural arrays (CSR / stencil tables).
    # Kept off the user-facing ``extras_global`` so it never pollutes it or leaks
    # into output datasets; merged into the model-facing extras at evaluation time.
    _prepared_structural: Optional[Dict[str, Any]] = None

    # ----------------------- Parameter / extras API -----------------------
    def set_params(self, *, theta_F: Optional[Array] = None, theta_D: Optional[Array] = None) -> None:
        """Bind PSF parameters to specialize models (PSF → SF).

        If `F` or `D` are `PSF`, these will be consumed during `initialize()`
        when the subclass calls :meth:`_bind_force` and :meth:`_setup_diffusion`.

        Notes
        -----
        We **do not** overwrite the user-provided `F` / `D` objects. Instead,
        we keep them unmodified and store specialized callables separately
        (e.g., `_F_sf`), derived from the pair (object, theta, extras).
        """
        if theta_F is not None:
            self.theta_F = theta_F
        if theta_D is not None:
            self.theta_D = theta_D

    def set_extras(
        self,
        *,
        extras_global: Optional[Dict[str, Any]] = None,
        extras_local: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Freeze or update extras dictionaries used when calling F and D.

        Parameters
        ----------
        extras_global :
            System-wide extras (geometry, neighbor lists, drive protocols,
            ...).  Time-dependent entries are supported: a
            :class:`~SFI.trajectory.TimeSeriesExtra` with one value per
            recorded frame of the next ``simulate`` call, or a plain
            callable ``f(t)`` of physical time (materialized at the frame
            times before the scan).
        extras_local :
            Per-particle extras (species labels, radii, ...), with the same
            time-dependence options.

        Notes
        -----
        Both dictionaries are merged into a single model-facing extras mapping
        that is passed as `extras=...` to *both* F and D. Local keys override
        global keys on conflicts.  Time-dependent values are held constant
        across the oversampling substeps of each frame (zeroth-order hold);
        the prerun uses the frame-0 value.
        """
        if extras_global is not None:
            self.extras_global = extras_global
        if extras_local is not None:
            self.extras_local = extras_local
        # If extras change, subclasses should re-run `initialize()` so any
        # cached JIT-compiled callables see the new extras.

        # Structural extras depend on geometry -> must be recomputed.
        self._invalidate_prepared_extras()

    # ----------------------- Public accessors -----------------------------

    @property
    def force_sf(self) -> "SF":
        """Bound force state function (read-only).

        Available after :meth:`initialize` has been called.  This is the
        same callable stored internally as ``_F_sf``; exposing it publicly
        avoids callers reaching into private attributes.

        Returns
        -------
        SF
            ``force_sf(X)`` evaluates the (vector) force at positions *X*.
        """
        if self._F_sf is None:
            raise RuntimeError("force_sf is not available before initialize() has been called.")
        return self._F_sf

    @property
    def diffusion_sf(self) -> "Optional[SF]":
        """Bound diffusion state function (read-only), or ``None``.

        Returns the diffusion *matrix* as an ``SF`` when available.
        For constant-scalar or constant-matrix diffusion that was not
        built from a ``Basis``/``PSF``, this returns ``None`` (since
        there is no callable ``SF``).

        Available after :meth:`initialize` has been called.

        Returns
        -------
        SF or None
            ``diffusion_sf(X)`` evaluates the diffusion matrix at *X*,
            or ``None`` if diffusion is not representable as an SF.
        """
        return getattr(self, "_D_sf", None)

    # NOTE: all model-facing calls (F, D) should go through this helper.
    def _model_extras(self) -> Optional[Dict[str, Any]]:
        """Merged extras dict passed to F and D (global + local).

        Time-dependent entries (:class:`TimeSeriesExtra`, plain callables
        ``f(t)``) are materialized **at frame 0** here — the value used by
        initialization probes, structural-extras preparation, and any
        model evaluation outside :meth:`simulate` (which overrides them
        per frame via schedules).
        """
        eg = self.extras_global
        el = self.extras_local
        required_reserved = self._model_required_extras() & RESERVED_NAMES

        if eg is None and el is None and not required_reserved and not self._prepared_structural:
            return None

        merged: Dict[str, Any] = dict(eg or {})
        merged.update(el or {})  # local overrides global on conflicts
        if self._prepared_structural:
            merged.update(self._prepared_structural)  # dispatcher-owned structural arrays

        from SFI.trajectory.dataset import FunctionExtra, TimeSeriesExtra

        out: Dict[str, Any] = {}
        for key, val in merged.items():
            if isinstance(val, TimeSeriesExtra):
                out[key] = jnp.asarray(val.data)[0]
            elif isinstance(val, FunctionExtra):
                out[key] = val
            elif callable(val):
                out[key] = jnp.asarray(val(0.0))
            else:
                out[key] = val
        # Reserved extras at frame 0 — present for probes and setup; simulate()
        # supplies the per-frame ``time`` through schedules.
        if required_reserved:
            probe = resolve_reserved(
                self._reserved_context(frame_times=jnp.asarray(0.0), duration=jnp.asarray(1.0))
            )
            for key in required_reserved:
                out.setdefault(key, probe[key])
        return out

    def _materialize_step_extras(
        self, *, dt: float, Nsteps: int
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Array], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Split extras into static values and per-frame schedules.

        Classification (per value, for both ``extras_global`` and
        ``extras_local``; local overrides global on key conflicts):

        - :class:`~SFI.trajectory.TimeSeriesExtra` — its array is the
          schedule; the leading axis must equal ``Nsteps`` (one value per
          recorded frame).
        - plain callable — interpreted as ``f(t)`` of *physical time* and
          materialized host-side at the frame times ``t = k·dt``.
        - :class:`~SFI.trajectory.FunctionExtra` and anything else — static.

        Returns
        -------
        (static_extras, schedules, extras_global_out, extras_local_out)
            ``static_extras`` is the merged model-facing dict without the
            scheduled keys; ``schedules`` maps key → ``(Nsteps, ...)``
            array; the two ``*_out`` dicts mirror the inputs with
            schedules wrapped as :class:`TimeSeriesExtra` (what the output
            collection carries).
        """
        from SFI.trajectory.dataset import FunctionExtra, TimeSeriesExtra, time_series_extra

        schedules: Dict[str, Array] = {}

        def _classify(src: Optional[Dict[str, Any]]):
            if src is None:
                return None, None
            static: Dict[str, Any] = {}
            out: Dict[str, Any] = {}
            for key, val in src.items():
                if isinstance(val, TimeSeriesExtra):
                    arr = jnp.asarray(val.data)
                    if int(arr.shape[0]) != int(Nsteps):
                        raise ValueError(
                            f"Time-dependent extra {key!r} has leading axis "
                            f"{int(arr.shape[0])} but simulate() records Nsteps={Nsteps} "
                            "frames — provide one value per recorded frame "
                            "(held constant across oversampling substeps)."
                        )
                    schedules[key] = arr
                    out[key] = val
                elif isinstance(val, FunctionExtra):
                    static[key] = val
                    out[key] = val
                elif callable(val):
                    arr = jnp.stack([jnp.asarray(val(k * dt)) for k in range(int(Nsteps))])
                    schedules[key] = arr
                    out[key] = time_series_extra(arr)
                else:
                    static[key] = val
                    out[key] = val
            return static, out

        static_g, out_g = _classify(self.extras_global)
        static_l, out_l = _classify(self.extras_local)

        if static_g is None and static_l is None and not self._prepared_structural:
            static_extras: Optional[Dict[str, Any]] = None
        else:
            static_extras = dict(static_g or {})
            static_extras.update(static_l or {})  # local overrides global
            if self._prepared_structural:
                # Dispatcher-owned structural arrays are static across frames.
                static_extras.update(self._prepared_structural)

        # Reserved extras a model reads are resolved from the same registry the
        # inference runtime uses, so a force sees the identical mapping whether
        # simulated or inferred: the per-frame ``time`` becomes a schedule, the
        # rest (``duration``, and — when bootstrapping an inferred force —
        # ``dataset_index`` / ``particle_index``) are static.
        required_reserved = self._model_required_extras() & RESERVED_NAMES
        if required_reserved:
            reserved = resolve_reserved(
                self._reserved_context(
                    frame_times=jnp.arange(Nsteps, dtype=float) * dt,
                    duration=jnp.asarray((Nsteps - 1) * dt if Nsteps > 1 else 1.0, dtype=float),
                )
            )
            if "time" in required_reserved:
                schedules["time"] = reserved["time"]
            statics = {k: reserved[k] for k in required_reserved if k != "time"}
            if statics:
                static_extras = dict(static_extras or {})
                static_extras.update(statics)

        return static_extras, schedules, out_g, out_l

    def _model_required_extras(self) -> set:
        """Extras keys read by the bound force or diffusion expression."""
        required: set = set()
        for sf in (getattr(self, "_F_sf", None), getattr(self, "_D_sf", None)):
            req = getattr(sf, "required_extras", None) if sf is not None else None
            if req:
                required |= set(req)
        return required

    def _reserved_context(self, *, frame_times, duration) -> ExtrasContext:
        """Context for resolving reserved extras during a simulation.

        ``dataset_index`` and the particle count come from
        ``_reserved_overrides`` (set, e.g., when bootstrapping an inferred force
        that reads them); the ``time`` / ``duration`` clock is built from the
        simulation's own frame schedule.
        """
        overrides = getattr(self, "_reserved_overrides", None) or {}
        pidx = overrides.get("particle_index")
        return ExtrasContext(
            n_particles=int(jnp.asarray(pidx).shape[0]) if pidx is not None else 1,
            dataset_index=int(overrides.get("dataset_index", 0)),
            frame_times=frame_times,
            duration=duration,
        )

    @staticmethod
    def _shift_schedules(schedules: Dict[str, Array]) -> Dict[str, Array]:
        """Shift schedules by one recorded step for the scan.

        The scan records the state *after* each step, so the step that
        produces frame ``k`` from frame ``k-1`` must consume ``a[k-1]``:
        ``xs[i] = a[i-1]`` with ``xs[0] = a[0]`` (the step from the initial
        state).  This makes ``dX[k] = X[k+1] - X[k]`` generated under
        ``a[k]`` — the pairing the inference layer assumes.
        """
        return {k: jnp.concatenate([v[:1], v[:-1]], axis=0) for k, v in schedules.items()}

    def _invalidate_prepared_extras(self) -> None:
        """Mark structural extras as stale.

        Call this whenever extras or the model graph changes (e.g. in set_extras
        or on rebind). Binding PSF parameters via :meth:`set_params` does *not*
        invalidate: structural arrays derive from geometry, not from
        ``theta_F``/``theta_D``. We keep it tiny so you can sprinkle it safely.
        """
        self._structural_extras_prepared = False
        self._prepared_structural = None
        self.extras_global = purge_cache_extras(self.extras_global)

    def _prepare_model_extras(
        self,
        *,
        x_probe,
        v_probe=None,
        mask_probe=None,
        exprs,  # list/tuple only
    ) -> Optional[Dict[str, Any]]:
        """Materialize structural extras (CSR/hyper tables, stencil masks, …) eagerly.

        This runs on the **Python side**, before the first JIT evaluation of any
        provided expression. The intended pattern is:

            self._F_sf = self._bind_force()
            extras = self._prepare_model_extras(x_probe=x0, exprs=[self._F_sf, ...])
            y = self._F_sf(x0, extras=extras)   # safe under JIT

        Contract
        --------
        This function expects that *somewhere* in the expression graph (or inside
        nodes/specs it contains) there is an eager hook:

            obj.prepare_extras(x, v=v, mask=mask, extras=extras) -> Optional[Dict]

        which:
          - may mutate `extras` in-place, and/or
          - may return a dict of extra keys to merge into model extras.

        If no such hook exists, this is a no-op and simply returns `_model_extras()`.
        """
        # 1) Build the merged model-facing extras dict
        extras = self._model_extras()
        if extras is None:
            # Ensure we have a mutable dict to inject into.
            if self.extras_global is None:
                self.extras_global = {}
            extras = self._model_extras()
            assert extras is not None  # now must exist

        if self._structural_extras_prepared:
            return extras

        if not isinstance(exprs, (list, tuple)):
            raise TypeError("exprs must be a list or tuple of expressions, e.g. [self._F_sf, self._D_sf].")

        additions: Optional[Dict[str, Any]] = None
        for expr in exprs:
            add_i = prepare_structural_extras_for_expr(expr, extras)
            if add_i:
                if additions is None:
                    additions = dict(add_i)
                else:
                    additions.update(add_i)

        # 2) Stash any returned additions in the private build-once store so
        # `_model_extras()` (and the simulation scan) see the prepared structural
        # keys later — without polluting the user-facing `extras_global`.
        if additions:
            self._prepared_structural = dict(additions)

        self._structural_extras_prepared = True
        return self._model_extras()

    # ------------------- Utilities for subclasses to use ------------------

    def _normalize_basis_to_psf(self) -> None:
        """Promote any ``Basis`` objects to their linear ``PSF`` form.

        Call at the top of ``initialize()`` so that all downstream isinstance
        checks naturally see a ``PSF``.  Raw-array ``theta_F`` / ``theta_D``
        are wrapped into ``{"coeff": array}`` so ``.bind()`` works directly.
        """
        if isinstance(self.F, Basis):
            self.F = self.F.to_psf()
            if self.theta_F is not None and not isinstance(self.theta_F, dict):
                self.theta_F = {"coeff": self.theta_F}
        if isinstance(self.D, Basis):
            self.D = self.D.to_psf()
            if self.theta_D is not None and not isinstance(self.theta_D, dict):
                self.theta_D = {"coeff": self.theta_D}

    @staticmethod
    def _is_psf(obj: Any) -> bool:
        """Robust PSF check (supports subclassing)."""
        return isinstance(obj, PSF)

    @staticmethod
    def _is_sf(obj: Any) -> bool:
        """Robust SF check (supports subclassing)."""
        return isinstance(obj, SF)

    def _bind_force(self) -> SF:
        """Return an SF for the force, binding `F` with `theta_F` if it is a PSF.

        Raises
        ------
        ValueError
            If `F` is PSF and `theta_F` is None (uninitialized).
        """
        if self._is_sf(self.F):
            return self.F  # type: ignore[return-value]
        if self._is_psf(self.F):
            if self.theta_F is None:
                # Fall back to any defaults carried on the PSF template.
                defaults = self.F.template.defaults()  # type: ignore[attr-defined]
                if defaults is None:
                    raise ValueError("Force PSF not bound: call set_params(theta_F=...).")
                return self.F.bind(defaults)  # type: ignore[attr-defined]
            # Keep F unmodified; store the specialized SF separately.
            return self.F.bind(self.theta_F)  # type: ignore[attr-defined]
        if isinstance(self.F, Basis):
            psf = self.F.to_psf()
            if self.theta_F is None:
                # Default behaviour: use defaults baked into to_psf() (coeff=1).
                return psf.bind()
            theta = self.theta_F if isinstance(self.theta_F, dict) else {"coeff": self.theta_F}
            return psf.bind(theta)
        raise TypeError("F must be a Basis, PSF, or SF.")

    def _setup_diffusion(self, *, d: int, with_v: bool = False) -> None:
        r"""Prepare diffusion representation and its sqrt(2D) mapping.

        .. physics:: Noise amplitude from diffusion tensor
           :label: noise-amplitude
           :category: Simulation

           .. math::

              B = \\sqrt{2\\,D}

           For scalar :math:`\\sigma`: :math:`B = \\sqrt{2\\sigma}\\,I_d`.
           For constant matrix :math:`D`: PSD matrix square root.
           For state-dependent :math:`D(x)`: evaluated at each step.

        Parameters
        ----------
        d : int
            Spatial dimension.
        with_v : bool, default=False
            Whether the simulator will pass velocity alongside x to D (underdamped).

        Side effects
        ------------
        Sets ``_D_kind``, ``_B_const``, ``_B_fn`` accordingly. ``_B_const``
        stores ``sqrt(2D)`` if diffusion is constant. ``_B_fn`` is a callable
        returning ``sqrt(2D(x))`` (or ``sqrt(2D(x, v))``) when diffusion is
        state-dependent.
        """
        Dobj = self.D

        # NoiseModel instance (conserved noise, composite, etc.)
        NM = _get_noise_model_class()
        if isinstance(Dobj, NM):
            self._D_kind = DiffusionKind.NOISE_MODEL
            self._noise_model = Dobj
            self._B_const = None
            self._B_fn = None
            # Store effective D for observables / inference approximation
            extras = self._model_extras()
            D_eff = Dobj.effective_D_per_site(extras)
            self._D_sf = constant_array(D_eff)
            return

        # Scalar constant: σ·I
        if isinstance(Dobj, (int, float)):
            self._D_kind = DiffusionKind.CONST_SCALAR
            sigma = float(Dobj)
            from SFI.utils.maths import as_default_float

            self._B_const = as_default_float(jnp.sqrt(2.0 * sigma) * jnp.eye(d))
            self._B_fn = None
            self._D_sf = constant_array(as_default_float(float(Dobj) * jnp.eye(d)))
            return

        # Constant matrix: D (d×d)
        if isinstance(Dobj, jnp.ndarray) and Dobj.ndim == 2:
            from SFI.utils.maths import as_default_float

            Dobj_c = as_default_float(Dobj)
            self._D_kind = DiffusionKind.CONST_MATRIX
            self._B_const = jnp.real(sqrtm_psd(2.0 * Dobj_c))
            self._B_fn = None
            self._D_sf = constant_array(Dobj_c)
            return

        # Callable: PSF or SF mapping state → (..., d, d)
        if self._is_sf(Dobj):
            D_sf = Dobj  # type: ignore[assignment]
        elif self._is_psf(Dobj):
            if self.theta_D is None:
                defaults = Dobj.template.defaults()  # type: ignore[attr-defined]
                if defaults is None:
                    raise ValueError("Diffusion PSF not bound: call set_params(theta_D=...).")
                D_sf = Dobj.bind(defaults)  # type: ignore[assignment, attr-defined]
            else:
                D_sf = Dobj.bind(self.theta_D)  # type: ignore[assignment, attr-defined]
        elif isinstance(Dobj, Basis):
            psf = Dobj.to_psf()
            if self.theta_D is None:
                D_sf = psf.bind()
            else:
                theta = self.theta_D if isinstance(self.theta_D, dict) else {"coeff": self.theta_D}
                D_sf = psf.bind(theta)
        else:
            raise TypeError("D must be a float, (d×d) array, or a Basis/PSF/SF.")

        self._D_kind = DiffusionKind.STATE_FUNC

        # Merge process-level extras once and close over them in the JITted
        # callable. For constant extras this is cheap and keeps them out of
        # the traced region.
        extras = self._model_extras()

        # The default extras are the merged static dict; per-step schedules
        # override them at call time via the explicit ``extras`` argument.
        if with_v:
            self._B_fn = jit(
                lambda x, v, extras=extras: jnp.real(sqrtm_psd(2.0 * D_sf(x, v=v, extras=extras)))
            )
        else:
            self._B_fn = jit(lambda x, extras=extras: jnp.real(sqrtm_psd(2.0 * D_sf(x, extras=extras))))
        self._B_const = None

    @staticmethod
    def _noise(key: Array, like: Array) -> Array:
        """Draw standard normal noise with the same leading shape as `like`.

        The trailing dimension of `like` is interpreted as the spatial axis (size d).
        The leading axes (possibly empty) can host particle indices i or any batch
        axes the PSF/SF expects.
        """
        return random.normal(key, shape=like.shape)

    @staticmethod
    def _apply_B(B: Array, xi: Array, *, state_dependent: bool) -> Array:
        """Compute (sqrt(2D) · ξ) with correct index conventions.

        Parameters
        ----------
        B : Array
            If constant: shape (m,n) representing the matrix with spatial indices.
            If state-dependent: shape (..., m, n) with leading axes matching `xi`.
        xi : Array
            Standard normal noise of shape (..., n), where "..." are particle/batch axes.
        state_dependent : bool
            Whether `B` depends on state (has leading axes) or is constant.

        Returns
        -------
        Array
            The product with shape (..., m).
        """
        if state_dependent:
            return jnp.einsum("...mn,...n->...m", B, xi)
        return jnp.einsum("mn,...n->...m", B, xi)

    # --------------------------- Scan machinery ---------------------------
    @staticmethod
    def _scan(
        initial_state: Any,
        *,
        step_fn: Callable[..., Any],
        dt: float,
        Nsteps: int,
        oversampling: int,
        prerun: int,
        key: Array,
        schedules: Optional[Dict[str, Array]] = None,
    ) -> Tuple[Any, dict]:
        """
        Generic simulate loop with oversampling + prerun using :func:`jax.lax.scan`.

        Parameters
        ----------
        initial_state
            State at t = 0 (shape determined by the concrete simulator).
        step_fn
            Function implementing one Euler–Maruyama substep:
            ``(state, ddt, key) -> state`` — or, when ``schedules`` is
            given, ``(state, ddt, key, sched) -> state`` with ``sched`` the
            per-frame slice of the scheduled extras.
        dt
            Output sampling time step for recorded frames.
        Nsteps
            Number of *recorded* steps (after oversampling/decimation).
        oversampling
            Number of substeps between recorded frames (must be ``>= 1``).
        prerun
            Number of recorded steps to discard before recording (burn-in).
        key
            PRNG key.
        schedules
            Optional per-frame extras schedules, key → ``(Nsteps, ...)``
            array.  Each recorded frame consumes one slice (zeroth-order
            hold across its substeps; the prerun and the first recorded
            step use the frame-0 values).  Entered as scan inputs, never
            carried.

        Returns
        -------
        traj, info
            ``traj`` is a pytree of recorded states with a leading time axis.
            ``info`` is a small dictionary with counters
            ``{"Nsteps", "oversampling", "prerun"}``.
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
                return (st, k), st

            if prerun > 0:
                (state, key), _ = lax.scan(
                    lambda c, _: one_recorded_step(c, sched0), (initial_state, key), None, length=prerun
                )
            else:
                state = initial_state

            if Nsteps > 0:
                xs = LangevinBase._shift_schedules(schedules)
                (state, key), traj = lax.scan(one_recorded_step, (state, key), xs, length=Nsteps)
            else:
                example = initial_state
                traj = jax.tree_util.tree_map(lambda a: a[None, ...][:0], example)

        else:

            def one_substep(carry, _):
                st, k = carry
                k, sub = random.split(k)
                st = step_fn(st, ddt, sub)
                return (st, k), None

            def one_recorded_step_plain(carry, _):
                st, k = carry
                (st, k), _ = lax.scan(one_substep, (st, k), None, length=oversampling)
                return (st, k), st

            # ------------- Burn-in (avoid zero-length scan under disable_jit) -----
            if prerun > 0:
                (state, key), _ = lax.scan(one_recorded_step_plain, (initial_state, key), None, length=prerun)
            else:
                state = initial_state

            # ------------- Recording (also guard Nsteps==0) -----------------------
            if Nsteps > 0:
                (state, key), traj = lax.scan(one_recorded_step_plain, (state, key), None, length=Nsteps)
            else:
                # Build an empty time axis (0, ...) matching the state pytree
                example = initial_state
                traj = jax.tree_util.tree_map(lambda a: a[None, ...][:0], example)

        info = {
            "Nsteps": int(Nsteps),
            "oversampling": int(oversampling),
            "prerun": int(prerun),
        }
        return traj, info

    # ----------------------------- Data export -----------------------------
    def _traj_to_collection(
        self,
        traj: Array,
        *,
        dt: float,
        meta: Optional[dict] = None,
        extras_global_out: Optional[Dict[str, Any]] = None,
        extras_local_out: Optional[Dict[str, Any]] = None,
    ):
        """
        Convert a trajectory of positions into a :class:`TrajectoryCollection`.

        Parameters
        ----------
        traj
            Recorded states with a leading time axis. Expected shapes are

            - ``(T, d)``   for single-trajectory simulations (no particle axis),
            - ``(T, P, d)`` for particle-based simulations (particle axis = 1).

            The shape must be consistent with the force model's ``pdepth``
            convention.
        dt
            Time step between successive recorded frames.
        meta
            Optional metadata dictionary to attach to the underlying dataset.
            A shallow copy is taken before storage.

        Returns
        -------
        TrajectoryCollection
            A collection with a single dataset, containing the positions only.
        """
        # Local import to avoid a hard dependency at module import time.
        from SFI.trajectory.collection import TrajectoryCollection

        meta_copy = None if meta is None else dict(meta)
        # Extras are attached directly to the dataset so that analysis can
        # access them via `extras` in the integration runtime.  Time-dependent
        # extras arrive through the *_out overrides as frame-aligned
        # TimeSeriesExtra values.
        eg = self.extras_global if extras_global_out is None else extras_global_out
        el = self.extras_local if extras_local_out is None else extras_local_out
        return TrajectoryCollection.from_arrays(
            X=traj,
            dt=dt,
            extras_global=purge_cache_extras(eg),
            extras_local=purge_cache_extras(el),
            meta=meta_copy,
        )
