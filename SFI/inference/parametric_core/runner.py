# SFI/inference/parametric_core/runner.py
"""
Exact-core runners (Stage-2, exploratory): per-point tensors → banded
whitening, chunked with carry threading.

Replaces the overlapping-window programs when the internal switch
``solve._core == "exact"`` is on.  Design:

* **prepare(θ)** — one pass per dataset computing the per-point tensors
  (:mod:`precompute`) on contiguous position blocks, chunked to bound
  memory.  For fixed-θ optimizations (the (D, Λ) profile) the prepared
  tensors are *cached* and re-evaluated against new noise matrices at
  linear-algebra cost only — no flow or basis re-evaluation.
* **evaluate(tensors, D, Λ)** — covariance blocks (:mod:`covariance`)
  → reversed-time banded whitening (:mod:`banded`), vmapped over
  particles with per-particle carries threaded across chunks
  (chunking-invariant by construction; ``test_parametric_core_banded``).

Masking: a residual is valid when every position it touches is unmasked
(OD: k, k+1; UD: n, n+1, n+2 — plus the shooting predecessor and, for
the instrument, the η-clean base points).  Gaps restart the recursion —
only ``bandwidth`` residuals are lost per gap, so fragmented data keeps
nearly all of its information.

Dataset weights follow the integrate engine's pooling (each dataset's
contribution scaled by its normalized weight; ``weight_by_dt=False`` as
everywhere in the parametric path).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .banded import init_carry, whiten_segment
from .covariance import build_od_blocks, build_ud_blocks
from .precompute import od_point_tensors, ud_point_tensors

__all__ = ["ExactRuns", "make_exact_runs_od", "make_exact_runs_ud"]


def _dataset_masks(ds):
    """(static, dynamic) validity, each ``(T, N)`` bool.

    *static* = position known (``dataset._M2d``), *dynamic* = increment
    reliable (``_dynamic_M2d``), falling back to the static mask when
    absent.  The residual-validity rule below applies the dynamic bit
    at the residual's centre and static bits at every other touched
    position.
    """
    X = jnp.asarray(ds.X)
    T = X.shape[0]
    N = X.shape[1] if X.ndim == 3 else 1

    def _norm(m):
        if m is None:
            return None
        m = jnp.asarray(m).astype(bool)
        return m[:, None] if m.ndim == 1 else m

    static = _norm(ds.mask)
    if static is None:
        static = jnp.ones((T, N), bool)
    dynamic = _norm(ds.dynamic_mask)
    if dynamic is None:
        dynamic = static
    return static, dynamic


def _dataset_dt_stream(ds, T):
    """``(T−1,)`` float64 per-interval steps of one dataset (host-side).

    ``t`` takes precedence (finite differences); a scalar ``dt`` is
    broadcast; a per-step ``dt`` array of length ``T`` (trailing entry
    unused, the storage convention) or ``T−1`` is passed through.
    """
    if ds.t is not None:
        return np.diff(np.asarray(ds.t, dtype=np.float64))
    dt = ds.dt
    if dt is None:
        raise ValueError("dataset carries neither dt nor t")
    arr = np.asarray(dt, dtype=np.float64).reshape(-1)
    if arr.size == 1:
        return np.full(T - 1, float(arr[0]))
    if arr.size == T:
        return arr[:T - 1]
    if arr.size == T - 1:
        return arr
    raise ValueError(
        f"per-step dt of length {arr.size} does not match T={T} positions")


class ExactRuns:
    """Bundles the exact-core runners for one (collection, model) pair.

    ``gram(θ, D, Λ)`` → packed ``concat[G.ravel(), f, H.ravel(), nll]``
    (the ``gram_run`` contract, consumed by ``unpack_gram``);
    ``nll(θ, D, Λ)`` → scalar.  ``prepare(θ)`` caches the fixed-θ Phase-A
    tensors, after which ``nll_cached(D, Λ)`` / ``profile_nll_vg`` (the
    (D,Λ) profile) and ``diffusion_nll_vg`` (state-dependent ``D(x[,v];
    θ_D)`` at fixed force) evaluate with zero force/basis calls per
    iteration.
    """

    def __init__(self, collection, F_psf, *, dynamics, dt=None, n_sub,
                 integrator, w=0.0, jitter=1e-7, jitter_chol=1e-10,
                 chunk_pts=200_000, chunk_target_bytes=None,
                 lyapunov=True, convexity="auto", huber_c=None,
                 aux_models=()):
        self._coll = collection
        self._F = F_psf
        # further models evaluated on the same blocks (the diffusion PSF):
        # their per-frame extras are materialised too
        self._aux_models = tuple(aux_models)
        self._dyn = dynamics                      # "od" | "ud"
        # dt: scalar override (tests / direct construction) or None —
        # per-interval streams are then derived per dataset (ds.t / ds.dt),
        # which handles per-step and cross-dataset non-uniform sampling.
        self._dt = dt
        self._n_sub = n_sub
        self._integrator = integrator
        self._w = float(w)
        self._jitter = jitter
        self._jitter_chol = jitter_chol
        self._chunk_pts = int(chunk_pts)
        self._chunk_bytes = (None if chunk_target_bytes is None
                             else int(chunk_target_bytes))
        self._n_params = int(F_psf.template.size)
        # Stage-3 upgrades (see the design note): Lyapunov-exact process
        # covariance; convexity-corrected residual mean (OD,
        # non-interacting; "auto" = on when the noise model is active,
        # i.e. on the instrument path); Huberized whitened score
        # (bounded influence; None = plain Gaussian).
        self._lyapunov = bool(lyapunov)
        self._convexity = bool(
            convexity is True or (convexity == "auto" and self._w > 0.0)
        ) and dynamics == "od" and not getattr(F_psf, "particles_input", False)
        self._huber_c = None if huber_c is None else float(huber_c)
        # jitted per-chunk pipelines, cached across GN iterations / profile
        # evaluations (keyed by with_gram; shapes + lead/n_res are static
        # arguments, so each distinct chunk geometry compiles once).
        self._chunk_fn = {}
        # measured chunk plans per (program, particle geometry, dtype) —
        # see _resolve_chunk_pts
        self._chunk_plans = {}
        # fixed-θ tensor cache (see prepare()); None until prepared
        self._prep = None
        self._cache_meta = None
        self._cache_data = None

    # ── phase A ─────────────────────────────────────────────────────────

    def _point_tensors(self, theta, X_block, extras, dt_block, *, with_psi,
                       with_inst, D=None, Lam=None):
        D_lyap = D if self._lyapunov else None
        if self._dyn == "od":
            return od_point_tensors(self._F, theta, X_block, extras,
                                    dt_block, self._n_sub, self._integrator,
                                    with_psi=with_psi, with_instrument=with_inst,
                                    D_lyap=D_lyap,
                                    conv_lambda=Lam if self._convexity else None)
        return ud_point_tensors(self._F, theta, X_block, extras,
                                dt_block, self._n_sub, self._integrator,
                                with_psi=with_psi, with_instrument=with_inst,
                                D_lyap=D_lyap)

    def _dataset_dt(self, ds, T):
        """``(T−1,)`` per-interval steps (the constructor scalar overrides)."""
        if self._dt is not None:
            return np.full(T - 1, float(self._dt))
        return _dataset_dt_stream(ds, T)

    def _dataset_extras(self, ds, i_ds):
        """Per-dataset extras ``(static, frames)``.

        ``static``: user ``extras_global`` and ``extras_local``
        (per-particle ``(N, ...)`` arrays — same shape class as the
        reserved ``particle_index``) plus the static reserved keys
        (``dataset_index``, ``particle_index``, ``duration`` — e.g. for
        shared-plus-per-dataset parameter models).

        ``frames``: the time-dependent values the model reads (a
        ``TimeSeriesExtra``, a time-generator callable, the reserved
        ``time``; dispatcher-owned structural tables such as per-frame
        neighbour lists included) as ``(T, ...)`` arrays, sliced per block by
        :meth:`_block_extras` and read per interval by the flow
        consumers (see :mod:`frame_extras`)."""
        from SFI.trajectory.reserved_extras import (
            INJECTED_NAMES,
            ExtrasContext,
            check_user_extras,
            resolve_reserved,
        )

        from .frame_extras import materialize_frame_extras, time_varying_keys

        X = np.asarray(ds.X)
        N = X.shape[1] if X.ndim == 3 else 1
        dts = self._dataset_dt(ds, X.shape[0])
        ctx = ExtrasContext(n_particles=N, dataset_index=i_ds,
                            frame_times=None, duration=float(np.sum(dts)))
        reserved = resolve_reserved(ctx)
        reserved.pop("time", None)          # per-frame — see ``frames``
        required = set()
        for model in (self._F,) + self._aux_models:
            req = getattr(model, "required_extras", None)
            req = req() if callable(req) else req
            if req is None:                  # undeclared: take every key
                required = None
                break
            required.update(req)
            # Dispatcher-owned structural tables (e.g. per-frame CSR neighbour
            # lists) are read too, though absent from ``required_extras``
            # (as in the integrate dispatcher's per-frame routing test).
            st = getattr(model, "structural_extras", None)
            st = st() if callable(st) else st
            required.update(st or ())
        tv = time_varying_keys(ds, required)
        # time-varying values never enter the static mapping (unread ones
        # are dropped: a TimeSeriesExtra is not a jit operand)
        tv_all = set(time_varying_keys(ds, None))
        out = {k: v for k, v in (ds.extras_global or {}).items()
               if k not in tv_all}
        out.update({k: v for k, v in
                    (getattr(ds, "extras_local", None) or {}).items()
                    if k not in tv_all})
        # A user value under a row-injected framework key (``pbc_ref``) would be
        # read as the framework's; the reserved keys are overwritten just below.
        check_user_extras(out, forbidden=INJECTED_NAMES)
        out.update(reserved)
        frames = materialize_frame_extras(ds, tv, dataset_index=i_ds)
        return out, frames

    def _block_extras(self, extras, lo, hi):
        """Model extras of the position block ``[lo, hi + bw)``: the
        static mapping plus the block's slice of every per-frame value."""
        from .frame_extras import FrameSeries

        static, frames = extras
        if not frames:
            return static
        bw = 1 if self._dyn == "od" else 2
        return {**static, **{k: FrameSeries(v[lo:hi + bw])
                             for k, v in frames.items()}}

    # ── chunk planning: measured working set per program ────────────────

    def _static_chunk_pts(self, ds, kind):
        """First-guess chunk size from a static per-residual estimate.

        Only the *probe size* of the measured planner
        (:meth:`_resolve_chunk_pts`) — it sets how often the planner must
        recompile, never the chunk it settles on.  Counts Phase-A point
        tensors (r/J/α/Lyapunov), the Gram program's ψ-wide tensors (per
        θ-flow, the per-stage recursion keeps ``Θ_s, dk_s`` for each of the
        ``S`` stages plus ``B_step``, ``J·B`` and ``B``; the residual flow,
        and two more for the overdamped instrument / three phase flows for
        the underdamped one; plus the ψ, instrument, blended and whitened
        column streams), Phase-B blocks, and the model's own per-stage
        working set under differentiation (statefunc ``memory_hint``), with
        a ×2 margin.
        """
        X = np.asarray(ds.X)
        N = X.shape[1] if X.ndim == 3 else 1
        d = X.shape[-1]
        n = self._n_params
        bw = 1 if self._dyn == "od" else 2
        n_mats = 1 if self._dyn == "od" else 4
        q_terms = (1 if self._dyn == "od" else 7) * d * d if self._lyapunov else 0
        cols = 0
        if kind == "gram":
            S = 4 if self._integrator == "rk4" else 1
            inst = self._w > 0.0
            if self._dyn == "od":
                theta_flows, width = 1 + (2 if inst else 0), 1
            else:                               # lifted (2d) phase flows
                theta_flows, width = 2 + (3 if inst else 0), 2
            cols = n * (width * theta_flows * (2 * S + 3)
                        + (5 if inst else 3))
        per_pt = N * d * (2 + n_mats * d + cols) * 8.0
        per_pt += N * (q_terms + 2 * (1 + bw) * d * d) * 8.0
        hint = 0
        est = getattr(self._F, "estimate_bytes_per_sample", None)
        if callable(est):
            try:
                from SFI.statefunc.memhint import SampleMeta

                hint = int(est(sample=SampleMeta(P=N), mode="grad"))
            except Exception:  # noqa: BLE001 — sizing must never crash a solve
                hint = 0
        flows = 1 if self._dyn == "od" else 2
        per_pt += flows * (2 + d) * max(hint, N * d * 8)
        return max(1, int(self._chunk_bytes // (2.0 * per_pt)))

    def _probe_program(self, kind, opts, ds, i_ds, C, dtype):
        """``(jitted program, operands)`` of one ``kind`` chunk of ``C``
        residuals — the geometry the runtime executes for an interior chunk
        (full instrument lead), so the probe's compilation is the one the
        runtime then reuses.  Operand *values* are placeholders: only shapes
        and dtypes enter the compiled buffer assignment."""
        lead_full = (1 if self._dyn == "od" else 2) if self._w > 0.0 else 0
        blk = None
        for bk in self._iter_blocks(ds, chunk_pts=C):   # lazy: ≤ 3 blocks
            if blk is None:
                blk = bk                                 # single-chunk case
            if bk[1] - bk[0] == C and bk[0] - bk[2] == lead_full:
                blk = bk
                break
        a, b, lo, hi, X_block, dt_block, ms, md = blk
        extras = self._block_extras(self._dataset_extras(ds, i_ds), lo, hi)
        N, d = X_block.shape[1], X_block.shape[-1]
        n = self._n_params
        theta = jnp.zeros((n,), dtype)
        D = jnp.eye(d, dtype=dtype)
        lead, n_res = a - lo, b - a
        if kind in ("gram", "nll"):
            with_gram = kind == "gram"
            carries = jax.vmap(lambda _: init_carry(
                d, n if with_gram else 0, dtype))(jnp.arange(N))
            return self._get_chunk_fn(with_gram), (
                theta, D, D, X_block, dt_block, ms, md, carries, extras,
                lead, n_res)
        if kind == "prepare":
            with_conv, with_base = opts
            return self._get_prepare_fn(with_conv, with_base), (
                theta, X_block, dt_block, ms, md, extras,
                D if with_conv else None, lead, n_res)
        if kind == "loss":
            carries = jax.vmap(lambda _: init_carry(d, 0, dtype))(
                jnp.arange(N))
            return self._get_loss_chunk_vg(), (
                theta, theta, D, D, X_block, dt_block, ms, md, carries,
                extras, lead, n_res)
        raise ValueError(f"unknown chunk program kind {kind!r}")

    def _resolve_chunk_pts(self, ds, kind="gram", opts=(), dtype=None):
        """Residuals per chunk for program ``kind`` on dataset ``ds``.

        Without a byte budget the fixed ``chunk_pts`` applies.  With one
        (``chunk_target_bytes`` — the engines' ``max_memory_gb``), the
        chunk is sized from XLA's buffer assignment of the program the
        runtime executes (temporaries + outputs, :func:`SFI.integrate.api.
        _xla_bytes`), separately for each program: the Gram (``"gram"``:
        ψ and instrument columns through the per-stage θ-recursion), the
        NLL (``"nll"``), the fixed-θ tensor cache (``"prepare"``) and the
        frozen-precision loss with its gradient (``"loss"``).  Procedure:
        compile at the static first guess (:meth:`_static_chunk_pts`),
        read the working set ``M(C)``, and rescale with the linear model
        ``M = a + b·C`` (secant through the measured points; ``a = 0`` from
        one point) until the chunk fits; a fitting probe is kept unless
        the budget admits at least twice as many residuals (each new size
        costs one compilation; the kept size's compilation is the one the
        runtime reuses).  Proportional growth from a fitting measurement
        cannot overshoot (``a ≥ 0``), so later datasets with the same
        particle geometry reuse the measurement without recompiling.

        The budget is a resource bound only: chunking leaves every result
        unchanged up to float reassociation (reverse-time carries thread
        across chunks).  There is no lower clamp beyond one residual —
        a model whose single residual exceeds the budget runs one residual
        per chunk (with a warning).  Falls back to the static guess when
        the backend reports no memory analysis.
        """
        if self._chunk_bytes is None:
            return self._chunk_pts
        X = jnp.asarray(ds.X)
        T = X.shape[0]
        N = X.shape[1] if X.ndim == 3 else 1
        d = X.shape[-1]
        bw = 1 if self._dyn == "od" else 2
        cap = max(1, min(self._chunk_pts, T - bw))
        if dtype is None:
            dtype = (X.dtype if jnp.issubdtype(X.dtype, jnp.floating)
                     else jnp.result_type(float))
        key = (kind, tuple(opts), N, d, jnp.dtype(dtype).name)
        budget = self._chunk_bytes
        plan = self._chunk_plans.get(key)
        if plan is None:
            i_ds = next((i for i, dd in enumerate(self._coll.datasets)
                         if dd is ds), 0)
            plan = self._measure_plan(kind, opts, ds, i_ds, cap, dtype, budget)
            self._chunk_plans[key] = plan
        C, M = plan
        if M is None or cap <= C or M > budget:
            return min(C, cap)
        # a longer dataset than the measured one (same particle geometry):
        # proportional growth from a fitting measurement stays in budget
        grown = int(budget * C // M)
        return C if grown < 2 * C else min(cap, grown)

    def _measure_plan(self, kind, opts, ds, i_ds, cap, dtype, budget):
        """Measured-chunk search (see :meth:`_resolve_chunk_pts`).
        Returns ``(C, bytes)``: the chunk the runtime uses and its measured
        working set (``None`` without a memory analysis)."""
        from SFI.integrate.api import _xla_bytes

        def measure(C):
            fn, args = self._probe_program(kind, opts, ds, i_ds, C, dtype)
            return _xla_bytes(fn, *args)

        C = min(cap, self._static_chunk_pts(ds, kind))
        M = measure(C)
        if M is None:
            return (C, None)
        pts = [(C, M)]
        for _ in range(4):
            if M <= budget and (C >= cap or 2 * M > budget):
                return (C, M)                    # fits; growth < 2× or capped
            if len(pts) >= 2 and pts[-1][0] != pts[-2][0]:
                (C1, M1), (C2, M2) = pts[-2], pts[-1]
                b = (M2 - M1) / (C2 - C1)
                a = max(M2 - b * C2, 0.0)
            else:
                a, b = 0.0, M / C
            C_new = cap if b <= 0 else int((budget - a) // b)
            C_new = int(np.clip(C_new, 1, cap))
            if C_new == C:
                break
            C, M = C_new, measure(C_new)
            if M is None:
                return (C, None)
            pts.append((C, M))
        if M > budget and C > 1:
            # the secant did not settle: shrink proportionally once more
            C = max(1, int(C * budget // M))
            M = measure(C) or M
        if M > budget:
            import warnings

            warnings.warn(
                f"parametric core: the {kind!r} program needs "
                f"{M / 2**30:.2f} GiB for a chunk of {C} residual(s), above "
                f"max_memory_gb={budget / 2**30:.3g}; running with this "
                "chunk.", RuntimeWarning, stacklevel=4)
        return (C, M)

    def _iter_blocks(self, ds, kind="gram", opts=(), dtype=None, *,
                     chunk_pts=None):
        """Yield ``(a, b, lo, hi, X_block, dt_block, ms_block, md_block)``
        for residual chunks ``[a, b)``.

        The Phase-A block covers residuals ``[lo, hi)`` — extended by up
        to ``lead`` residuals at the front (η-clean instrument bases) and
        ``bw`` at the tail (the cross-chunk coupling blocks
        ``Cov(r_{b−1}, r_b)`` need the next residual's propagators) — so
        every kept residual has complete couplings.  The overlap work is
        O(bw + lead) points per chunk.  ``dt_block`` holds the
        per-interval steps of the block's ``P`` positions (length
        ``P−1``); ``ms/md`` are the static/dynamic masks.

        The chunk size is planned per program ``kind`` (``opts`` and the
        operand ``dtype`` select the measured plan, see
        :meth:`_resolve_chunk_pts`); ``chunk_pts`` overrides it."""
        bw = 1 if self._dyn == "od" else 2
        lead = (1 if self._dyn == "od" else 2) if self._w > 0.0 else 0
        X = jnp.asarray(ds.X)
        if X.ndim == 2:
            X = X[:, None, :]
        T = X.shape[0]
        n_res_total = T - bw                      # residual indices [0, T−bw)
        ms, md = _dataset_masks(ds)
        dts = jnp.asarray(self._dataset_dt(ds, T))
        if chunk_pts is None:
            chunk_pts = self._resolve_chunk_pts(ds, kind, opts, dtype)
        for a in range(0, max(n_res_total, 1), chunk_pts):
            b = min(a + chunk_pts, n_res_total)
            if b <= a:
                continue
            lo = max(a - lead, 0)
            hi = min(b + bw, n_res_total)
            yield (a, b, lo, hi, X[lo:hi + bw], dts[lo:hi + bw - 1],
                   ms[lo:hi + bw], md[lo:hi + bw])

    # ── phase B ─────────────────────────────────────────────────────────

    def _residual_validity(self, ms_block, md_block, n_res, lead):
        """(n_res, N) validity — the centre/neighbour mask blend.

        The dynamic bit ("increment reliable") applies at the residual's
        *centre* (OD: the increment base ``k``; UD: the interior
        position ``n+1``), static bits ("position known") at the other
        touched positions — the recursion needs only the residual's own
        points.
        """
        if self._dyn == "od":
            return (md_block[lead:lead + n_res]
                    & ms_block[lead + 1:lead + 1 + n_res])
        return (ms_block[lead:lead + n_res]
                & md_block[lead + 1:lead + 1 + n_res]
                & ms_block[lead + 2:lead + 2 + n_res])

    def _instrument_validity(self, ms_block, n_res, lead):
        """(n_res, N) static-mask bits of the instrument's lagged base
        positions — ``X_{k−1}`` (OD) / ``(Y_{n−2}, Y_{n−1})`` (UD).

        After a mid-data mask gap the banded recursion restarts, but the
        first residual(s) of the new segment would otherwise blend an
        instrument built from masked (garbage) positions; where a base
        bit is False the blend falls back to ψ for that residual and
        particle."""
        N = ms_block.shape[1]
        pad = jnp.zeros((2, N), bool)
        msp = jnp.concatenate([pad, ms_block], axis=0)   # position j → j+2
        k = lead + 2
        if self._dyn == "od":
            return msp[k - 1:k - 1 + n_res]
        return (msp[k - 2:k - 2 + n_res] & msp[k - 1:k - 1 + n_res])

    def _blocks_for(self, pt, D, Lambda, dt_res):
        # state-dependent diffusion: D per (residual, particle), (n, N, d, d)
        per_pt_D = jnp.ndim(D) == 4
        if self._dyn == "od":
            if "Q" in pt:
                # the Lyapunov Q already carries D; the D argument of
                # build_od_blocks is unused on this branch
                D_arg = jnp.zeros(pt["r"].shape[-1:] * 2,
                                  dtype=pt["r"].dtype) if per_pt_D else D
                return jax.vmap(
                    lambda J, Q: build_od_blocks(J, D_arg, Lambda, dt_res,
                                                 jitter=self._jitter, Q=Q),
                    in_axes=(1, 1), out_axes=0)(pt["J"], pt["Q"])
            if per_pt_D:
                return jax.vmap(
                    lambda J, Dp: build_od_blocks(J, Dp, Lambda, dt_res,
                                                  jitter=self._jitter),
                    in_axes=(1, 1), out_axes=0)(pt["J"], D)
            return jax.vmap(
                lambda J: build_od_blocks(J, D, Lambda, dt_res,
                                          jitter=self._jitter),
                in_axes=1, out_axes=0)(pt["J"])
        if "qing" in pt:
            from .covariance import build_ud_blocks_exact

            qax = {k: 1 for k in pt["qing"]}
            return jax.vmap(
                lambda ap, a0, am, qing: build_ud_blocks_exact(
                    ap, a0, am, qing, Lambda, jitter=self._jitter),
                in_axes=(1, 1, 1, qax), out_axes=0)(
                    pt["ap"], pt["a0"], pt["am"], pt["qing"])
        # leading-order (4/3, 1/3)Δt³ blocks: valid for uniform dt only
        # (guarded upstream); dt_res[1:] matches the (P−2,) residual count.
        if per_pt_D:
            return jax.vmap(
                lambda ap, a0, am, Dp: build_ud_blocks(ap, a0, am, Dp, Lambda,
                                                       dt_res[1:],
                                                       jitter=self._jitter),
                in_axes=(1, 1, 1, 1), out_axes=0)(
                    pt["ap"], pt["a0"], pt["am"], D)
        return jax.vmap(
            lambda ap, a0, am: build_ud_blocks(ap, a0, am, D, Lambda,
                                               dt_res[1:],
                                               jitter=self._jitter),
            in_axes=(1, 1, 1), out_axes=0)(pt["ap"], pt["a0"], pt["am"])

    def _whiten_chunk(self, pt_ext, D, Lambda, valid_res, inst_ok, carries,
                      dt_block, lead, n_res, *, with_gram):
        """One chunk: blocks on the extended residual set → slice the kept
        range [lead, lead+n_res) → per-particle whitening → partials.

        The couplings are per-step (``c1[i] = Cov(r_i, r_{i+1})``), taken
        from the extended block set so the cross-chunk entries are real;
        missing tail entries (trajectory end) are zero-padded and killed
        by the fresh carry's validity bit."""
        bw = 1 if self._dyn == "od" else 2
        blocks = self._blocks_for(pt_ext, D, Lambda, dt_block)
        d = pt_ext["r"].shape[-1]
        N = pt_ext["r"].shape[1]
        dtype = pt_ext["r"].dtype
        sl = slice(lead, lead + n_res)

        A = blocks.A[:, sl]                             # (N, n_res, d, d)
        pad1 = jnp.zeros((N, 1, d, d), dtype)
        c1 = jnp.concatenate([blocks.offdiag[0], pad1], axis=1)[:, sl]
        if bw == 2:
            pad2 = jnp.zeros((N, 2, d, d), dtype)
            c2 = jnp.concatenate([blocks.offdiag[1], pad2], axis=1)[:, sl]
        else:
            c2 = None
        r = jnp.swapaxes(pt_ext["r"], 0, 1)[:, sl]      # (N, n_res, d)

        if with_gram:
            psi = jnp.swapaxes(pt_ext["psi"], 0, 1)[:, sl]
            if self._w > 0.0:
                inst = jnp.swapaxes(pt_ext["psi_inst"], 0, 1)[:, sl]
                # blend only where the instrument is trustworthy: past the
                # dataset front (positional bit) AND with unmasked lagged
                # base positions (per-particle bits — the post-gap fix).
                iv = (pt_ext["inst_valid"][None, sl, None, None]
                      & jnp.swapaxes(inst_ok, 0, 1)[:, :, None, None])
                # Where the instrument is NOT trustworthy, EXCLUDE the
                # residual from the IV estimating equation (zero left row)
                # rather than fall back to ψ: under measurement noise ψ is
                # the η-DIRTY regressor, and each mask gap contributes a few
                # ψ-fallback residuals whose noise correlates with the score
                # — measured on the home-range system as a +82% friction
                # bias at 0.4%-noise × 3%-gaps (γ 1.6 → 2.9), collapsing to
                # the +7% kinematic floor with the exclusion.  The whitened
                # left columns are scale-only (no carry propagation), so a
                # zero row contributes exactly nothing.
                left = jnp.where(
                    iv, (1.0 - self._w) * psi + self._w * inst, 0.0)
            else:
                left = None
        else:
            psi = jnp.zeros((N, n_res, d, 0), dtype)
            left = None

        valid = jnp.swapaxes(valid_res, 0, 1)           # (N, n_res)

        huber_c = self._huber_c

        def _per_particle(A_p, c1_p, c2_p, r_p, psi_p, left_p, v_p, carry):
            e_t, cols_t, raw_t, logdet, v_out, carry_out = whiten_segment(
                A_p, c1_p, c2_p, r_p, psi_p, v_p, raw_cols=left_p,
                carry=carry, jitter=self._jitter_chol, reverse=True)
            if huber_c is None:
                wgt = None
                rho = 0.5 * jnp.sum(e_t * e_t)
            else:
                # bounded-influence (Huber) score on the whitened
                # innovations: componentwise weight min(1, c/|ẽ|)
                a = jnp.abs(e_t)
                wgt = jnp.minimum(1.0, huber_c / jnp.maximum(a, 1e-300))
                rho = jnp.sum(jnp.where(
                    a <= huber_c, 0.5 * e_t * e_t,
                    huber_c * a - 0.5 * huber_c**2))
            out = {
                "nll": rho + jnp.sum(logdet),
                "n": jnp.sum(v_out.astype(r_p.dtype)),
            }
            if with_gram:
                lt = cols_t if left_p is None else raw_t
                ew = e_t if wgt is None else wgt * e_t
                out["f"] = jnp.einsum("kdn,kd->n", lt, ew)
                if wgt is None:
                    out["G"] = jnp.einsum("kdn,kdm->nm", lt, cols_t)
                    out["H"] = jnp.einsum("kdn,kdm->nm", lt, lt)
                else:                                # IRLS-weighted Gram
                    out["G"] = jnp.einsum("kdn,kd,kdm->nm", lt, wgt, cols_t)
                    out["H"] = jnp.einsum("kdn,kd,kdm->nm", lt, wgt, lt)
            return out, carry_out

        in_axes = (0, 0, None if c2 is None else 0, 0, 0,
                   None if left is None else 0, 0, 0)
        out, carries = jax.vmap(_per_particle, in_axes=in_axes)(
            A, c1, c2, r, psi, left, valid, carries)
        return jax.tree_util.tree_map(lambda x: jnp.sum(x, axis=0), out), carries

    def _get_chunk_fn(self, with_gram):
        """One jitted (Phase A → Phase B) pipeline per with_gram mode.

        ``dt_block`` is a traced operand (per-interval steps of the
        block's positions), so uniform and non-uniform sampling share one
        compiled program per chunk geometry."""
        if with_gram not in self._chunk_fn:
            def fn(theta, D, Lambda, X_block, dt_block, ms_block, md_block,
                   carries, extras, lead, n_res):
                pt_ext = self._point_tensors(
                    theta, X_block, extras, dt_block,
                    with_psi=with_gram, with_inst=with_gram and self._w > 0.0,
                    D=D, Lam=Lambda)
                valid_res = self._residual_validity(ms_block, md_block,
                                                    n_res, lead)
                inst_ok = self._instrument_validity(ms_block, n_res, lead)
                return self._whiten_chunk(pt_ext, D, Lambda, valid_res,
                                          inst_ok, carries, dt_block,
                                          lead, n_res, with_gram=with_gram)

            self._chunk_fn[with_gram] = jax.jit(fn, static_argnums=(9, 10))
        return self._chunk_fn[with_gram]

    # ── public runners ──────────────────────────────────────────────────

    def _run(self, theta, D, Lambda, *, with_gram):
        n = self._n_params
        dtype = jnp.asarray(D).dtype
        total = {"nll": jnp.zeros((), dtype), "n": jnp.zeros((), dtype)}
        if with_gram:
            total.update(G=jnp.zeros((n, n), dtype), f=jnp.zeros((n,), dtype),
                         H=jnp.zeros((n, n), dtype))
        weights = np.asarray(self._coll.weights)

        for i_ds, ds in enumerate(self._coll.datasets):
            w_ds = float(weights[i_ds])
            X = jnp.asarray(ds.X)
            N = X.shape[1] if X.ndim == 3 else 1
            d = X.shape[-1]
            n_cols = n if with_gram else 0
            carries = jax.vmap(lambda _: init_carry(d, n_cols, dtype))(
                jnp.arange(N))
            chunk_fn = self._get_chunk_fn(with_gram)
            # static per-dataset and per-particle extras (extras_global /
            # extras_local + the static reserved keys) plus the block's
            # slice of the per-frame (time-dependent) values.
            ex_ds = self._dataset_extras(ds, i_ds)
            chunks = list(self._iter_blocks(
                ds, "gram" if with_gram else "nll", dtype=dtype))
            for ci in range(len(chunks) - 1, -1, -1):   # reversed carry order
                a, b, lo, hi, X_block, dt_block, ms_block, md_block = chunks[ci]
                extras = self._block_extras(ex_ds, lo, hi)
                part, carries = chunk_fn(theta, D, Lambda, X_block, dt_block,
                                         ms_block, md_block, carries, extras,
                                         a - lo, b - a)
                for k in total:
                    total[k] = total[k] + w_ds * part[k]
        return total

    def gram(self, theta, D, Lambda):
        t = self._run(theta, D, Lambda, with_gram=True)
        return jnp.concatenate([t["G"].ravel(), t["f"], t["H"].ravel(),
                                t["nll"][None]])

    def nll(self, theta, D, Lambda):
        return self._run(theta, D, Lambda, with_gram=False)["nll"]

    # ── fixed-θ tensor cache (the (D,Λ)-profile / diffusion fast path) ──

    def _cache_bytes_per_point(self, N, d):
        S = 4 if self._integrator == "rk4" else 1
        if self._dyn == "od":
            per = d + d * d + d                       # r, J, base point
            if self._lyapunov:
                per += self._n_sub * S * d * d        # stage Jacobians
        else:
            per = d + 3 * d * d + d + 2 * d           # r, α's, v̂, base
            if self._lyapunov:
                per += 2 * (self._n_sub * S * 4 + 2) * d * d  # lifted stages + N, Jxv
        return N * per * 8.0

    def _prepare_pt(self, theta, X_block, dt_block, extras, conv, with_base):
        """Phase-A tensors for the fixed-θ cache (traceable)."""
        if self._dyn == "od":
            pt = od_point_tensors(self._F, theta, X_block, extras,
                                  dt_block, self._n_sub, self._integrator,
                                  with_psi=False, with_instrument=False,
                                  conv_lambda=conv,
                                  with_stages=self._lyapunov)
            if with_base:
                pt["X_base"] = X_block[:-1]
        else:
            pt = ud_point_tensors(self._F, theta, X_block, extras,
                                  dt_block, self._n_sub, self._integrator,
                                  with_psi=False, with_instrument=False,
                                  with_stages=self._lyapunov)
            if with_base:
                pt["Y_base"] = X_block[1:-1]
        return pt

    def _get_prepare_fn(self, with_conv, with_base):
        key = ("prep", with_conv, with_base)
        if key not in self._chunk_fn:
            def fn(theta, X_block, dt_block, ms_block, md_block, extras, conv,
                   lead, n_res):
                pt = self._prepare_pt(theta, X_block, dt_block, extras,
                                      conv if with_conv else None, with_base)
                valid_res = self._residual_validity(ms_block, md_block,
                                                    n_res, lead)
                return pt, valid_res

            self._chunk_fn[key] = jax.jit(fn, static_argnums=(7, 8))
        return self._chunk_fn[key]

    def prepare(self, theta, *, conv_lambda=None, with_base=False):
        """Cache the fixed-θ Phase-A tensors per chunk.

        Serves :meth:`nll_cached` (the (D,Λ) profile — zero force/basis
        evaluations per iteration) and :meth:`diffusion_nll_vg` (the
        diffusion solve; ``with_base=True`` additionally stores the
        D-evaluation base points).  With the Lyapunov upgrade active the
        per-interval *stage Jacobians* are cached instead of Q, so the
        process covariance is recomputable for any ``D`` via
        :func:`jacobians.lyapunov_from_stages` — identical arithmetic to
        the fused path.  ``conv_lambda`` freezes the convexity correction
        of the residuals at the given Λ (the profile entry value); the
        final Gram re-evaluates it at the profiled Λ.

        Budget-gated: when the estimated cache exceeds
        ``max(2·chunk_target_bytes, 1 GiB)`` nothing is stored and the
        cached entry points transparently recompute the flow.
        """
        theta = jnp.asarray(theta)
        self._prep = (theta, conv_lambda, with_base)
        bw = 1 if self._dyn == "od" else 2
        total = 0.0
        for ds in self._coll.datasets:
            X = np.asarray(ds.X)
            N = X.shape[1] if X.ndim == 3 else 1
            total += max(X.shape[0] - bw, 0) * self._cache_bytes_per_point(
                N, X.shape[-1])
        budget = max(2 * (self._chunk_bytes or 0), 1 << 30)
        if total > budget:
            self._cache_meta = None
            self._cache_data = None
            return self

        weights = np.asarray(self._coll.weights)
        fn = self._get_prepare_fn(conv_lambda is not None, with_base)
        meta, data = [], []
        for i_ds, ds in enumerate(self._coll.datasets):
            ex_ds = self._dataset_extras(ds, i_ds)
            X = jnp.asarray(ds.X)
            N = X.shape[1] if X.ndim == 3 else 1
            ch_meta, ch_data = [], []
            for a, b, lo, hi, X_block, dt_block, ms, md in self._iter_blocks(
                    ds, "prepare", (conv_lambda is not None, with_base),
                    theta.dtype):
                extras = self._block_extras(ex_ds, lo, hi)
                pt, valid_res = fn(theta, X_block, dt_block, ms, md, extras,
                                   conv_lambda, a - lo, b - a)
                ch_meta.append((a - lo, b - a))
                ch_data.append((pt, dt_block, valid_res, extras))
            meta.append((float(weights[i_ds]), N, int(X.shape[-1]),
                         ch_meta))
            data.append(ch_data)
        self._cache_meta = meta
        self._cache_data = data
        return self

    def _q_from_stages(self, stages, h_vec, D, lifted_d=None):
        """Per-residual Q from cached stages for the given D — a ``(d, d)``
        matrix or ``(n, N, d, d)`` state-dependent field."""
        from .jacobians import lyapunov_from_stages

        def _noise(Dv):
            twoD = Dv + jnp.swapaxes(Dv, -1, -2)
            if lifted_d is None:
                return twoD
            m = 2 * lifted_d
            B = jnp.zeros(twoD.shape[:-2] + (m, m), dtype=twoD.dtype)
            return B.at[..., lifted_d:, lifted_d:].set(twoD)

        B = _noise(jnp.asarray(D))
        if jnp.ndim(D) == 2:
            return jax.vmap(lambda st, hk: lyapunov_from_stages(
                st, hk, B, integrator=self._integrator))(stages, h_vec)
        return jax.vmap(lambda st, hk, Bk: lyapunov_from_stages(
            st, hk, Bk, integrator=self._integrator))(stages, h_vec, B)

    def _materialize(self, pt, D, dt_block):
        """Cached tensors → block-builder inputs, with Q/qing recomputed
        for the given D (constant or per-point)."""
        if self._dyn == "od":
            if "stages" in pt:
                Q = self._q_from_stages(pt["stages"],
                                        dt_block / self._n_sub, D)
                return {"r": pt["r"], "J": pt["J"], "Q": Q}
            return {"r": pt["r"], "J": pt["J"]}
        out = {k: pt[k] for k in ("r", "ap", "a0", "am", "vhat")}
        if "cache" in pt:
            c = pt["cache"]
            d = pt["r"].shape[-1]
            Q_in = self._q_from_stages(c["stages_in"],
                                       dt_block[:-1] / self._n_sub, D,
                                       lifted_d=d)
            Q_out = self._q_from_stages(c["stages_out"],
                                        dt_block[1:] / self._n_sub, D,
                                        lifted_d=d)
            out["qing"] = {
                "Qxx_in": Q_in[..., :d, :d], "Qxv_in": Q_in[..., :d, d:],
                "Qvv_in": Q_in[..., d:, d:],
                "Qxx_out": Q_out[..., :d, :d], "Qxv_out": Q_out[..., :d, d:],
                "N": c["N"], "Jxv_out": c["Jxv_out"],
            }
        return out

    def _nll_from_cache(self, D, Lambda, data):
        """Cache pytree → scalar NLL (traceable; cache passed as runtime
        arguments so it is not baked into the jitted graph as constants)."""
        dtype = jnp.asarray(D).dtype
        total = jnp.zeros((), dtype)
        for (w_ds, N, d, ch_meta), ch_data in zip(self._cache_meta, data):
            carries = jax.vmap(lambda _: init_carry(d, 0, dtype))(
                jnp.arange(N))
            for ci in range(len(ch_data) - 1, -1, -1):
                lead, n_res = ch_meta[ci]
                pt, dt_block, valid_res, _extras = ch_data[ci]
                pt2 = self._materialize(pt, D, dt_block)
                part, carries = self._whiten_chunk(
                    pt2, D, Lambda, valid_res, jnp.zeros_like(valid_res),
                    carries, dt_block, lead, n_res, with_gram=False)
                total = total + w_ds * part["nll"]
        return total

    def nll_cached(self, D, Lambda):
        """NLL at the prepared θ — zero basis evaluations on a cache hit;
        transparent full recompute when the cache did not fit."""
        if self._cache_data is None:
            return self.nll(self._prep[0], D, Lambda)
        return self._nll_from_cache(D, Lambda, self._cache_data)

    def profile_nll_vg(self, make_mats):
        """``vg(z)`` for the (D,Λ) profile at the prepared θ.

        ``make_mats(z) -> (D, Λ)`` decodes the profile vector (supplied by
        the solver, which owns the Cholesky-vec convention).  The cache is
        passed as runtime arguments of the jitted graph.
        """
        if self._cache_data is not None:
            @jax.jit
            def _vg(z, data):
                def f(zz):
                    D_, L_ = make_mats(zz)
                    return self._nll_from_cache(D_, L_, data)
                return jax.value_and_grad(f)(z)

            data = self._cache_data
            return lambda z: _vg(z, data)

        th = self._prep[0]

        @jax.jit
        def _vg0(z):
            def f(zz):
                D_, L_ = make_mats(zz)
                return self.nll(th, D_, L_)
            return jax.value_and_grad(f)(z)

        return _vg0

    # ── frozen-precision IRLS loss (the nonlinear-θ L-BFGS path) ────────

    def _live_residuals(self, theta_live, X_block, dt_block, extras):
        """Residuals at θ_live only — one primal flow pass, no ψ/J."""
        if self._dyn == "od":
            from .flow import flow_displacement
            from .frame_extras import frame_window, split_frames, with_frame

            struct = self._F.unflatten_params(theta_live)
            static, frames = split_frames(extras)

            def _r(x0, x1, dtk, fk):
                def _drift(Xf):
                    return self._F(Xf, params=struct,
                                   extras=with_frame(static, fk))

                return x1 - x0 - flow_displacement(
                    _drift, x0, dtk, self._n_sub, self._integrator)

            per_step = jnp.ndim(dt_block) > 0
            return jax.vmap(_r, in_axes=(0, 0, 0 if per_step else None, 0))(
                X_block[:-1], X_block[1:], dt_block,
                frame_window(frames, 0, -1))
        from .flow_multi import ud_multi_step_residuals_with_psi

        return ud_multi_step_residuals_with_psi(
            self._F, theta_live, X_block, extras, dt_block,
            self._n_sub, self._integrator)[0]

    def loss(self, theta_live, theta_frozen, D, Lambda):
        """Frozen-precision quadratic for the IRLS driver: covariance
        blocks at ``θ_frozen``, residuals at ``θ_live``.

        Keeps quasi-score semantics — the
        ``∂Σ/∂θ`` terms of the full likelihood are deliberately absent,
        so the fixed point is the same estimating-equation root as the
        Gauss–Newton path (the full-NLL θ-optimum would import the
        O(γΔt) covariance-model error into θ̂, most visibly as an
        underdamped damping bias).  The frozen log-det rides along as a
        θ_live-independent constant.

        Overdamped only in practice: for the underdamped shooting
        residual this quadratic is *unbounded* along the damping
        direction (the velocity re-solved at θ_live absorbs the misfit
        of over-damped models faster than the exact couplings penalise
        it), so the UD solvers route ``inner="lbfgs"`` to Gauss–Newton
        instead — see ``solve_force_ud``.

        Differentiated (the L-BFGS inner objective), a multi-chunk pass
        rematerialises each chunk's forward in the backward sweep
        (``jax.checkpoint``), so the gradient's working set is one chunk's
        — the planned ``"loss"`` chunk (:meth:`_resolve_chunk_pts`) — rather
        than every chunk's stored residuals at once; a single-chunk pass
        (the data fit the budget) keeps plain reverse mode.  Same
        arithmetic either way.
        """
        dtype = jnp.asarray(D).dtype
        total = jnp.zeros((), dtype)
        weights = np.asarray(self._coll.weights)
        for i_ds, ds in enumerate(self._coll.datasets):
            w_ds = float(weights[i_ds])
            ex_ds = self._dataset_extras(ds, i_ds)
            X = jnp.asarray(ds.X)
            N = X.shape[1] if X.ndim == 3 else 1
            d = X.shape[-1]
            carries = jax.vmap(lambda _: init_carry(d, 0, dtype))(
                jnp.arange(N))
            chunks = list(self._iter_blocks(ds, "loss", dtype=dtype))
            chunk_loss = self._loss_chunk
            if len(chunks) > 1:
                chunk_loss = jax.checkpoint(self._loss_chunk,
                                            static_argnums=(10, 11))
            for ci in range(len(chunks) - 1, -1, -1):
                a, b, lo, hi, X_block, dt_block, ms, md = chunks[ci]
                extras = self._block_extras(ex_ds, lo, hi)
                nll_c, carries = chunk_loss(
                    theta_live, theta_frozen, D, Lambda, X_block, dt_block,
                    ms, md, carries, extras, a - lo, b - a)
                total = total + w_ds * nll_c
        return total

    def _loss_chunk(self, theta_live, theta_frozen, D, Lambda, X_block,
                    dt_block, ms, md, carries, extras, lead, n_res):
        """One chunk of :meth:`loss` → ``(nll, carries)`` (traceable)."""
        pt = self._prepare_pt(theta_frozen, X_block, dt_block, extras, None,
                              False)
        pt2 = self._materialize(pt, D, dt_block)
        pt2["r"] = self._live_residuals(theta_live, X_block, dt_block, extras)
        valid_res = self._residual_validity(ms, md, n_res, lead)
        part, carries = self._whiten_chunk(
            pt2, D, Lambda, valid_res, jnp.zeros_like(valid_res), carries,
            dt_block, lead, n_res, with_gram=False)
        return part["nll"], carries

    def _get_loss_chunk_vg(self):
        """Jitted value-and-θ_live-gradient of one :meth:`_loss_chunk` —
        the per-chunk working set of the differentiated loss, measured by
        the chunk planner (the runtime loss runs inside the caller's
        whole-trajectory program)."""
        key = ("loss_vg",)
        if key not in self._chunk_fn:
            def fn(theta_live, theta_frozen, D, Lambda, X_block, dt_block,
                   ms, md, carries, extras, lead, n_res):
                return jax.value_and_grad(lambda tl: self._loss_chunk(
                    tl, theta_frozen, D, Lambda, X_block, dt_block, ms, md,
                    carries, extras, lead, n_res)[0])(theta_live)

            self._chunk_fn[key] = jax.jit(fn, static_argnums=(10, 11))
        return self._chunk_fn[key]

    # ── diffusion: exact banded NLL over θ_D at the prepared θ_F ────────

    def _eval_D(self, D_psf, D_struct, pt, extras):
        """Per-(residual, particle) diffusion field at the cached base
        points — ``(n, N, d, d)``; per-frame extras are read at the base
        frame (OD: ``k``; UD: the residual centre ``n+1``)."""
        from .frame_extras import frame_window, split_frames, with_frame

        static, frames = split_frames(extras)
        if self._dyn == "od":
            return jax.vmap(lambda Xf, fk: D_psf(
                Xf, params=D_struct, extras=with_frame(static, fk)))(
                pt["X_base"], frame_window(frames, 0, -1))
        return jax.vmap(lambda Xf, Vf, fk: D_psf(
            Xf, v=Vf, params=D_struct, extras=with_frame(static, fk)))(
            pt["Y_base"], pt["vhat"], frame_window(frames, 1, -1))

    def _diffusion_nll(self, theta_D, D_psf, Lambda, data, *, raw):
        """Banded NLL over θ_D (traceable).  ``raw=True``: ``data`` holds
        raw chunk inputs and Phase A is recomputed inside the trace (the
        over-budget fallback); else ``data`` is the prepared cache."""
        D_struct = D_psf.unflatten_params(theta_D)
        dtype = jnp.asarray(theta_D).dtype
        total = jnp.zeros((), dtype)
        for (w_ds, N, d, ch_meta), ch_data in zip(self._cache_meta, data):
            carries = jax.vmap(lambda _: init_carry(d, 0, dtype))(
                jnp.arange(N))
            for ci in range(len(ch_data) - 1, -1, -1):
                lead, n_res = ch_meta[ci]
                if raw:
                    X_block, dt_block, ms, md, extras = ch_data[ci]
                    pt = self._prepare_pt(self._prep[0], X_block, dt_block,
                                          extras, None, True)
                    valid_res = self._residual_validity(ms, md, n_res, lead)
                else:
                    pt, dt_block, valid_res, extras = ch_data[ci]
                Dk = self._eval_D(D_psf, D_struct, pt, extras)
                pt2 = self._materialize(pt, Dk, dt_block)
                part, carries = self._whiten_chunk(
                    pt2, Dk, Lambda, valid_res, jnp.zeros_like(valid_res),
                    carries, dt_block, lead, n_res, with_gram=False)
                total = total + w_ds * part["nll"]
        return total

    def diffusion_nll_vg(self, D_psf, Lambda):
        """``vg(θ_D)`` minimising the exact banded NLL at the prepared θ_F
        (call :meth:`prepare` with ``with_base=True`` first)."""
        raw, data = self._diffusion_nll_operands()

        @jax.jit
        def _vg(td, dat):
            return jax.value_and_grad(
                lambda t: self._diffusion_nll(t, D_psf, Lambda, dat,
                                              raw=raw))(td)

        return lambda td: _vg(td, data)

    def diffusion_nll_hessian(self, D_psf, Lambda):
        """``H(θ_D)``: Hessian of the exact banded NLL over θ_D at the
        prepared θ_F — the observed Fisher information of the diffusion
        parameters (θ_F and Λ held fixed)."""
        raw, data = self._diffusion_nll_operands()

        @jax.jit
        def _h(td, dat):
            return jax.hessian(
                lambda t: self._diffusion_nll(t, D_psf, Lambda, dat,
                                              raw=raw))(td)

        return lambda td: _h(td, data)

    def _diffusion_nll_operands(self):
        """``(raw, data)`` operands of :meth:`_diffusion_nll`."""
        raw = self._cache_data is None
        if raw:
            weights = np.asarray(self._coll.weights)
            meta, data = [], []
            for i_ds, ds in enumerate(self._coll.datasets):
                ex_ds = self._dataset_extras(ds, i_ds)
                X = jnp.asarray(ds.X)
                N = X.shape[1] if X.ndim == 3 else 1
                ch_meta, ch_data = [], []
                for a, b, lo, hi, X_block, dt_block, ms, md in \
                        self._iter_blocks(ds, "prepare", (False, True),
                                          self._prep[0].dtype):
                    ch_meta.append((a - lo, b - a))
                    ch_data.append((X_block, dt_block, ms, md,
                                    self._block_extras(ex_ds, lo, hi)))
                meta.append((float(weights[i_ds]), N, int(X.shape[-1]),
                             ch_meta))
                data.append(ch_data)
            self._cache_meta = meta
        else:
            data = self._cache_data
        return raw, data


def make_exact_runs_od(collection, F_psf, *, dt, n_substeps, integrator,
                       w=0.0, **kw):
    return ExactRuns(collection, F_psf, dynamics="od", dt=dt,
                     n_sub=n_substeps, integrator=integrator, w=w, **kw)


def make_exact_runs_ud(collection, F_psf, *, dt, n_substeps, integrator,
                       w=0.0, **kw):
    return ExactRuns(collection, F_psf, dynamics="ud", dt=dt,
                     n_sub=n_substeps, integrator=integrator, w=w, **kw)
