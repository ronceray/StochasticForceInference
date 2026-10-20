"""Noise shared between particles: model-free force corrections and graph-noise intensities (EXPERIMENTAL).

.. warning:: Experimental -- not exported from :mod:`SFI.inference`; the API may change.

Everything here reuses SFI's instantaneous diffusion estimators ("stencils"), extended
from one particle to pairs of particles.  Each stencil is a symmetric quadratic form in
a few increment vectors ``y_q`` of the whole configuration (``dX``, ``dX_minus``, ...),

    D_t = sum_q L_q y_q^T ,      D_t,ij = sum_q L_q,i y_q,j^T ,

with ``L_q = sum_p Q_pq y_p`` (``Q`` symmetric, possibly time-step dependent).  Its
diagonal blocks ``D_t,ii`` are exactly the per-particle estimators of
:mod:`SFI.inference.overdamped` / :mod:`SFI.inference.underdamped`
(:func:`stencil_directions`); the same formula on ``i != j`` estimates the cross-particle
blocks ``D_ij``, with the same treatment of the time step and of localisation noise
(which is independent between particles and does not enter ``i != j`` at all).

* **Force corrections.**  OLI's Stratonovich moments and ULI subtract
  ``<D_t : A^{-1} grad b>`` using each particle's own block and the same-particle
  Jacobian.  The all-pairs contraction is
  ``sum_j D_t,ij : A^{-1} d b_i / d y_j = sum_q L_q,i . A^{-1} (J y_q)_i``, i.e. one
  forward-mode derivative of the basis along each increment direction ``y_q`` (an
  :class:`~SFI.integrate.integrand.ExprOperand` with a tangent): the increments are the
  probes.  No diffusion model, no Jacobian block, and the sum over ``j`` is restricted
  automatically to the particles the basis depends on.  :class:`SharedNoiseOLI`,
  :class:`SharedNoiseULI`.
* **Intensities of graph-noise families** (``D = sum_q theta_q D^q``): least squares
  of the stencil on ``D^q``, ``<D^q, D_t>_F = sum_q' L_q' . D^q y_q'`` (one matrix-vector
  product with ``D^q``, a per-slot interactor fed ``y`` through its velocity slot), and
  the Gram ``<D^q, D^r>_F = E_z[(D^q z).(D^r z)]`` with random probes ``z`` drawn from
  integrate's per-row keys (Hutchinson).  Runs entirely through
  :func:`~SFI.integrate.integrate`: chunking, masks, datasets and per-frame topology
  as for forces.  :func:`estimate_graph_noise`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from SFI.integrate.timeops import TimeOp

# ---------------------------------------------------------------------------
# Stencils as quadratic forms in increment vectors
# ---------------------------------------------------------------------------


def _bc(a, ref):
    """Broadcast a per-row scalar (``()`` or ``(K,)``) against ``(..., N, d)``."""
    while a.ndim < ref.ndim:
        a = a[..., None]
    return a


def _valid(streams, ref):
    """Zero the increments of particles whose position the row does not see (NaN-safe).

    Uses the state mask ``mask_state`` (position known at every frame the row reads),
    not the fit mask ``mask_out``: a particle excluded from the fit by a dynamic mask
    (e.g. a lattice cell next to a missing pixel) is still observed, and its increments
    enter its neighbours' all-pairs contractions.  Falls back to ``mask_out`` for
    producers that do not emit ``mask_state``.
    """
    m = streams.get("mask_state", None)
    if m is None:
        m = streams.get("mask_out", None)
    if m is None:
        return ref
    return jnp.where(m[..., None], ref, 0.0)


class _StencilOp(TimeOp):
    """A stencil time operator that also sees the row's masks (when there are some):
    :func:`_valid` zeroes the increments of unobserved particles, which the all-pairs
    contractions sum over inside a row, before the integrator's per-particle fit mask."""

    def __call__(self, **streams):
        kw = {k: streams[k] for k in self.requires}
        for key in ("mask_out", "mask_state"):
            if key in streams:
                kw[key] = streams[key]
        return self.fn(**kw)


def _op(fn, requires, name):
    return _StencilOp(fn, name=name, batch_safe=True, requires=frozenset(requires))


def stencil_directions(method: str, dynamics: str) -> List[Tuple[TimeOp, TimeOp]]:
    """``[(L_q, y_q)]`` with ``D_t,ij = sum_q L_q,i y_q,j^T`` for an instantaneous estimator.

    ``dynamics="od"``: ``"MSD"``, ``"WeakNoise"``, ``"noisy"`` (Vestergaard) of
    :mod:`SFI.inference.overdamped`, and ``"noisy_lag2"`` (Vestergaard plus three zero-mean
    lag-2/lag-3 products, the default Stratonovich correction of the robust preset); ``dynamics="ud"``: ``"MSD"``, ``"WeakNoise"``,
    ``"noisy"`` of :mod:`SFI.inference.underdamped` (ULI).  The diagonal blocks reproduce
    those estimators exactly, including their non-uniform-``dt`` forms.
    """
    if dynamics == "od":
        # The overdamped stencils are attached to their first point (as _D_noisy / _D_weaknoise in
        # SFI.inference.overdamped): increments dX ("early") and dX_plus ("late") at X.
        e, l, dte, dtl = "dX", "dX_plus", "dt", "dt_plus"

        def invdt(s, ref):  # two-interval mean, as the per-particle estimators
            return _bc(2.0 / (s[dte] + s[dtl]), ref)

        if method == "MSD":
            y = _op(lambda **s: _valid(s, s["dX"]), {"dX"}, "y[dX]")
            L = _op(lambda **s: 0.5 * _valid(s, s["dX"]) / _bc(s["dt"], s["dX"]), {"dX", "dt"}, "L_msd")
            return [(L, y)]
        if method == "WeakNoise":
            def dd(s):
                return _valid(s, s[l] - s[e])

            y = _op(lambda **s: dd(s), {e, l}, "y[dX+-dX]")
            L = _op(lambda **s: 0.25 * dd(s) * invdt(s, s[e]), {e, l, dte, dtl}, "L_wn")
            return [(L, y)]
        if method in ("noisy", "noisy_lag2"):
            req = {e, l, dte, dtl}
            y1 = _op(lambda **s: _valid(s, s[l]), {l}, f"y[{l}]")
            y2 = _op(lambda **s: _valid(s, s[e]), {e}, f"y[{e}]")
            L2 = _op(lambda **s: 0.25 * _valid(s, 2.0 * s[l] + s[e]) * invdt(s, s[e]), req, "L_noisy2")
            L1 = _op(lambda **s: 0.25 * _valid(s, s[l] + 2.0 * s[e]) * invdt(s, s[e]), req, "L_noisy1")
            if method == "noisy":
                return [(L1, y1), (L2, y2)]
            # Second-order variant, forward-attached (a = dX, b = dX_plus, c = X_p3 - X_plusplus,
            # m = dX_minus; left index on the moment side): Vestergaard plus
            # [-a (x) c - c (x) a + b (x) m] / (2 dt).  Lambda-free (no shared frame), zero-mean in
            # the noise, and it cancels the O(dt) term of the Stratonovich + shift-Gram estimator
            # for nonlinear dynamics as well (a single shifted lag-2 product does so only for
            # linear dynamics with this attachment).
            def c_inc(s):
                return _valid(s, s["X_p3"] - s["X_plusplus"])

            def idt(s, ref):
                return 1.0 / _bc(s["dt"], ref)

            req2 = req | {"X_p3", "X_plusplus", "dX_minus", "dt"}
            L2o2 = _op(lambda **s: L2.fn(**s) - 0.5 * c_inc(s) * idt(s, s[e]), req2, "L_noisy_lag2_a")
            Lc = _op(lambda **s: -0.5 * _valid(s, s[e]) * idt(s, s[e]), req2, "L_noisy_lag2_c")
            yc = _op(lambda **s: c_inc(s), {"X_p3", "X_plusplus"}, "y[X_p3-X_plusplus]")
            Lm = _op(lambda **s: 0.5 * _valid(s, s[l]) * idt(s, s[e]), req2, "L_noisy_lag2_m")
            ym = _op(lambda **s: _valid(s, s["dX_minus"]), {"dX_minus"}, "y[dX_minus]")
            return [(L1, y1), (L2o2, y2), (Lc, yc), (Lm, ym)]
    elif dynamics == "ud":
        if method == "MSD":
            def u(s):
                return _valid(s, s["dX"] - s["dX_minus"])

            y = _op(lambda **s: u(s), {"dX", "dX_minus"}, "y[ddX]")
            L = _op(lambda **s: 0.75 * u(s) / _bc(s["dt"], s["dX"]) ** 3, {"dX", "dX_minus", "dt"}, "L_msd_uli")
            return [(L, y)]
        if method == "WeakNoise":
            def w(s):
                return _valid(s, 2.0 * s["dX"] - s["dX_minus"] - s["dX_plus"])

            req = {"dX", "dX_minus", "dX_plus"}
            y = _op(lambda **s: w(s), req, "y[dddX]")
            L = _op(lambda **s: 0.5 * w(s) / _bc(s["dt"], s["dX"]) ** 3, req | {"dt"}, "L_wn_uli")
            return [(L, y)]
        if method == "noisy":
            req = {"dX", "dX_minus", "dX_plus", "dt", "dt_minus", "dt_plus"}
            keys = ("dX_minus", "dX", "dX_plus")

            def Q(s):
                dt, dtm, dtp = s["dt"], s["dt_minus"], s["dt_plus"]
                dv, dw = dtm - dt, dtp - dt
                c = 0.5 * (dv * dv + dw * dw - 3.0 * dv * dw) / (dtp * dtm)  # non-uniform ballistic term
                dt_bar = dt + ((dtm - dt) + (dtp - dt)) / 3.0
                f = 3.0 / (11.0 * dt_bar ** 3)
                one = jnp.ones_like(dt)
                return f, [[one, 0.5 * one, -1.5 * one - c], [0.5 * one, -one, 0.5 * one], [-1.5 * one - c, 0.5 * one, one]]

            def L_q(q):
                def fn(**s):
                    f, Qm = Q(s)
                    ys = [_valid(s, s[k]) for k in keys]
                    out = sum(_bc(Qm[p][q], ys[0]) * ys[p] for p in range(3))
                    return _bc(f, ys[0]) * out
                return fn

            return [(_op(L_q(q), req, f"L_noisy_uli{q}"), _op((lambda k: (lambda **s: _valid(s, s[k])))(k), {k}, f"y[{k}]"))
                    for q, k in enumerate(keys)]
    raise ValueError(f"unknown stencil {method!r} for dynamics {dynamics!r}")


def stencil_correction_program(basis, A_inv, directions, *, x, v=None, wrt: str = "x", scale: float = 1.0):
    """Integrand of ``c_i,a = sum_j D_t,ij : A^{-1} d b_i[., a] / d y_j`` (all particle pairs).

    One tangent operand per direction: ``(J y_q)_i = sum_j d b_i / d y_j . y_q,j`` (``wrt``
    ``"x"``: positions, ``"v"``: velocities), contracted as ``L_q,i . A^{-1} (J y_q)_i``.
    Output ``(N, F)`` per row, scaled by ``scale``.
    """
    from SFI.integrate.integrand import ConstOperand, ExprOperand, Integrand, Term, TimeOperand

    A = ConstOperand(jnp.asarray(A_inv), alias="Ainv")
    exprs, times, terms = [], [], []
    for q, (L, y) in enumerate(directions):
        tang = {"tx": y} if wrt == "x" else {"tv": y}
        exprs.append(ExprOperand(expr=basis, x=x, v=v, alias=f"Jy{q}", **tang))
        times.append(TimeOperand(L, alias=f"L{q}"))
        terms.append(Term(eq="in,no,ioa->ia", ops=(f"L{q}", "Ainv", f"Jy{q}"), scale=float(scale)))
    return Integrand(exprs=exprs, times=times, consts=[A], terms=terms)


def diagonal_remainder_program(basis, A_inv, D_diag, D_cross, *, x, v=None, wrt: str = "x", scale: float = 1.0):
    """Integrand of ``(D_diag,t - D_cross,t)_ii : A^{-1} d b_i / d y_i`` (same-particle Jacobian).

    Added to :func:`stencil_correction_program` with the ``D_cross`` stencil, it puts
    ``D_diag`` on the diagonal blocks and ``D_cross`` on the cross-particle ones.
    ``D_diag``/``D_cross`` are SFI instantaneous-diffusion time operators (``(N, d, d)``).
    """
    from SFI.integrate.integrand import ConstOperand, ExprOperand, Integrand, Term, TimeOperand

    diff = TimeOp(lambda **s: D_diag(**s) - D_cross(**s), name="D_diag-D_cross", batch_safe=False,
                  requires=frozenset(D_diag.requires) | frozenset(D_cross.requires))
    same = getattr(basis, "pdepth", 0) >= 1
    J = basis.d_x(same_particle=same) if wrt == "x" else basis.d_v(same_particle=same)
    return Integrand(exprs=[ExprOperand(expr=J, x=x, v=v, alias="Jself")], times=[TimeOperand(diff, alias="dD")],
                     consts=[ConstOperand(jnp.asarray(A_inv), alias="Ainv_self")],
                     terms=[Term(eq="imn,no,imoa->ia", ops=("dD", "Ainv_self", "Jself"), scale=float(scale))])


# ---------------------------------------------------------------------------
# Model-free force inference with noise shared between particles
# ---------------------------------------------------------------------------
class CrossIncrementTrapezoid:
    """``c_i[a] = 1/2 v_i . A^{-1} . sum_{j != i} d b_i[., a] / d x_j . v_j`` (OD, for comparison).

    With ``v = dX/dt`` this is the cross-particle part of the Stratonovich trapezoid to
    first order: removing it puts the cross-particle derivatives in Ito form (MSD
    stencil on ``i != j``) while the own-particle term keeps the engine's stencil.
    """

    pdepth = 1
    rank = 0
    particles_input = True

    def __init__(self, basis, A_inv):
        self.basis = basis
        self.A_inv = jnp.asarray(A_inv)
        self.dim = int(basis.dim)
        self.n_features = int(basis.n_features)
        self.required_extras = tuple(getattr(basis, "required_extras", ()) or ())
        self.structural_extras = tuple(getattr(basis, "structural_extras", ()) or ())
        same = getattr(basis, "pdepth", 0) >= 1
        self._jac_self = basis.d_x(same_particle=same)

    def _validate_extras_presence(self, extras):
        missing = [k for k in self.required_extras if extras is None or k not in extras]
        if missing:
            raise KeyError(f"missing extras {missing}")

    def _one(self, x, v, extras):
        _, jv = jax.jvp(lambda xx: self.basis(xx, extras=extras), (x,), (v,))  # (N, d, F)
        J = self._jac_self(x, extras=extras)  # (N, m, o, F)
        jv = jv - jnp.einsum("imoa,im->ioa", J, v)
        return 0.5 * jnp.einsum("in,no,ioa->ia", v, self.A_inv, jv)

    def __call__(self, x, *, v=None, mask=None, extras=None, params=None):
        if x.ndim == 2:
            return self._one(x, v, extras)
        lead = x.shape[:-2]
        xf = x.reshape((-1,) + x.shape[-2:])
        vf = v.reshape((-1,) + v.shape[-2:])
        out = jax.vmap(lambda a, b: self._one(a, b, extras))(xf, vf)
        return out.reshape(lead + out.shape[1:])


def _make_shared_oli():
    from SFI.inference.overdamped import OverdampedLangevinInference
    from SFI.integrate import stream
    from SFI.integrate.api import integrate
    from SFI.integrate.integrand import ExprOperand, Integrand, Term, TimeOperand
    from SFI.integrate.timeops import velocity

    class SharedNoiseOLI(OverdampedLangevinInference):
        """OLI whose Stratonovich gradient term includes the cross-particle diffusion blocks.

        ``cross="stencil"`` (default): the Stratonovich term's own instantaneous estimator
        (``"noisy"``) on every particle pair, by tangent operands along the increments.
        ``cross="ito"``: the cross-particle part of the trapezoid is removed instead
        (MSD stencil on ``i != j``; for comparison).  ``cross=None``: standard OLI.
        Ito moments carry no ``D`` term and are left unchanged.
        """

        _cross = "stencil"
        _cross_stencil = None
        _stencil = "noisy"

        def set_cross_particle(self, cross="stencil", cross_stencil=None, stencil="noisy"):
            """``stencil``: the estimator on all blocks (``"noisy"``, or ``"noisy_lag2"``: second order for
            linear dynamics); ``cross_stencil``: a different one on the ``i != j`` blocks (experiments)."""
            if cross not in ("stencil", "ito", None):
                raise ValueError("cross must be 'stencil', 'ito' or None")
            self._cross = cross
            self._cross_stencil = cross_stencil
            self._stencil = stencil
            return self

        def _force_moments(self, *, program_only: bool = False):
            if self._cross is None or getattr(self, "__force_M_mode__", "Ito") != "Strato":
                return super()._force_moments(program_only=program_only)
            if self._cross == "ito":
                C = ExprOperand(expr=CrossIncrementTrapezoid(self.force_basis, self.A_inv), x=stream("X"),
                                v=velocity("dX", "dt"), alias="C")
                DT = TimeOperand(stream("dt"), alias="DT")
                prog = Integrand(exprs=[C], times=[DT], terms=[Term(eq="ia,->ia", ops=("C", "DT"))])
                if program_only:
                    return list(super()._force_moments(program_only=True)) + [prog]
                total = super()._force_moments()
                self._force_cross_trapezoid = integrate(self.data, prog, reduce="sum",
                                                        chunk_target_bytes=self._chunk_target_bytes)
                self.force_v_moments = self.force_v_moments - self._force_cross_trapezoid
                return total - self._force_cross_trapezoid
            cs = self._cross_stencil or self._stencil
            prog = stencil_correction_program(self.force_basis, self.A_inv, stencil_directions(cs, "od"),
                                              x=stream("X"), wrt="x")
            if cs != self._stencil:
                prog = prog + diagonal_remainder_program(
                    self.force_basis, self.A_inv, self.get_diffusion_timeop("noisy").op,
                    self.get_diffusion_timeop(cs).op, x=stream("X"), wrt="x")
            if program_only:
                return [self._strato_v_moments(program_only=True), prog]
            self.force_v_moments = self._strato_v_moments()
            self._force_D_grad_b_average = integrate(self.data, prog, reduce="sum",
                                                     chunk_target_bytes=self._chunk_target_bytes)
            return self.force_v_moments - self._force_D_grad_b_average

    return SharedNoiseOLI


def _make_shared_uli():
    from SFI.inference.underdamped import UnderdampedLangevinInference
    from SFI.integrate.api import integrate
    from SFI.integrate.integrand import ConstOperand, ExprOperand, Integrand, Term, TimeOperand

    class SharedNoiseULI(UnderdampedLangevinInference):
        """ULI whose correction term includes the cross-particle diffusion blocks.

        The correction ``w <D_t : A^{-1} d_v b>`` is evaluated with the force preset's own
        instantaneous estimator (``__force_diffusion_method__``) on every particle pair,
        by tangent operands along the increments; ``cross=None`` restores standard ULI.
        """

        _cross = "stencil"
        _cross_stencil = None

        def set_cross_particle(self, cross="stencil", cross_stencil=None):
            """``cross_stencil``: estimator for the ``i != j`` blocks (default: the preset's own)."""
            if cross not in ("stencil", None):
                raise ValueError("cross must be 'stencil' or None")
            self._cross = cross
            self._cross_stencil = cross_stencil
            return self

        def _force_moments(self, *, program_only: bool = False):
            if self._cross is None:
                return super()._force_moments(program_only=program_only)
            mode = getattr(self, "__force_M_mode__", "symmetric")
            x_hat, v_hat, a_hat = (self._get_x_hat(M_mode=mode), self._get_v_hat(M_mode=mode),
                                   self._get_a_hat(M_mode=mode))
            prog_phi = Integrand(exprs=[ExprOperand(expr=self.force_basis, x=x_hat, v=v_hat, alias="B")],
                                 times=[TimeOperand(a_hat, alias="a")],
                                 consts=[ConstOperand(self.A_inv, alias="A")],
                                 terms=[Term(eq="im,mn,ina->ia", ops=("a", "A", "B"))])
            lam = {"symmetric": 1.0, "early": 0.0, "anticipated": -0.5}[mode]
            w_pref = (1.0 + 2.0 * lam) / 3.0
            if w_pref == 0.0:
                if program_only:
                    return [prog_phi]
                self._phi_moments = integrate(self.data, prog_phi, reduce="sum",
                                              chunk_target_bytes=self._chunk_target_bytes)
                return self._phi_moments
            method = getattr(self, "__force_diffusion_method__", "noisy")
            cs = self._cross_stencil or method
            prog = stencil_correction_program(self.force_basis, self.A_inv, stencil_directions(cs, "ud"),
                                              x=x_hat, v=v_hat, wrt="v", scale=-w_pref)
            if cs != method:
                prog = prog + diagonal_remainder_program(
                    self.force_basis, self.A_inv, self.get_diffusion_timeop(method).op,
                    self.get_diffusion_timeop(cs).op, x=x_hat, v=v_hat, wrt="v", scale=-w_pref)
            if program_only:
                return [prog_phi, prog]
            self._phi_moments = integrate(self.data, prog_phi, reduce="sum", chunk_target_bytes=self._chunk_target_bytes)
            self._w_moments = integrate(self.data, prog, reduce="sum", chunk_target_bytes=self._chunk_target_bytes)
            return self._phi_moments + self._w_moments

    return SharedNoiseULI


# ---------------------------------------------------------------------------
# Graph-noise intensities from the stencils
# ---------------------------------------------------------------------------
def family_matvec(fam, dim: int, *, chunk_size=None):
    """State expression ``(x, v=u) -> D^q(x) u`` of one family (unit intensity), shape ``(N, d, 1)``.

    ``D^q u = sum_s G_s (G_s^T u)``: one per-slot interactor whose velocity slot carries
    ``u`` -- per source, ``w_s = sum_k G_s,k^T u_k``, then ``G_s,k w_s`` scattered to
    every member (``owners="all"``).  White family: ``u`` itself.
    """
    from SFI.statefunc import make_basis, make_interactor
    from SFI.statefunc.nodes.interactions import FromExtrasHyperCSR

    if fam.per_particle:
        return make_basis(lambda x, v: v[:, None], dim=dim, rank=1, n_features=1, needs_v=True,
                          labels=(fam.label,))

    def kernel(Xk, v, mask, extras):
        G = fam.amplitudes(Xk, mask, extras)  # (K, d, m), zero on invalid slots
        w = jnp.einsum("kdm,kd->m", G, jnp.where(mask[:, None], v, 0.0))
        return jnp.einsum("kdm,m->kd", G, w)[..., None]

    leaf = make_interactor(kernel, dim=dim, rank=1, Kmax=fam.Kmax, n_features=1, needs_v=True, per_slot=True,
                           extras_keys=tuple(fam.extras_keys), labels=(fam.label,))
    return leaf.dispatch(FromExtrasHyperCSR(fam.indptr_key, fam.indices_key), owners="all", chunk_size=chunk_size)


@dataclass
class GraphNoiseFit:
    """Result of :func:`estimate_graph_noise`."""

    theta: np.ndarray
    A: np.ndarray
    b: np.ndarray
    labels: tuple
    stencil: str
    dynamics: str

    def __repr__(self) -> str:
        body = ", ".join(f"{l}={t:.4g}" for l, t in zip(self.labels, self.theta))
        return f"GraphNoiseFit({body}; {self.dynamics}/{self.stencil})"


def estimate_graph_noise(collection, families: Sequence, *, dynamics: str = "od", stencil: str = "noisy",
                         n_probes: int = 2, seed: int = 0, chunk_target_bytes: int = 512 * 1024**2,
                         chunk_size=None) -> GraphNoiseFit:
    """Least-squares intensities ``theta_q`` of ``D = sum_q theta_q D^q`` from an instantaneous stencil.

    ``families`` is a sequence of :class:`~SFI.langevin.graph_noise.NoiseFamily` or a
    :class:`~SFI.langevin.graph_noise.GraphNoise` (its families; topology tables it carries,
    e.g. :class:`~SFI.langevin.graph_noise.LatticeConservedNoise`'s bonds, are added to the
    data's extras where missing).

    Minimises ``sum_t || D_t - sum_q theta_q D^q(X_t) ||_F^2`` over the full ``Nd x Nd``
    matrices (entries outside the families' support do not enter), i.e.
    ``A theta = b`` with ``b_q = sum_t <D^q, D_t>_F = sum_t sum_q' L_q' . D^q y_q'`` and
    ``A_qr = sum_t <D^q, D^r>_F``, the latter by Hutchinson probes ``z ~ N(0, I)``
    (``n_probes`` per row, keys from ``seed``).  ``stencil`` is any of
    :func:`stencil_directions`: ``"noisy"`` is unbiased by localisation noise on every
    block; ``dynamics="ud"`` uses the ULI estimators (velocity noise).  Per point, at
    ``X`` (``weight_by_dt=False``), as SFI's diffusion inference.
    """
    from SFI.integrate import stream
    from SFI.integrate.api import integrate, required_frames, shared_rows
    from SFI.integrate.integrand import ExprOperand, Integrand, Term, TimeOperand
    from SFI.integrate.timeops import normal_probe

    own = dict(getattr(families, "extras", None) or {})
    families = list(getattr(families, "families", families))
    if own:
        import dataclasses

        from SFI.trajectory import TrajectoryCollection

        collection = TrajectoryCollection(
            datasets=[dataclasses.replace(ds, extras_global={**own, **dict(ds.extras_global or {})})
                      for ds in collection.datasets], weights=collection.weights)
    dim = int(collection.datasets[0].X.shape[-1])
    mats = [family_matvec(f, dim, chunk_size=chunk_size) for f in families]
    M = mats[0]
    for m in mats[1:]:
        M = M & m  # (N, d, Q)
    x = stream("X")
    dirs = stencil_directions(stencil, dynamics)

    exprs = [ExprOperand(expr=M, x=x, v=y, alias=f"Dy{q}") for q, (_, y) in enumerate(dirs)]
    times = [TimeOperand(L, alias=f"L{q}") for q, (L, _) in enumerate(dirs)]
    terms = [Term(eq="in,ina->ia", ops=(f"L{q}", f"Dy{q}")) for q in range(len(dirs))]
    prog_b = Integrand(exprs=exprs, times=times, terms=terms)
    exprs = [ExprOperand(expr=M, x=x, v=normal_probe("X", index=k), alias=f"Dz{k}") for k in range(n_probes)]
    terms = [Term(eq="ina,inb->iab", ops=(f"Dz{k}", f"Dz{k}"), scale=1.0 / n_probes) for k in range(n_probes)]
    prog_A = Integrand(exprs=exprs, terms=terms)
    # Gram and moments on the same rows (see ``shared_rows``).
    with shared_rows(required_frames(prog_b, prog_A)):
        b = integrate(collection, prog_b, reduce="sum", weight_by_dt=False, chunk_target_bytes=chunk_target_bytes)
        A = integrate(collection, prog_A, reduce="sum", weight_by_dt=False, seed=seed,
                      chunk_target_bytes=chunk_target_bytes)

    A = np.asarray(A, float)
    b = np.asarray(b, float).reshape(-1)
    theta = np.linalg.lstsq(A, b, rcond=None)[0]
    return GraphNoiseFit(theta=theta, A=A, b=b, labels=tuple(f.label for f in families), stencil=stencil,
                         dynamics=dynamics)


_LAZY = {"SharedNoiseOLI": _make_shared_oli, "SharedNoiseULI": _make_shared_uli}
_BUILT: dict = {}


def __getattr__(name):  # lazy: avoid importing the inference engines at module import
    if name in _LAZY:
        if name not in _BUILT:
            _BUILT[name] = _LAZY[name]()
        return _BUILT[name]
    raise AttributeError(name)
