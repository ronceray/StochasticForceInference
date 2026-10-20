"""Per-frame neighbour lists (time-series structural extras) are read per frame.

CSR tables consumed by a ``dispatch_pairs_from_extras`` dispatcher are
structural extras, absent from ``required_extras``.  The batched runtime
gathers extras at a chunk's first frame, so such programs must be routed to
the per-frame runtime; otherwise every row of a chunk used one topology.
"""
import jax.numpy as jnp
import numpy as np

from SFI import TrajectoryCollection
from SFI.bases.pairs import compact_kernels, radial_pair_basis
from SFI.integrate import integrate, stream
from SFI.integrate.integrand import ExprOperand, Integrand, Term
from SFI.trajectory import TimeSeriesExtra
from SFI.utils.neighbors import build_neighbor_csr, pad_neighbor_csr

L, RC = 10.0, 3.0


def test_per_frame_csr_matches_dense_pairs():
    rng = np.random.default_rng(0)
    X = np.cumsum(0.3 * rng.standard_normal((40, 12, 2)), axis=0) + rng.uniform(0, L, (12, 2))
    ips, ids = [], []
    for t in range(len(X)):
        ip, ix = build_neighbor_csr(X[t] % L, RC, np.array([L, L]))
        ips.append(ip)
        ids.append(ix)
    nnz = max(len(i) for i in ids)
    ids = [pad_neighbor_csr(ip, ix, nnz)[1] for ip, ix in zip(ips, ids)]
    ex = {"box": jnp.array([L, L]), "nb/indptr": TimeSeriesExtra(jnp.asarray(np.stack(ips), jnp.int32)),
          "nb/indices": TimeSeriesExtra(jnp.asarray(np.stack(ids), jnp.int32))}
    coll = TrajectoryCollection.from_arrays(X=X, dt=0.1, extras_global=ex)
    # Compact kernels vanish beyond RC: the list at frame t gives the dense value at X_t.
    inter = radial_pair_basis(compact_kernels([0, 1], RC), dim=2, box="extras")
    kw = dict(exclude_self=True, owners="focal", reducer="sum")
    out = []
    for B in (inter.dispatch_pairs_from_extras(indptr_key="nb/indptr", indices_key="nb/indices", **kw),
              inter.dispatch_pairs(return_as="basis", **kw)):
        prog = Integrand(exprs=[ExprOperand(expr=B, x=stream("X"), alias="B")], terms=[Term(eq="ina->ina", ops=("B",))])
        out.append(np.asarray(integrate(coll, prog, reduce="sum", reduce_over_particles=False, weight_by_dt=False)))
    np.testing.assert_allclose(out[0], out[1], atol=1e-4)
