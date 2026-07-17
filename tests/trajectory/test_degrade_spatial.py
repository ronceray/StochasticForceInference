# TODO: review this file
# tests/test_spatial_degrade_invariants.py
#
# Goal: pinpoint why inference/bootstrapping breaks only when downscale>1.
# These tests *only* exercise the spatial degradation path and its contracts:
#   - grid_shape/dx are read from collection extras (never passed explicitly)
#   - dim-axis (u,v) must never be reshaped/mixed with spatial axes
#   - flattening order must be consistent (C-order by convention)
#   - box extras must update consistently (grid_shape, dx)
#
# If any of these fail, you will typically see the “striated bootstrap” artefact.

from __future__ import annotations

import numpy as np
import pytest

from SFI.trajectory.collection import TrajectoryCollection
from SFI.trajectory.dataset import TrajectoryDataset
from SFI.trajectory.degrade import degrade_spatial_data

# ------------------------------- helpers ---------------------------------


def _box_keys(prefix: str = "box") -> tuple[str, str]:
    return (f"{prefix}/grid_shape", f"{prefix}/dx")


def _get_box_extras(
    coll: TrajectoryCollection, *, prefix: str = "box"
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Read (grid_shape, dx) from the *collection* extras (dataset[0] assumed)."""
    ds = coll.datasets[0]
    eg = ds.extras_global or {}
    k_g, k_dx = _box_keys(prefix)
    if k_g not in eg or k_dx not in eg:
        raise KeyError(
            f"Missing {k_g!r} or {k_dx!r} in extras_global; keys={list(eg.keys())[:20]}"
        )
    g = eg[k_g]
    dx = eg[k_dx]
    gshape = tuple(
        int(v) for v in (g if isinstance(g, (tuple, list)) else np.asarray(g).tolist())
    )
    dx_tuple = tuple(
        float(v)
        for v in (dx if isinstance(dx, (tuple, list)) else np.asarray(dx).tolist())
    )
    return gshape, dx_tuple


def _make_coll_from_grid_fields(
    *,
    grid_shape: tuple[int, int],
    dx: tuple[float, float],
    U: np.ndarray,
    V: np.ndarray,
    T: int = 4,
    prefix: str = "box",
) -> TrajectoryCollection:
    """Build a single-dataset collection with X=(T,N,2) from 2D U,V fields."""
    Nx, Ny = grid_shape
    assert U.shape == (Nx, Ny)
    assert V.shape == (Nx, Ny)
    Xg = np.stack([U, V], axis=-1).astype(np.float32)  # (Nx,Ny,2)
    X = np.repeat(Xg.reshape((1, Nx * Ny, 2)), T, axis=0)  # (T,N,2)
    mask = np.ones((T, Nx * Ny), dtype=bool)

    eg = {}
    k_g, k_dx = _box_keys(prefix)
    eg[k_g] = grid_shape
    eg[k_dx] = dx

    ds = TrajectoryDataset.from_arrays(
        X=X, dt=1.0, t=None, mask=mask, extras_global=eg, extras_local={}, meta={}
    )
    return TrajectoryCollection.from_dataset(ds, weights="pool")


def _downscale_mean_2d(u: np.ndarray, fac: tuple[int, int]) -> np.ndarray:
    """Reference block-mean downscale for 2D arrays (no mask)."""
    fx, fy = fac
    Nx, Ny = u.shape
    assert Nx % fx == 0 and Ny % fy == 0
    Nx2, Ny2 = Nx // fx, Ny // fy
    out = np.empty((Nx2, Ny2), dtype=u.dtype)
    for I in range(Nx2):
        for J in range(Ny2):
            blk = u[I * fx : (I + 1) * fx, J * fy : (J + 1) * fy]
            out[I, J] = blk.mean()
    return out


def _extract_uv_grid(
    coll: TrajectoryCollection, *, prefix: str = "box", ti: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Extract U,V as 2D arrays using grid_shape from extras."""
    gshape, _dx = _get_box_extras(coll, prefix=prefix)
    assert len(gshape) == 2
    Nx, Ny = gshape
    X = np.asarray(coll.datasets[0]._X3d())[ti]  # (N,2)
    U = X[:, 0].reshape((Nx, Ny))
    V = X[:, 1].reshape((Nx, Ny))
    return U, V


# ------------------------------- tests -----------------------------------


@pytest.mark.parametrize("downscale", [2, (2, 3), (3, 2)])
def test_degrade_updates_box_extras_from_extras_only(downscale):
    prefix = "box"
    grid_shape = (24, 18)
    dx = (0.25, 0.50)

    # simple fields
    ii = np.arange(grid_shape[0])[:, None]
    jj = np.arange(grid_shape[1])[None, :]
    U = (ii + 0.01 * jj).astype(np.float32)
    V = (100.0 + jj + 0.01 * ii).astype(np.float32)

    coll = _make_coll_from_grid_fields(
        grid_shape=grid_shape, dx=dx, U=U, V=V, T=4, prefix=prefix
    )

    # IMPORTANT: never pass grid_shape or dx; degrade must read them from extras
    coll2 = degrade_spatial_data(
        coll,
        downscale=downscale,
        method="mean",
        blur_radius=0,
        data_loss_fraction=0.0,
        noise=None,
        bc="pbc",
        seed=0,
        prefix=prefix,
    )

    g2, dx2 = _get_box_extras(coll2, prefix=prefix)

    if isinstance(downscale, int):
        fac = (downscale, downscale)
    else:
        fac = tuple(int(v) for v in downscale)

    assert g2 == (grid_shape[0] // fac[0], grid_shape[1] // fac[1])
    assert dx2 == (dx[0] * fac[0], dx[1] * fac[1])

    # particle count must match prod(grid_shape)
    N2 = int(np.prod(g2))
    X2 = np.asarray(coll2.datasets[0]._X3d())
    assert X2.shape[1] == N2
    assert X2.shape[2] == 2  # dim preserved


@pytest.mark.parametrize("downscale", [(2, 2), (2, 3), (3, 2)])
def test_degrade_never_mixes_u_and_v_channels(downscale):
    """
    Directly targets u/v mixing.

    Construct U and V with different spatial signatures:
      - U varies only with i (rows)
      - V varies only with j (cols) and has a large offset
    After block-mean downscale, each channel must match its own reference.
    """
    prefix = "box"
    grid_shape = (24, 18)
    dx = (1.0, 1.0)
    fx, fy = downscale

    Nx, Ny = grid_shape
    ii = np.arange(Nx, dtype=np.float32)[:, None]  # (Nx,1)
    jj = np.arange(Ny, dtype=np.float32)[None, :]  # (1,Ny)

    # Make them explicitly (Nx,Ny) to avoid broadcasting surprises in the fixture.
    U = np.broadcast_to(ii, (Nx, Ny)).copy()
    V = np.broadcast_to(1000.0 + jj, (Nx, Ny)).copy()

    coll = _make_coll_from_grid_fields(
        grid_shape=grid_shape, dx=dx, U=U, V=V, T=4, prefix=prefix
    )
    coll2 = degrade_spatial_data(
        coll,
        downscale=downscale,
        method="mean",
        blur_radius=0,
        data_loss_fraction=0.0,
        noise=None,
        bc="pbc",
        seed=0,
        prefix=prefix,
    )

    U2, V2 = _extract_uv_grid(coll2, prefix=prefix, ti=0)

    U_ref = _downscale_mean_2d(U, (fx, fy))
    V_ref = _downscale_mean_2d(V, (fx, fy))

    np.testing.assert_allclose(U2, U_ref, rtol=0, atol=0)
    np.testing.assert_allclose(V2, V_ref, rtol=0, atol=0)

    # Any swap/mix is catastrophic due to the +1000 offset.
    assert np.max(np.abs(U2 - V_ref)) > 100.0
    assert np.max(np.abs(V2 - U_ref)) > 100.0


@pytest.mark.parametrize("downscale", [(2, 2), (2, 3), (3, 2)])
def test_degrade_preserves_C_order_flattening_convention(downscale):
    """
    Catch the most common subtle bug: a wrong reshape/flatten order (C vs F)
    or an axis permutation before/after downscale.

    Field definition:
      U[p] = p  where p is the *C-order* flat index of (i,j): p = i*Ny + j.
    For block-mean downscale, each coarse cell must equal the mean of p over its block.
    """
    prefix = "box"
    grid_shape = (24, 18)
    dx = (1.0, 1.0)

    Nx, Ny = grid_shape
    ii = np.arange(Nx)[:, None]
    jj = np.arange(Ny)[None, :]
    p = (ii * Ny + jj).astype(np.float32)

    U = p
    V = 0.0 * p  # irrelevant
    coll = _make_coll_from_grid_fields(
        grid_shape=grid_shape, dx=dx, U=U, V=V, T=4, prefix=prefix
    )

    coll2 = degrade_spatial_data(
        coll,
        downscale=downscale,
        method="mean",
        blur_radius=0,
        data_loss_fraction=0.0,
        noise=None,
        bc="pbc",
        seed=0,
        prefix=prefix,
    )

    U2, _V2 = _extract_uv_grid(coll2, prefix=prefix, ti=0)

    if isinstance(downscale, int):
        fac = (downscale, downscale)
    else:
        fac = downscale
    U_ref = _downscale_mean_2d(U, (int(fac[0]), int(fac[1])))

    # If you accidentally used F-order flattening or swapped axes, this fails sharply.
    np.testing.assert_allclose(U2, U_ref, rtol=0, atol=0)


@pytest.mark.parametrize("downscale", [(2, 2), (2, 3)])
def test_degrade_downscale_does_not_touch_dim_axis(downscale):
    """
    Very explicit dim-axis invariant:
    - build a field where U and V have different *statistics* and verify they remain so.
    This catches bugs where downscale averages over the last axis or reshapes as (..., -1)
    and later reinterprets it incorrectly.
    """
    prefix = "box"
    grid_shape = (30, 24)
    dx = (1.0, 1.0)

    rng = np.random.default_rng(0)
    U = rng.normal(size=grid_shape).astype(np.float32)
    V = (10.0 + 2.0 * rng.normal(size=grid_shape)).astype(
        np.float32
    )  # shifted/scaled stats

    coll = _make_coll_from_grid_fields(
        grid_shape=grid_shape, dx=dx, U=U, V=V, T=4, prefix=prefix
    )
    coll2 = degrade_spatial_data(
        coll,
        downscale=downscale,
        method="mean",
        blur_radius=0,
        data_loss_fraction=0.0,
        noise=None,
        bc="pbc",
        seed=0,
        prefix=prefix,
    )

    U2, V2 = _extract_uv_grid(coll2, prefix=prefix, ti=0)

    # Means/vars should remain separated (block-mean changes variance, but shift separation stays huge)
    assert abs(float(np.nanmean(V2)) - float(np.nanmean(U2))) > 5.0
    assert float(np.nanmean(V2)) > float(np.nanmean(U2))


"""
How to use the failures
-----------------------
- If `test_degrade_updates_box_extras...` fails: extras propagation/update is wrong (grid_shape/dx).
- If `test_degrade_preserves_C_order...` fails: you have an order/axis bug in reshape/flatten or downscale.
- If `test_degrade_never_mixes_u_and_v...` fails: you are mixing dim with spatial axes somewhere in degrade.
  This is the most plausible cause for “striped bootstrap” if your inferred operator is correct on non-degraded data.
"""
