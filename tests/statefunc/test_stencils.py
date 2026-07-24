# TODO: review this file
import inspect

import numpy as np
import pytest

from SFI.statefunc.nodes.interactions import stencils as S


def _call_hyperfixed_square_stencil(grid_shape, offsets, bc):
    fn = S.hyperfixed_square_stencil
    sig = inspect.signature(fn)
    names = list(sig.parameters)

    kwargs = {"grid_shape": grid_shape, "offsets": offsets, "bc": bc}
    if "include_slot_extras" in names:
        kwargs["include_slot_extras"] = True
    if "include_slot_extras_offset" in names:
        kwargs["include_slot_extras_offset"] = True
    if "include_center" in names:
        kwargs["include_center"] = True
    result = fn(**kwargs)
    # Function returns a tuple (hyper, slot_mask, slot_extras_offset).
    # Wrap in a namespace for cleaner test access.
    class _Spec:
        pass
    s = _Spec()
    s.hyper = result[0]
    s.slot_mask = result[1]
    s.slot_extras_offset = result[2] if len(result) > 2 else None
    return s


@pytest.mark.parametrize("bc", ["noflux", "drop", "pbc"])
def test_hyperfixed_square_stencil_shape_and_mask(bc):
    grid_shape = (3, 2)
    offsets = S.square_cross_offsets(2, include_center=True)
    spec = _call_hyperfixed_square_stencil(grid_shape, offsets, bc)

    P = int(np.prod(grid_shape))
    K = int(np.asarray(offsets).shape[0])

    hyper = np.asarray(spec.hyper)
    assert hyper.shape == (P, K)

    slot_mask = np.asarray(spec.slot_mask)
    assert slot_mask.shape == (P, K)

    if bc == "pbc":
        assert slot_mask.all()
    else:
        assert slot_mask[:, 0].all()  # center always valid


def test_hyperfixed_square_stencil_oob_slots_marked():
    grid_shape = (2, 2)
    offsets = S.square_cross_offsets(2, include_center=True)

    # --- noflux: OOB neighbors reflected to focal, slot_mask stays True ---
    spec_nf = _call_hyperfixed_square_stencil(grid_shape, offsets, "noflux")
    hyper_nf = np.asarray(spec_nf.hyper)
    mask_nf = np.asarray(spec_nf.slot_mask)

    p = 0  # focal at (0,0)
    # ordering: [center, +x, -x, +y, -y]
    assert hyper_nf[p, 0] == p
    assert mask_nf[p, 0]

    assert mask_nf[p, 1]  # +x in
    assert mask_nf[p, 3]  # +y in
    # noflux: all slots are marked valid (ghost reflection handled via index)
    assert mask_nf[p, 2]  # -x reflected
    assert mask_nf[p, 4]  # -y reflected
    # OOB neighbors are replaced by the focal index (Neumann ghost)
    assert hyper_nf[p, 2] == p
    assert hyper_nf[p, 4] == p

    # --- drop: OOB neighbors are masked out ---
    spec_dr = _call_hyperfixed_square_stencil(grid_shape, offsets, "drop")
    hyper_dr = np.asarray(spec_dr.hyper)
    mask_dr = np.asarray(spec_dr.slot_mask)

    assert mask_dr[p, 0]       # center always valid
    assert mask_dr[p, 1]       # +x in
    assert mask_dr[p, 3]       # +y in
    assert not mask_dr[p, 2]   # -x out-of-bounds
    assert not mask_dr[p, 4]   # -y out-of-bounds
