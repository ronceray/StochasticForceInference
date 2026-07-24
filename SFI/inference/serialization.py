# SFI/inference/serialization.py
"""
Save / load helpers for inference results.

Two complementary routes:

1. **Lightweight** (`save_results` / `load_results`):
   Saves the scalar + array summary produced by ``report_dict()`` to a
   NumPy ``.npz`` archive with a JSON sidecar.  Useful for archival and
   cross-language interop.  Does **not** preserve the callable model.

2. **Equinox model** (`save` / `load` on ``InferenceResultSF``):
   Serializes the full JAX pytree (leaf arrays + metadata) so the fitted
   model can be reloaded and called for prediction.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from .result import InferenceResultSF

logger = logging.getLogger(__name__)


# =====================================================================
# 1. Lightweight save/load  (report_dict → .npz + .json)
# =====================================================================


def _split_for_npz(d: dict) -> tuple[dict, dict]:
    """Split a dict into (arrays, scalars) suitable for np.savez / JSON."""
    arrays: dict = {}
    scalars: dict = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            arrays[k] = v
        elif isinstance(v, dict):
            # Recurse one level (e.g. metadata)
            sub_a, sub_s = _split_for_npz(v)
            for sk, sv in sub_a.items():
                arrays[f"{k}.{sk}"] = sv
            scalars[k] = sub_s
        else:
            scalars[k] = v
    return arrays, scalars


def _merge_from_npz(arrays: dict, scalars: dict) -> dict:
    """Inverse of _split_for_npz."""
    out: dict = {}
    # First, add scalars
    for k, v in scalars.items():
        out[k] = v
    # Then inject arrays back, re-nesting dotted keys
    for k, v in arrays.items():
        if "." in k:
            parent, child = k.split(".", 1)
            out.setdefault(parent, {})
            if isinstance(out[parent], dict):
                out[parent][child] = v
            else:
                out[k] = v
        else:
            out[k] = v
    return out


def save_results(inferer, path: str | Path) -> Path:
    """Save ``inferer.report_dict()`` to ``<path>.npz`` + ``<path>.json``.

    Parameters
    ----------
    inferer : BaseLangevinInference
        Any inference object that exposes ``report_dict()``.
    path : str or Path
        Base path (without extension).  Two files are created:
        ``<path>.npz`` for arrays, ``<path>.json`` for scalars/metadata.

    Returns
    -------
    Path
        The base path used.
    """
    path = Path(path)
    d = inferer.report_dict()
    arrays, scalars = _split_for_npz(d)

    np.savez(str(path.with_suffix(".npz")), **arrays)

    # JSON-safe conversion
    def _jsonable(obj):
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (list, tuple)):
            return [_jsonable(x) for x in obj]
        if isinstance(obj, dict):
            return {k: _jsonable(v) for k, v in obj.items()}
        return obj

    try:
        from importlib.metadata import version as _pkg_version

        sfi_version = _pkg_version("StochasticForceInference")
    except Exception:
        sfi_version = "unknown"
    scalars["_sfi_version"] = sfi_version

    with open(str(path.with_suffix(".json")), "w") as f:
        json.dump(_jsonable(scalars), f, indent=2)

    logger.info("Results saved to %s.{npz,json}", path)
    return path


def load_results(path: str | Path) -> dict:
    """Reload a dict saved by :func:`save_results`.

    Parameters
    ----------
    path : str or Path
        Base path (without extension).

    Returns
    -------
    dict
        The merged dictionary of arrays and scalars.
    """
    path = Path(path)
    npz = np.load(str(path.with_suffix(".npz")), allow_pickle=False)
    arrays = dict(npz)

    with open(str(path.with_suffix(".json"))) as f:
        scalars = json.load(f)

    return _merge_from_npz(arrays, scalars)


# =====================================================================
# 2. Model save/load  (InferenceResultSF → .params.npz + .meta.json)
# =====================================================================


def save_model(result_sf, path: str | Path) -> Path:
    """Serialize an ``InferenceResultSF`` (params array + metadata).

    Creates ``<path>.params.npz`` (the fitted parameter arrays) and
    ``<path>.meta.json`` (param_cov + non-array metadata).

    The PSF/Basis structure is **not** saved — it must be supplied again
    as a ``template`` when calling :func:`load_model`.

    Parameters
    ----------
    result_sf : InferenceResultSF
        The fitted model to save.
    path : str or Path
        Base path (without extension).

    Returns
    -------
    Path
        The base path used.
    """
    path = Path(path)

    # Save parameter arrays (the only dynamic content)
    params_np = {k: np.asarray(v) for k, v in result_sf.params.items()}
    np.savez(str(path.with_suffix(".params.npz")), **params_np)

    # Persist param_cov and meta in JSON sidecar
    sidecar: Dict[str, Any] = {}
    if result_sf.meta:
        safe_meta: Dict[str, Any] = {}
        for k, v in result_sf.meta.items():
            if isinstance(v, (jnp.ndarray, np.ndarray)):
                safe_meta[k] = np.asarray(v).tolist()
            elif isinstance(v, dict):
                safe_meta[k] = {
                    kk: np.asarray(vv).tolist() if isinstance(vv, (jnp.ndarray, np.ndarray)) else vv
                    for kk, vv in v.items()
                }
            else:
                safe_meta[k] = v
        sidecar["meta"] = safe_meta
    if result_sf.param_cov is not None:
        sidecar["param_cov"] = np.asarray(result_sf.param_cov).tolist()

    with open(str(path.with_suffix(".meta.json")), "w") as f:
        json.dump(sidecar, f, indent=2)

    logger.info("Model saved to %s.{params.npz,meta.json}", path)
    return path


def load_model(path: str | Path, template) -> "InferenceResultSF":
    """Reload an ``InferenceResultSF`` from files written by :func:`save_model`.

    Parameters
    ----------
    path : str or Path
        Base path (without extension).
    template : InferenceResultSF
        A *skeleton* instance providing the PSF / Basis structure.
        Typically built from the same PSF / Basis used at training time:
        ``template = InferenceResultSF(SF(psf, dummy_params))``.

    Returns
    -------
    InferenceResultSF
    """
    from SFI.statefunc import SF

    from .result import InferenceResultSF

    path = Path(path)
    npz = np.load(str(path.with_suffix(".params.npz")))
    params = {k: jnp.array(npz[k]) for k in npz.files}

    sf = SF(template._psf_ref, params, drop_features=template.drop_features)
    loaded = InferenceResultSF(sf)

    meta_path = path.with_suffix(".meta.json")
    if meta_path.exists():
        with open(str(meta_path)) as f:
            sidecar = json.load(f)
        meta = sidecar.get("meta", {})
        param_cov_list = sidecar.get("param_cov", None)
        param_cov = jnp.array(param_cov_list) if param_cov_list is not None else None
        object.__setattr__(loaded, "meta", meta)
        object.__setattr__(loaded, "param_cov", param_cov)

    logger.info("Model loaded from %s", path)
    return loaded
