from typing import Any, Callable, Dict

import equinox as eqx

# === JIT control & harness ====================================================
_JIT_ENABLED = True

# Per-root compiled function cache. Keyed by object identity of `root`.
# We intentionally use identity (id(root)) rather than a structural hash to
# keep lookups O(1) and avoid expensive tree-walk hashing or serialisation.
_COMPILED_CACHE: Dict[int, Callable[..., Any]] = {}


def set_jit(enabled: bool = True):
    """Globally enable/disable JIT for Basis/PSF/SF __call__."""
    global _JIT_ENABLED
    _JIT_ENABLED = bool(enabled)


def _eager_eval(root, x, v, mask, extras, params):
    # Plain Python call; used when JIT is disabled (set_jit(False)).
    return root(x, params=params, v=v, mask=mask, extras=extras)


def _compiled_for_root(root) -> Callable[..., Any]:
    """
    Return a compiled callable specialised to the given `root`.
    The compiled function *closes over* `root`, so `root` is static
    without being an argument (reduces tracing/dispatch overhead).
    """
    key = id(root)
    fn = _COMPILED_CACHE.get(key)
    if fn is not None:
        return fn

    # Compile once for this root: dynamic args are (x, v, mask, extras, params)
    @eqx.filter_jit
    def _call(x, v, mask, extras, params):
        return root(x, params=params, v=v, mask=mask, extras=extras)

    _COMPILED_CACHE[key] = _call
    return _call


def _jitted_eval(root, x, v, mask, extras, params):
    """Dispatch to the per-root compiled callable (JIT-enabled path)."""
    return _compiled_for_root(root)(x, v, mask, extras, params)
