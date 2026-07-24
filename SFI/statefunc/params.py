from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Tuple

import equinox as eqx
import jax.numpy as jnp


# ---------------------------------------------------------------------
# ParamSpec  - static description of one parameter block
# ---------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ParamSpec:
    name: str  # <-- KEY for sharing
    shape: Tuple[int, ...]
    dtype: Any = jnp.float32
    init: Callable | str = "zeros"  # PRNG->array or keyword
    default: Any = None  # optional concrete value (scalar or array-like)

    @property
    def size(self) -> int:
        from functools import reduce
        from operator import mul

        return reduce(mul, self.shape, 1)

    def compatible_with(self, other: "ParamSpec") -> bool:
        """Shareable iff shape and dtype match exactly."""
        return (self.shape == other.shape) and (self.dtype == other.dtype)

    def merged_with(self, other: "ParamSpec") -> "ParamSpec":
        """
        Return a single spec representing the shared parameter.
        Requires compatibility; keeps `self.init` by default.
        """
        if not self.compatible_with(other):
            raise ValueError(
                f"ParamSpec mismatch for '{self.name}': {self.shape}/{self.dtype} vs {other.shape}/{other.dtype}"
            )
        # deterministically keep `self`, but salvage a default from `other`
        if self.default is None and other.default is not None:
            from dataclasses import replace

            return replace(self, default=other.default)
        return self


class ParamSuite(eqx.Module):
    """Immutable container holding a set of ``ParamSpec`` objects."""

    specs: tuple[ParamSpec, ...] = eqx.field(static=True)
    _lookup: dict[str, int] = eqx.field(static=True)  # name → idx

    # ---------- construction ---------- #
    def __init__(self, specs: Iterable[ParamSpec]):
        specs = tuple(specs)
        names = [ps.name for ps in specs]
        if len(set(names)) != len(names):
            dup = {n for n in names if names.count(n) > 1}
            raise ValueError(f"Duplicate parameter names: {dup}")
        object.__setattr__(self, "specs", specs)
        object.__setattr__(self, "_lookup", {ps.name: i for i, ps in enumerate(specs)})

    @classmethod
    def from_specs(cls, *specs: ParamSpec) -> "ParamSuite":
        return cls(specs)

    # ---------- basic info ---------- #
    def __iter__(self):
        return iter(self.specs)

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, name: str) -> ParamSpec:
        return self.specs[self._lookup[name]]

    @property
    def size(self) -> int:
        return sum(ps.size for ps in self.specs)

    # ---------- convenience constructors ---------- #
    def zeros(self) -> dict[str, jnp.ndarray]:
        """Return a parameter dict with all values initialized to zero."""
        return {ps.name: jnp.zeros(ps.shape, dtype=ps.dtype) for ps in self.specs}

    @property
    def has_defaults(self) -> bool:
        """True iff every spec in this suite carries a concrete ``default``."""
        return len(self.specs) > 0 and all(ps.default is not None for ps in self.specs)

    def defaults(self) -> dict[str, jnp.ndarray] | None:
        """Return a parameter dict from spec ``default`` values, or None if any
        spec has no default. Values are broadcast to the declared shape."""
        if not self.has_defaults:
            return None
        out: dict[str, jnp.ndarray] = {}
        for ps in self.specs:
            arr = jnp.asarray(ps.default, dtype=ps.dtype)
            if arr.shape != ps.shape:
                arr = jnp.broadcast_to(arr, ps.shape)
            out[ps.name] = arr
        return out

    # ---------- param ↔ vector helpers ---------- #
    def materialize(
        self,
        vector: jnp.ndarray,
        *,
        dtype_overrides: dict[str, jnp.dtype] | None = None,
    ):
        if vector.ndim != 1 or vector.size != self.size:
            raise ValueError("Flat vector has wrong length")
        out, i = {}, 0
        for ps in self.specs:
            n = ps.size
            arr = (
                vector[i : i + n]
                .reshape(ps.shape)
                .astype(dtype_overrides.get(ps.name, ps.dtype) if dtype_overrides else ps.dtype)
            )
            out[ps.name] = arr
            i += n
        return out

    def vectorize(self, tree: dict[str, jnp.ndarray]) -> jnp.ndarray:
        parts = [jnp.ravel(tree[ps.name]).astype(ps.dtype) for ps in self.specs]
        return jnp.concatenate(parts, axis=0)

    # ---------- merging (parameter sharing) ---------- #
    def merge(self, other: "ParamSuite | None") -> "ParamSuite":
        """
        Union with sharing-by-name:
          - If a name appears in both suites and specs are compatible (shape/dtype),
            they are **tied** (kept once).
          - If incompatible → error.
        """
        if other is None:
            return self
        if not isinstance(other, ParamSuite):
            raise TypeError(f"merge expects ParamSuite or None, got {type(other).__name__}")
        if not self.specs:
            return other
        if not other.specs:
            return self

        left = {ps.name: ps for ps in self.specs}
        right = {ps.name: ps for ps in other.specs}

        out: dict[str, ParamSpec] = dict(left)  # copy
        for name, ps_r in right.items():
            if name in out:
                ps_l = out[name]
                out[name] = ps_l.merged_with(ps_r)  # validates compatibility
            else:
                out[name] = ps_r
        return ParamSuite(out.values())

    @classmethod
    def merge_many(cls, *suites: "ParamSuite | None") -> "ParamSuite | None":
        """Merge any number of suites, sharing parameters by name (shape/dtype must match)."""
        merged: ParamSuite | None = None
        for s in suites:
            if s is None:
                continue
            merged = s if merged is None else merged.merge(s)
        return merged

    # ---------- PyTree protocol ---------- #
    def tree_flatten(self):
        children = ()  # no array leaves
        aux = self.specs  # static
        return children, aux

    # ---------- universal parser ---------- #
    @classmethod
    def parse(cls, params) -> "ParamSuite | None":
        """
        Normalize various user-facing descriptions into a ParamSuite.

        Accepts:

        - ``None`` -- returns ``None``
        - ``ParamSuite`` -- returned as-is
        - ``dict[name -> array | shape]`` -- infer shape/dtype
        - ``iterable[ParamSpec]`` -- from_specs
        - ``iterable[str]`` -- scalar specs for each name

        Shapes may be ``()``, ``(k,)``, ``(m, n, ...)`` or an integer k
        (interpreted as ``(k,)``).
        """
        from collections.abc import Iterable as _Iterable

        if params is None:
            return None
        if isinstance(params, ParamSuite):
            return params
        # dict path: values can be arrays (sample), shapes, ints, ParamSpec, or None→scalar
        if isinstance(params, dict):
            specs: list[ParamSpec] = []
            for name, val in params.items():
                if isinstance(val, ParamSpec):
                    specs.append(val)
                    continue
                if hasattr(val, "shape"):
                    shape = tuple(val.shape)
                    dtype = getattr(val, "dtype", jnp.float32)
                    specs.append(ParamSpec(name, shape, dtype=dtype))
                    continue
                if val is None:
                    specs.append(ParamSpec(name, ()))
                    continue
                if isinstance(val, int):
                    specs.append(ParamSpec(name, (int(val),)))
                    continue
                if isinstance(val, tuple) and all(isinstance(n, int) for n in val):
                    specs.append(ParamSpec(name, tuple(int(n) for n in val)))
                    continue
                raise TypeError(f"ParamSuite.parse: unsupported dict value for '{name}': {type(val).__name__}")
            return cls.from_specs(*specs)
        # iterable path: ParamSpec or str
        if isinstance(params, _Iterable):
            items = list(params)
            specs: list[ParamSpec] = []
            for it in items:
                if isinstance(it, ParamSpec):
                    specs.append(it)
                elif isinstance(it, str):
                    specs.append(ParamSpec(it, ()))
                else:
                    raise TypeError(
                        f"ParamSuite.parse: iterable must contain ParamSpec or str (got {type(it).__name__})"
                    )
            return cls.from_specs(*specs)
        raise TypeError(f"ParamSuite.parse: unsupported params of type {type(params).__name__}")

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(aux)

    def coerce(
        self,
        params: dict,
        *,
        allow_scalar_for_scalar: bool = True,
        allow_scalar_to_len1: bool = True,
        cast_dtype: bool = True,
    ) -> dict[str, jnp.ndarray]:
        """
        Normalize a user param dict into JAX arrays matching this suite.

        Rules:
          - If spec.shape == (), accept Python scalars / 0-d arrays (if allowed).
          - If spec.shape == (1,), optionally accept a scalar and expand to (1,).
          - Otherwise, require exact shape; dtype is cast if `cast_dtype` is True.
        Returns a NEW dict with normalized arrays.
        """
        out = {}
        look = self._lookup
        missing = [k for k in look if k not in params]
        if missing:
            raise KeyError(f"Missing params: {missing}")

        for spec in self:
            name, shape, dtype = spec.name, spec.shape, spec.dtype
            val = params[name]

            arr = jnp.asarray(val, dtype=dtype if cast_dtype else None)

            # Handle scalar vs length-1 convenience
            if shape == ():  # true scalar
                # 0-d OK; (1,) OK if allow_scalar_for_scalar and allow squeezing
                if arr.shape == ():
                    pass
                elif arr.shape == (1,) and allow_scalar_for_scalar:
                    arr = jnp.reshape(arr, ())
                else:
                    raise TypeError(f"Param '{name}': expected scalar (), got {arr.shape}")
            elif shape == (1,):
                # accept scalar and expand to (1,) if asked
                if arr.shape == ():
                    if allow_scalar_to_len1:
                        arr = jnp.reshape(arr, (1,))
                    else:
                        raise TypeError(f"Param '{name}': expected (1,), got scalar ()")
                elif arr.shape == (1,):
                    pass
                else:
                    raise TypeError(f"Param '{name}': expected {shape}, got {arr.shape}")
            else:
                if arr.shape != shape:
                    raise TypeError(f"Param '{name}': expected {shape}, got {arr.shape}")

            # Final dtype enforcement
            if cast_dtype:
                arr = arr.astype(dtype)
            elif arr.dtype != dtype:
                raise TypeError(f"Param '{name}': expected dtype {dtype}, got {arr.dtype}")

            out[name] = arr
        return out

    def stamp(self, params_dict: dict) -> tuple:
        """
        Build a stable stamp for (shape, dtype) of each param in template order.
        Caller is expected to pass an already-coerced dict.
        """
        return tuple((spec.name, params_dict[spec.name].shape, params_dict[spec.name].dtype) for spec in self)
