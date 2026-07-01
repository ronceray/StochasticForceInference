"""
Noise models for Langevin simulators.

This module provides a hierarchy of noise models that go beyond the
simple diagonal diffusion ``D = σ I``.  In particular, it supports
**conserved noise** needed by SPDE models such as Active Model B+, where
the noise takes the form ``∇·(σ η)`` and preserves the spatial integral
of the field.

Class hierarchy
---------------
- :class:`NoiseModel` — abstract base;
- :class:`WhiteNoise` — i.i.d. per-site Gaussian (recovers current behaviour);
- :class:`ConservedNoise` — ``sqrt(-σ² ∇²) ξ`` via FFT on a periodic grid;
- :class:`CompositeNoise` — different noise models on different field components.

Usage
-----
Pass a ``NoiseModel`` instance as ``D=`` to :class:`~SFI.langevin.OverdampedProcess`
(or any ``LangevinBase`` subclass).  The simulator detects the noise model and
delegates noise generation to it instead of the traditional ``sqrt(2D)·ξ`` path.

.. code-block:: python

   from SFI.langevin.noise import ConservedNoise

   noise = ConservedNoise(sigma=0.3, grid_shape=(64, 64), dx=1.0)
   proc  = OverdampedProcess(BASIS, D=noise)
   proc.set_params(theta_F=theta)
   proc.set_extras(extras_global=box_extras)
   proc.initialize(X0)
   coll = proc.simulate(dt=0.02, Nsteps=3000, key=key, oversampling=4)
"""

from __future__ import annotations

import abc
from typing import List, Sequence, Tuple, Union

import jax.numpy as jnp
from jax import random

Array = jnp.ndarray


# ============================================================================
# Abstract base
# ============================================================================


class NoiseModel(abc.ABC):
    """Abstract base for noise models used by Langevin simulators.

    Subclasses must implement :meth:`sample` and :meth:`effective_D_per_site`.
    The simulator calls ``sample(key, x, extras)`` once per Euler–Maruyama
    substep to obtain the noise increment (already scaled by ``sqrt(2)`` so
    that the step becomes ``x += dt*F + sqrt(dt)*sample(key, x, extras)``).

    Parameters
    ----------
    n_fields : int
        Number of field components per grid site (= ``dim`` in the force
        contract).  E.g. 1 for a scalar field, 2 for a 2-component system.
    """

    def __init__(self, *, n_fields: int) -> None:
        self._n_fields = n_fields

    @property
    def n_fields(self) -> int:
        """Number of field components per site."""
        return self._n_fields

    # Alias for the force-contract convention
    @property
    def dim(self) -> int:
        return self._n_fields

    @abc.abstractmethod
    def sample(self, key: Array, x: Array, extras: dict) -> Array:
        r"""Draw one noise increment.

        Parameters
        ----------
        key : PRNG key
        x : Array, shape ``(P, d)`` or ``(d,)``
            Current state (used only for shape; not accessed by white/conserved
            noise, but may be needed by multiplicative noise subclasses).
        extras : dict
            Process extras (contains grid geometry, etc.).

        Returns
        -------
        Array, same shape as *x*
            Noise increment already multiplied by ``sqrt(2)`` so the
            integrator applies ``x += dt*F + sqrt(dt) * sample(...)``.
        """
        ...

    @abc.abstractmethod
    def effective_D_per_site(self, extras: dict) -> Array:
        r"""Return an approximate per-site diffusion matrix ``(d, d)``.

        This is used by the inference pipeline as a *pragmatic approximation*
        when the noise is not white-in-space.  For ``WhiteNoise(σ)`` this
        returns exactly ``σ·I``.  For ``ConservedNoise`` it returns the
        spatially-averaged effective variance per site.

        Returns
        -------
        Array, shape ``(d, d)``
        """
        ...

    @property
    def noise_kind(self) -> str:
        """Short string tag for the noise type."""
        return self.__class__.__name__


# ============================================================================
# White noise  (recovers current constant-scalar behaviour)
# ============================================================================


class WhiteNoise(NoiseModel):
    r"""Spatially uncorrelated Gaussian noise: ``B = sqrt(2σ) I``.

    Each grid site receives i.i.d. ``N(0, 2σ dt)`` noise per component.
    This recovers the current ``D = σ`` (scalar constant) behaviour.

    Parameters
    ----------
    sigma : float
        Scalar diffusion coefficient (the *D* in ``dx = F dt + sqrt(2D) dW``).
    n_fields : int
        Number of field components per site.
    """

    def __init__(self, sigma: float, *, n_fields: int = 1) -> None:
        super().__init__(n_fields=n_fields)
        if sigma < 0:
            raise ValueError(f"sigma must be non-negative, got {sigma}")
        self._sigma = float(sigma)
        # Precompute sqrt(2σ) for efficiency
        self._sqrt2sigma = float(jnp.sqrt(2.0 * sigma))

    @property
    def sigma(self) -> float:
        return self._sigma

    def sample(self, key: Array, x: Array, extras: dict) -> Array:
        xi = random.normal(key, shape=x.shape)
        return self._sqrt2sigma * xi

    def effective_D_per_site(self, extras: dict) -> Array:
        return self._sigma * jnp.eye(self._n_fields)

    def __repr__(self) -> str:
        return f"WhiteNoise(sigma={self._sigma}, n_fields={self._n_fields})"


# ============================================================================
# Conserved noise  (sqrt(-σ² ∇²) via FFT on periodic grids)
# ============================================================================


def _build_freq_amplitudes(
    grid_shape: Sequence[int],
    dx: Union[float, Sequence[float]],
    ndim: int,
) -> Array:
    r"""Build the Fourier-space multiplier ``|k|`` for conserved noise.

    For a periodic grid with spacing *dx*, the wavenumbers along axis α are

    .. math::

        k_\alpha = \frac{2\pi\,n_\alpha}{N_\alpha\,\Delta x_\alpha}

    and the multiplier is ``|k| = sqrt(sum_α k_α²)``, which corresponds to
    the operator ``sqrt(-∇²)`` in Fourier space.

    We use ``rfft`` along the last spatial axis, so the returned array has
    shape ``(N_0, N_1, ..., N_{d-2}, N_{d-1}//2+1)`` for an *ndim*-D grid.

    The ``k = 0`` mode is set to zero (conserved noise has zero mean).

    Returns
    -------
    Array, real, shape matching rfft output
        ``|k|`` on the half-complex grid.
    """
    grid_shape = tuple(int(n) for n in grid_shape)
    if isinstance(dx, (int, float)):
        dx_arr = [float(dx)] * ndim
    else:
        dx_arr = [float(d) for d in dx]

    # Build k² = sum_α k_α²
    k_sq = jnp.zeros(grid_shape[:-1] + (grid_shape[-1] // 2 + 1,))
    for axis in range(ndim):
        N = grid_shape[axis]
        if axis < ndim - 1:
            # Full-size axis: use fftfreq
            freq = jnp.fft.fftfreq(N, d=dx_arr[axis])  # shape (N,)
        else:
            # Last axis: rfft convention → only non-negative frequencies
            freq = jnp.fft.rfftfreq(N, d=dx_arr[axis])  # shape (N//2+1,)

        k_alpha = 2 * jnp.pi * freq  # angular wavenumber

        # Broadcast to the full rfft grid shape
        shape = [1] * ndim
        if axis < ndim - 1:
            shape[axis] = N
        else:
            shape[axis] = grid_shape[-1] // 2 + 1
        k_alpha = k_alpha.reshape(shape)

        k_sq = k_sq + k_alpha**2

    # |k| = sqrt(k²), with k=0 mode zeroed out
    k_abs = jnp.sqrt(k_sq)
    return k_abs


class ConservedNoise(NoiseModel):
    r"""Conserved (divergence-form) noise on a periodic square grid.

    Implements noise of the form

    .. math::

        \eta(x, t) = \nabla \cdot \bigl(\sigma\, \vec{\Lambda}(x,t)\bigr)

    where :math:`\vec{\Lambda}` is spatiotemporal white vector noise.
    In Fourier space this is equivalent to multiplying each mode by
    :math:`|k|`:

    .. math::

        \hat{\eta}_k = \sigma\,|k|\,\hat{\xi}_k

    This noise **conserves the spatial average** of the field
    (:math:`\sum_i \phi_i` is a constant of the noise process), as
    required by Model B / Active Model B+ dynamics.

    Parameters
    ----------
    sigma : float
        Noise amplitude (the :math:`\sigma` in the equations above).
        This is the *continuum* amplitude; the grid discretisation is
        handled internally.
    grid_shape : sequence of int
        Grid dimensions ``(Nx, Ny, ...)`` — must match the simulation grid.
    dx : float or sequence of float
        Grid spacing (uniform or per-axis).
    n_fields : int
        Number of field components per site.

    Notes
    -----
    The ``sample`` method uses real FFT (``rfftn`` / ``irfftn``) for
    efficiency.  It draws white noise in real space, transforms to
    Fourier space, multiplies by :math:`\sigma\,|k|\,\sqrt{2/\Delta V}`
    (where :math:`\Delta V = \prod \Delta x_\alpha` is the cell volume),
    and transforms back.

    The factor :math:`1/\sqrt{\Delta V}` provides the correct continuum
    limit: the noise covariance
    :math:`\langle\eta_i\,\eta_j\rangle = -\sigma^2 \nabla^2 \delta_{ij} / \Delta V`
    is independent of grid resolution when *sigma* is held fixed.
    """

    def __init__(
        self,
        sigma: float,
        *,
        grid_shape: Sequence[int],
        dx: Union[float, Sequence[float]] = 1.0,
        n_fields: int = 1,
    ) -> None:
        super().__init__(n_fields=n_fields)
        if sigma < 0:
            raise ValueError(f"sigma must be non-negative, got {sigma}")
        self._sigma = float(sigma)
        self._grid_shape = tuple(int(n) for n in grid_shape)
        self._ndim = len(self._grid_shape)

        if isinstance(dx, (int, float)):
            self._dx = tuple([float(dx)] * self._ndim)
        else:
            self._dx = tuple(float(d) for d in dx)

        self._P = 1
        for n in self._grid_shape:
            self._P *= n

        # Cell volume for continuum-limit normalisation
        dV = 1.0
        for d in self._dx:
            dV *= d
        self._dV = dV

        # Precompute |k| array for rfft
        self._k_abs = _build_freq_amplitudes(self._grid_shape, self._dx, self._ndim)

        # Combined multiplier: sigma * |k| * sqrt(2 / dV)
        # The sqrt(2) enters because the integrator step is
        #   x += sqrt(dt) * sample(...)
        # and we need <sample_i sample_j> = 2 * D_eff * delta_{ij}
        # where D_eff is the noise operator.
        self._multiplier = self._sigma * self._k_abs * jnp.sqrt(2.0 / self._dV)

        # Effective per-site D for inference approximation:
        # Var(noise_i) = sigma² * <|k|²> / dV
        # where <|k|²> = (1/N) sum_k |k|² (excluding k=0)
        k_sq_mean = float(jnp.mean(self._k_abs**2))
        self._D_eff = float(self._sigma**2 * k_sq_mean / self._dV)

    @property
    def sigma(self) -> float:
        return self._sigma

    @property
    def grid_shape(self) -> Tuple[int, ...]:
        return self._grid_shape

    def sample(self, key: Array, x: Array, extras: dict) -> Array:
        r"""Draw one conserved-noise increment.

        Steps:
        1. Draw white noise ξ ~ N(0,1) on the grid, shape (Nx, Ny, ..., d)
        2. rFFT along spatial axes
        3. Multiply by :math:`\sigma\,|k|\,\sqrt{2/\Delta V}` (the ``_multiplier``)
        4. iFFT back to real space
        5. Flatten back to (P, d)

        The k=0 mode of ``_multiplier`` is zero, so sum_i η_i = 0 exactly.
        """
        d = self._n_fields
        grid_d = self._grid_shape + (d,)

        # Draw white noise on the grid
        xi = random.normal(key, shape=grid_d)

        # FFT axes = spatial only (not the field axis)
        fft_axes = tuple(range(self._ndim))

        # Forward real FFT along spatial axes
        xi_hat = jnp.fft.rfftn(xi, axes=fft_axes)

        # Multiply by the precomputed |k|*sigma*sqrt(2/dV)
        # multiplier shape: (Nx, ..., Ny//2+1) — broadcast over field axis d
        eta_hat = xi_hat * self._multiplier[..., None]

        # Inverse real FFT
        eta = jnp.fft.irfftn(eta_hat, s=self._grid_shape, axes=fft_axes)

        # Flatten spatial axes: (Nx, Ny, ..., d) → (P, d)
        return eta.reshape(self._P, d)

    def effective_D_per_site(self, extras: dict) -> Array:
        return self._D_eff * jnp.eye(self._n_fields)

    def __repr__(self) -> str:
        return (
            f"ConservedNoise(sigma={self._sigma}, "
            f"grid_shape={self._grid_shape}, dx={self._dx}, "
            f"n_fields={self._n_fields})"
        )


# ============================================================================
# Composite noise (different models on different field components)
# ============================================================================


class CompositeNoise(NoiseModel):
    r"""Apply different noise models to different field components.

    Useful when some fields have conserved dynamics (e.g. concentration)
    and others have non-conserved dynamics (e.g. velocity).

    Parameters
    ----------
    components : list of ``(NoiseModel, field_indices)`` pairs
        Each element specifies a noise model and the field indices it
        applies to.  ``field_indices`` is a list of ints.  Together the
        indices must cover ``range(n_fields)`` exactly once.

    Example
    -------
    >>> conserved = ConservedNoise(sigma=0.3, grid_shape=(64, 64), n_fields=1)
    >>> white = WhiteNoise(sigma=0.1, n_fields=2)
    >>> composite = CompositeNoise(
    ...     components=[(conserved, [0]), (white, [1, 2])],
    ...     n_fields=3,
    ... )
    """

    def __init__(
        self,
        *,
        components: List[Tuple[NoiseModel, List[int]]],
        n_fields: int,
    ) -> None:
        super().__init__(n_fields=n_fields)

        # Validate coverage
        all_indices: set[int] = set()
        for model, indices in components:
            idx_set = set(indices)
            if idx_set & all_indices:
                raise ValueError(f"Overlapping field indices: {idx_set & all_indices}")
            all_indices |= idx_set
        if all_indices != set(range(n_fields)):
            raise ValueError(f"Field indices must cover range({n_fields}), got {sorted(all_indices)}")

        self._components = components

    def sample(self, key: Array, x: Array, extras: dict) -> Array:
        result = jnp.zeros_like(x)
        for i, (model, indices) in enumerate(self._components):
            key, sub = random.split(key)
            # Extract the sub-state for this component's fields
            idx = jnp.array(indices)
            x_sub = x[..., idx]
            noise_sub = model.sample(sub, x_sub, extras)
            # Place back into the full field
            result = result.at[..., idx].set(noise_sub)
        return result

    def effective_D_per_site(self, extras: dict) -> Array:
        D = jnp.zeros((self._n_fields, self._n_fields))
        for model, indices in self._components:
            D_sub = model.effective_D_per_site(extras)
            for i_local, i_global in enumerate(indices):
                for j_local, j_global in enumerate(indices):
                    D = D.at[i_global, j_global].set(D_sub[i_local, j_local])
        return D

    def __repr__(self) -> str:
        parts = [f"({m!r}, {idx})" for m, idx in self._components]
        return f"CompositeNoise(components=[{', '.join(parts)}], n_fields={self._n_fields})"
