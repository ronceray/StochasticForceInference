"""Basis façade: dictionary of deterministic functions."""

from .psf import PSF
from .stateexpr import StateExpr


# ============================================================================
#  BASIS  – deterministic dictionary
# ============================================================================
class Basis(StateExpr):
    """Deterministic dictionary façade (no parameters)."""

    def __init__(self, root):
        if root.param_suite is not None:
            raise ValueError("Basis root must be parameter-free")
        super().__init__(root)

    def __call__(self, x, *, v=None, mask=None, extras=None):
        """Evaluate the basis on a single or batched input.

        Parameters
        ----------
        x : array
            If `particles_input=False`: shape `batch · dim`.
            If `particles_input=True`:  shape `batch · P · dim`.
            `batch` can be empty (single input).
        v : array | None
            Optional velocity matching `x.shape` (required if `needs_v=True`).
        mask : array | scalar | None
            Broadcasts to the prefix of `x` **including** the particle axis.
            Boolean or numeric masks are accepted.
        extras : dict | None
            Optional per-batch metadata. Presence is enforced according to the
            aggregated node requirements; values must broadcast over **batch only**.

        Returns
        -------
        array
            Shape `batch · [P]^pdepth · (dim)^rank · n_features`.
        """
        self._validate_extras_presence(extras)
        return self._caller(x, v, mask, extras, None)

    # convenience
    @property
    def labels(self):
        _, labs, _ = self.root.flatten()
        return labs

    def __len__(self):
        return self.n_features

    # Upgrade-aware dispatch: if a derived node carries parameters,
    # the result is a PSF (e.g. Basis * PSF, PSF - Basis, …).
    def _with_node(self, new_root):
        if new_root.param_suite is not None:
            return PSF(new_root)
        return Basis(new_root)

    # factory → linear-coeff PSF
    def to_psf(self, coeff_key: str = "coeff", drop_features=True):
        """
        Return a **parametric state function** whose value is a linear combination
        of this Basis' features:

            F(x; θ) = Σ_j θ_j · f_j(x)

            Note that use cases are rare within SFI, since the PSF's features axis is typically
            used for nonlinearities and/or vector/tensor components.  But this can be useful for
            quick prototyping of linear models, benchmark comparisons of linear vs nonlinear
            solvers, or as a building block for more complex PSFs.

        Parameters
        ----------
        coeff_key : str
            Key name for the coefficient vector in the parameter dict.
        drop_features: bool
            Whether to remove the trailing size-1 feature axis (default True).

        Notes
        -----
        The resulting `PSF` shares the same spatial contract (rank/dim/pdepth,
        particles_input) as this `Basis`, and does not have a features axis.
        """
        from .nodes import CoeffNode

        return PSF(CoeffNode(self.root, coeff_key=coeff_key), drop_features=drop_features)
