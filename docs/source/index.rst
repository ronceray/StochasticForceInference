.. _intro:

SFI documentation
=================

**Stochastic Force Inference** (SFI) is a JAX-based Python library for
inferring force and diffusion fields from stochastic trajectory
data. Its mathematical backbone is the same-name method introduced in
`Frishman & Ronceray (2020) <https://doi.org/10.1103/PhysRevX.10.021009>`_,
with performance improvements and extensions including sparse model selection
(PASTIS), underdamped inference, noise-robust parametric estimation,
interacting-particle systems, and (experimentally) spatially-extended
SPDE systems.  See :doc:`whatsnew` for what changed in v2.

Given a set of recorded positions — tracked particles, migrating cells,
swimming microorganisms, colloidal probes, or any system whose state evolves
stochastically — SFI reconstructs the underlying dynamical equations without
requiring a prior model.

The physical picture
--------------------

Many systems in physics, biophysics, chemistry, ecology, and more can be
described by Langevin dynamics, an equation governing the evolution of a
*state variable* :math:`\mathbf{x}_t` (position, orientation, concentration
field, …) driven by a deterministic drift and thermal or active noise:

.. math::

   \mathrm{d}\mathbf{x}_t
   = \mathbf{F}(\mathbf{x}_t)\,\mathrm{d}t
   + \sqrt{2\,\mathbf{D}(\mathbf{x}_t)}\;\mathrm{d}W_t

where :math:`\mathbf{F}(\mathbf{x})` is the drift (force) field,
:math:`\mathbf{D}(\mathbf{x})` the diffusion tensor, and
:math:`\mathrm{d}W_t` a Wiener increment.  SFI estimates both
:math:`\mathbf{F}` and :math:`\mathbf{D}` directly from trajectory data,
using a principled projection onto user-chosen basis functions.


What makes SFI different?
-------------------------

**Robust to data imperfections.**
SFI is designed for real experimental data: it handles coarse temporal sampling,
missing frames, and high measurement noise.  Two first-class estimator
families share one API — fast closed-form **linear estimators**, and
**parametric estimators** that model measurement noise and finite
sampling explicitly (see :ref:`choosing-an-estimator`).  The
built-in trajectory toolkit provides standardised degradation models for
quantifying sensitivity to these imperfections.

**No initial guess required.**
The linear estimators solve an exact regression — no iterative
optimisation, no loss landscape to navigate, no sensitivity to initial
conditions — and they hand the parametric estimators a reliable
starting point when you need the robust path.

**Automatic model selection.**
The PASTIS information criterion tells you *which* basis terms the data
actually supports, preventing both underfitting and overfitting.  A sparse
Pareto-beam search finds the optimal support even in high-dimensional
feature spaces.

**Built on JAX.**
All core operations are JIT-compiled and auto-differentiated.  The same
code runs transparently on CPU and GPU, and scales to large particle
systems through adaptive chunking.

**Composable state functions.**
Forces, diffusion tensors, and basis dictionaries are first-class objects
with explicit shape contracts and a small algebra (``+``, ``*``, ``&``).
You can mix polynomial, pair-interaction, and neural-network components
freely.

**From inference to simulation and back.**
Inferred models are immediately usable as Langevin simulators, enabling
bootstrapped validation and synthetic-data benchmarks without any glue
code.


A minimal example
-----------------

.. code-block:: python

    import jax.numpy as jnp
    from jax import random

    from SFI import OverdampedLangevinInference
    from SFI.bases import X, monomials_up_to
    from SFI.langevin import OverdampedProcess

    # 1. Simulate — a 2D Ornstein–Uhlenbeck process
    F_true = -1.33 * X(dim=2)
    proc = OverdampedProcess(F_true, D=jnp.eye(2))
    proc.initialize(jnp.zeros(2))
    coll = proc.simulate(dt=0.1, Nsteps=200, key=random.PRNGKey(0))

    # 2. Infer — linear force regression + PASTIS selection + error estimation
    inf = OverdampedLangevinInference(coll)
    inf.compute_diffusion_constant()
    B = monomials_up_to(order=3, dim=2, rank='vector')
    inf.infer_force_linear(B)
    inf.sparsify_force()
    inf.compute_force_error()
    inf.print_report()

    # 3. Validate — compare to ground truth
    inf.compare_to_exact(model_exact=proc)

Intended audience
-----------------

SFI is designed for **experimentalists, computational scientists and theorists working with experimental data** in
any field where systems evolve stochastically and trajectory data are
available:

- biophysics: single-molecule tracking (optical traps, SPT), cell
  migration, tissue dynamics, intracellular traffic...
- stochastic thermodynamics: dissipation, currents and broken detailed
  balance measured from trajectories,
- active matter: self-propelled colloids, bacteria, active nematics,
- soft matter: colloidal probes in complex fluids, polymer dynamics,
- coarse-grained molecular simulations and enhanced-sampling methods,
- climate and geophysics: particle tracking in turbulent flows,
- quantitative ecology: microbial populations, animal behavior, 
- any other field where the observable dynamics can be modelled as
  overdamped or underdamped Langevin equations.

SFI does not require a pre-existing model: it learns the equations of
motion directly from data and a broad library of candidate functions.


How to cite
-----------

If you use SFI in your research, please cite the **software** itself (via its
archived Zenodo release; a machine-readable record is provided in
``CITATION.cff``) together with the method paper(s) relevant to the features you
use.

**Software (Zenodo archive):**

   Ronceray, P., *Stochastic Force Inference*, v2.0.1, Zenodo (2026).
   `DOI: 10.5281/zenodo.XXXXXXX <https://doi.org/10.5281/zenodo.XXXXXXX>`_
   (placeholder — the DOI is minted when the first GitHub release is archived to
   Zenodo).

**Method papers.** Please also cite the original method paper:

   Frishman, A. & Ronceray, P., *Learning force fields from stochastic
   trajectories*, **Physical Review X** 10, 021009 (2020).
   `DOI: 10.1103/PhysRevX.10.021009 <https://doi.org/10.1103/PhysRevX.10.021009>`_

This paper also introduces the **entropy-production estimator**
(:meth:`~SFI.inference.OverdampedLangevinInference.compute_entropy_production`) —
if you use it, the same citation covers it.

If you use the **underdamped (ULI)** inference mode, please also cite:

   Brückner, D. B., Ronceray, P. & Broedersz, C. P.,
   *Inferring the dynamics of underdamped stochastic systems*,
   **Physical Review Letters** 125, 058103 (2020).
   `DOI: 10.1103/PhysRevLett.125.058103 <https://doi.org/10.1103/PhysRevLett.125.058103>`_

If you use the overdamped **trapeze (trapezoidal) integration scheme**
(``G_mode="trapeze"``, part of the default settings of
:meth:`~SFI.inference.OverdampedLangevinInference.infer_force_linear`), please also cite:

   Amri, S., Zhang, Y., Gerardos, A., Sykes, C. & Ronceray, P.,
   *Inferring geometrical dynamics of cell nucleus translocation*,
   **Physical Review Research** 6, 043030 (2024).
   `DOI: 10.1103/PhysRevResearch.6.043030 <https://doi.org/10.1103/PhysRevResearch.6.043030>`_

If you use the **PASTIS** (Parsimonious Stochastic Inference) sparse
selection criterion, please also cite:

   Gerardos, A. & Ronceray, P.,
   *Principled model selection for stochastic dynamics*,
   **Physical Review Letters** 135, 167401 (2025).
   `DOI: 10.1103/ltdt-hvh7 <https://doi.org/10.1103/ltdt-hvh7>`_

If you use the parametric estimators (:meth:`~SFI.inference.OverdampedLangevinInference.infer_force` and :meth:`~SFI.inference.OverdampedLangevinInference.infer_diffusion`), please use the following placeholder citation and contact me before submitting your work:

   Ronceray, P.,
   *Nonlinear Stochastic Force Inference*,
   **In preparation** (2026).

License and maintainer
----------------------

SFI is released under the **MIT licence**.

*Maintainer:* Pierre Ronceray (pierre.ronceray@univ-amu.fr).
Contributions, requests for additional features and bug reports are welcome — see the :doc:`install` page for instructions.


How to read these docs
----------------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Section
     - What you'll find
   * - :doc:`start_here`
     - Quick scope check and the right entry point for experimental,
       underdamped, interacting, spatial, and nonlinear problems
   * - :doc:`install`
     - Installation, dependencies, optional GPU setup
   * - :doc:`gallery/ou_demo`
     - End-to-end synthetic walkthrough on an Ornstein-Uhlenbeck process
   * - :doc:`whatsnew`
     - The two estimator families, lineage, and breaking changes in v2
   * - **User guides**
     - Practical how-to per task: trajectory I/O, running inference,
       measurement noise & coarse sampling, underdamped systems, sparse
       selection, diagnostics, bases and models, particle systems,
       simulation
   * - :doc:`gallery/index`
     - Ready-to-run examples across real data, underdamped, interacting,
       and spatial problems
   * - :doc:`physics_reference`
     - All mathematical formulas collected in one place
   * - **API reference**
     - Full function signatures and docstrings


.. seealso::

   Already have data? Start with :doc:`start_here`.
   If you want a synthetic walkthrough first, continue with
   :doc:`install` and then the :doc:`gallery/ou_demo` tutorial.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   start_here
   install
   Tutorial <gallery/ou_demo>
   whatsnew

.. toctree::
   :maxdepth: 2
   :caption: User guides

   trajectory/user_guide
   inference/user_guide
   inference/noise_and_sampling
   inference/underdamped
   inference/sparsity_guide
   diagnostics
   bases/user_guide
   statefunc/user_guide
   particles/user_guide
   langevin/user_guide
   spde/index

.. toctree::
   :maxdepth: 2
   :caption: Gallery

   gallery/index

.. toctree::
   :maxdepth: 2
   :caption: Concepts & theory

   inference/parametric_concept
   inference/parametric_algorithm
   inference/sparsity_theory
   inference/stochastic_thermodynamics
   statefunc/overview

.. toctree::
   :maxdepth: 1
   :caption: Reference

   physics_reference
   api_frontend
   inference/reference
   statefunc/reference
   bases/reference
   spde/reference
   trajectory/reference
   trajectory/data_formats
   langevin/reference
   glossary

.. toctree::
   :maxdepth: 1
   :caption: Development

   dev_notes
   agent_playbooks/index
