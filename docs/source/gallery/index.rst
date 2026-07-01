:orphan:

Gallery
=======

SFI examples gallery.  Each script demonstrates a complete inference pipeline:
simulate (or load) data, infer forces and diffusion, and validate the results.
Use the **tags** below each thumbnail to filter by topic.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" data-sgtags='["synthetic", "overdamped", "linear", "1D"]' tooltip="This is the recommended starting point for new users: a complete, runnable Stochastic Force Inference (SFI) workflow on the simplest non-trivial system.  Every step below is executed when the documentation is built, so the numbers and figures on this page are real output.">

.. only:: html

  .. image:: /gallery/images/thumb/sphx_glr_ou_demo_thumb.png
    :alt:

  :doc:`/gallery/ou_demo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Getting started: end-to-end inference (Ornstein–Uhlenbeck)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" data-sgtags='["synthetic", "overdamped", "nonlinear", "non-equilibrium", "2D"]' tooltip="Infer the force field of a 2D nonlinear system with a stable limit cycle.  The radial force \dot{r} = r(1 - r^2) drives the system to a unit circle, while an angular drift \dot{\theta} = \omega generates rotation — a non-equilibrium steady state with non-zero probability currents.">

.. only:: html

  .. image:: /gallery/images/thumb/sphx_glr_limitcycle_demo_thumb.png
    :alt:

  :doc:`/gallery/limitcycle_demo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">2D limit cycle — nonlinear overdamped inference</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" data-sgtags='["synthetic", "overdamped", "nonlinear", "sparsity", "3D", "benchmark"]' tooltip="Infer the force field of a 3D Lorenz system from a single simulated trajectory using polynomial basis functions.">

.. only:: html

  .. image:: /gallery/images/thumb/sphx_glr_lorenz_demo_thumb.png
    :alt:

  :doc:`/gallery/lorenz_demo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Lorenz attractor — overdamped inference</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" data-sgtags='["synthetic", "overdamped", "nonlinear", "sparsity", "ecology", "custom-basis"]' tooltip="Recover the sparse interaction network of a 6-species Lotka–Volterra ecosystem from a heavily downsampled stochastic trajectory, using a custom polynomial-of-exponential basis and PASTIS model selection.">

.. only:: html

  .. image:: /gallery/images/thumb/sphx_glr_lotka_volterra_demo_thumb.png
    :alt:

  :doc:`/gallery/lotka_volterra_demo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Lotka–Volterra ecosystem — sparse network recovery</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" data-sgtags='["real-data", "overdamped", "experimental-workflow"]' tooltip="The recommended workflow for applying SFI to real experimental data. This demo loads a 2D optical-tweezer trajectory from a CSV file, infers the force and diffusion, sparsifies with PASTIS, and validates with a bootstrap trajectory.">

.. only:: html

  .. image:: /gallery/images/thumb/sphx_glr_experimental_workflow_demo_thumb.png
    :alt:

  :doc:`/gallery/experimental_workflow_demo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Experimental-data workflow template</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" data-sgtags='["synthetic", "overdamped", "2D", "custom-basis", "extras", "multi-experiment"]' tooltip="Build a hand-crafted basis with make_basis that reads experiment-specific metadata from extras.">

.. only:: html

  .. image:: /gallery/images/thumb/sphx_glr_custom_basis_demo_thumb.png
    :alt:

  :doc:`/gallery/custom_basis_demo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Custom basis with extras — multi-experiment traps</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" data-sgtags='["synthetic", "overdamped", "multiplicative-noise", "diffusion-field", "1D"]' tooltip="Recover a state-dependent diffusion field D(x) together with the force from a single overdamped trajectory.  The system is the classic Landauer blowtorch: a bistable particle whose bath is hotter on one side.  The temperature gradient redistributes the well occupancies away from the naive Boltzmann weights — an effect that only a joint inference of F(x) and D(x) can capture.">

.. only:: html

  .. image:: /gallery/images/thumb/sphx_glr_multiplicative_diffusion_demo_thumb.png
    :alt:

  :doc:`/gallery/multiplicative_diffusion_demo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Multiplicative noise — the Landauer blowtorch</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" data-sgtags='["synthetic", "overdamped", "multiplicative-noise", "diffusion-field", "anisotropic", "2D"]' tooltip="Recover a position-dependent, anisotropic diffusion tensor D(\mathbf{x}) from a single 2D trajectory — model-free.  A tracer in a ring trap experiences fluctuations whose radial component grows with distance from the center (think of a probe in a radially stretched gel, or near a topological defect in an active film): the local noise ellipse rotates with position.  A polynomial tensor basis recovers the full field, including its principal axes.">

.. only:: html

  .. image:: /gallery/images/thumb/sphx_glr_anisotropic_diffusion_demo_thumb.png
    :alt:

  :doc:`/gallery/anisotropic_diffusion_demo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Anisotropic diffusion tensor field</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" data-sgtags='["synthetic", "underdamped", "nonlinear", "1D"]' tooltip="Infer the force field of a 1D Van der Pol oscillator from position-only data.  Velocities are reconstructed internally by the ULI estimators.">

.. only:: html

  .. image:: /gallery/images/thumb/sphx_glr_van_der_pol_demo_thumb.png
    :alt:

  :doc:`/gallery/van_der_pol_demo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Van der Pol oscillator — underdamped inference</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" data-sgtags='["synthetic", "underdamped", "multiplicative-noise", "diffusion-field", "1D"]' tooltip="Recover a velocity-dependent diffusion field D(v) for an inertial particle, from positions only.  Many active and driven systems fluctuate harder the faster they move — motility noise, flight-force fluctuations, turbulent drag.  Underdamped SFI reconstructs the unobserved velocity and infers F(x,v) and D(x,v) jointly; here the noise amplitude doubles within the explored speed range and is recovered model-free.">

.. only:: html

  .. image:: /gallery/images/thumb/sphx_glr_velocity_dependent_noise_demo_thumb.png
    :alt:

  :doc:`/gallery/velocity_dependent_noise_demo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Velocity-dependent noise — underdamped multiplicative diffusion</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" data-sgtags='["synthetic", "underdamped", "multi-particle", "2D", "interactions", "per-particle"]' tooltip="Infer individual, anisotropic home ranges for a colony of interacting agents that also share a landscape — meandering river valleys read off a known topographic map — from corrupted, positions-only data.">

.. only:: html

  .. image:: /gallery/images/thumb/sphx_glr_home_range_demo_thumb.png
    :alt:

  :doc:`/gallery/home_range_demo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Home ranges in a shared landscape, from noisy gappy data</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" data-sgtags='["synthetic", "overdamped", "linear", "1D", "time-dependent"]' tooltip="Infer a force law that depends on a known time-dependent protocol — here a trap whose stiffness is switched between two values — from a single 1D trajectory.">

.. only:: html

  .. image:: /gallery/images/thumb/sphx_glr_time_dependent_forcing_demo_thumb.png
    :alt:

  :doc:`/gallery/time_dependent_forcing_demo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Time-dependent forcing — protocols as extras</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" data-sgtags='["synthetic", "overdamped", "linear", "1D", "time-dependent"]' tooltip="Recover an unknown, time-varying force law from trajectories alone, by expanding time in a Fourier dictionary.">

.. only:: html

  .. image:: /gallery/images/thumb/sphx_glr_time_fourier_demo_thumb.png
    :alt:

  :doc:`/gallery/time_fourier_demo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Learning a time-dependent force field — time-Fourier basis</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" data-sgtags='["diagnostics", "overdamped", "linear", "1D", "synthetic"]' tooltip="Once a fit is in hand, the next question is should we trust it? The SFI.diagnostics submodule answers that by recomputing standardised innovations from the fitted state function and the inferred constant diffusion, then running a battery of statistical tests.">

.. only:: html

  .. image:: /gallery/images/thumb/sphx_glr_diagnostics_demo_thumb.png
    :alt:

  :doc:`/gallery/diagnostics_demo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Diagnostics — assessing fit quality</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" data-sgtags='["diagnostics", "overdamped", "underdamped", "1D", "2D", "non-equilibrium", "synthetic"]' tooltip="Before fitting a force, you must choose an engine: OverdampedLangevinInference assumes first-order (overdamped) dynamics, UnderdampedLangevinInference assumes second-order (inertial) dynamics.  SFI.classify_dynamics reads the data and decides which regime you are in — directly from positions, with no velocities required.">

.. only:: html

  .. image:: /gallery/images/thumb/sphx_glr_dynamics_order_demo_thumb.png
    :alt:

  :doc:`/gallery/dynamics_order_demo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Overdamped or underdamped? Classifying dynamics from data</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" data-sgtags='["synthetic", "overdamped", "multi-particle", "linear", "interactions"]' tooltip="Infer pairwise interaction forces in a system of aligning active Brownian particles (ABPs) using generic pair-interaction building blocks from SFI.bases.pairs.">

.. only:: html

  .. image:: /gallery/images/thumb/sphx_glr_abp_align_demo_thumb.png
    :alt:

  :doc:`/gallery/abp_align_demo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Aligning active Brownian particles — generic pairs API</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" data-sgtags='["synthetic", "overdamped", "multi-particle", "linear", "interactions", "large-scale", "nonreciprocal"]' tooltip="Infer three pairwise interaction kernels — repulsion, alignment, and pursuit — in a system of 3 000 active Brownian particles whose interactions are nonreciprocal and vision-gated.">

.. only:: html

  .. image:: /gallery/images/thumb/sphx_glr_abp_nonreciprocal_demo_thumb.png
    :alt:

  :doc:`/gallery/abp_nonreciprocal_demo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Nonreciprocal ABPs at large scale — 3 000 particles</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" data-sgtags='["synthetic", "overdamped", "multi-particle", "linear", "spde", "experimental", "pastis", "interactions"]' tooltip="   Uses the experimental SPDE toolbox — see /spde/index.">

.. only:: html

  .. image:: /gallery/images/thumb/sphx_glr_abp_to_spde_demo_thumb.png
    :alt:

  :doc:`/gallery/abp_to_spde_demo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Discovering Toner–Tu hydrodynamics from agent-based flocking</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" data-sgtags='["synthetic", "overdamped", "SPDE", "experimental", "reaction-diffusion", "2D", "sparsification", "multi-experiment"]' tooltip="   Uses the experimental SPDE toolbox — see /spde/index.">

.. only:: html

  .. image:: /gallery/images/thumb/sphx_glr_gray_scott_demo_thumb.png
    :alt:

  :doc:`/gallery/gray_scott_demo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Gray-Scott reaction-diffusion: SPDE inference</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /gallery/ou_demo
   /gallery/limitcycle_demo
   /gallery/lorenz_demo
   /gallery/lotka_volterra_demo
   /gallery/experimental_workflow_demo
   /gallery/custom_basis_demo
   /gallery/multiplicative_diffusion_demo
   /gallery/anisotropic_diffusion_demo
   /gallery/van_der_pol_demo
   /gallery/velocity_dependent_noise_demo
   /gallery/home_range_demo
   /gallery/time_dependent_forcing_demo
   /gallery/time_fourier_demo
   /gallery/diagnostics_demo
   /gallery/dynamics_order_demo
   /gallery/abp_align_demo
   /gallery/abp_nonreciprocal_demo
   /gallery/abp_to_spde_demo
   /gallery/gray_scott_demo

Advanced
========

These examples push the parametric estimators further: neural-network
force fields, multi-experiment fitting with shared parameters, and
underdamped multi-particle systems.  Start with the main gallery above
if you are new to SFI; the regime table in the
:doc:`Running-inference guide </inference/user_guide>` tells you when
these tools are the right choice.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" data-sgtags='["synthetic", "overdamped", "nonlinear", "neural-network", "2D"]' tooltip="Infer the 2D force field of the Müller-Brown potential energy surface with a neural network (multi-layer perceptron), built entirely from SFI&#x27;s compositional basis operations, and compare with polynomial-basis inference.">

.. only:: html

  .. image:: /gallery/advanced/images/thumb/sphx_glr_nn_force_demo_thumb.png
    :alt:

  :doc:`/gallery/advanced/nn_force_demo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Neural-network force field — Müller-Brown potential</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" data-sgtags='["synthetic", "overdamped", "multi-particle", "multi-experiment", "nonlinear", "interactions"]' tooltip="Infer a shared interaction model from multiple independent ABP experiments that differ in both particle number and box size.">

.. only:: html

  .. image:: /gallery/advanced/images/thumb/sphx_glr_multi_experiment_demo_thumb.png
    :alt:

  :doc:`/gallery/advanced/multi_experiment_demo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Multi-experiment ABP inference</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" data-sgtags='["synthetic", "underdamped", "multi-particle", "interactions", "solver-comparison"]' tooltip="End-to-end demonstration of underdamped multi-particle parametric inference on a 3D flocking-class system with both position and velocity pairwise coupling — a translation-invariant flock held together by pairwise cohesion plus velocity alignment:">

.. only:: html

  .. image:: /gallery/advanced/images/thumb/sphx_glr_flocking_3d_demo_thumb.png
    :alt:

  :doc:`/gallery/advanced/flocking_3d_demo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">3D flocking — underdamped multi-particle inference</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /gallery/advanced/nn_force_demo
   /gallery/advanced/multi_experiment_demo
   /gallery/advanced/flocking_3d_demo



.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
