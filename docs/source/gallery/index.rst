:orphan:

Gallery
=======

SFI examples gallery.  Each script demonstrates a complete inference pipeline:
simulate (or load) data, infer forces and diffusion, and validate the results.
Use the **tag bar** above the thumbnails to filter examples by topic.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" data-sgtags='["synthetic", "overdamped", "linear", "1D"]' tooltip="The recommended starting point: the complete Stochastic Force Inference (SFI) workflow on the simplest non-trivial system, a particle in a harmonic trap.  Its position obeys the overdamped Langevin (Ornstein–Uhlenbeck) equation">

.. only:: html

  .. image:: /gallery/images/thumb/sphx_glr_ou_demo_thumb.png
    :alt:

  :doc:`/gallery/ou_demo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Getting started: end-to-end inference (Ornstein–Uhlenbeck)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" data-sgtags='["synthetic", "overdamped", "nonlinear", "non-equilibrium", "2D"]' tooltip="A noisy oscillator: a radial force pulls the system onto the unit circle while a constant angular drift turns it around — a non-equilibrium steady state with a circulating probability current. From one trajectory, we recover the full nonlinear force field with a generic cubic dictionary, check it with residual diagnostics and simulations of the inferred model, and then repeat the inference on positions blurred by localisation noise.">

.. only:: html

  .. image:: /gallery/images/thumb/sphx_glr_limitcycle_demo_thumb.png
    :alt:

  :doc:`/gallery/limitcycle_demo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">2D limit cycle — nonlinear overdamped inference</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" data-sgtags='["synthetic", "overdamped", "nonlinear", "sparsity", "3D", "benchmark"]' tooltip="A 3D Lorenz-type system below its chaotic threshold: without noise it would settle on one of two stable foci.  Strong noise kicks it around each focus and, every so often, across to the other.  From a single recording of 10 time units, can we recover the seven Lorenz terms among 30 polynomial candidates — and a model that hops like the data?">

.. only:: html

  .. image:: /gallery/images/thumb/sphx_glr_lorenz_demo_thumb.png
    :alt:

  :doc:`/gallery/lorenz_demo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Noise-driven Lorenz system — hopping between two foci</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" data-sgtags='["synthetic", "overdamped", "nonlinear", "sparsity", "ecology"]' tooltip="Ecological time series are short, sparse and noisy.  How little data does it take to map who inhibits whom in a community?  Here, eight replicate cultures of a 6-species Lotka–Volterra community are inoculated at random low densities and counted every two time units, with a 10% counting error: 20 censuses per culture, 160 noisy snapshots in total, taken at an interval as long as the species&#x27; growth time.  From these, the parametric force estimator and PASTIS model selection recover the sparse interaction network.">

.. only:: html

  .. image:: /gallery/images/thumb/sphx_glr_lotka_volterra_demo_thumb.png
    :alt:

  :doc:`/gallery/lotka_volterra_demo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Lotka–Volterra ecosystem — a food web from a few noisy censuses</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" data-sgtags='["synthetic", "overdamped", "nonlinear", "sparsity", "diagnostics", "2D"]' tooltip="The recommended workflow for applying SFI to experimental trajectories: load a recording, look at it, infer the force with a generic basis, let PASTIS select the model, and validate the result without any ground truth — residual diagnostics, a noise-aware refit, a data-length check and a bootstrap.">

.. only:: html

  .. image:: /gallery/images/thumb/sphx_glr_experimental_workflow_demo_thumb.png
    :alt:

  :doc:`/gallery/experimental_workflow_demo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Experimental-data workflow template</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" data-sgtags='["synthetic", "overdamped", "2D", "custom-basis", "extras", "multi-experiment"]' tooltip="When does a force model need hand-written code?  Rarely: per-experiment constants, recorded drives and dataset-specific coefficients are all expressible with built-in symbols.  The exception is physics the symbol algebra cannot express — typically a measured object such as an image.  This page shows both, on one dataset.">

.. only:: html

  .. image:: /gallery/images/thumb/sphx_glr_custom_basis_demo_thumb.png
    :alt:

  :doc:`/gallery/custom_basis_demo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Custom basis with extras — traps in measured light fields</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" data-sgtags='["synthetic", "overdamped", "multiplicative-noise", "1D"]' tooltip="A particle hops in a symmetric double well, yet it spends most of its time on the left.  The reason is invisible in the force: the bath is hotter on the right — Landauer&#x27;s blowtorch.  This page recovers the state-dependent diffusion field D(x) together with the force from a single trajectory, lets PASTIS decide which terms D(x) needs, and shows that the inferred force and diffusion field together — and only together — predict the lopsided occupancy.  The last section repeats the analysis on a recording with localisation noise.">

.. only:: html

  .. image:: /gallery/images/thumb/sphx_glr_multiplicative_diffusion_demo_thumb.png
    :alt:

  :doc:`/gallery/multiplicative_diffusion_demo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Multiplicative noise — the Landauer blowtorch</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" data-sgtags='["synthetic", "overdamped", "multiplicative-noise", "anisotropic", "sparsity", "2D"]' tooltip="Recover a position-dependent, anisotropic diffusion tensor — and the force — from a single 2D trajectory.  A tracer in a ring trap experiences fluctuations whose radial component grows with distance from the center (think of a probe in a radially stretched gel, or near a topological defect in an active film): the local noise ellipse rotates with position.">

.. only:: html

  .. image:: /gallery/images/thumb/sphx_glr_anisotropic_diffusion_demo_thumb.png
    :alt:

  :doc:`/gallery/anisotropic_diffusion_demo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Anisotropic diffusion tensor field</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" data-sgtags='["synthetic", "underdamped", "nonlinear", "sparsity", "1D"]' tooltip="Infer the force field of a stochastic Van der Pol oscillator from position-only data.  The system has inertia, but only positions are observed — as is typical of experimental tracking — so the velocity is reconstructed from the positions by the underdamped estimators.">

.. only:: html

  .. image:: /gallery/images/thumb/sphx_glr_van_der_pol_demo_thumb.png
    :alt:

  :doc:`/gallery/van_der_pol_demo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Van der Pol oscillator — underdamped inference</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" data-sgtags='["synthetic", "underdamped", "multiplicative-noise", "sparsity", "1D"]' tooltip="Recover a velocity-dependent diffusion field D(v) for an inertial particle, from positions only.  Many active and driven systems fluctuate harder the faster they move — motility noise, flight-force fluctuations, turbulent drag.  Underdamped SFI reconstructs the unobserved velocity and infers F(x,v) and D(x,v) jointly; here the noise amplitude grows several-fold across the explored speeds.  We fit D on a generic polynomial basis, let PASTIS pick the noise model, cross-check with the likelihood-based estimator, and close on the regime where velocity-dependent noise becomes genuinely hard.">

.. only:: html

  .. image:: /gallery/images/thumb/sphx_glr_velocity_dependent_noise_demo_thumb.png
    :alt:

  :doc:`/gallery/velocity_dependent_noise_demo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Velocity-dependent noise — underdamped multiplicative diffusion</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" data-sgtags='["synthetic", "underdamped", "multi-particle", "2D", "per-particle"]' tooltip="Four tracked agents roam overlapping territories crossed by winding river valleys.  From their noisy, gappy, positions-only tracks we ask: where does each agent live — the centre, size and orientation of its home range — how strongly do they avoid each other, and how much does the terrain steer them?">

.. only:: html

  .. image:: /gallery/images/thumb/sphx_glr_home_range_demo_thumb.png
    :alt:

  :doc:`/gallery/home_range_demo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Home ranges in a shared landscape, from noisy gappy data</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" data-sgtags='["synthetic", "overdamped", "linear", "1D", "time-dependent"]' tooltip="An experiment often comes with a recorded protocol — a stiffness, a field, a temperature switched or ramped during the run.  Here a trap is switched between two stiffnesses, and we infer from a single 1D trajectory how the force depends on the protocol.">

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

    <div class="sphx-glr-thumbcontainer" data-sgtags='["diagnostics", "overdamped", "nonlinear", "1D", "synthetic"]' tooltip="An inference always returns numbers; the question is whether the model behind them is consistent with the data.  SFI.diagnostics answers it without ground truth: it recomputes the standardised residuals of a fit and tests whether they look like independent standard Gaussian noise, and it checks that the sampling interval is fine enough for the estimator used.">

.. only:: html

  .. image:: /gallery/images/thumb/sphx_glr_diagnostics_demo_thumb.png
    :alt:

  :doc:`/gallery/diagnostics_demo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Diagnostics — should we trust a fit?</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" data-sgtags='["diagnostics", "overdamped", "underdamped", "1D", "2D", "non-equilibrium", "synthetic"]' tooltip="Before fitting a force, you must choose an engine: OverdampedLangevinInference assumes first-order (overdamped) dynamics, UnderdampedLangevinInference second-order (inertial) dynamics.  SFI.classify_dynamics decides from the positions alone — no velocities needed, and robust to localisation noise.">

.. only:: html

  .. image:: /gallery/images/thumb/sphx_glr_dynamics_order_demo_thumb.png
    :alt:

  :doc:`/gallery/dynamics_order_demo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Overdamped or underdamped? Classifying dynamics from data</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" data-sgtags='["synthetic", "overdamped", "linear", "2D", "non-equilibrium", "thermodynamics"]' tooltip="A colloidal bead in an optical trap looks like the textbook equilibrium system — yet real tweezers exert a small non-conservative force: the scattering force has a curl component that silently stirs the bead (Roichman et al., PRL 2008; Wu et al., PRL 2009).  The bead obeys">

.. only:: html

  .. image:: /gallery/images/thumb/sphx_glr_entropy_production_demo_thumb.png
    :alt:

  :doc:`/gallery/entropy_production_demo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Measuring dissipation: entropy production in a driven optical trap</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" data-sgtags='["synthetic", "underdamped", "linear", "2D", "non-equilibrium", "thermodynamics"]' tooltip="The driven optical trap of the overdamped entropy-production page, but the bead now keeps its inertia — and the camera still records only its positions.  How much does it dissipate, and can we tell from positions alone?">

.. only:: html

  .. image:: /gallery/images/thumb/sphx_glr_entropy_production_underdamped_demo_thumb.png
    :alt:

  :doc:`/gallery/entropy_production_underdamped_demo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Inertial dissipation: entropy production from positions only</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" data-sgtags='["synthetic", "overdamped", "multi-particle", "linear", "sparsity"]' tooltip="Infer the pair interactions of a population of aligning active Brownian particles (ABPs) from tracking data — positions and headings corrupted by localisation noise, with a few detections missing — using a generic, over-complete dictionary of pair kernels from SFI.bases.pairs.">

.. only:: html

  .. image:: /gallery/images/thumb/sphx_glr_abp_align_demo_thumb.png
    :alt:

  :doc:`/gallery/abp_align_demo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Aligning active Brownian particles — interaction kernels from noisy tracks</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" data-sgtags='["synthetic", "overdamped", "multi-particle", "linear", "large-scale", "nonreciprocal"]' tooltip="Infer three pairwise interaction kernels — repulsion, alignment and pursuit — from the tracks of 10 000 self-propelled particles whose interactions are nonreciprocal: each particle reacts to what it sees ahead of it, not to what is behind.  Pursuit and alignment organise the swarm into worm-like streams of chevrons that sweep through the box — the kind of structure seen in tracked swarms of cells, insects or robots.">

.. only:: html

  .. image:: /gallery/images/thumb/sphx_glr_abp_nonreciprocal_demo_thumb.png
    :alt:

  :doc:`/gallery/abp_nonreciprocal_demo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Nonreciprocal active particles at scale — 10 000 particles</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" data-sgtags='["real-data", "overdamped", "multi-particle", "linear", "custom-basis", "2D", "non-equilibrium"]' tooltip="Self-propelled Janus colloids interact through electrostatic and hydrodynamic forces that cannot be computed from first principles.  Hem, Poncet, Ronceray, Nishiguchi and Démery (Soft Matter 2025, doi:10.1039/D5SM00655D) learned them from tracking data instead, with a generic basis: a Fourier series in the two relative orientations times smooth radial functions, for the radial velocity, the transverse velocity and the torque that one colloid induces on another.">

.. only:: html

  .. image:: /gallery/images/thumb/sphx_glr_janus_colloids_demo_thumb.png
    :alt:

  :doc:`/gallery/janus_colloids_demo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Janus colloids — learning general pair interactions from experimental tracks</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" data-sgtags='["synthetic", "overdamped", "multi-particle", "linear", "SPDE", "sparsity"]' tooltip="From particles to equations.  We simulate 10 000 self-propelled particles that align with their neighbours, coarse-grain their positions and headings into two continuum fields — the density and the polarisation density — and let SFI infer the stochastic equations that drive these fields.  We then read the result as a physicist would: which terms matter, what they mean, how they compare with the microscopic model and with the Toner–Tu theory of flocking — and whether the inferred equations, run on their own, make flocks like the particles do.">

.. only:: html

  .. image:: /gallery/images/thumb/sphx_glr_abp_to_spde_demo_thumb.png
    :alt:

  :doc:`/gallery/abp_to_spde_demo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Discovering Toner–Tu hydrodynamics from agent-based flocking</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" data-sgtags='["synthetic", "overdamped", "SPDE", "reaction-diffusion", "2D", "sparsity"]' tooltip="A few drops of an autocatalyst V in a bath of feed U grow into a colony of self-replicating spots: each spot swells, splits in two, and the daughters push apart until they tile the dish.  The dynamics is the stochastic Gray-Scott model">

.. only:: html

  .. image:: /gallery/images/thumb/sphx_glr_gray_scott_demo_thumb.png
    :alt:

  :doc:`/gallery/gray_scott_demo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Gray-Scott: equations from a corrupted movie</div>
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
   /gallery/entropy_production_demo
   /gallery/entropy_production_underdamped_demo
   /gallery/abp_align_demo
   /gallery/abp_nonreciprocal_demo
   /gallery/janus_colloids_demo
   /gallery/abp_to_spde_demo
   /gallery/gray_scott_demo

Advanced
========

These examples combine several tools at once: neural-network energy
landscapes, joint fits across experiments with shared and
experiment-specific parameters, and underdamped multi-particle systems.
Start with the main gallery above if you are new to SFI; the regime table
in the :doc:`Running-inference guide </inference/user_guide>` tells you
when these tools are the right choice.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" data-sgtags='["synthetic", "overdamped", "nonlinear", "neural-network", "thermodynamics", "2D"]' tooltip="Can a small neural network learn an energy landscape from a single trajectory?  A particle hops between the three wells of the Müller–Brown surface; we parametrize its potential by a network built from SFI&#x27;s compositional basis operations and obtain the force as minus its gradient, by automatic differentiation — every fitted force is conservative by construction.">

.. only:: html

  .. image:: /gallery/advanced/images/thumb/sphx_glr_nn_force_demo_thumb.png
    :alt:

  :doc:`/gallery/advanced/nn_force_demo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Neural-network energy landscape — Müller–Brown potential</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" data-sgtags='["synthetic", "overdamped", "multi-particle", "multi-experiment", "linear", "sparsity"]' tooltip="Infer one interaction model from several recordings made under different conditions — here three active-particle experiments at different densities and box sizes, whose particles also swim at different speeds (think of different temperatures, fuel or illumination levels).">

.. only:: html

  .. image:: /gallery/advanced/images/thumb/sphx_glr_multi_experiment_demo_thumb.png
    :alt:

  :doc:`/gallery/advanced/multi_experiment_demo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Multi-experiment ABP inference — shared interactions, per-condition motility</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" data-sgtags='["synthetic", "underdamped", "multi-particle", "linear", "sparsity", "large-scale", "3D"]' tooltip="Two hundred fish swim around a landmark and organise into a mill: a rotating torus in which everyone circles the same axis in the same direction, a state many fish schools adopt.  A 3D tracking rig records only their positions.  From these tracks alone, can we learn the rules each individual follows — how it swims, what holds it near the centre, how it reacts to its neighbours — with error bars, and check that these rules, simulated, rebuild the mill?">

.. only:: html

  .. image:: /gallery/advanced/images/thumb/sphx_glr_flocking_3d_demo_thumb.png
    :alt:

  :doc:`/gallery/advanced/flocking_3d_demo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">3D milling school — the interaction rules of 200 swimmers</div>
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
