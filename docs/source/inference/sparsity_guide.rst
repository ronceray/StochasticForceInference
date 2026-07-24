.. _sparsity-user-guide:

Sparse model selection
======================

When the basis library is larger than the true model, many inferred
coefficients should be zero.  SFI's sparse model selection module
identifies which basis functions are genuinely needed and sets the rest
to zero, yielding **interpretable, parsimonious models** with
controlled overfitting.


Quick start
-----------

After running :meth:`~SFI.inference.BaseLangevinInference.infer_force_linear`,
call :meth:`~SFI.inference.BaseLangevinInference.sparsify_force`:

.. code-block:: python

   inf.infer_force_linear(basis)
   result = inf.sparsify_force(criterion="PASTIS")

This performs a **beam search** (the default method) over all possible
subsets of the basis, builds a Pareto front of
information-gain vs. model size, and selects the best model according
to the PASTIS information criterion.

The return value is a :class:`~SFI.inference.sparse.SparsityResult` that
holds the full Pareto front and can be queried for any criterion:

.. code-block:: python

   # Re-select with a different criterion without re-running the search
   # (BIC and EBIC require the trajectory time tau)
   k, support, score, coeffs = result.select_by_ic("BIC", tau=tau)

   # Summary table of all criteria (tau enables BIC/EBIC)
   result.all_ic(tau=tau, verbose=True)


Choosing a search method
------------------------

Four algorithms are available via the ``method`` keyword.  Each
explores the combinatorial space of supports differently:

.. list-table::
   :header-rows: 1
   :widths: 15 35 25 25

   * - Method
     - Description
     - Strengths
     - When to use
   * - ``"beam"``
     - Bidirectional beam search (PASTIS original).
       Expands ±1 neighbours of the best supports per
       cardinality and retains only the *beam_width* best.
     - Good coverage of the support lattice; second-best
       tracking for robustness.
     - Default choice for moderate libraries (:math:`p \lesssim 100`).
   * - ``"greedy"``
     - Forward, backward, or bidirectional stepwise selection.
       Adds or drops one feature per step.
     - Very fast; one support per cardinality.
     - First pass / screening on large libraries.
   * - ``"hillclimb"``
     - Stochastic hill-climbing with random add/remove moves
       (Gerardos & Ronceray, 2025).  One chain per cardinality.
     - Escapes greedy traps; works at every complexity level.
     - Good complement to beam/greedy for difficult problems.
   * - ``"stlsq"``
     - Sequential Thresholded Least Squares (SINDy-style).
       Iteratively solves and thresholds small coefficients.
     - Familiar to SINDy users; works well with clean data.
     - Benchmarking against SINDy-style pipelines (see note).
   * - ``"lasso"``
     - :math:`\ell_1`-penalised coordinate descent on the normal
       equations.  Sweeps a regularisation path.
     - Continuous shrinkage; warm-started path is efficient.
     - Benchmarking against LASSO pipelines (see note).

.. note:: **STLSQ and LASSO are for benchmarking, not the default.**

   These two are deterministic-regression sparsifiers, included as a
   convenience for *comparison* — not as a competitor to SINDy or LASSO
   toolchains.  They come from the deterministic-dynamics literature,
   where the data is essentially noise-free.  SFI's own territory is
   **stochastic** dynamics, and there the principled choice is a beam or
   hill-climbing search under :term:`PASTIS`, which is calibrated to the
   fluctuations of the residual.

   SFI *can* sparsify a deterministic system, but on noise-free data its
   error estimates and noise-calibrated criteria (PASTIS, the predicted
   NMSE) lose their statistical meaning — there is no residual
   distribution to calibrate against.  Reach for STLSQ/LASSO when you
   want a like-for-like comparison with deterministic methods, and
   prefer ``"beam"`` or ``"hillclimb"`` with PASTIS for genuine
   stochastic data.


Examples
^^^^^^^^

.. code-block:: python

   # Beam search with wider beam
   result = inf.sparsify_force(
       criterion="PASTIS",
       method="beam",
       beam_width=10,
   )

   # Greedy forward selection
   result = inf.sparsify_force(
       criterion="AIC",
       method="greedy",
       direction="forward",
   )

   # STLSQ with automatic threshold sweep
   result = inf.sparsify_force(
       criterion="BIC",
       method="stlsq",
       n_thresholds=30,
   )

   # Stochastic hill-climbing (PASTIS paper algorithm)
   result = inf.sparsify_force(
       criterion="PASTIS",
       method="hillclimb",
       patience=200,
       seed=42,
   )

   # LASSO path
   result = inf.sparsify_force(
       criterion="PASTIS",
       method="lasso",
       n_alphas=50,
   )


Method-specific parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Beam search** (``method="beam"``)

.. list-table::
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``beam_width``
     - 3
     - Number of candidates retained per cardinality level.
   * - ``aic_patience``
     - 2
     - Stop early when AIC has declined for this many
       consecutive closed levels.

**Greedy stepwise** (``method="greedy"``)

.. list-table::
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``direction``
     - ``"forward"``
     - ``"forward"``, ``"backward"``, or ``"both"``
       (merge both paths).

**STLSQ** (``method="stlsq"``)

.. list-table::
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``threshold``
     - None (auto sweep)
     - Fixed threshold value, or ``None`` for a log-spaced
       sweep.
   * - ``mode``
     - ``"relative"``
     - ``"relative"`` (fraction of :math:`\max|C|`) or
       ``"absolute"`` (fixed value).
   * - ``n_thresholds``
     - 30
     - Number of threshold values in the automatic sweep.
   * - ``max_iter``
     - 50
     - Maximum STLSQ iterations per threshold.

**Hill-climbing** (``method="hillclimb"``)

.. list-table::
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``ic``
     - ``"PASTIS"``
     - Information criterion used as the acceptance objective.
   * - ``patience``
     - 200
     - Stop a chain after this many consecutive rejected moves.
   * - ``seed``
     - None
     - Random seed for reproducibility.
   * - ``p_param``
     - 1e-3
     - PASTIS significance level :math:`p_0`.
   * - ``tau``
     - None
     - Total trajectory time (required when ``ic`` is BIC or EBIC).
   * - ``gamma``
     - 0.5
     - EBIC tuning parameter :math:`\gamma \in [0,1]`.

**LASSO** (``method="lasso"``)

.. list-table::
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``alpha``
     - None (auto path)
     - Fixed :math:`\alpha`, or ``None`` for a log-spaced path
       from :math:`\alpha_{\max}` down to
       :math:`10^{-4}\alpha_{\max}`.
   * - ``n_alphas``
     - 50
     - Number of :math:`\alpha` values in the path.
   * - ``max_iter``
     - 1000
     - Maximum coordinate-descent iterations per :math:`\alpha`.
   * - ``tol``
     - 1e-7
     - Convergence tolerance for coordinate descent.


Information criteria
--------------------

After the search, a Pareto front of **(cardinality, information gain)**
pairs is available.  The information criterion trades model fit against
complexity:

.. math::

   \text{IC}(k) \;=\; \mathcal{I}(k) \;-\; \text{penalty}(k)

where :math:`\mathcal{I}(k)` is the log-likelihood gain with :math:`k`
basis terms out of :math:`p` candidates.

.. list-table::
   :header-rows: 1
   :widths: 12 35 53

   * - Criterion
     - Penalty
     - Use case
   * - ``"AIC"``
     - :math:`k`
     - When :math:`n_0` is small relative to data size
       (Akaike, 1974).
   * - ``"BIC"``
     - :math:`\frac{1}{2}\,k\,\ln\tau`
     - Bayesian model comparison; requires the total trajectory
       time :math:`\tau` (Schwarz, 1978; continuous-time form
       from Gerardos & Ronceray, 2025).
   * - ``"EBIC"``
     - :math:`\frac{1}{2}\,k\,\ln\tau + 2\gamma\,\ln\binom{n_0}{k}`
     - Extended BIC for large libraries; :math:`\gamma\!\in\![0,1]`
       tunes the combinatorial penalty (Chen & Chen, 2008).
       Requires :math:`\tau`.
   * - ``"PASTIS"``
     - :math:`k\,\ln(n_0 / p_0)`
     - Default for SFI.  :math:`p_0` is the significance level
       (default 1e-3) controlling overfitting tolerance
       (Gerardos & Ronceray, 2025).
.. note::

   BIC and EBIC require the total trajectory time :math:`\tau`.
   Pass ``tau=...`` to :meth:`select_by_ic` or :meth:`all_ic`.  When
   ``tau`` is not provided, :meth:`all_ic` includes only AIC and
   PASTIS.

.. tip::

   **Which criterion?**  ``"PASTIS"`` is recommended for typical
   experimental data.  For very small bases, ``"AIC"`` or ``"BIC"``
   may be more appropriate since the PASTIS penalty scales as
   :math:`\ln n_0` and can be too harsh when :math:`n_0 < 10`.  For
   large libraries, ``"EBIC"`` with :math:`\gamma \approx 0.5` adds a
   combinatorial correction that improves selection consistency.


Inspecting the Pareto front
----------------------------

The :class:`~SFI.inference.sparse.SparsityResult` exposes the full
Pareto front for plotting or further analysis:

.. code-block:: python

   import matplotlib.pyplot as plt

   result = inf.sparsify_force(criterion="PASTIS", method="beam")

   # Plot information gain vs. model size
   ks = range(len(result.best_info_by_k))
   plt.plot(ks, result.best_info_by_k, "o-")
   plt.xlabel("Model size k")
   plt.ylabel("Information gain I(k)")
   plt.title("Pareto front")

   # Overlay IC-selected model
   k_star, support, score, coeffs = result.select_by_ic("PASTIS")
   plt.axvline(k_star, color="r", ls="--", label=f"PASTIS: k*={k_star}")
   plt.legend()

.. code-block:: python

   # Full IC comparison table (logged at INFO level)
   summary = result.all_ic(verbose=True)


Benchmarking against ground truth
----------------------------------

When the true support and coefficients are known (synthetic data):

.. code-block:: python

   from SFI.inference.sparse import overlap_metrics, predictive_nmse

   om = overlap_metrics(true_support, support)
   print(f"TP={om['TP']}, FP={om['FP']}, FN={om['FN']}, "
         f"precision={om['prec']:.2f}, recall={om['rec']:.2f}")

   # Predictive NMSE on held-out design matrix
   nmse = predictive_nmse(Phi_test, true_support, true_coeffs,
                          support, coeffs)
   print(f"Predictive NMSE: {nmse:.4f}")

The ``result.all_ic()`` method can optionally compute these metrics
for all criteria at once:

.. code-block:: python

   summary = result.all_ic(
       true_support=true_support,
       true_coeffs=true_coeffs,
       Phi_test=Phi_test,
       verbose=True,
   )


Advanced: using strategies directly
------------------------------------

For fine-grained control, instantiate a strategy and a
:class:`~SFI.inference.sparse.SparseScorer` manually:

.. code-block:: python

   from SFI.inference.sparse import SparseScorer, BeamSearchStrategy

   scorer = inf.force_scorer   # created during infer_force_linear
   strategy = BeamSearchStrategy(beam_width=20)
   result = strategy.run(scorer, max_k=15)

   # Or build a scorer from raw (M, G)
   scorer = SparseScorer(M=M_vec, G=G_mat)

You can also write your own strategy by subclassing
:class:`~SFI.inference.sparse.SparsityStrategy` and implementing
:meth:`run(scorer, *, max_k) -> SparsityResult`.


Troubleshooting
---------------

**All coefficients are pruned by PASTIS.**
The signal is too weak for the library size.  Reduce the basis,
increase the trajectory length, or try ``criterion="AIC"`` which
has a weaker penalty.

**Beam search is slow.**
Reduce ``beam_width`` or ``max_k``.  For libraries with :math:`p > 50`,
consider using ``method="greedy"`` or ``method="lasso"`` as a fast
first pass.

**STLSQ and LASSO disagree with beam search.**
This is expected.  Different algorithms explore different parts of
the support lattice.  The beam search is the most thorough; STLSQ and
LASSO provide complementary perspectives.  Compare their Pareto
fronts.

**LASSO selects too many features.**
Increase ``n_alphas`` for a finer regularisation path, or try
``method="stlsq"`` which performs hard thresholding.


Further reading
---------------

- :ref:`sparsity-theory` — mathematical background for the information
  criteria and search algorithms.
