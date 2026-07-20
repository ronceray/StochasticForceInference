.. _sparsity-theory:

Sparse model selection — Theory & design
=========================================

This page describes the mathematical foundations and software
architecture of the :mod:`SFI.inference.sparse` sub-package.  It
complements the :ref:`user guide <sparsity-user-guide>` with deeper
technical details.


Mathematical framework
----------------------

Normal equations and information gain
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After projecting the force onto a basis library
:math:`\{\phi_a\}_{a=1}^p`, the inference problem reduces to a
linear system in normal-equation form:

.. math::

   G\,C = M

where

.. math::

   G_{ab} = \bigl\langle \phi_a \cdot \bar D^{-1} \cdot \phi_b \bigr\rangle,
   \qquad
   M_a    = \bigl\langle v_t \cdot \bar D^{-1} \cdot \phi_a \bigr\rangle.

Given a *support* :math:`B \subseteq \{1, \dots, p\}`, the restricted
solution :math:`C_B = G_{BB}^{-1}\,M_B` yields an **information gain**

.. math::

   \mathcal{I}(B) = \tfrac{1}{2}\,C_B^\top M_B
   = \tfrac{1}{2}\,M_B^\top G_{BB}^{-1}\,M_B.

This quantity measures how much the data support the selected terms.
The full problem (:math:`B = \{1,\dots,p\}`) gives
:math:`\mathcal{I}_{\mathrm{total}}`.

.. note::

   All sparsity algorithms in SFI operate **only on** :math:`(M, G)`.
   The raw trajectory data never enters the sparse search, ensuring a
   clean separation between data processing and model selection.


Sparse model selection as combinatorial optimisation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We seek the support :math:`B^*(k)` that maximises
:math:`\mathcal{I}(B)` for each cardinality :math:`|B| = k`:

.. math::

   B^*(k) = \arg\max_{|B|=k} \mathcal{I}(B).

The set of all :math:`(k, \mathcal{I}(B^*(k)))` pairs defines the
**Pareto front**.  This is a combinatorial optimisation problem —
exhaustive search is intractable for :math:`p > 20` — so approximate
algorithms are used.


Information criteria
~~~~~~~~~~~~~~~~~~~~

The optimal model size is selected by maximising an
information criterion that trades fit against complexity:

.. math::

   \text{AIC}(k)    &= \mathcal{I}(k) - k, \\
   \text{BIC}(k)    &= \mathcal{I}(k) - \tfrac{1}{2}\,k\,\ln\tau, \\
   \text{EBIC}(k)   &= \text{BIC}(k) - 2\gamma\,\ln\binom{n_0}{k}, \\
   \text{PASTIS}(k) &= \mathcal{I}(k) - k\,\ln(n_0 / p_0).

Here :math:`\tau` is the total trajectory time, :math:`n_0` the library
size, :math:`p_0` the PASTIS significance level, and
:math:`\gamma \in [0,1]` the EBIC tuning parameter.

**AIC** (Akaike, 1974) penalises by the number of parameters.

**BIC** (Schwarz, 1978) arises from a Laplace approximation of the
marginal likelihood.  The continuous-time formulation uses
:math:`\ln\tau` rather than :math:`\ln n` (number of data points),
which is the correct scaling when :math:`\Delta t \to 0`
(Gerardos & Ronceray, 2025).

**EBIC** (Chen & Chen, 2008) augments BIC with a combinatorial
correction :math:`2\gamma\ln\binom{n_0}{k}` that accounts for the
number of candidate subsets.  Setting :math:`\gamma=0` recovers BIC;
:math:`\gamma=1` gives the strongest penalty.

**PASTIS** (Gerardos & Ronceray, 2025) combines likelihood statistics
with extreme-value theory.  The penalty :math:`k\ln(n_0/p_0)` controls
the probability of retaining at least one superfluous term.

.. admonition:: Choosing the criterion
   :class: tip

   PASTIS is recommended for typical experimental data where :math:`n_0`
   is moderate (10–100).  For very small bases, AIC or BIC may be
   better calibrated.  For large libraries, EBIC with
   :math:`\gamma \approx 0.5` improves selection consistency.

.. note::

   BIC and EBIC require the trajectory time :math:`\tau` to be passed
   explicitly via the ``tau`` keyword argument.


Algorithms
----------

Beam search (PASTIS original)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The bidirectional beam search explores the support lattice by *expanding*
(add one index) and *contracting* (drop one index) every support in the
current frontier, then retaining only the ``beam_width`` best supports
per cardinality.

**Algorithm outline:**

1. Initialise with the empty support :math:`B = \emptyset`.
2. For each support in the frontier at cardinality :math:`k`:

   a. Generate all *add-one* children (cardinality :math:`k+1`).
   b. Generate all *drop-one* children (cardinality :math:`k-1`).

3. Score all children via :math:`\mathcal{I}(B)` (batched using
   ``vmap``).
4. For each target cardinality, retain only the ``beam_width`` highest-
   scoring supports (min-heap).
5. Repeat until no new supports are generated or the AIC early-stop
   criterion triggers (AIC has declined for ``aic_patience`` consecutive
   closed levels).

The bidirectionality allows the search to correct early mistakes by
removing terms that were added sub-optimally.

**Complexity:** Each generation scores
:math:`O(p \times \text{beam\_width})` candidates at each of
:math:`O(p)` cardinality levels.  In practice, the AIC early stop
limits the number of generations well below :math:`p`.

**Second-best tracking:** The beam naturally retains the top-2 supports
per cardinality, providing a robustness diagnostic.


Greedy stepwise selection
~~~~~~~~~~~~~~~~~~~~~~~~~

Stepwise selection adds (forward) or removes (backward) one feature at
a time, always choosing the action that maximises information gain.

**Forward** (starting from :math:`B = \emptyset`):

Step :math:`k`: evaluate all :math:`p - k + 1` candidates obtained by
adding one index to the current support; keep the best.

**Backward** (starting from :math:`B = \{1,\dots,p\}`):

Step :math:`k`: evaluate all :math:`k` candidates obtained by
removing one index; keep the best.

**Bidirectional**: run both paths and merge their Pareto fronts.

**Complexity:** :math:`O(p^2)` evaluations total (each done in a batched
``vmap`` call).  This is the cheapest algorithm and scales well to large
:math:`p`.

.. note::

   Greedy selection produces exactly one candidate per cardinality,
   so the Pareto front is always monotone.  It does not track
   second-best alternatives.


Sequential Thresholded Least Squares (STLSQ)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

STLSQ (Brunton *et al.*, 2016) is an iterative hard-thresholding
algorithm:

1. Solve the full system :math:`G\,C = M` for all :math:`p` coefficients.
2. Zero out coefficients whose magnitude falls below a threshold
   :math:`\tau`.
3. Re-solve on the surviving support.
4. Repeat until convergence.

The threshold can be:

* **Relative** (default): :math:`\tau_{\mathrm{eff}} = \tau \cdot \max|C|`.
* **Absolute**: :math:`\tau_{\mathrm{eff}} = \tau`.

Running the algorithm over a log-spaced sweep of thresholds produces
supports at different sparsity levels, from which a Pareto front is
assembled.

**Complexity:** Each threshold requires :math:`O(p)` least-squares
solves (one per iteration, each of size at most :math:`p`).  With
:math:`n_\tau` thresholds, the total cost is
:math:`O(n_\tau \cdot \text{max\_iter} \cdot p^3)`, but each solve is
small.


Stochastic hill-climbing
~~~~~~~~~~~~~~~~~~~~~~~~

The stochastic hill-climbing strategy implements the search algorithm
of Gerardos & Ronceray (2025): starting from an initial model, it
accepts random single-parameter additions or removals if they increase
a chosen information criterion, and stops after a configurable number
of consecutive failures (*patience*).

Multiple independent chains run in parallel, one per cardinality
:math:`k \in \{0, \ldots, \text{max\_k}\}`.  Each chain starts from a
uniformly random support of size :math:`k` (this includes the null and
full models as special cases).

**Algorithm outline:**

1. For each :math:`k = 0, \ldots, \text{max\_k}`, draw a random
   support :math:`B_0` of size :math:`k`.
2. At each step, propose a random add or remove move:

   * **Add:** pick a random index :math:`j \notin B` and form
     :math:`B' = B \cup \{j\}`.
   * **Remove:** pick a random index :math:`j \in B` and form
     :math:`B' = B \setminus \{j\}`.

3. If :math:`\text{IC}(B') > \text{IC}(B)`, accept the move.
4. Stop after *patience* consecutive rejections.
5. Record the best support found by each chain into the Pareto front.

**Complexity:** Each chain performs at most
:math:`O(\text{patience} \times \text{max\_k})` single-support
evaluations.  Since each evaluation is JIT-compiled, the overhead is
dominated by the number of evaluations.

.. note::

   Unlike beam search and greedy selection, hill-climbing is
   *stochastic* — the result depends on the random seed.  Use the
   ``seed`` parameter for reproducibility.


LASSO (ℓ₁-penalised regression)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The LASSO minimises the penalised normal-equations objective:

.. math::

   \hat C = \arg\min_C \;
   \tfrac{1}{2}\,C^\top G\,C \;-\; M^\top C
   \;+\; \alpha\,\|C\|_1

using **proximal coordinate descent**.  No access to the raw
design matrix :math:`\Phi` is needed — only :math:`(M, G)`.

**Coordinate descent update:**  For each :math:`j = 1, \dots, p`:

.. math::

   C_j \leftarrow \frac{S_\alpha\!\bigl(
       M_j - \sum_{i \neq j} G_{ji} C_i
   \bigr)}{G_{jj}}

where :math:`S_\alpha(x) = \text{sign}(x)\max(|x| - \alpha, 0)` is
the soft-thresholding operator.

**Regularisation path:** sweeping :math:`\alpha` from
:math:`\alpha_{\max} = \max|M|` (all-zero solution) down to
:math:`10^{-4}\alpha_{\max}` with warm-starting produces a set of
supports at varying sparsity.

**De-biased re-solve:** for each support found by the LASSO, we
re-solve the unrestricted problem on that support to remove the
:math:`\ell_1` shrinkage bias.  The de-biased coefficients and
information gain are stored in the Pareto front.


Software architecture
---------------------

The :mod:`SFI.inference.sparse` sub-package separates three concerns:

.. list-table::
   :widths: 15 25 60
   :header-rows: 1

   * - Component
     - Class
     - Responsibility
   * - **Scoring**
     - :class:`~SFI.inference.sparse.SparseScorer`
     - Owns ``(M, G)``.  Scores any support :math:`B` by
       solving :math:`G_{BB} C_B = M_B` and returning
       :math:`(\mathcal{I}(B), C_B)`.  JIT-compiled and
       ``vmap``-batched for efficiency.
   * - **Search**
     - :class:`~SFI.inference.sparse.SparsityStrategy` (ABC)
     - Abstract base class.  Subclasses implement
       ``run(scorer, max_k) → SparsityResult``.
       Concrete strategies: :class:`~SFI.inference.sparse.BeamSearchStrategy`,
       :class:`~SFI.inference.sparse.GreedyStepwiseStrategy`, :class:`~SFI.inference.sparse.HillClimbStrategy`,
       :class:`~SFI.inference.sparse.STLSQStrategy`, :class:`~SFI.inference.sparse.LassoStrategy`.
   * - **Result**
     - :class:`~SFI.inference.sparse.SparsityResult`
     - Immutable container for the Pareto front.  Provides
       ``select_by_ic(name)`` and ``all_ic()`` for criterion-based
       model selection.

This factoring means:

* **New algorithms** only need to implement the ``SparsityStrategy``
  interface — they receive a scorer and return a result.
* **Scoring** is decoupled from the search.  The same scorer can be
  reused by multiple strategies without re-computation.
* **Selection** is decoupled from both scoring and searching.  A
  ``SparsityResult`` from any algorithm can be queried for any IC.


Module map
~~~~~~~~~~

.. code-block:: text

   SFI/inference/sparse/
   ├── __init__.py      # re-exports public API
   ├── scorer.py        # SparseScorer — (M, G) scoring
   ├── result.py        # SparsityResult — Pareto front + IC
   ├── base.py          # SparsityStrategy ABC
   ├── beam.py          # BeamSearchStrategy
   ├── greedy.py        # GreedyStepwiseStrategy
   ├── hillclimb.py     # HillClimbStrategy
   ├── stlsq.py         # STLSQStrategy
   ├── lasso.py         # LassoStrategy
   └── metrics.py       # overlap_metrics, predictive_nmse


Integration with the inference engines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The inference base class (``BaseLangevinInference``) creates a
``SparseScorer`` during ``infer_force_linear()`` and stores it as
``self.force_scorer``.  Calling ``sparsify_force(method=...)``
dispatches to the chosen strategy and stores the result as
``self.force_sparsity_result``.

.. code-block:: text

   infer_force_linear(basis)
       └── builds SparseScorer(M=..., G=...)
               stored as self.force_scorer

   sparsify_force(method="beam", criterion="PASTIS")
       ├── strategy = BeamSearchStrategy(...)
       ├── result = strategy.run(self.force_scorer, max_k=...)
       ├── k*, support, coeffs = result.select_by_ic("PASTIS")
       ├── updates force coefficients via _update_force_coefficients
       └── stores result as self.force_sparsity_result


Extending with a custom strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To add a new search algorithm:

.. code-block:: python

   from SFI.inference.sparse import SparsityStrategy, SparsityResult

   class MyStrategy(SparsityStrategy):
       name = "my_algo"

       def run(self, scorer, *, max_k, **kwargs):
           # ... implement search logic using scorer.info_and_coeffs
           #     and/or scorer.vmap_info ...
           return SparsityResult(
               p=scorer.p,
               total_info=float(scorer.total_info),
               method=self.name,
               best_info_by_k=...,
               best_support_by_k=...,
               best_coeffs_by_k=...,
           )

Then use it directly:

.. code-block:: python

   scorer = inf.force_scorer
   result = MyStrategy().run(scorer, max_k=20)
   k, support, score, coeffs = result.select_by_ic("PASTIS")


References
----------

* **AIC:** Akaike, H. (1974). "A new look at the statistical model
  identification." *IEEE Trans. Automat. Control*, 19(6), 716–723.
* **BIC:** Schwarz, G. (1978). "Estimating the dimension of a model."
  *Ann. Statist.*, 6(2), 461–464.
* **EBIC:** Chen, J. & Chen, Z. (2008). "Extended Bayesian information
  criteria for model selection with large model spaces." *Biometrika*,
  95(3), 759–771.
* **PASTIS (Parsimonious Stochastic Inference):** Gerardos, A. & Ronceray, P. (2025).
  "Principled model selection for stochastic dynamics."
  *Phys. Rev. Lett.* 135, 167401.
  `DOI: 10.1103/ltdt-hvh7 <https://doi.org/10.1103/ltdt-hvh7>`_
* **SINDy / STLSQ:** Brunton, S. L., Proctor, J. L. & Kutz, J. N.
  (2016). "Discovering governing equations from data by sparse
  identification of nonlinear dynamical systems."
  *Proc. Natl. Acad. Sci.* 113(15), 3932–3937.
* **LASSO:** Tibshirani, R. (1996). "Regression shrinkage and selection
  via the lasso." *J. R. Statist. Soc. B* 58(1), 267–288.
