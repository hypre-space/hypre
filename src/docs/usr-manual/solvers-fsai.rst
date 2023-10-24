.. Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
   HYPRE Project Developers. See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (Apache-2.0 OR MIT)

.. _fsai:

FSAI
==============================================================================

FSAI is a parallel implementation of the Factorized Sparse Approximate Inverse preconditioner,
initially proposed by [KoYe1993]_. Given a symmetric positive definite matrix :math:`A`, FSAI
computes a triangular matrix :math:`G` that approximates the inverse of the lower Cholesky
factor (:math:`L`) of :math:`A`. This computation is done by minimizing the Frobenius norm
:math:`|| I - G L ||_F` without explicit knowledge of :math:`L`. The resulting preconditioner
preserves the positive definiteness of :math:`A` and is given by :math:`M^{-1} = G^{T} G`.

One of the critical factors determining the quality of sparse approximate inverse
preconditioners lies in choosing the sparsity pattern of :math:`G`. While ParaSails
employs *a priori* sparsity patterns, FSAI uses an iterative strategy that generates
sparsity patterns on the fly, i.e., while computing their nonzero coefficient values
concurrently. At every step of the iterative process, the sparsity pattern of a row of
:math:`G` is augmented with a fixed number of entries, leading to the most significant
reduction of the conditioning number of :math:`G A G^T`. Such a strategy is also called
"adaptive FSAI" or "dynamic FSAI" and it can lead to more robust sparse approximate
inverses than ParaSails. For more details on how it works, see [JaFe2015]_.

Parameter Settings
------------------------------------------------------------------------------

The accuracy and cost of FSAI are determined by three configurations parameters as shown
in the table below

   =================  =======  =============  =================  =======
   param              type     range          sug. values        default
   =================  =======  =============  =================  =======
   ``max_steps``      int      :math:`\ge 0`  5, 10, 30          5
   ``max_step_size``  int      :math:`\ge 0`  1, 3, 6            3
   ``kap_tolerance``  real     :math:`\ge 0`  0.0, 1.E-2, 1.E-3  1.E-3
   =================  =======  =============  =================  =======

The first parameter, ``max_steps``, controls the number of maximum steps used in the iterative
algorithm. The second parameter, ``max_step_size``, gives the maximum number of indices added
to the sparsity pattern of :math:`G` at each step. Lastly, the third parameter,
``kap_tolerance``, is a floating-point value used to stop the inclusion of new indices to the
sparsity pattern of :math:`G` when the conditioning number of :math:`G A G^T`
stagnates. This can be disabled by setting ``kap_tolerance = 0``. Naturally, the
preconditioner quality increases for denser sparsity patterns of :math:`G`, but so do
its setup and solve costs. For a reasonable balance between accuracy and cost,
we recommend that :math:`max\_steps * max\_step\_size \leq 30`. The configuration
parameters of FSAI can be set via the following calls:

.. code-block:: c

   HYPRE_FSAISetMaxSteps(HYPRE_Solver solver, HYPRE_Int max_steps);
   HYPRE_FSAISetMaxStepSize(HYPRE_Solver solver, HYPRE_Int max_step_size);
   HYPRE_FSAISetKapTolerance(HYPRE_Solver solver, HYPRE_Real kap_tolerance);

.. _fsai-amg-smoother:

FSAI as Smoother to BoomerAMG
------------------------------------------------------------------------------

As discussed in [PaFa2019]_, the factorized sparse approximate inverse method can be an
effective smoother to AMG for several reasons. Particularly, it leads to a symmetric operator,
and thus allows AMG to be used as a preconditioner for the conjugate gradient solver. In
hypre, FSAI can be used as a complex smoother to BoomerAMG by calling the functions:

.. code-block:: c

   HYPRE_BoomerAMGSetSmoothType(HYPRE_Solver solver, 4);
   HYPRE_BoomerAMGSetSmoothNumLevels(HYPRE_Solver solver, HYPRE_Int num_levels);

where ``num_levels`` is the last multigrid level where FSAI is used. The configuration
parameters of the FSAI smoother, as described above, can be set via the following calls:

.. code-block:: c

   HYPRE_BoomerAMGSetFSAIMaxSteps(HYPRE_Solver solver, HYPRE_Int max_steps);
   HYPRE_BoomerAMGSetFSAIMaxStepSize(HYPRE_Solver solver, HYPRE_Int max_step_size);
   HYPRE_BoomerAMGSetFSAIKapTolerance(HYPRE_Solver solver, HYPRE_Real kap_tolerance);

Implementation Notes
------------------------------------------------------------------------------

* When the matrix :math:`A` is distributed across MPI tasks, FSAI considers only the
  block diagonal portions of :math:`A` for computing :math:`G`. The resulting
  preconditioner is effectively a block-Jacobi sparse approximate inverse in the MPI
  sense. Although this strategy reduces communication costs, it can degrade convergence
  performance when several tasks are used, especially when FSAI is employed as a
  preconditioner to a Krylov solver.

* The CPU version of FSAI supports threading via OpenMP. To enable it, users need to
  compile hypre with OpenMP support via the configure option ``--with-openmp``. In this
  case, FSAI relies on an implementation of BLAS/LAPACK that is thread-safe. The one
  distributed internally with hypre fulfills this criterion, but care must be taken when
  linking hypre to external BLAS/LAPACK libraries. In HPC platforms, we recommend using
  vendor implementations of BLAS/LAPACK for better setup performance of FSAI, regardless
  of whether using OpenMP or not.

* The GPU version of FSAI is under development.
