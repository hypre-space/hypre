.. Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
   HYPRE Project Developers. See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (Apache-2.0 OR MIT)


SSAMG
==============================================================================

SSAMG is a semi-structured algebraic multigrid solver designed for problems
defined through hypre's SStruct interface [MaFaYa23]_. It targets applications
with grids composed of multiple logically rectangular parts (e.g., block-structured,
overset, and structured adaptive mesh refinement grids) and supports coupling
between parts. The system matrix is naturally viewed as

.. math::

   A = S + U,

where ``S`` represents structured, stencil-based intra-part couplings and ``U``
captures unstructured inter-part couplings.

SSAMG builds a multilevel hierarchy algebraically, combining structured,
directional coarsening within each part with algebraic treatment of couplings
across parts. Compared to fully unstructured AMG, SSAMG exploits regularity for
performance while allowing more general problem structure than strictly
structured methods like PFMG.

.. note::
   SSAMG currently targets single-type variables on each part. Support for
   multiple variables per part is limited.

Configuration options
------------------------------------------------------------------------------

Below is a summary of configuration routines. See the reference manual for full
details and defaults.

- ``HYPRE_SStructSSAMGSetTol``: Convergence tolerance on the relative residual.
- ``HYPRE_SStructSSAMGSetMaxIter``: Maximum number of SSAMG iterations.
- ``HYPRE_SStructSSAMGSetMaxLevels``: Limit the multigrid hierarchy depth.
- ``HYPRE_SStructSSAMGSetRelChange``: Require relative change of iterates to be
  small in addition to the residual criterion.
- ``HYPRE_SStructSSAMGSetZeroGuess`` / ``SetNonZeroGuess``: Specify initial
  guess semantics.
- ``HYPRE_SStructSSAMGSetInterpType``: Interpolation choice within parts:

  - ``-1``: Structured interpolation only (default)
  - ``0``: Structured plus classical modified unstructured interpolation

- ``HYPRE_SStructSSAMGSetRelaxType``: Smoother selection:

  - ``0``: Jacobi
  - ``1``: Weighted Jacobi (default)
  - ``2``: L1-Jacobi
  - ``10``: Red/Black Gauss–Seidel (symmetric RB/BR)

- ``HYPRE_SStructSSAMGSetRelaxWeight``: Jacobi weight (used by Jacobi variants).
- ``HYPRE_SStructSSAMGSetNumPreRelax`` / ``SetNumPostRelax``: Number of sweeps
  before/after coarse correction on each level.
- ``HYPRE_SStructSSAMGSetNumCoarseRelax``: Number of sweeps on the coarsest
  level (when not delegating to BoomerAMG).
- ``HYPRE_SStructSSAMGSetSkipRelax``: Enable skip-relax on levels whose
  coarsening directions repeat those from previous cycles (useful for isotropic
  problems).
- ``HYPRE_SStructSSAMGSetCoarseSolverType``: Coarse solver selection:

  - ``0``: Weighted Jacobi (default)
  - ``1``: BoomerAMG (hybrid hierarchy on coarse levels)

- ``HYPRE_SStructSSAMGSetDxyz``: Provide a per-part grid-spacing metric used to
  guide coarsening direction choices.
- ``HYPRE_SStructSSAMGSetMaxCoarseSize``: Limit the maximum coarse problem size
  (set to zero to disable).
- ``HYPRE_SStructSSAMGSetLogging``: Enable internal logging.
- ``HYPRE_SStructSSAMGSetPrintLevel`` / ``SetPrintFreq``: Control verbosity and
  output frequency.
- ``HYPRE_SStructSSAMGGetNumIterations`` / ``GetFinalRelativeResidualNorm``:
  Retrieve iteration count and final relative residual norm.


Algorithm notes
------------------------------------------------------------------------------

- **Per-part semi-coarsening**: Each part selects a dominant coupling direction
  (heuristically related to an effective grid spacing metric) and coarsens by a
  factor of two in that direction. This adapts to anisotropy that varies across
  parts.
- **Structured interpolation**: Prolongation is constructed within each part
  from the structured component of the operator and uses the transpose as
  restriction. This limits stencil growth and captures heterogeneity.
- **Hybrid coarse solve**: Delegating coarse levels to BoomerAMG can reduce the
  overall number of levels in the multigrid hierarchy while retaining efficiency
  on fine levels.


Minimal C example
------------------------------------------------------------------------------

The snippet below shows how to create and configure an SSAMG solver and solves
a problem with an assembled SStruct matrix ``A`` and vectors ``b``,
``x``. Only solver creation and configuration are shown. For better robustness,
we recommend using SSAMG as a preconditioner for a Krylov solver.

.. code-block:: c

   HYPRE_SStructSolver ssamg;
   HYPRE_SStructSSAMGCreate(MPI_COMM_WORLD, &ssamg);

   HYPRE_SStructSSAMGSetTol(ssamg, 1e-8);
   HYPRE_SStructSSAMGSetMaxIter(ssamg, 50);
   HYPRE_SStructSSAMGSetInterpType(ssamg, -1);      /* Structured-only interpolation */
   HYPRE_SStructSSAMGSetRelaxType(ssamg, 1);        /* Weighted Jacobi */
   HYPRE_SStructSSAMGSetNumPreRelax(ssamg, 1);
   HYPRE_SStructSSAMGSetNumPostRelax(ssamg, 1);
   HYPRE_SStructSSAMGSetSkipRelax(ssamg, 1);        /* Skip relaxation on certain levels */
   HYPRE_SStructSSAMGSetCoarseSolverType(ssamg, 1); /* Switch to BoomerAMG on coarse level */

   HYPRE_SStructSSAMGSetup(ssamg, A, b, x);
   HYPRE_SStructSSAMGSolve(ssamg, A, b, x);

   HYPRE_Int iters; HYPRE_Real relres;
   HYPRE_SStructSSAMGGetNumIterations(ssamg, &iters);
   HYPRE_SStructSSAMGGetFinalRelativeResidualNorm(ssamg, &relres);

   HYPRE_SStructSSAMGDestroy(ssamg);

