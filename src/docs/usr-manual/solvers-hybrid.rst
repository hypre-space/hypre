.. Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
   HYPRE Project Developers. See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (Apache-2.0 OR MIT)


Hybrid
==============================================================================

The hybrid solver is designed to detect whether a multigrid preconditioner is
needed when solving a linear system and possibly avoid the expensive setup of a
preconditioner if a system can be solved efficiently with a diagonally scaled
Krylov solver, e.g. a strongly diagonally dominant system.  It first uses a
diagonally scaled Krylov solver, which can be chosen by the user (the default is
conjugate gradient, but one should use GMRES if the matrix of the linear system
to be solved is nonsymmetric). It monitors how fast the Krylov solver converges.
If there is not sufficient progress, the algorithm switches to a preconditioned
Krylov solver.

If used through the ``Struct`` interface, the solver is called StructHybrid and
can be used with the preconditioners SMG and PFMG (default).  It is called
ParCSRHybrid, if used through the ``IJ`` interface and is used here with
BoomerAMG.  The user can determine the average convergence speed by setting a
convergence tolerance :math:`0 \leq \theta < 1` via the routine
``HYPRE_StructHybridSetConvergenceTol`` or
``HYPRE_ParCSRHybridSetConvergenceTol``.  The default setting is 0.9.

The average convergence factor :math:`\rho_i = \left({{\| r_i \|} \over {\| r_0
\|}}\right)^{1/i}` is monitored within the chosen Krylov solver, where
:math:`r_i = b - Ax_{i}` is the :math:`i`-th residual.  Convergence is
considered too slow when

.. math::

   \left( 1 - {{|\rho_i - \rho_{i-1}|} \over { \max(\rho_i, \rho_{i-1})}} \right) \rho_i > \theta .

When this condition is fulfilled the hybrid solver switches from a diagonally
scaled Krylov solver to a preconditioned solver.

.. Add blank lines to help with navigation pane formatting
|
|
|
|
|
|
|
|
|
|
|
|
