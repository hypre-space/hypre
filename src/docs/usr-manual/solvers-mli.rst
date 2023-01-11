.. Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
   HYPRE Project Developers. See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (Apache-2.0 OR MIT)


The MLI Package
==============================================================================
                                                                                   
MLI is an object-oriented module that implements the class of algebraic
multigrid algorithms based on Vanek and Brezina's smoothed aggregation method
[VaMB1996]_, [VaBM2001]_.  There are two main algorithms in this module - the
original smoothed aggregation algorithm and the modified version that uses the
finite element substructure matrices to construct the prolongation operators.
As such, the later algorithm can only be used in the finite element context via
the finite element interface.  In addition, the nodal coordinates obtained via
the finite element interface can be used to construct a better prolongation
operator than the pure translation modes.

Below is an example on how to set up MLI as a preconditioner for conjugate
gradient.

.. code-block:: c
   
   HYPRE_LSI_MLICreate(MPI_COMM_WORLD, &pcg_precond);
   
   HYPRE_LSI_MLISetParams(pcg_precond, "MLI strengthThreshold 0.08");
   ...
   
   HYPRE_PCGSetPrecond(pcg_solver,
                       (HYPRE_PtrToSolverFcn) HYPRE_LSI_MLISolve,
                       (HYPRE_PtrToSolverFcn) HYPRE_LSI_MLISetup,
                       pcg_precond);

Note that parameters are set via ``HYPRE_LSI_MLISetParams``. A list of valid
parameters that can be set using this routine can be found in the FEI section of
the reference manual.

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
|
|
|
