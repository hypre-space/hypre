.. Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
   HYPRE Project Developers. See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (Apache-2.0 OR MIT)

.. _ilu:

ILU
==============================================================================

ILU is a suite of parallel incomplete LU factorization algorithms featuring dual threshold
(ILUT) and level-based (ILUK) variants. The implementation is based on a domain
decomposition framework for achieving distributed parallelism. ILU can be used as a
standalone iterative solver (this is not recommended), preconditioner for Krylov subspace
methods, or smoother for multigrid methods such as BoomerAMG and MGR.

.. note::
   ILU is currently only supported by the IJ interface.

Overview
------------------------------------------------------------------------------

ILU utilizes a domain decomposition framework. A basic block-Jacobi approach involves
performing inexact solutions within the local domains owned by the processes, using
parallel local ILU factorizations. In a more advanced approach, the unknowns are
partitioned into interior and interface points, where the interface points separate the
interior points in adjacent domains. In an algebraic context, this is equivalent to
dividing the matrix rows into local (processor-owned) and external (off-processor-owned)
blocks. The overall parallel ILU strategy is a two-level method that consists of ILU
solves within the local blocks and a global solve involving the Schur complement system,
which various iterative approaches in this framework can solve.

User-level functions
------------------------------------------------------------------------------

A list of user-level functions for configuring ILU is given below, where each block
of functions is marked as *Required*, *Recommended*, *Optional*, or *Exclusively
required*. Note that the last two blocks of function calls are exclusively required, i.e.,
the first block should be called only when ILU is used as a standalone solver, while
the second block should be called only when it is used as a preconditioner to GMRES. In
the last case, other Krylov methods can be chosen. We refer the reader to
:ref:`ch-Solvers` for more information.

.. code-block:: c

 /* (Required) Create ILU solver */
 HYPRE_ILUCreate(&ilu_solver);

 /* (Recommended) General solver options */
 HYPRE_ILUSetType(ilu_solver, ilu_type); /* 0, 1, 10, 11, 20, 21, 30, 31, 40, 41, 50 */
 HYPRE_ILUSetMaxIter(ilu_solver, max_iter);
 HYPRE_ILUSetTol(ilu_solver, tol);
 HYPRE_ILUSetLocalReordering(ilu_solver, reordering); /* 0: none, 1: RCM */
 HYPRE_ILUSetPrintLevel(ilu_solver, print_level);

 /* (Optional) Function calls for ILUK variants */
 HYPRE_ILUSetLevelOfFill(ilu_solver, fill);

 /* (Optional) Function calls for ILUT variants */
 HYPRE_ILUSetMaxNnzPerRow(ilu_solver, max_nnz_row);
 HYPRE_ILUSetDropThreshold(ilu_solver, threshold);

 /* (Optional) Function calls for GMRES-ILU or NSH-ILU */
 HYPRE_ILUSetNSHDropThreshold(ilu_solver, threshold);
 HYPRE_ILUSetSchurMaxIter(ilu_solver, schur_max_iter);

 /* (Optional) Function calls for iterative ILU variants */
 HYPRE_ILUSetTriSolve(ilu_solver, 0);
 HYPRE_ILUSetLowerJacobiIters(ilu_solver, ljac_iters);
 HYPRE_ILUSetUpperJacobiIters(ilu_solver, ujac_iters);

 /* (Exclusively required) Function calls for using ILU as standalone solver */
 HYPRE_ILUSetup(ilu_solver, parcsr_M, b, x);
 HYPRE_ILUSolve(ilu_solver, parcsr_A, b, x);

 /* (Exclusively required) Function calls for using ILU as preconditioner to GMRES */
 HYPRE_GMRESSetup(gmres_solver, (HYPRE_Matrix)A, (HYPRE_Vector)b, (HYPRE_Vector)x);
 HYPRE_GMRESSolve(gmres_solver, (HYPRE_Matrix)A, (HYPRE_Vector)b, (HYPRE_Vector)x);

 /* (Required) Free memory */
 HYPRE_ILUDestroy(ilu_solver);

A short explanation for each of those functions and its parameters is given next.

* ``HYPRE_ILUCreate`` Create the hypre_ILU solver object.
* ``HYPRE_ILUDestroy`` Destroy the hypre_ILU solver object.
* ``HYPRE_ILUSetType`` Set the type of ILU factorization. Options are:

  * 0:  Block-Jacobi ILUK (BJ-ILUK).
  * 1:  Block-Jacobi ILUT (BJ-ILUT).
  * 10: GMRES with ILUK (GMRES-ILUK).
  * 11: GMRES with ILUT (GMRES-ILUT).
  * 20: NSH with ILUK (NSH-ILUK).
  * 21: NSH with ILUT (NSH-ILUT).
  * 30: RAS with ILUK (RAS-ILUK).
  * 31: RAS with ILUT (RAS-ILUT).
  * 40: ddPQ-GMRES with ILUK (ddPQ-GMRES-ILUK).
  * 41: ddPQ-GMRES with ILUT (ddPQ-GMRES-ILUT).
  * 50: GMRES with RAP-ILU0 with modified ILU0 (GMRES-RAP-ILU0).
* ``HYPRE_ILUSetMaxIter`` Set the maximum number of ILU iterations. We recommend setting
  this value to one when ILU is used as a preconditioner or smoother.
* ``HYPRE_ILUSetTol`` Set the convergence tolerance for ILU. We recommend setting
  this value to zero when ILU is used as a preconditioner or smoother.
* ``HYPRE_ILUSetLocalReordering`` Set the local matrix reordering algorithm.

  * 0: No reordering.
  * 1: Reverse Cuthill–McKee (RCM).
* ``HYPRE_ILUSetPrintLevel`` Set the verbosity level for algorithm statistics.

  * 0: No output.
  * 1: Print setup info.
  * 2: Print solve info.
  * 3: Print setup and solve info.
* ``HYPRE_ILUSetLevelOfFill`` Set the level of fill used by the level-based ILUK
  strategy.
* ``HYPRE_ILUSetMaxNnzPerRow`` Set the maximum number of nonzero entries per row in the
  triangular factors for ILUT.
* ``HYPRE_ILUSetDropThreshold`` Set the threshold for dropping nonzero entries during the
  construction of the triangular factors for ILUT.
* ``HYPRE_ILUSetNSHDropThreshold`` Set the threshold for dropping nonzero entries during the
  computation of the approximate inverse matrix via NSH-ILU.
* ``HYPRE_ILUSetSchurMaxIter`` Set the maximum number of iterations for solving
  the Schur complement system (GMRES-ILU or NSH-ILU).
* ``HYPRE_ILUSetTriSolve`` Set triangular solve method used in ILU's solve phase. Option zero
  refers to the iterative approach, which leads to good performance in GPUs, and option
  one refers to the direct (exact) approach.
* ``HYPRE_ILUSetLowerJacobiIters`` Set the number of iterations for solving the lower
  triangular linear system. This option makes sense when enabling the iterative triangular
  solve approach.
* ``HYPRE_ILUSetUpperJacobiIters`` Same as previous function, but for the upper
  triangular factor.
* ``HYPRE_ILUSetup`` Setup a hypre_ILU solver object.
* ``HYPRE_ILUSolve`` Solve the linear system with hypre_ILU.
* ``HYPRE_ILUDestroy`` Destroy the hypre_ILU solver object.

.. note::
   For more details about ILU options and parameters, including their default
   values, we refer the reader to hypre's reference manual or section :ref:`sec-ParCSR-Solvers`.

.. _ilu-amg-smoother:

ILU as Smoother for BoomerAMG
------------------------------------------------------------------------------

The following functions can be used to configure ILU as a smoother to BoomerAMG:

.. code-block:: c

 /* (Required) Set ILU as smoother to BoomerAMG */
 HYPRE_BoomerAMGSetSmoothType(amg_solver, 5);
 HYPRE_BoomerAMGSetSmoothNumLevels(amg_solver, num_levels);

 /* (Optional) General ILU configuration parameters */
 HYPRE_BoomerAMGSetILUType(amg_solver, ilu_type);
 HYPRE_BoomerAMGSetILUMaxIter(amg_solver, ilu_max_iter);
 HYPRE_BoomerAMGSetILULocalReordering(amg_solver, ilu_reordering);

 /* (Optional) Function calls for ILUK smoother variants */
 HYPRE_BoomerAMGSetILULevel(amg_solver, ilu_fill);

 /* (Optional) Function calls for ILUT smoother variants */
 HYPRE_BoomerAMGSetILUDroptol(amg_solver, ilu_threshold);
 HYPRE_BoomerAMGSetILUMaxRowNnz(amg_solver, ilu_max_nnz_row);

 /* (Optional) Function calls for iterative ILU smoother variants */
 HYPRE_BoomerAMGSetILUTriSolve(amg_solver, 0);
 HYPRE_BoomerAMGSetILULowerJacobiIters(amg_solver, ilu_ljac_iters);
 HYPRE_BoomerAMGSetILUUpperJacobiIters(amg_solver, ilu_ujac_iters);

where:

* ``HYPRE_BoomerAMGSetSmoothNumLevels`` Enable smoothing in the first ``num_levels``
  levels of AMG.
* ``HYPRE_BoomerAMGSetILUType`` Set the type of ILU factorization. See ``HYPRE_ILUSetType``.
* ``HYPRE_BoomerAMGSetILUMaxIter`` Set the number of ILU smoother sweeps.
* ``HYPRE_BoomerAMGSetILULocalReordering`` Set the local matrix reordering algorithm.
* ``HYPRE_BoomerAMGSetILULevel`` Set ILUK's fill level.
* ``HYPRE_BoomerAMGSetILUDroptol`` Set ILUT's threshold.
* ``HYPRE_BoomerAMGSetILUMaxRowNnz`` Set ILUT's maximum number of nonzero entries per row.
* ``HYPRE_BoomerAMGSetILUTriSolve`` Set triangular solve method. See ``HYPRE_ILUSetTriSolve``.
* ``HYPRE_BoomerAMGSetILULowerJacobiIters`` Set the number of iterations for the L factor.
* ``HYPRE_BoomerAMGSetILUUpperJacobiIters`` Same as previous function, but for the U factor.

GPU support
------------------------------------------------------------------------------

The addition of GPU support to ILU is ongoing work. A few algorithm types have already
been fully ported to the CUDA and HIP backends, i.e., both their setup (factorization) and
solve phases are executed on the device. Below is a detailed list of which phases (setup
and solve) of the various ILU algorithms have been ported to GPUs. In the table,
*UVM-Setup* indicates that the setup phase is executed on the CPU (host); at the same
time, the triangular factors are stored in a memory space that is accessible from the GPU
(device) via unified memory. This feature must be enabled during hypre's configuration.

.. list-table::
   :widths: 20 20 20 20
   :header-rows: 1

   * -
     - CUDA (NVIDIA GPUs)
     - HIP (AMD GPUs)
     - SYCL (Intel GPUs)
   * - **BJ-ILU0**
     - Setup and Solve
     - Setup and Solve
     - UVM-Setup and Solve
   * - **BJ-ILU(K/T)**
     - UVM-Setup and Solve
     - UVM-Setup and Solve
     - UVM-Setup and Solve
   * - **GMRES-ILU0**
     - Setup and Solve
     - Setup and Solve
     - UVM-Setup and Solve
   * - **GMRES-RAP-ILU0**
     - UVM-Setup and Solve
     - UVM-Setup and Solve
     - UVM-Setup and Solve
   * - **GMRES-ILU(K/T)**
     - UVM-Setup and Solve
     - UVM-Setup and Solve
     - UVM-Setup and Solve
   * - **ddPQ-GMRES-ILU(K/T)**
     - UVM-Setup and Solve
     - UVM-Setup and Solve
     - UVM-Setup and Solve
   * - **NSH-ILU(K/T)**
     - UVM-Setup and Solve
     - UVM-Setup and Solve
     - UVM-Setup and Solve
   * - **RAS-ILU(K/T)**
     - UVM-Setup and Solve
     - UVM-Setup and Solve
     - UVM-Setup and Solve

.. hint::
   For better setup performance on GPUs, disable local reordering by passing option
   zero to ``HYPRE_ILUSetLocalReordering`` or
   ``HYPRE_BoomerAMGSetILULocalReordering``. This may degrade convergence of the iterative
   solver.

.. note::
   hypre must be built with ``cuSPARSE`` support when running ILU on NVIDIA
   GPUs, ``rocSPARSE`` when running on AMD GPUs, or ``oneMKL sparse`` when
   running on Intel GPUs.
