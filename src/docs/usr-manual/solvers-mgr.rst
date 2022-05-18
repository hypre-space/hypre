.. Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
   HYPRE Project Developers. See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (Apache-2.0 OR MIT)


Multigrid Reduction (MGR)
==============================================================================
                                                                                   
MGR is a parallel multigrid reduction solver and preconditioner designed to take
advantage of use-provided information to solve systems of equations with
multiple vatiable types.  The algorithm is similar to two-stage preconditioner
strategies and other reduction techniques like ARMS, but in a standard multigrid
framework.

The MGR algorithm accepts information about the variables in block form from the
user and uses it to define the appropriate C/F splitting for the multigrid
scheme.  The linear system solve proceeds with an F-relaxation solve on the F
points, folowed by a coarse grid correction. The coarse grid solve is handled by
scalar AMG (BoomerAMG). MGR provides users with more control over the coarsening
process, and can potentially be a starting point for designing multigrid-based
physics-based preconditioners.

The following represents a minimal set of functions, and some optional
functions, to call to use the MGR solver. For simplicity, we ignore the function
parameters here, and refer the reader to the reference manual for more details
on the parameters and their defaults.


* ``HYPRE_MGRCreate:`` Create the MGR solver object.
* ``HYPRE_MGRSetCpointsByBlock:`` Set up block data with information about
  coarse indexes for reduction. Here, the user specifies the number of reduction
  levels, as well as the the coarse nodes for each level of the reduction. These
  coarse nodes are indexed by their index in the block of unknowns.  This is
  used internally to tag the appropriate indexes of the linear system matrix as
  coarse nodes.
* (Optional) ``HYPRE_MGRSetReservedCoarseNodes:`` Prescribe a subset of nodes to
  be kept as coarse nodes until the coarsest level. These nodes are transferred
  onto the coarsest grid of the BoomerAMG coarse grid solver.
* (Optional) ``HYPRE_MGRSetNonCpointsToFpoints:`` Set points not prescribed as C
  points to be fixed as F points for intermediate levels. Setting this to 1 uses
  the user input to define the C/F splitting.  Otherwise, a BoomerAMG coarsening
  routine is used to determine the C/F splitting for intermediate levels.
* (Optional) ``HYPRE_MGRSetCoarseSolver:`` This function sets the BoomerAMG
  solver to be used for the solve on the coarse grid. The user can define their
  own BoomerAMG solver with their preferred options and pass this to the MGR
  solver. Otherwise, an internal BoomerAMG solver is used as the coarse grid
  solver instead.
* ``HYPRE_MGRSetup:`` Setup and MGR solver object.
* ``HYPRE_MGRSolve:`` Solve the linear system.
* ``HYPRE_MGRDestroy:`` Destroy the MGR solver object

For more details about additional solver options and parameters, please refer to
the reference manual.  NOTE: The MGR solver is currently only supported by the
IJ interface.

