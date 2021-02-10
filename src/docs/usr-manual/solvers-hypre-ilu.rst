.. Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
   HYPRE Project Developers. See the top-level COPYRIGHT file for details.

   SPDX-License-Identifier: (Apache-2.0 OR MIT)


hypre-ILU
==============================================================================

The hypre-ILU solver is a parallel ILU solver based on a domain decomposition 
framework. It may be use iteratively as a standalone solver or smoother, as well as a
preconditioner for accelerators like Krylov subspace methods. This solver 
implements various parallel variants of the dual threshold (truncation) incomplete 
LU factorization - ILUT, and the level-based incomplete LU factorization - ILUk.


Overview
------------------------------------------------------------------------------
The parallel hypre-ILU solver follows a domain decomposition approach for solving 
distributed sparse linear systems of equations. The strategy is to decompose the 
domain into interior and interface nodes, where an interface node separates two 
interior nodes from adjacent domains. In the purely algebraic setting, this is 
equivalent to partitioning the matrix row data into local (processor-owned) data 
and external (off-processor-owned) data. The resulting global view of the 
partitioned matrix has (diagonal) blocks corresponding to local data, and 
off-diagonal blocks corresponding to non-local data. The resulting parallel ILU 
strategy is composed of a (local) block factorization and a (global) Schur 
complement solve. Several strategies provided to efficiently solve the Schur 
complement system. 

The following represents a minimal set of functions, and some optional
functions, to call to use the hypre_ILU solver. For simplicity, we ignore the function
parameters here, and refer the reader to the reference manual for more details
on the parameters and their defaults.


* ``HYPRE_ILUCreate:`` Create the hypre_ILU solver object.
* ``HYPRE_ILUSetType:`` Set the type of ILU factorization to do. Here, the user specifies 
  one of several flavors of parallel ILU based on the different combinations of local 
  factorizations and global Schur complement solves. Please refer to the reference manual 
  for more details about the different options available to the user.
* (Optional) ``HYPRE_ILUSetLevelOfFill:`` Set the level of fill used by the level-based ILUk strategy.
* (Optional) ``HYPRE_ILUSetSchurMaxIter:`` Set the maximum number of iterations for solving 
  the Schur complement system.
* (Optional) ``HYPRE_ILUSetMaxIter:`` Set the maximum number of iterations when used as a 
  solver or smoother.
* ``HYPRE_ILUSetup:`` Setup and hypre_ILU solver object.
* ``HYPRE_ILUSolve:`` Solve the linear system.
* ``HYPRE_ILUDestroy:`` Destroy the hypre_ILU solver object

For more details about additional solver options and parameters, please refer to
the reference manual.  NOTE: The hypre_ILU solver is currently only supported by the
IJ interface.
