/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_sstruct_ls.h"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructBiCGSTABCreate( MPI_Comm             comm,
                             HYPRE_SStructSolver *solver )
{
   HYPRE_UNUSED_VAR(comm);

   hypre_BiCGSTABFunctions * bicgstab_functions =
      hypre_BiCGSTABFunctionsCreate(
         hypre_SStructKrylovCreateVector,
         hypre_SStructKrylovDestroyVector, hypre_SStructKrylovMatvecCreate,
         hypre_SStructKrylovMatvec, hypre_SStructKrylovMatvecDestroy,
         hypre_SStructKrylovInnerProd, hypre_SStructKrylovCopyVector,
         hypre_SStructKrylovClearVector,
         hypre_SStructKrylovScaleVector, hypre_SStructKrylovAxpy,
         hypre_SStructKrylovCommInfo,
         hypre_SStructKrylovIdentitySetup, hypre_SStructKrylovIdentity );

   *solver = ( (HYPRE_SStructSolver) hypre_BiCGSTABCreate( bicgstab_functions ) );

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructBiCGSTABDestroy( HYPRE_SStructSolver solver )
{
   return ( hypre_BiCGSTABDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructBiCGSTABSetup( HYPRE_SStructSolver solver,
                            HYPRE_SStructMatrix A,
                            HYPRE_SStructVector b,
                            HYPRE_SStructVector x )
{
   return ( HYPRE_BiCGSTABSetup( (HYPRE_Solver) solver,
                                 (HYPRE_Matrix) A,
                                 (HYPRE_Vector) b,
                                 (HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructBiCGSTABSolve( HYPRE_SStructSolver solver,
                            HYPRE_SStructMatrix A,
                            HYPRE_SStructVector b,
                            HYPRE_SStructVector x )
{
   return ( HYPRE_BiCGSTABSolve( (HYPRE_Solver) solver,
                                 (HYPRE_Matrix) A,
                                 (HYPRE_Vector) b,
                                 (HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructBiCGSTABSetTol( HYPRE_SStructSolver solver,
                             HYPRE_Real          tol )
{
   return ( HYPRE_BiCGSTABSetTol( (HYPRE_Solver) solver, tol ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructBiCGSTABSetAbsoluteTol( HYPRE_SStructSolver solver,
                                     HYPRE_Real          tol )
{
   return ( HYPRE_BiCGSTABSetAbsoluteTol( (HYPRE_Solver) solver, tol ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructBiCGSTABSetMinIter( HYPRE_SStructSolver solver,
                                 HYPRE_Int           min_iter )
{
   return ( HYPRE_BiCGSTABSetMinIter( (HYPRE_Solver) solver, min_iter ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructBiCGSTABSetMaxIter( HYPRE_SStructSolver solver,
                                 HYPRE_Int           max_iter )
{
   return ( HYPRE_BiCGSTABSetMaxIter( (HYPRE_Solver) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructBiCGSTABSetStopCrit( HYPRE_SStructSolver solver,
                                  HYPRE_Int           stop_crit )
{
   return ( HYPRE_BiCGSTABSetStopCrit( (HYPRE_Solver) solver, stop_crit ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructBiCGSTABSetPrecond( HYPRE_SStructSolver          solver,
                                 HYPRE_PtrToSStructSolverFcn  precond,
                                 HYPRE_PtrToSStructSolverFcn  precond_setup,
                                 void *          precond_data )
{
   return ( HYPRE_BiCGSTABSetPrecond( (HYPRE_Solver) solver,
                                      (HYPRE_PtrToSolverFcn) precond,
                                      (HYPRE_PtrToSolverFcn) precond_setup,
                                      (HYPRE_Solver) precond_data ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructBiCGSTABSetLogging( HYPRE_SStructSolver solver,
                                 HYPRE_Int           logging )
{
   return ( HYPRE_BiCGSTABSetLogging( (HYPRE_Solver) solver, logging ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructBiCGSTABSetPrintLevel( HYPRE_SStructSolver solver,
                                    HYPRE_Int           print_level )
{
   return ( HYPRE_BiCGSTABSetPrintLevel( (HYPRE_Solver) solver, print_level ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructBiCGSTABGetNumIterations( HYPRE_SStructSolver  solver,
                                       HYPRE_Int           *num_iterations )
{
   return ( HYPRE_BiCGSTABGetNumIterations( (HYPRE_Solver) solver,
                                            num_iterations ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructBiCGSTABGetFinalRelativeResidualNorm( HYPRE_SStructSolver  solver,
                                                   HYPRE_Real          *norm )
{
   return ( HYPRE_BiCGSTABGetFinalRelativeResidualNorm( (HYPRE_Solver) solver,
                                                        norm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructBiCGSTABGetResidual
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructBiCGSTABGetResidual( HYPRE_SStructSolver  solver,
                                  void          **residual)
{
   return ( HYPRE_BiCGSTABGetResidual( (HYPRE_Solver) solver, residual ) );
}
