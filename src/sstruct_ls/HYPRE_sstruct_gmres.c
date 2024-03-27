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
HYPRE_SStructGMRESCreate( MPI_Comm             comm,
                          HYPRE_SStructSolver *solver )
{
   HYPRE_UNUSED_VAR(comm);

   hypre_GMRESFunctions * gmres_functions =
      hypre_GMRESFunctionsCreate(
         hypre_SStructKrylovCAlloc, hypre_SStructKrylovFree, hypre_SStructKrylovCommInfo,
         hypre_SStructKrylovCreateVector,
         hypre_SStructKrylovCreateVectorArray,
         hypre_SStructKrylovDestroyVector, hypre_SStructKrylovMatvecCreate,
         hypre_SStructKrylovMatvec, hypre_SStructKrylovMatvecDestroy,
         hypre_SStructKrylovInnerProd, hypre_SStructKrylovCopyVector,
         hypre_SStructKrylovClearVector,
         hypre_SStructKrylovScaleVector, hypre_SStructKrylovAxpy,
         hypre_SStructKrylovIdentitySetup, hypre_SStructKrylovIdentity );

   *solver = ( (HYPRE_SStructSolver) hypre_GMRESCreate( gmres_functions ) );

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructGMRESDestroy( HYPRE_SStructSolver solver )
{
   return ( hypre_GMRESDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructGMRESSetup( HYPRE_SStructSolver solver,
                         HYPRE_SStructMatrix A,
                         HYPRE_SStructVector b,
                         HYPRE_SStructVector x )
{
   return ( HYPRE_GMRESSetup( (HYPRE_Solver) solver,
                              (HYPRE_Matrix) A,
                              (HYPRE_Vector) b,
                              (HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructGMRESSolve( HYPRE_SStructSolver solver,
                         HYPRE_SStructMatrix A,
                         HYPRE_SStructVector b,
                         HYPRE_SStructVector x )
{
   return ( HYPRE_GMRESSolve( (HYPRE_Solver) solver,
                              (HYPRE_Matrix) A,
                              (HYPRE_Vector) b,
                              (HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructGMRESSetKDim( HYPRE_SStructSolver solver,
                           HYPRE_Int           k_dim )
{
   return ( HYPRE_GMRESSetKDim( (HYPRE_Solver) solver, k_dim ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructGMRESSetTol( HYPRE_SStructSolver solver,
                          HYPRE_Real          tol )
{
   return ( HYPRE_GMRESSetTol( (HYPRE_Solver) solver, tol ) );
}
/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructGMRESSetAbsoluteTol( HYPRE_SStructSolver solver,
                                  HYPRE_Real          atol )
{
   return ( HYPRE_GMRESSetAbsoluteTol( (HYPRE_Solver) solver, atol ) );
}
/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructGMRESSetMinIter( HYPRE_SStructSolver solver,
                              HYPRE_Int           min_iter )
{
   return ( HYPRE_GMRESSetMinIter( (HYPRE_Solver) solver, min_iter ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructGMRESSetMaxIter( HYPRE_SStructSolver solver,
                              HYPRE_Int           max_iter )
{
   return ( HYPRE_GMRESSetMaxIter( (HYPRE_Solver) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructGMRESSetStopCrit( HYPRE_SStructSolver solver,
                               HYPRE_Int           stop_crit )
{
   return ( HYPRE_GMRESSetStopCrit( (HYPRE_Solver) solver, stop_crit ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructGMRESSetPrecond( HYPRE_SStructSolver          solver,
                              HYPRE_PtrToSStructSolverFcn  precond,
                              HYPRE_PtrToSStructSolverFcn  precond_setup,
                              void *          precond_data )
{
   return ( HYPRE_GMRESSetPrecond( (HYPRE_Solver) solver,
                                   (HYPRE_PtrToSolverFcn) precond,
                                   (HYPRE_PtrToSolverFcn) precond_setup,
                                   (HYPRE_Solver) precond_data ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructGMRESSetLogging( HYPRE_SStructSolver solver,
                              HYPRE_Int           logging )
{
   return ( HYPRE_GMRESSetLogging( (HYPRE_Solver) solver, logging ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructGMRESSetPrintLevel( HYPRE_SStructSolver solver,
                                 HYPRE_Int           level )
{
   return ( HYPRE_GMRESSetPrintLevel( (HYPRE_Solver) solver, level ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructGMRESGetNumIterations( HYPRE_SStructSolver  solver,
                                    HYPRE_Int           *num_iterations )
{
   return ( HYPRE_GMRESGetNumIterations( (HYPRE_Solver) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructGMRESGetFinalRelativeResidualNorm( HYPRE_SStructSolver  solver,
                                                HYPRE_Real          *norm )
{
   return ( HYPRE_GMRESGetFinalRelativeResidualNorm( (HYPRE_Solver) solver, norm ) );
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructGMRESGetResidual( HYPRE_SStructSolver  solver,
                               void              **residual )
{
   return ( HYPRE_GMRESGetResidual( (HYPRE_Solver) solver, residual ) );
}
