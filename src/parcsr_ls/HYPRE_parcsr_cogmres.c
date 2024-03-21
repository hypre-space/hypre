/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESCreate
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRCOGMRESCreate( MPI_Comm comm, HYPRE_Solver *solver )
{
   HYPRE_UNUSED_VAR(comm);

   hypre_COGMRESFunctions * cogmres_functions;

   if (!solver)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   cogmres_functions =
      hypre_COGMRESFunctionsCreate(
         hypre_ParKrylovCAlloc,
         hypre_ParKrylovFree,
         hypre_ParKrylovCommInfo,
         hypre_ParKrylovCreateVector,
         hypre_ParKrylovCreateVectorArray,
         hypre_ParKrylovDestroyVector,
         hypre_ParKrylovMatvecCreate,
         hypre_ParKrylovMatvec,
         hypre_ParKrylovMatvecDestroy,
         hypre_ParKrylovInnerProd,
         hypre_ParKrylovMassInnerProd,
         hypre_ParKrylovMassDotpTwo,
         hypre_ParKrylovCopyVector,
         //hypre_ParKrylovCopyVector,
         hypre_ParKrylovClearVector,
         hypre_ParKrylovScaleVector,
         hypre_ParKrylovAxpy,
         hypre_ParKrylovMassAxpy,
         hypre_ParKrylovIdentitySetup,
         hypre_ParKrylovIdentity );
   *solver = ( (HYPRE_Solver) hypre_COGMRESCreate( cogmres_functions ) );

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRCOGMRESDestroy( HYPRE_Solver solver )
{
   return ( hypre_COGMRESDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRCOGMRESSetup( HYPRE_Solver solver,
                          HYPRE_ParCSRMatrix A,
                          HYPRE_ParVector b,
                          HYPRE_ParVector x      )
{
   return ( HYPRE_COGMRESSetup( solver,
                                (HYPRE_Matrix) A,
                                (HYPRE_Vector) b,
                                (HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESSolve
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRCOGMRESSolve( HYPRE_Solver solver,
                          HYPRE_ParCSRMatrix A,
                          HYPRE_ParVector b,
                          HYPRE_ParVector x      )
{
   return ( HYPRE_COGMRESSolve( solver,
                                (HYPRE_Matrix) A,
                                (HYPRE_Vector) b,
                                (HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESSetKDim
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRCOGMRESSetKDim( HYPRE_Solver solver,
                            HYPRE_Int             k_dim    )
{
   return ( HYPRE_COGMRESSetKDim( solver, k_dim ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESSetUnroll
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRCOGMRESSetUnroll( HYPRE_Solver solver,
                              HYPRE_Int             unroll    )
{
   return ( HYPRE_COGMRESSetUnroll( solver, unroll ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESSetCGS
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRCOGMRESSetCGS( HYPRE_Solver solver,
                           HYPRE_Int             cgs    )
{
   return ( HYPRE_COGMRESSetCGS( solver, cgs ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESSetTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRCOGMRESSetTol( HYPRE_Solver solver,
                           HYPRE_Real         tol    )
{
   return ( HYPRE_COGMRESSetTol( solver, tol ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESSetAbsoluteTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRCOGMRESSetAbsoluteTol( HYPRE_Solver solver,
                                   HYPRE_Real         a_tol    )
{
   return ( HYPRE_COGMRESSetAbsoluteTol( solver, a_tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESSetMinIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRCOGMRESSetMinIter( HYPRE_Solver solver,
                               HYPRE_Int          min_iter )
{
   return ( HYPRE_COGMRESSetMinIter( solver, min_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESSetMaxIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRCOGMRESSetMaxIter( HYPRE_Solver solver,
                               HYPRE_Int          max_iter )
{
   return ( HYPRE_COGMRESSetMaxIter( solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESSetPrecond
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRCOGMRESSetPrecond( HYPRE_Solver          solver,
                               HYPRE_PtrToParSolverFcn  precond,
                               HYPRE_PtrToParSolverFcn  precond_setup,
                               HYPRE_Solver          precond_solver )
{
   return ( HYPRE_COGMRESSetPrecond( solver,
                                     (HYPRE_PtrToSolverFcn) precond,
                                     (HYPRE_PtrToSolverFcn) precond_setup,
                                     precond_solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESGetPrecond
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRCOGMRESGetPrecond( HYPRE_Solver  solver,
                               HYPRE_Solver *precond_data_ptr )
{
   return ( HYPRE_COGMRESGetPrecond( solver, precond_data_ptr ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESSetLogging
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRCOGMRESSetLogging( HYPRE_Solver solver,
                               HYPRE_Int logging)
{
   return ( HYPRE_COGMRESSetLogging( solver, logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESSetPrintLevel
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRCOGMRESSetPrintLevel( HYPRE_Solver solver,
                                  HYPRE_Int print_level)
{
   return ( HYPRE_COGMRESSetPrintLevel( solver, print_level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESGetNumIterations
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRCOGMRESGetNumIterations( HYPRE_Solver  solver,
                                     HYPRE_Int    *num_iterations )
{
   return ( HYPRE_COGMRESGetNumIterations( solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRCOGMRESGetFinalRelativeResidualNorm( HYPRE_Solver  solver,
                                                 HYPRE_Real   *norm   )
{
   return ( HYPRE_COGMRESGetFinalRelativeResidualNorm( solver, norm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCOGMRESGetResidual
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRCOGMRESGetResidual( HYPRE_Solver  solver,
                                HYPRE_ParVector *residual)
{
   return ( HYPRE_COGMRESGetResidual( solver, (void *) residual ) );
}
