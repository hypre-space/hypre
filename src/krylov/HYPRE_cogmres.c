/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_COGMRES interface
 *
 *****************************************************************************/
#include "krylov.h"

/*--------------------------------------------------------------------------
 * HYPRE_COGMRESDestroy
 *--------------------------------------------------------------------------*/
/* to do, not trivial */
/*
HYPRE_Int
HYPRE_ParCSRCOGMRESDestroy( HYPRE_Solver solver )
{
   return( hypre_COGMRESDestroy( (void *) solver ) );
}
*/

/*--------------------------------------------------------------------------
 * HYPRE_COGMRESSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_COGMRESSetup( HYPRE_Solver solver,
                    HYPRE_Matrix A,
                    HYPRE_Vector b,
                    HYPRE_Vector x      )
{
   return ( hypre_COGMRESSetup( solver, A, b, x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_COGMRESSolve
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_COGMRESSolve( HYPRE_Solver solver,
                    HYPRE_Matrix A,
                    HYPRE_Vector b,
                    HYPRE_Vector x      )
{
   return ( hypre_COGMRESSolve( solver, A, b, x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_COGMRESSetKDim, HYPRE_COGMRESGetKDim
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_COGMRESSetKDim( HYPRE_Solver solver,
                      HYPRE_Int             k_dim    )
{
   return ( hypre_COGMRESSetKDim( (void *) solver, k_dim ) );
}

HYPRE_Int
HYPRE_COGMRESGetKDim( HYPRE_Solver solver,
                      HYPRE_Int           * k_dim    )
{
   return ( hypre_COGMRESGetKDim( (void *) solver, k_dim ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_COGMRESSetUnroll, HYPRE_COGMRESGetUnroll
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_COGMRESSetUnroll( HYPRE_Solver solver,
                        HYPRE_Int             unroll    )
{
   return ( hypre_COGMRESSetUnroll( (void *) solver, unroll ) );
}

HYPRE_Int
HYPRE_COGMRESGetUnroll( HYPRE_Solver solver,
                        HYPRE_Int           * unroll    )
{
   return ( hypre_COGMRESGetUnroll( (void *) solver, unroll ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_COGMRESSetCGS, HYPRE_COGMRESGetCGS
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_COGMRESSetCGS( HYPRE_Solver solver,
                     HYPRE_Int             cgs    )
{
   return ( hypre_COGMRESSetCGS( (void *) solver, cgs ) );
}

HYPRE_Int
HYPRE_COGMRESGetCGS( HYPRE_Solver solver,
                     HYPRE_Int           * cgs    )
{
   return ( hypre_COGMRESGetCGS( (void *) solver, cgs ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_COGMRESSetTol, HYPRE_COGMRESGetTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_COGMRESSetTol( HYPRE_Solver solver,
                     HYPRE_Real         tol    )
{
   return ( hypre_COGMRESSetTol( (void *) solver, tol ) );
}

HYPRE_Int
HYPRE_COGMRESGetTol( HYPRE_Solver solver,
                     HYPRE_Real       * tol    )
{
   return ( hypre_COGMRESGetTol( (void *) solver, tol ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_COGMRESSetAbsoluteTol, HYPRE_COGMRESGetAbsoluteTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_COGMRESSetAbsoluteTol( HYPRE_Solver solver,
                             HYPRE_Real         a_tol    )
{
   return ( hypre_COGMRESSetAbsoluteTol( (void *) solver, a_tol ) );
}

HYPRE_Int
HYPRE_COGMRESGetAbsoluteTol( HYPRE_Solver solver,
                             HYPRE_Real       * a_tol    )
{
   return ( hypre_COGMRESGetAbsoluteTol( (void *) solver, a_tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_COGMRESSetConvergenceFactorTol, HYPRE_COGMRESGetConvergenceFactorTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_COGMRESSetConvergenceFactorTol( HYPRE_Solver solver,
                                      HYPRE_Real         cf_tol    )
{
   return ( hypre_COGMRESSetConvergenceFactorTol( (void *) solver, cf_tol ) );
}

HYPRE_Int
HYPRE_COGMRESGetConvergenceFactorTol( HYPRE_Solver solver,
                                      HYPRE_Real       * cf_tol    )
{
   return ( hypre_COGMRESGetConvergenceFactorTol( (void *) solver, cf_tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_COGMRESSetMinIter, HYPRE_COGMRESGetMinIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_COGMRESSetMinIter( HYPRE_Solver solver,
                         HYPRE_Int          min_iter )
{
   return ( hypre_COGMRESSetMinIter( (void *) solver, min_iter ) );
}

HYPRE_Int
HYPRE_COGMRESGetMinIter( HYPRE_Solver solver,
                         HYPRE_Int        * min_iter )
{
   return ( hypre_COGMRESGetMinIter( (void *) solver, min_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_COGMRESSetMaxIter, HYPRE_COGMRESGetMaxIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_COGMRESSetMaxIter( HYPRE_Solver solver,
                         HYPRE_Int          max_iter )
{
   return ( hypre_COGMRESSetMaxIter( (void *) solver, max_iter ) );
}

HYPRE_Int
HYPRE_COGMRESGetMaxIter( HYPRE_Solver solver,
                         HYPRE_Int        * max_iter )
{
   return ( hypre_COGMRESGetMaxIter( (void *) solver, max_iter ) );
}



/*--------------------------------------------------------------------------
 * HYPRE_COGMRESSetPrecond
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_COGMRESSetPrecond( HYPRE_Solver          solver,
                         HYPRE_PtrToSolverFcn  precond,
                         HYPRE_PtrToSolverFcn  precond_setup,
                         HYPRE_Solver          precond_solver )
{
   return ( hypre_COGMRESSetPrecond( (void *) solver,
                                     (HYPRE_Int (*)(void*, void*, void*, void*))precond,
                                     (HYPRE_Int (*)(void*, void*, void*, void*))precond_setup,
                                     (void *) precond_solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_COGMRESGetPrecond
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_COGMRESGetPrecond( HYPRE_Solver  solver,
                         HYPRE_Solver *precond_data_ptr )
{
   return ( hypre_COGMRESGetPrecond( (void *)     solver,
                                     (HYPRE_Solver *) precond_data_ptr ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_COGMRESSetPrintLevel, HYPRE_COGMRESGetPrintLevel
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_COGMRESSetPrintLevel( HYPRE_Solver solver,
                            HYPRE_Int          level )
{
   return ( hypre_COGMRESSetPrintLevel( (void *) solver, level ) );
}

HYPRE_Int
HYPRE_COGMRESGetPrintLevel( HYPRE_Solver solver,
                            HYPRE_Int        * level )
{
   return ( hypre_COGMRESGetPrintLevel( (void *) solver, level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_COGMRESSetLogging, HYPRE_COGMRESGetLogging
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_COGMRESSetLogging( HYPRE_Solver solver,
                         HYPRE_Int          level )
{
   return ( hypre_COGMRESSetLogging( (void *) solver, level ) );
}

HYPRE_Int
HYPRE_COGMRESGetLogging( HYPRE_Solver solver,
                         HYPRE_Int        * level )
{
   return ( hypre_COGMRESGetLogging( (void *) solver, level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_COGMRESGetNumIterations
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_COGMRESGetNumIterations( HYPRE_Solver  solver,
                               HYPRE_Int                *num_iterations )
{
   return ( hypre_COGMRESGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_COGMRESGetConverged
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_COGMRESGetConverged( HYPRE_Solver  solver,
                           HYPRE_Int                *converged )
{
   return ( hypre_COGMRESGetConverged( (void *) solver, converged ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_COGMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_COGMRESGetFinalRelativeResidualNorm( HYPRE_Solver  solver,
                                           HYPRE_Real         *norm   )
{
   return ( hypre_COGMRESGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_COGMRESGetResidual
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_COGMRESGetResidual( HYPRE_Solver solver, void *residual )
{
   /* returns a pointer to the residual vector */
   return hypre_COGMRESGetResidual( (void *) solver, (void **) residual );

}

/*--------------------------------------------------------------------------
 * HYPRE_COGMRESSetModifyPC
 *--------------------------------------------------------------------------*/


HYPRE_Int HYPRE_COGMRESSetModifyPC( HYPRE_Solver  solver,
                                    HYPRE_Int (*modify_pc)(HYPRE_Solver, HYPRE_Int, HYPRE_Real) )
{
   return hypre_COGMRESSetModifyPC( (void *) solver, (HYPRE_Int(*)(void*, HYPRE_Int,
                                                                   HYPRE_Real))modify_pc);

}


