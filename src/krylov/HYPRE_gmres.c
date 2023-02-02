/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_GMRES interface
 *
 *****************************************************************************/
#include "krylov.h"

/*--------------------------------------------------------------------------
 * HYPRE_GMRESDestroy
 *--------------------------------------------------------------------------*/
/* to do, not trivial */
/*
HYPRE_Int
HYPRE_ParCSRGMRESDestroy( HYPRE_Solver solver )
{
   return( hypre_GMRESDestroy( (void *) solver ) );
}
*/

/*--------------------------------------------------------------------------
 * HYPRE_GMRESSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_GMRESSetup( HYPRE_Solver solver,
                  HYPRE_Matrix A,
                  HYPRE_Vector b,
                  HYPRE_Vector x      )
{
   return ( hypre_GMRESSetup( solver, A, b, x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_GMRESSolve
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_GMRESSolve( HYPRE_Solver solver,
                  HYPRE_Matrix A,
                  HYPRE_Vector b,
                  HYPRE_Vector x      )
{
   return ( hypre_GMRESSolve( solver, A, b, x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_GMRESSetKDim, HYPRE_GMRESGetKDim
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_GMRESSetKDim( HYPRE_Solver solver,
                    HYPRE_Int             k_dim    )
{
   return ( hypre_GMRESSetKDim( (void *) solver, k_dim ) );
}

HYPRE_Int
HYPRE_GMRESGetKDim( HYPRE_Solver solver,
                    HYPRE_Int           * k_dim    )
{
   return ( hypre_GMRESGetKDim( (void *) solver, k_dim ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_GMRESSetTol, HYPRE_GMRESGetTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_GMRESSetTol( HYPRE_Solver solver,
                   HYPRE_Real         tol    )
{
   return ( hypre_GMRESSetTol( (void *) solver, tol ) );
}

HYPRE_Int
HYPRE_GMRESGetTol( HYPRE_Solver solver,
                   HYPRE_Real       * tol    )
{
   return ( hypre_GMRESGetTol( (void *) solver, tol ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_GMRESSetAbsoluteTol, HYPRE_GMRESGetAbsoluteTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_GMRESSetAbsoluteTol( HYPRE_Solver solver,
                           HYPRE_Real         a_tol    )
{
   return ( hypre_GMRESSetAbsoluteTol( (void *) solver, a_tol ) );
}

HYPRE_Int
HYPRE_GMRESGetAbsoluteTol( HYPRE_Solver solver,
                           HYPRE_Real       * a_tol    )
{
   return ( hypre_GMRESGetAbsoluteTol( (void *) solver, a_tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_GMRESSetConvergenceFactorTol, HYPRE_GMRESGetConvergenceFactorTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_GMRESSetConvergenceFactorTol( HYPRE_Solver solver,
                                    HYPRE_Real         cf_tol    )
{
   return ( hypre_GMRESSetConvergenceFactorTol( (void *) solver, cf_tol ) );
}

HYPRE_Int
HYPRE_GMRESGetConvergenceFactorTol( HYPRE_Solver solver,
                                    HYPRE_Real       * cf_tol    )
{
   return ( hypre_GMRESGetConvergenceFactorTol( (void *) solver, cf_tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_GMRESSetMinIter, HYPRE_GMRESGetMinIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_GMRESSetMinIter( HYPRE_Solver solver,
                       HYPRE_Int          min_iter )
{
   return ( hypre_GMRESSetMinIter( (void *) solver, min_iter ) );
}

HYPRE_Int
HYPRE_GMRESGetMinIter( HYPRE_Solver solver,
                       HYPRE_Int        * min_iter )
{
   return ( hypre_GMRESGetMinIter( (void *) solver, min_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_GMRESSetMaxIter, HYPRE_GMRESGetMaxIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_GMRESSetMaxIter( HYPRE_Solver solver,
                       HYPRE_Int          max_iter )
{
   return ( hypre_GMRESSetMaxIter( (void *) solver, max_iter ) );
}

HYPRE_Int
HYPRE_GMRESGetMaxIter( HYPRE_Solver solver,
                       HYPRE_Int        * max_iter )
{
   return ( hypre_GMRESGetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_GMRESSetStopCrit, HYPRE_GMRESGetStopCrit - OBSOLETE
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_GMRESSetStopCrit( HYPRE_Solver solver,
                        HYPRE_Int          stop_crit )
{
   return ( hypre_GMRESSetStopCrit( (void *) solver, stop_crit ) );
}

HYPRE_Int
HYPRE_GMRESGetStopCrit( HYPRE_Solver solver,
                        HYPRE_Int        * stop_crit )
{
   return ( hypre_GMRESGetStopCrit( (void *) solver, stop_crit ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_GMRESSetRelChange, HYPRE_GMRESGetRelChange
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_GMRESSetRelChange( HYPRE_Solver solver,
                         HYPRE_Int                rel_change )
{
   return ( hypre_GMRESSetRelChange( (void *) solver, rel_change ) );
}

HYPRE_Int
HYPRE_GMRESGetRelChange( HYPRE_Solver solver,
                         HYPRE_Int              * rel_change )
{
   return ( hypre_GMRESGetRelChange( (void *) solver, rel_change ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_GMRESSetSkipRealResidualCheck, HYPRE_GMRESGetSkipRealResidualCheck
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_GMRESSetSkipRealResidualCheck( HYPRE_Solver solver,
                                     HYPRE_Int skip_real_r_check )
{
   return ( hypre_GMRESSetSkipRealResidualCheck( (void *) solver, skip_real_r_check ) );
}

HYPRE_Int
HYPRE_GMRESGetSkipRealResidualCheck( HYPRE_Solver solver,
                                     HYPRE_Int *skip_real_r_check )
{
   return ( hypre_GMRESGetSkipRealResidualCheck( (void *) solver, skip_real_r_check ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_GMRESSetPrecond
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_GMRESSetPrecond( HYPRE_Solver          solver,
                       HYPRE_PtrToSolverFcn  precond,
                       HYPRE_PtrToSolverFcn  precond_setup,
                       HYPRE_Solver          precond_solver )
{
   return ( hypre_GMRESSetPrecond( (void *) solver,
                                   (HYPRE_Int (*)(void*, void*, void*, void*))precond,
                                   (HYPRE_Int (*)(void*, void*, void*, void*))precond_setup,
                                   (void *) precond_solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_GMRESGetPrecond
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_GMRESGetPrecond( HYPRE_Solver  solver,
                       HYPRE_Solver *precond_data_ptr )
{
   return ( hypre_GMRESGetPrecond( (void *)     solver,
                                   (HYPRE_Solver *) precond_data_ptr ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_GMRESSetPrintLevel, HYPRE_GMRESGetPrintLevel
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_GMRESSetPrintLevel( HYPRE_Solver solver,
                          HYPRE_Int          level )
{
   return ( hypre_GMRESSetPrintLevel( (void *) solver, level ) );
}

HYPRE_Int
HYPRE_GMRESGetPrintLevel( HYPRE_Solver solver,
                          HYPRE_Int        * level )
{
   return ( hypre_GMRESGetPrintLevel( (void *) solver, level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_GMRESSetLogging, HYPRE_GMRESGetLogging
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_GMRESSetLogging( HYPRE_Solver solver,
                       HYPRE_Int          level )
{
   return ( hypre_GMRESSetLogging( (void *) solver, level ) );
}

HYPRE_Int
HYPRE_GMRESGetLogging( HYPRE_Solver solver,
                       HYPRE_Int        * level )
{
   return ( hypre_GMRESGetLogging( (void *) solver, level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_GMRESGetNumIterations
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_GMRESGetNumIterations( HYPRE_Solver  solver,
                             HYPRE_Int                *num_iterations )
{
   return ( hypre_GMRESGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_GMRESGetConverged
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_GMRESGetConverged( HYPRE_Solver  solver,
                         HYPRE_Int                *converged )
{
   return ( hypre_GMRESGetConverged( (void *) solver, converged ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_GMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_GMRESGetFinalRelativeResidualNorm( HYPRE_Solver  solver,
                                         HYPRE_Real         *norm   )
{
   return ( hypre_GMRESGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_GMRESGetResidual
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_GMRESGetResidual( HYPRE_Solver solver, void *residual )
{
   /* returns a pointer to the residual vector */
   return hypre_GMRESGetResidual( (void *) solver, (void **) residual );
}

