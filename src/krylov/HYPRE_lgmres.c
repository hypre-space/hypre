/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_LGMRES interface
 *
 *****************************************************************************/
#include "krylov.h"

/*--------------------------------------------------------------------------
 * HYPRE_LGMRESDestroy
 *--------------------------------------------------------------------------*/
/* to do, not trivial */
/*
HYPRE_Int
HYPRE_ParCSRLGMRESDestroy( HYPRE_Solver solver )
{
   return( hypre_LGMRESDestroy( (void *) solver ) );
}
*/

/*--------------------------------------------------------------------------
 * HYPRE_LGMRESSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_LGMRESSetup( HYPRE_Solver solver,
                   HYPRE_Matrix A,
                   HYPRE_Vector b,
                   HYPRE_Vector x      )
{
   return ( hypre_LGMRESSetup( solver, A, b, x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_LGMRESSolve
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_LGMRESSolve( HYPRE_Solver solver,
                   HYPRE_Matrix A,
                   HYPRE_Vector b,
                   HYPRE_Vector x      )
{
   return ( hypre_LGMRESSolve( solver, A, b, x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_LGMRESSetKDim, HYPRE_LGMRESGetKDim
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_LGMRESSetKDim( HYPRE_Solver solver,
                     HYPRE_Int             k_dim    )
{
   return ( hypre_LGMRESSetKDim( (void *) solver, k_dim ) );
}

HYPRE_Int
HYPRE_LGMRESGetKDim( HYPRE_Solver solver,
                     HYPRE_Int           * k_dim    )
{
   return ( hypre_LGMRESGetKDim( (void *) solver, k_dim ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_LGMRESSetAugDim, HYPRE_LGMRESGetAugDim
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_LGMRESSetAugDim( HYPRE_Solver solver,
                       HYPRE_Int             aug_dim    )
{
   return ( hypre_LGMRESSetAugDim( (void *) solver, aug_dim ) );
}

HYPRE_Int
HYPRE_LGMRESGetAugDim( HYPRE_Solver solver,
                       HYPRE_Int           * aug_dim    )
{
   return ( hypre_LGMRESGetAugDim( (void *) solver, aug_dim ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_LGMRESSetTol, HYPRE_LGMRESGetTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_LGMRESSetTol( HYPRE_Solver solver,
                    HYPRE_Real         tol    )
{
   return ( hypre_LGMRESSetTol( (void *) solver, tol ) );
}

HYPRE_Int
HYPRE_LGMRESGetTol( HYPRE_Solver solver,
                    HYPRE_Real       * tol    )
{
   return ( hypre_LGMRESGetTol( (void *) solver, tol ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_LGMRESSetAbsoluteTol, HYPRE_LGMRESGetAbsoluteTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_LGMRESSetAbsoluteTol( HYPRE_Solver solver,
                            HYPRE_Real         a_tol    )
{
   return ( hypre_LGMRESSetAbsoluteTol( (void *) solver, a_tol ) );
}

HYPRE_Int
HYPRE_LGMRESGetAbsoluteTol( HYPRE_Solver solver,
                            HYPRE_Real       * a_tol    )
{
   return ( hypre_LGMRESGetAbsoluteTol( (void *) solver, a_tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_LGMRESSetConvergenceFactorTol, HYPRE_LGMRESGetConvergenceFactorTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_LGMRESSetConvergenceFactorTol( HYPRE_Solver solver,
                                     HYPRE_Real         cf_tol    )
{
   return ( hypre_LGMRESSetConvergenceFactorTol( (void *) solver, cf_tol ) );
}

HYPRE_Int
HYPRE_LGMRESGetConvergenceFactorTol( HYPRE_Solver solver,
                                     HYPRE_Real       * cf_tol    )
{
   return ( hypre_LGMRESGetConvergenceFactorTol( (void *) solver, cf_tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_LGMRESSetMinIter, HYPRE_LGMRESGetMinIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_LGMRESSetMinIter( HYPRE_Solver solver,
                        HYPRE_Int          min_iter )
{
   return ( hypre_LGMRESSetMinIter( (void *) solver, min_iter ) );
}

HYPRE_Int
HYPRE_LGMRESGetMinIter( HYPRE_Solver solver,
                        HYPRE_Int        * min_iter )
{
   return ( hypre_LGMRESGetMinIter( (void *) solver, min_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_LGMRESSetMaxIter, HYPRE_LGMRESGetMaxIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_LGMRESSetMaxIter( HYPRE_Solver solver,
                        HYPRE_Int          max_iter )
{
   return ( hypre_LGMRESSetMaxIter( (void *) solver, max_iter ) );
}

HYPRE_Int
HYPRE_LGMRESGetMaxIter( HYPRE_Solver solver,
                        HYPRE_Int        * max_iter )
{
   return ( hypre_LGMRESGetMaxIter( (void *) solver, max_iter ) );
}



/*--------------------------------------------------------------------------
 * HYPRE_LGMRESSetPrecond
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_LGMRESSetPrecond( HYPRE_Solver          solver,
                        HYPRE_PtrToSolverFcn  precond,
                        HYPRE_PtrToSolverFcn  precond_setup,
                        HYPRE_Solver          precond_solver )
{
   return ( hypre_LGMRESSetPrecond( (void *) solver,
                                    (HYPRE_Int (*)(void*, void*, void*, void*))precond,
                                    (HYPRE_Int (*)(void*, void*, void*, void*))precond_setup,
                                    (void *) precond_solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_LGMRESGetPrecond
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_LGMRESGetPrecond( HYPRE_Solver  solver,
                        HYPRE_Solver *precond_data_ptr )
{
   return ( hypre_LGMRESGetPrecond( (void *)     solver,
                                    (HYPRE_Solver *) precond_data_ptr ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_LGMRESSetPrintLevel, HYPRE_LGMRESGetPrintLevel
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_LGMRESSetPrintLevel( HYPRE_Solver solver,
                           HYPRE_Int          level )
{
   return ( hypre_LGMRESSetPrintLevel( (void *) solver, level ) );
}

HYPRE_Int
HYPRE_LGMRESGetPrintLevel( HYPRE_Solver solver,
                           HYPRE_Int        * level )
{
   return ( hypre_LGMRESGetPrintLevel( (void *) solver, level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_LGMRESSetLogging, HYPRE_LGMRESGetLogging
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_LGMRESSetLogging( HYPRE_Solver solver,
                        HYPRE_Int          level )
{
   return ( hypre_LGMRESSetLogging( (void *) solver, level ) );
}

HYPRE_Int
HYPRE_LGMRESGetLogging( HYPRE_Solver solver,
                        HYPRE_Int        * level )
{
   return ( hypre_LGMRESGetLogging( (void *) solver, level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_LGMRESGetNumIterations
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_LGMRESGetNumIterations( HYPRE_Solver  solver,
                              HYPRE_Int                *num_iterations )
{
   return ( hypre_LGMRESGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_LGMRESGetConverged
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_LGMRESGetConverged( HYPRE_Solver  solver,
                          HYPRE_Int                *converged )
{
   return ( hypre_LGMRESGetConverged( (void *) solver, converged ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_LGMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_LGMRESGetFinalRelativeResidualNorm( HYPRE_Solver  solver,
                                          HYPRE_Real         *norm   )
{
   return ( hypre_LGMRESGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_LGMRESGetResidual
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_LGMRESGetResidual( HYPRE_Solver solver, void *residual )
{
   /* returns a pointer to the residual vector */
   return hypre_LGMRESGetResidual( (void *) solver, (void **) residual );
}

