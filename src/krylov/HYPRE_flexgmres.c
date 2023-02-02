/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_FlexGMRES interface
 *
 *****************************************************************************/
#include "krylov.h"

/*--------------------------------------------------------------------------
 * HYPRE_FlexGMRESDestroy
 *--------------------------------------------------------------------------*/
/* to do, not trivial */
/*
HYPRE_Int
HYPRE_ParCSRFlexGMRESDestroy( HYPRE_Solver solver )
{
   return( hypre_FlexGMRESDestroy( (void *) solver ) );
}
*/

/*--------------------------------------------------------------------------
 * HYPRE_FlexGMRESSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_FlexGMRESSetup( HYPRE_Solver solver,
                      HYPRE_Matrix A,
                      HYPRE_Vector b,
                      HYPRE_Vector x      )
{
   return ( hypre_FlexGMRESSetup( solver, A, b, x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FlexGMRESSolve
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_FlexGMRESSolve( HYPRE_Solver solver,
                      HYPRE_Matrix A,
                      HYPRE_Vector b,
                      HYPRE_Vector x      )
{
   return ( hypre_FlexGMRESSolve( solver, A, b, x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FlexGMRESSetKDim, HYPRE_FlexGMRESGetKDim
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_FlexGMRESSetKDim( HYPRE_Solver solver,
                        HYPRE_Int             k_dim    )
{
   return ( hypre_FlexGMRESSetKDim( (void *) solver, k_dim ) );
}

HYPRE_Int
HYPRE_FlexGMRESGetKDim( HYPRE_Solver solver,
                        HYPRE_Int           * k_dim    )
{
   return ( hypre_FlexGMRESGetKDim( (void *) solver, k_dim ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FlexGMRESSetTol, HYPRE_FlexGMRESGetTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_FlexGMRESSetTol( HYPRE_Solver solver,
                       HYPRE_Real         tol    )
{
   return ( hypre_FlexGMRESSetTol( (void *) solver, tol ) );
}

HYPRE_Int
HYPRE_FlexGMRESGetTol( HYPRE_Solver solver,
                       HYPRE_Real       * tol    )
{
   return ( hypre_FlexGMRESGetTol( (void *) solver, tol ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_FlexGMRESSetAbsoluteTol, HYPRE_FlexGMRESGetAbsoluteTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_FlexGMRESSetAbsoluteTol( HYPRE_Solver solver,
                               HYPRE_Real         a_tol    )
{
   return ( hypre_FlexGMRESSetAbsoluteTol( (void *) solver, a_tol ) );
}

HYPRE_Int
HYPRE_FlexGMRESGetAbsoluteTol( HYPRE_Solver solver,
                               HYPRE_Real       * a_tol    )
{
   return ( hypre_FlexGMRESGetAbsoluteTol( (void *) solver, a_tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FlexGMRESSetConvergenceFactorTol, HYPRE_FlexGMRESGetConvergenceFactorTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_FlexGMRESSetConvergenceFactorTol( HYPRE_Solver solver,
                                        HYPRE_Real         cf_tol    )
{
   return ( hypre_FlexGMRESSetConvergenceFactorTol( (void *) solver, cf_tol ) );
}

HYPRE_Int
HYPRE_FlexGMRESGetConvergenceFactorTol( HYPRE_Solver solver,
                                        HYPRE_Real       * cf_tol    )
{
   return ( hypre_FlexGMRESGetConvergenceFactorTol( (void *) solver, cf_tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FlexGMRESSetMinIter, HYPRE_FlexGMRESGetMinIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_FlexGMRESSetMinIter( HYPRE_Solver solver,
                           HYPRE_Int          min_iter )
{
   return ( hypre_FlexGMRESSetMinIter( (void *) solver, min_iter ) );
}

HYPRE_Int
HYPRE_FlexGMRESGetMinIter( HYPRE_Solver solver,
                           HYPRE_Int        * min_iter )
{
   return ( hypre_FlexGMRESGetMinIter( (void *) solver, min_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FlexGMRESSetMaxIter, HYPRE_FlexGMRESGetMaxIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_FlexGMRESSetMaxIter( HYPRE_Solver solver,
                           HYPRE_Int          max_iter )
{
   return ( hypre_FlexGMRESSetMaxIter( (void *) solver, max_iter ) );
}

HYPRE_Int
HYPRE_FlexGMRESGetMaxIter( HYPRE_Solver solver,
                           HYPRE_Int        * max_iter )
{
   return ( hypre_FlexGMRESGetMaxIter( (void *) solver, max_iter ) );
}



/*--------------------------------------------------------------------------
 * HYPRE_FlexGMRESSetPrecond
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_FlexGMRESSetPrecond( HYPRE_Solver          solver,
                           HYPRE_PtrToSolverFcn  precond,
                           HYPRE_PtrToSolverFcn  precond_setup,
                           HYPRE_Solver          precond_solver )
{
   return ( hypre_FlexGMRESSetPrecond( (void *) solver,
                                       (HYPRE_Int (*)(void*, void*, void*, void*))precond,
                                       (HYPRE_Int (*)(void*, void*, void*, void*))precond_setup,
                                       (void *) precond_solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FlexGMRESGetPrecond
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_FlexGMRESGetPrecond( HYPRE_Solver  solver,
                           HYPRE_Solver *precond_data_ptr )
{
   return ( hypre_FlexGMRESGetPrecond( (void *)     solver,
                                       (HYPRE_Solver *) precond_data_ptr ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FlexGMRESSetPrintLevel, HYPRE_FlexGMRESGetPrintLevel
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_FlexGMRESSetPrintLevel( HYPRE_Solver solver,
                              HYPRE_Int          level )
{
   return ( hypre_FlexGMRESSetPrintLevel( (void *) solver, level ) );
}

HYPRE_Int
HYPRE_FlexGMRESGetPrintLevel( HYPRE_Solver solver,
                              HYPRE_Int        * level )
{
   return ( hypre_FlexGMRESGetPrintLevel( (void *) solver, level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FlexGMRESSetLogging, HYPRE_FlexGMRESGetLogging
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_FlexGMRESSetLogging( HYPRE_Solver solver,
                           HYPRE_Int          level )
{
   return ( hypre_FlexGMRESSetLogging( (void *) solver, level ) );
}

HYPRE_Int
HYPRE_FlexGMRESGetLogging( HYPRE_Solver solver,
                           HYPRE_Int        * level )
{
   return ( hypre_FlexGMRESGetLogging( (void *) solver, level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FlexGMRESGetNumIterations
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_FlexGMRESGetNumIterations( HYPRE_Solver  solver,
                                 HYPRE_Int                *num_iterations )
{
   return ( hypre_FlexGMRESGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FlexGMRESGetConverged
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_FlexGMRESGetConverged( HYPRE_Solver  solver,
                             HYPRE_Int                *converged )
{
   return ( hypre_FlexGMRESGetConverged( (void *) solver, converged ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FlexGMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_FlexGMRESGetFinalRelativeResidualNorm( HYPRE_Solver  solver,
                                             HYPRE_Real         *norm   )
{
   return ( hypre_FlexGMRESGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_FlexGMRESGetResidual
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_FlexGMRESGetResidual( HYPRE_Solver solver, void *residual )
{
   /* returns a pointer to the residual vector */
   return hypre_FlexGMRESGetResidual( (void *) solver, (void **) residual );
}

/*--------------------------------------------------------------------------
 * HYPRE_FlexGMRESSetModifyPC
 *--------------------------------------------------------------------------*/


HYPRE_Int HYPRE_FlexGMRESSetModifyPC( HYPRE_Solver  solver,
                                      HYPRE_Int (*modify_pc)(HYPRE_Solver, HYPRE_Int, HYPRE_Real) )

{
   return hypre_FlexGMRESSetModifyPC( (void *) solver, (HYPRE_Int(*)(void*, HYPRE_Int,
                                                                     HYPRE_Real))modify_pc);

}




