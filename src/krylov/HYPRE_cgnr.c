/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * HYPRE_CGNR interface
 *
 *****************************************************************************/
#include "krylov.h"

/*--------------------------------------------------------------------------
 * HYPRE_CGNRCreate does not exist.  Call the appropriate function which
 * also specifies the vector type, e.g. HYPRE_ParCSRCGNRCreate
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * HYPRE_CGNRDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_CGNRDestroy( HYPRE_Solver solver )
{
   return ( hypre_CGNRDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CGNRSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_CGNRSetup( HYPRE_Solver solver,
                 HYPRE_Matrix A,
                 HYPRE_Vector b,
                 HYPRE_Vector x      )
{
   return ( hypre_CGNRSetup( solver, A, b, x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CGNRSolve
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_CGNRSolve( HYPRE_Solver solver,
                 HYPRE_Matrix A,
                 HYPRE_Vector b,
                 HYPRE_Vector x      )
{
   return ( hypre_CGNRSolve( solver, A, b, x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CGNRSetTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_CGNRSetTol( HYPRE_Solver solver,
                  HYPRE_Real         tol    )
{
   return ( hypre_CGNRSetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CGNRSetMinIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_CGNRSetMinIter( HYPRE_Solver solver,
                      HYPRE_Int                min_iter )
{
   return ( hypre_CGNRSetMinIter( (void *) solver, min_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CGNRSetMaxIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_CGNRSetMaxIter( HYPRE_Solver solver,
                      HYPRE_Int                max_iter )
{
   return ( hypre_CGNRSetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CGNRSetStopCrit
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_CGNRSetStopCrit( HYPRE_Solver solver,
                       HYPRE_Int                stop_crit )
{
   return ( hypre_CGNRSetStopCrit( (void *) solver, stop_crit ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CGNRSetPrecond
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_CGNRSetPrecond( HYPRE_Solver         solver,
                      HYPRE_PtrToSolverFcn precond,
                      HYPRE_PtrToSolverFcn precondT,
                      HYPRE_PtrToSolverFcn precond_setup,
                      HYPRE_Solver         precond_solver )
{
   return ( hypre_CGNRSetPrecond( (void *) solver,
                                  (HYPRE_Int (*)(void*, void*, void*, void*))precond,
                                  (HYPRE_Int (*)(void*, void*, void*, void*))precondT,
                                  (HYPRE_Int (*)(void*, void*, void*, void*))precond_setup,
                                  (void *) precond_solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CGNRGetPrecond
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_CGNRGetPrecond( HYPRE_Solver   solver,
                      HYPRE_Solver  *precond_data_ptr )
{
   return ( hypre_CGNRGetPrecond( (void *)         solver,
                                  (HYPRE_Solver *) precond_data_ptr ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CGNRSetLogging
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_CGNRSetLogging( HYPRE_Solver solver,
                      HYPRE_Int logging)
{
   return ( hypre_CGNRSetLogging( (void *) solver, logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CGNRGetNumIterations
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_CGNRGetNumIterations( HYPRE_Solver  solver,
                            HYPRE_Int                *num_iterations )
{
   return ( hypre_CGNRGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CGNRGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_CGNRGetFinalRelativeResidualNorm( HYPRE_Solver  solver,
                                        HYPRE_Real         *norm   )
{
   return ( hypre_CGNRGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}
