/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.2 $
 ***********************************************************************EHEADER*/





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
int 
HYPRE_ParCSRLGMRESDestroy( HYPRE_Solver solver )
{
   return( hypre_LGMRESDestroy( (void *) solver ) );
}
*/

/*--------------------------------------------------------------------------
 * HYPRE_LGMRESSetup
 *--------------------------------------------------------------------------*/

int 
HYPRE_LGMRESSetup( HYPRE_Solver solver,
                        HYPRE_Matrix A,
                        HYPRE_Vector b,
                        HYPRE_Vector x      )
{
   return( hypre_LGMRESSetup( solver,
                             A,
                             b,
                             x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_LGMRESSolve
 *--------------------------------------------------------------------------*/

int 
HYPRE_LGMRESSolve( HYPRE_Solver solver,
                        HYPRE_Matrix A,
                        HYPRE_Vector b,
                        HYPRE_Vector x      )
{
   return( hypre_LGMRESSolve( solver,
                             A,
                             b,
                             x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_LGMRESSetKDim, HYPRE_LGMRESGetKDim
 *--------------------------------------------------------------------------*/

int
HYPRE_LGMRESSetKDim( HYPRE_Solver solver,
                          int             k_dim    )
{
   return( hypre_LGMRESSetKDim( (void *) solver, k_dim ) );
}

int
HYPRE_LGMRESGetKDim( HYPRE_Solver solver,
                          int           * k_dim    )
{
   return( hypre_LGMRESGetKDim( (void *) solver, k_dim ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_LGMRESSetAugDim, HYPRE_LGMRESGetAugDim
 *--------------------------------------------------------------------------*/

int
HYPRE_LGMRESSetAugDim( HYPRE_Solver solver,
                          int             aug_dim    )
{
   return( hypre_LGMRESSetAugDim( (void *) solver, aug_dim ) );
}

int
HYPRE_LGMRESGetAugDim( HYPRE_Solver solver,
                          int           * aug_dim    )
{
   return( hypre_LGMRESGetAugDim( (void *) solver, aug_dim ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_LGMRESSetTol, HYPRE_LGMRESGetTol
 *--------------------------------------------------------------------------*/

int
HYPRE_LGMRESSetTol( HYPRE_Solver solver,
                         double             tol    )
{
   return( hypre_LGMRESSetTol( (void *) solver, tol ) );
}

int
HYPRE_LGMRESGetTol( HYPRE_Solver solver,
                         double           * tol    )
{
   return( hypre_LGMRESGetTol( (void *) solver, tol ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_LGMRESSetAbsoluteTol, HYPRE_LGMRESGetAbsoluteTol
 *--------------------------------------------------------------------------*/

int
HYPRE_LGMRESSetAbsoluteTol( HYPRE_Solver solver,
                         double             a_tol    )
{
   return( hypre_LGMRESSetAbsoluteTol( (void *) solver, a_tol ) );
}

int
HYPRE_LGMRESGetAbsoluteTol( HYPRE_Solver solver,
                         double           * a_tol    )
{
   return( hypre_LGMRESGetAbsoluteTol( (void *) solver, a_tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_LGMRESSetConvergenceFactorTol, HYPRE_LGMRESGetConvergenceFactorTol
 *--------------------------------------------------------------------------*/

int
HYPRE_LGMRESSetConvergenceFactorTol( HYPRE_Solver solver,
                         double             cf_tol    )
{
   return( hypre_LGMRESSetConvergenceFactorTol( (void *) solver, cf_tol ) );
}

int
HYPRE_LGMRESGetConvergenceFactorTol( HYPRE_Solver solver,
                         double           * cf_tol    )
{
   return( hypre_LGMRESGetConvergenceFactorTol( (void *) solver, cf_tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_LGMRESSetMinIter, HYPRE_LGMRESGetMinIter
 *--------------------------------------------------------------------------*/

int
HYPRE_LGMRESSetMinIter( HYPRE_Solver solver,
                             int          min_iter )
{
   return( hypre_LGMRESSetMinIter( (void *) solver, min_iter ) );
}

int
HYPRE_LGMRESGetMinIter( HYPRE_Solver solver,
                             int        * min_iter )
{
   return( hypre_LGMRESGetMinIter( (void *) solver, min_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_LGMRESSetMaxIter, HYPRE_LGMRESGetMaxIter
 *--------------------------------------------------------------------------*/

int
HYPRE_LGMRESSetMaxIter( HYPRE_Solver solver,
                             int          max_iter )
{
   return( hypre_LGMRESSetMaxIter( (void *) solver, max_iter ) );
}

int
HYPRE_LGMRESGetMaxIter( HYPRE_Solver solver,
                             int        * max_iter )
{
   return( hypre_LGMRESGetMaxIter( (void *) solver, max_iter ) );
}



/*--------------------------------------------------------------------------
 * HYPRE_LGMRESSetPrecond
 *--------------------------------------------------------------------------*/

int
HYPRE_LGMRESSetPrecond( HYPRE_Solver          solver,
                             HYPRE_PtrToSolverFcn  precond,
                             HYPRE_PtrToSolverFcn  precond_setup,
                             HYPRE_Solver          precond_solver )
{
   return( hypre_LGMRESSetPrecond( (void *) solver,
                                  precond, precond_setup,
                                  (void *) precond_solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_LGMRESGetPrecond
 *--------------------------------------------------------------------------*/

int
HYPRE_LGMRESGetPrecond( HYPRE_Solver  solver,
                             HYPRE_Solver *precond_data_ptr )
{
   return( hypre_LGMRESGetPrecond( (void *)     solver,
                                  (HYPRE_Solver *) precond_data_ptr ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_LGMRESSetPrintLevel, HYPRE_LGMRESGetPrintLevel
 *--------------------------------------------------------------------------*/

int
HYPRE_LGMRESSetPrintLevel( HYPRE_Solver solver,
                        int          level )
{
   return( hypre_LGMRESSetPrintLevel( (void *) solver, level ) );
}

int
HYPRE_LGMRESGetPrintLevel( HYPRE_Solver solver,
                        int        * level )
{
   return( hypre_LGMRESGetPrintLevel( (void *) solver, level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_LGMRESSetLogging, HYPRE_LGMRESGetLogging
 *--------------------------------------------------------------------------*/

int
HYPRE_LGMRESSetLogging( HYPRE_Solver solver,
                     int          level )
{
   return( hypre_LGMRESSetLogging( (void *) solver, level ) );
}

int
HYPRE_LGMRESGetLogging( HYPRE_Solver solver,
                     int        * level )
{
   return( hypre_LGMRESGetLogging( (void *) solver, level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_LGMRESGetNumIterations
 *--------------------------------------------------------------------------*/

int
HYPRE_LGMRESGetNumIterations( HYPRE_Solver  solver,
                                   int                *num_iterations )
{
   return( hypre_LGMRESGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_LGMRESGetConverged
 *--------------------------------------------------------------------------*/

int
HYPRE_LGMRESGetConverged( HYPRE_Solver  solver,
                         int                *converged )
{
   return( hypre_LGMRESGetConverged( (void *) solver, converged ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_LGMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int
HYPRE_LGMRESGetFinalRelativeResidualNorm( HYPRE_Solver  solver,
                                               double             *norm   )
{
   return( hypre_LGMRESGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_LGMRESGetResidual
 *--------------------------------------------------------------------------*/

int HYPRE_LGMRESGetResidual( HYPRE_Solver solver, void **residual )
{
   /* returns a pointer to the residual vector */
   return hypre_LGMRESGetResidual( (void *) solver, residual );
}

