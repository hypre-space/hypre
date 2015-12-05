/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.19 $
 ***********************************************************************EHEADER*/





/******************************************************************************
 *
 * HYPRE_PCG interface
 *
 *****************************************************************************/
#include "krylov.h"

/*--------------------------------------------------------------------------
 * HYPRE_PCGCreate does not exist.  Call the appropriate function which
 * also specifies the vector type, e.g. HYPRE_ParCSRPCGCreate
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * HYPRE_PCGDestroy
 *--------------------------------------------------------------------------*/

/*
HYPRE_Int 
HYPRE_PCGDestroy( HYPRE_Solver solver )*/
/* >>> This is something we can't do without knowing the vector_type.
   We can't save it in and pull it out of solver because that isn't
   really a known struct. */
/*
{
   if ( vector_type=="ParCSR" ) {
      return HYPRE_ParCSRPCGDestroy( HYPRE_Solver solver );
   }
   else {
      return 0;
   }
}*/

/*--------------------------------------------------------------------------
 * HYPRE_PCGSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_PCGSetup( HYPRE_Solver solver,
                HYPRE_Matrix A,
                HYPRE_Vector b,
                HYPRE_Vector x      )
{
   return( hypre_PCGSetup( solver,
                           A,
                           b,
                           x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_PCGSolve
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_PCGSolve( HYPRE_Solver solver,
                HYPRE_Matrix A,
                HYPRE_Vector b,
                HYPRE_Vector x      )
{
   return( hypre_PCGSolve( (void *) solver,
                           (void *) A,
                           (void *) b,
                           (void *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_PCGSetTol, HYPRE_PCGGetTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_PCGSetTol( HYPRE_Solver solver,
                 double             tol    )
{
   return( hypre_PCGSetTol( (void *) solver, tol ) );
}

HYPRE_Int
HYPRE_PCGGetTol( HYPRE_Solver solver,
                 double           * tol    )
{
   return( hypre_PCGGetTol( (void *) solver, tol ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_PCGSetAbsoluteTol, HYPRE_PCGGetAbsoluteTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_PCGSetAbsoluteTol( HYPRE_Solver solver,
                 double             a_tol    )
{
   return( hypre_PCGSetAbsoluteTol( (void *) solver, a_tol ) );
}

HYPRE_Int
HYPRE_PCGGetAbsoluteTol( HYPRE_Solver solver,
                 double           * a_tol    )
{
   return( hypre_PCGGetAbsoluteTol( (void *) solver, a_tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_PCGSetResidualTol, HYPRE_PCGGetResidualTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_PCGSetResidualTol( HYPRE_Solver solver,
                         double rtol )
{
   return( hypre_PCGSetResidualTol( (void *) solver, rtol ) );
}

HYPRE_Int
HYPRE_PCGGetResidualTol( HYPRE_Solver solver,
                         double * rtol )
{
   return( hypre_PCGGetResidualTol( (void *) solver, rtol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_PCGSetAbsoluteTolFactor, HYPRE_PCGGetAbsoluteTolFactor
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_PCGSetAbsoluteTolFactor( HYPRE_Solver solver,
                               double abstolf )
{
   return( hypre_PCGSetAbsoluteTolFactor( (void *) solver, abstolf ) );
}

HYPRE_Int
HYPRE_PCGGetAbsoluteTolFactor( HYPRE_Solver solver,
                               double * abstolf )
{
   return( hypre_PCGGetAbsoluteTolFactor( (void *) solver, abstolf ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_PCGSetConvergenceFactorTol, HYPRE_PCGGetConvergenceFactorTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_PCGSetConvergenceFactorTol( HYPRE_Solver solver,
                                  double cf_tol )
{
   return hypre_PCGSetConvergenceFactorTol( (void *) solver,
                                            cf_tol   );
}

HYPRE_Int
HYPRE_PCGGetConvergenceFactorTol( HYPRE_Solver solver,
                                  double * cf_tol )
{
   return hypre_PCGGetConvergenceFactorTol( (void *) solver,
                                            cf_tol   );
}

/*--------------------------------------------------------------------------
 * HYPRE_PCGSetMaxIter, HYPRE_PCGGetMaxIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_PCGSetMaxIter( HYPRE_Solver solver,
                     HYPRE_Int                max_iter )
{
   return( hypre_PCGSetMaxIter( (void *) solver, max_iter ) );
}

HYPRE_Int
HYPRE_PCGGetMaxIter( HYPRE_Solver solver,
                     HYPRE_Int              * max_iter )
{
   return( hypre_PCGGetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_PCGSetStopCrit, HYPRE_PCGGetStopCrit
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_PCGSetStopCrit( HYPRE_Solver solver,
                      HYPRE_Int          stop_crit )
{
   return( hypre_PCGSetStopCrit( (void *) solver, stop_crit ) );
}

HYPRE_Int
HYPRE_PCGGetStopCrit( HYPRE_Solver solver,
                      HYPRE_Int        * stop_crit )
{
   return( hypre_PCGGetStopCrit( (void *) solver, stop_crit ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_PCGSetTwoNorm, HYPRE_PCGGetTwoNorm
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_PCGSetTwoNorm( HYPRE_Solver solver,
                     HYPRE_Int                two_norm )
{
   return( hypre_PCGSetTwoNorm( (void *) solver, two_norm ) );
}

HYPRE_Int
HYPRE_PCGGetTwoNorm( HYPRE_Solver solver,
                     HYPRE_Int              * two_norm )
{
   return( hypre_PCGGetTwoNorm( (void *) solver, two_norm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_PCGSetRelChange, HYPRE_PCGGetRelChange
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_PCGSetRelChange( HYPRE_Solver solver,
                       HYPRE_Int                rel_change )
{
   return( hypre_PCGSetRelChange( (void *) solver, rel_change ) );
}

HYPRE_Int
HYPRE_PCGGetRelChange( HYPRE_Solver solver,
                       HYPRE_Int              * rel_change )
{
   return( hypre_PCGGetRelChange( (void *) solver, rel_change ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_PCGSetRecomputeResidual, HYPRE_PCGGetRecomputeResidual
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_PCGSetRecomputeResidual( HYPRE_Solver solver,
                       HYPRE_Int                recompute_residual )
{
   return( hypre_PCGSetRecomputeResidual( (void *) solver, recompute_residual ) );
}

HYPRE_Int
HYPRE_PCGGetRecomputeResidual( HYPRE_Solver solver,
                       HYPRE_Int              * recompute_residual )
{
   return( hypre_PCGGetRecomputeResidual( (void *) solver, recompute_residual ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_PCGSetRecomputeResidualP, HYPRE_PCGGetRecomputeResidualP
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_PCGSetRecomputeResidualP( HYPRE_Solver solver,
                       HYPRE_Int                recompute_residual_p )
{
   return( hypre_PCGSetRecomputeResidualP( (void *) solver, recompute_residual_p ) );
}

HYPRE_Int
HYPRE_PCGGetRecomputeResidualP( HYPRE_Solver solver,
                       HYPRE_Int              * recompute_residual_p )
{
   return( hypre_PCGGetRecomputeResidualP( (void *) solver, recompute_residual_p ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_PCGSetPrecond
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_PCGSetPrecond( HYPRE_Solver         solver,
                     HYPRE_PtrToSolverFcn precond,
                     HYPRE_PtrToSolverFcn precond_setup,
                     HYPRE_Solver         precond_solver )
{
   return( hypre_PCGSetPrecond( (void *) solver,
                                precond, precond_setup,
                                (void *) precond_solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_PCGGetPrecond
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_PCGGetPrecond( HYPRE_Solver  solver,
                     HYPRE_Solver *precond_data_ptr )
{
   return( hypre_PCGGetPrecond( (void *)     solver,
                                (HYPRE_Solver *) precond_data_ptr ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_PCGSetLogging, HYPRE_PCGGetLogging
 * SetLogging sets both the print and log level, for backwards compatibility.
 * Soon the SetPrintLevel call should be deleted.
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_PCGSetLogging( HYPRE_Solver solver,
                     HYPRE_Int          level )
{
   return ( hypre_PCGSetLogging( (void *) solver, level ) );
}

HYPRE_Int
HYPRE_PCGGetLogging( HYPRE_Solver solver,
                     HYPRE_Int        * level )
{
   return ( hypre_PCGGetLogging( (void *) solver, level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_PCGSetPrintLevel, HYPRE_PCGGetPrintLevel
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_PCGSetPrintLevel( HYPRE_Solver solver,
                        HYPRE_Int          level )
{
   return( hypre_PCGSetPrintLevel( (void *) solver, level ) );
}

HYPRE_Int
HYPRE_PCGGetPrintLevel( HYPRE_Solver solver,
                        HYPRE_Int        * level )
{
   return( hypre_PCGGetPrintLevel( (void *) solver, level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_PCGGetNumIterations
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_PCGGetNumIterations( HYPRE_Solver  solver,
                           HYPRE_Int                *num_iterations )
{
   return( hypre_PCGGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_PCGGetConverged
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_PCGGetConverged( HYPRE_Solver  solver,
                       HYPRE_Int                *converged )
{
   return( hypre_PCGGetConverged( (void *) solver, converged ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_PCGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_PCGGetFinalRelativeResidualNorm( HYPRE_Solver  solver,
                                       double             *norm   )
{
   return( hypre_PCGGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_PCGGetResidual
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_PCGGetResidual( HYPRE_Solver solver, void **residual )
{
   /* returns a pointer to the residual vector */
   return hypre_PCGGetResidual( (void *) solver, residual );
}

