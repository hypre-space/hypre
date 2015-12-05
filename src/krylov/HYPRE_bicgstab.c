/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.8 $
 ***********************************************************************EHEADER*/





/******************************************************************************
 *
 * HYPRE_BiCGSTAB interface
 *
 *****************************************************************************/
#include "krylov.h"

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABCreate does not exist.  Call the appropriate function which
 * also specifies the vector type, e.g. HYPRE_ParCSRBiCGSTABCreate
 *--------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_BiCGSTABDestroy( HYPRE_Solver solver )
{
   return( hypre_BiCGSTABDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_BiCGSTABSetup( HYPRE_Solver solver,
                        HYPRE_Matrix A,
                        HYPRE_Vector b,
                        HYPRE_Vector x      )
{
   return( hypre_BiCGSTABSetup( (void *) solver,
                             (void *) A,
                             (void *) b,
                             (void *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABSolve
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_BiCGSTABSolve( HYPRE_Solver solver,
                        HYPRE_Matrix A,
                        HYPRE_Vector b,
                        HYPRE_Vector x      )
{
   return( hypre_BiCGSTABSolve( (void *) solver,
                             (void *) A,
                             (void *) b,
                             (void *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABSetTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BiCGSTABSetTol( HYPRE_Solver solver,
                         double             tol    )
{
   return( hypre_BiCGSTABSetTol( (void *) solver, tol ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABSetAbsoluteTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BiCGSTABSetAbsoluteTol( HYPRE_Solver solver,
                         double             a_tol    )
{
   return( hypre_BiCGSTABSetAbsoluteTol( (void *) solver, a_tol ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABSetConvergenceFactorTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BiCGSTABSetConvergenceFactorTol( HYPRE_Solver solver,
                         double             cf_tol    )
{
   return( hypre_BiCGSTABSetConvergenceFactorTol( (void *) solver, cf_tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABSetMinIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BiCGSTABSetMinIter( HYPRE_Solver solver,
                             HYPRE_Int          min_iter )
{
   return( hypre_BiCGSTABSetMinIter( (void *) solver, min_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABSetMaxIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BiCGSTABSetMaxIter( HYPRE_Solver solver,
                             HYPRE_Int          max_iter )
{
   return( hypre_BiCGSTABSetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABSetStopCrit
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BiCGSTABSetStopCrit( HYPRE_Solver solver,
                              HYPRE_Int          stop_crit )
{
   return( hypre_BiCGSTABSetStopCrit( (void *) solver, stop_crit ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABSetPrecond
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BiCGSTABSetPrecond( HYPRE_Solver         solver,
                                HYPRE_PtrToSolverFcn precond,
                                HYPRE_PtrToSolverFcn precond_setup,
                                HYPRE_Solver         precond_solver )
{
   return( hypre_BiCGSTABSetPrecond( (void *) solver,
                                     precond, precond_setup,
                                     (void *) precond_solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABGetPrecond
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BiCGSTABGetPrecond( HYPRE_Solver  solver,
                             HYPRE_Solver *precond_data_ptr )
{
   return( hypre_BiCGSTABGetPrecond( (void *)     solver,
                                  (HYPRE_Solver *) precond_data_ptr ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABSetLogging
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BiCGSTABSetLogging( HYPRE_Solver solver,
                             HYPRE_Int logging)
{
   return( hypre_BiCGSTABSetLogging( (void *) solver, logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABSetPrintLevel
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BiCGSTABSetPrintLevel( HYPRE_Solver solver,
                             HYPRE_Int print_level)
{
   return( hypre_BiCGSTABSetPrintLevel( (void *) solver, print_level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABGetNumIterations
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BiCGSTABGetNumIterations( HYPRE_Solver  solver,
                                   HYPRE_Int                *num_iterations )
{
   return( hypre_BiCGSTABGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BiCGSTABGetFinalRelativeResidualNorm( HYPRE_Solver  solver,
                                               double             *norm   )
{
   return( hypre_BiCGSTABGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_BiCGSTABGetResidual
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_BiCGSTABGetResidual( HYPRE_Solver  solver,
                            void             **residual  )
{
   return( hypre_BiCGSTABGetResidual( (void *) solver, residual ) );
}
