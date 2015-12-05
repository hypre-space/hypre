/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.7 $
 ***********************************************************************EHEADER*/





/******************************************************************************
 *
 * HYPRE_ParCSRCGNR interface
 *
 *****************************************************************************/
#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRCreate
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRCGNRCreate( MPI_Comm comm, HYPRE_Solver *solver )
{
   hypre_CGNRFunctions * cgnr_functions =
      hypre_CGNRFunctionsCreate(
         hypre_ParKrylovCommInfo,
         hypre_ParKrylovCreateVector,
         hypre_ParKrylovDestroyVector, hypre_ParKrylovMatvecCreate,
         hypre_ParKrylovMatvec, hypre_ParKrylovMatvecT,
         hypre_ParKrylovMatvecDestroy,
         hypre_ParKrylovInnerProd,
         hypre_ParKrylovCopyVector, hypre_ParKrylovClearVector,
         hypre_ParKrylovScaleVector, hypre_ParKrylovAxpy,
         hypre_ParKrylovIdentitySetup,
         hypre_ParKrylovIdentity, hypre_ParKrylovIdentity );

   *solver = ( (HYPRE_Solver) hypre_CGNRCreate( cgnr_functions) );
   if (!solver) hypre_error_in_arg(2);
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_ParCSRCGNRDestroy( HYPRE_Solver solver )
{
   return( hypre_CGNRDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_ParCSRCGNRSetup( HYPRE_Solver solver,
                        HYPRE_ParCSRMatrix A,
                        HYPRE_ParVector b,
                        HYPRE_ParVector x      )
{
   return( HYPRE_CGNRSetup( solver,
                             (HYPRE_Matrix) A,
                             (HYPRE_Vector) b,
                             (HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRSolve
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_ParCSRCGNRSolve( HYPRE_Solver solver,
                        HYPRE_ParCSRMatrix A,
                        HYPRE_ParVector b,
                        HYPRE_ParVector x      )
{
   return( HYPRE_CGNRSolve( solver,
                            (HYPRE_Matrix) A,
                            (HYPRE_Vector) b,
                            (HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRSetTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRCGNRSetTol( HYPRE_Solver solver,
                         double             tol    )
{
   return( HYPRE_CGNRSetTol( solver, tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRSetMinIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRCGNRSetMinIter( HYPRE_Solver solver,
                             HYPRE_Int                min_iter )
{
   return( HYPRE_CGNRSetMinIter( solver, min_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRSetMaxIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRCGNRSetMaxIter( HYPRE_Solver solver,
                             HYPRE_Int                max_iter )
{
   return( HYPRE_CGNRSetMaxIter( solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRSetStopCrit
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRCGNRSetStopCrit( HYPRE_Solver solver,
                             HYPRE_Int                stop_crit )
{
   return( HYPRE_CGNRSetStopCrit( solver, stop_crit ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRSetPrecond
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRCGNRSetPrecond( HYPRE_Solver         solver,
                            HYPRE_PtrToParSolverFcn precond,
                            HYPRE_PtrToParSolverFcn precondT,
                            HYPRE_PtrToParSolverFcn precond_setup,
                            HYPRE_Solver         precond_solver )
{
   return( HYPRE_CGNRSetPrecond( solver,
                                 (HYPRE_PtrToSolverFcn) precond,
                                 (HYPRE_PtrToSolverFcn) precondT,
                                 (HYPRE_PtrToSolverFcn) precond_setup,
                                 precond_solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRGetPrecond
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRCGNRGetPrecond( HYPRE_Solver   solver,
                            HYPRE_Solver  *precond_data_ptr )
{
   return( HYPRE_CGNRGetPrecond( solver, precond_data_ptr ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRSetLogging
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRCGNRSetLogging( HYPRE_Solver solver,
                             HYPRE_Int logging)
{
   return( HYPRE_CGNRSetLogging( solver, logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRGetNumIterations
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRCGNRGetNumIterations( HYPRE_Solver  solver,
                                   HYPRE_Int                *num_iterations )
{
   return( HYPRE_CGNRGetNumIterations( solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRCGNRGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRCGNRGetFinalRelativeResidualNorm( HYPRE_Solver  solver,
                                               double             *norm   )
{
   return( HYPRE_CGNRGetFinalRelativeResidualNorm( solver, norm ) );
}
