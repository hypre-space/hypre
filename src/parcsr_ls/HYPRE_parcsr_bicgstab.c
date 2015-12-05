/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.9 $
 ***********************************************************************EHEADER*/





/******************************************************************************
 *
 * HYPRE_ParCSRBiCGSTAB interface
 *
 *****************************************************************************/
#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABCreate
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRBiCGSTABCreate( MPI_Comm comm, HYPRE_Solver *solver )
{
   hypre_BiCGSTABFunctions * bicgstab_functions =
      hypre_BiCGSTABFunctionsCreate(
         hypre_ParKrylovCreateVector,
         hypre_ParKrylovDestroyVector, hypre_ParKrylovMatvecCreate,
         hypre_ParKrylovMatvec, hypre_ParKrylovMatvecDestroy,
         hypre_ParKrylovInnerProd, hypre_ParKrylovCopyVector,
         hypre_ParKrylovClearVector,
         hypre_ParKrylovScaleVector, hypre_ParKrylovAxpy,
         hypre_ParKrylovCommInfo,
         hypre_ParKrylovIdentitySetup, hypre_ParKrylovIdentity );

   *solver = ( (HYPRE_Solver) hypre_BiCGSTABCreate( bicgstab_functions) );
   if (!solver)
      hypre_error_in_arg(2);
    
   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_ParCSRBiCGSTABDestroy( HYPRE_Solver solver )
{
   return( hypre_BiCGSTABDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_ParCSRBiCGSTABSetup( HYPRE_Solver solver,
                        HYPRE_ParCSRMatrix A,
                        HYPRE_ParVector b,
                        HYPRE_ParVector x      )
{
   return( HYPRE_BiCGSTABSetup( solver,
                                (HYPRE_Matrix) A,
                                (HYPRE_Vector) b,
                                (HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABSolve
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_ParCSRBiCGSTABSolve( HYPRE_Solver solver,
                        HYPRE_ParCSRMatrix A,
                        HYPRE_ParVector b,
                        HYPRE_ParVector x      )
{
   return( HYPRE_BiCGSTABSolve( solver,
                                (HYPRE_Matrix) A,
                                (HYPRE_Vector) b,
                                (HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABSetTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRBiCGSTABSetTol( HYPRE_Solver solver,
                         double             tol    )
{
   return( HYPRE_BiCGSTABSetTol( solver, tol ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABSetAbsoluteTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRBiCGSTABSetAbsoluteTol( HYPRE_Solver solver,
                         double             a_tol    )
{
   return( HYPRE_BiCGSTABSetAbsoluteTol( solver, a_tol ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABSetMinIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRBiCGSTABSetMinIter( HYPRE_Solver solver,
                             HYPRE_Int          min_iter )
{
   return( HYPRE_BiCGSTABSetMinIter( solver, min_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABSetMaxIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRBiCGSTABSetMaxIter( HYPRE_Solver solver,
                             HYPRE_Int          max_iter )
{
   return( HYPRE_BiCGSTABSetMaxIter( solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABSetStopCrit
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRBiCGSTABSetStopCrit( HYPRE_Solver solver,
                              HYPRE_Int          stop_crit )
{
   return( HYPRE_BiCGSTABSetStopCrit( solver, stop_crit ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABSetPrecond
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRBiCGSTABSetPrecond( HYPRE_Solver         solver,
                                HYPRE_PtrToParSolverFcn precond,
                                HYPRE_PtrToParSolverFcn precond_setup,
                                HYPRE_Solver         precond_solver )
{
   return( HYPRE_BiCGSTABSetPrecond( solver,
                                     (HYPRE_PtrToSolverFcn) precond,
                                     (HYPRE_PtrToSolverFcn) precond_setup,
                                     precond_solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABGetPrecond
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRBiCGSTABGetPrecond( HYPRE_Solver  solver,
                             HYPRE_Solver *precond_data_ptr )
{
   return( HYPRE_BiCGSTABGetPrecond( solver, precond_data_ptr ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABSetLogging
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRBiCGSTABSetLogging( HYPRE_Solver solver,
                             HYPRE_Int logging)
{
   return( HYPRE_BiCGSTABSetLogging( solver, logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABSetPrintLevel
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRBiCGSTABSetPrintLevel( HYPRE_Solver solver,
                             HYPRE_Int print_level)
{
   return( HYPRE_BiCGSTABSetPrintLevel( solver, print_level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABGetNumIterations
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRBiCGSTABGetNumIterations( HYPRE_Solver  solver,
                                   HYPRE_Int                *num_iterations )
{
   return( HYPRE_BiCGSTABGetNumIterations( solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRBiCGSTABGetFinalRelativeResidualNorm( HYPRE_Solver  solver,
                                               double             *norm   )
{
   return( HYPRE_BiCGSTABGetFinalRelativeResidualNorm( solver, norm ) );
}
