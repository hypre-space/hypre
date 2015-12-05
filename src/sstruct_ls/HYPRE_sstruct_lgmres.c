/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.5 $
 ***********************************************************************EHEADER*/





/******************************************************************************
 *
 * HYPRE_SStructLGMRES interface
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_SStructLGMRESCreate
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructLGMRESCreate( MPI_Comm             comm,
                          HYPRE_SStructSolver *solver )
{
   hypre_LGMRESFunctions * lgmres_functions =
      hypre_LGMRESFunctionsCreate(
         hypre_CAlloc, hypre_SStructKrylovFree, hypre_SStructKrylovCommInfo,
         hypre_SStructKrylovCreateVector,
         hypre_SStructKrylovCreateVectorArray,
         hypre_SStructKrylovDestroyVector, hypre_SStructKrylovMatvecCreate,
         hypre_SStructKrylovMatvec, hypre_SStructKrylovMatvecDestroy,
         hypre_SStructKrylovInnerProd, hypre_SStructKrylovCopyVector,
         hypre_SStructKrylovClearVector,
         hypre_SStructKrylovScaleVector, hypre_SStructKrylovAxpy,
         hypre_SStructKrylovIdentitySetup, hypre_SStructKrylovIdentity );

   *solver = ( (HYPRE_SStructSolver) hypre_LGMRESCreate( lgmres_functions ) );

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructLGMRESDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_SStructLGMRESDestroy( HYPRE_SStructSolver solver )
{
   return( hypre_LGMRESDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructLGMRESSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_SStructLGMRESSetup( HYPRE_SStructSolver solver,
                         HYPRE_SStructMatrix A,
                         HYPRE_SStructVector b,
                         HYPRE_SStructVector x )
{
   return( HYPRE_LGMRESSetup( (HYPRE_Solver) solver,
                             (HYPRE_Matrix) A,
                             (HYPRE_Vector) b,
                             (HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructLGMRESSolve
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_SStructLGMRESSolve( HYPRE_SStructSolver solver,
                         HYPRE_SStructMatrix A,
                         HYPRE_SStructVector b,
                         HYPRE_SStructVector x )
{
   return( HYPRE_LGMRESSolve( (HYPRE_Solver) solver,
                             (HYPRE_Matrix) A,
                             (HYPRE_Vector) b,
                             (HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructLGMRESSetKDim
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructLGMRESSetKDim( HYPRE_SStructSolver solver,
                           HYPRE_Int           k_dim )
{
   return( HYPRE_LGMRESSetKDim( (HYPRE_Solver) solver, k_dim ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_SStructLGMRESSetAugDim
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructLGMRESSetAugDim( HYPRE_SStructSolver solver,
                           HYPRE_Int           aug_dim )
{
   return( HYPRE_LGMRESSetAugDim( (HYPRE_Solver) solver, aug_dim ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_SStructLGMRESSetTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructLGMRESSetTol( HYPRE_SStructSolver solver,
                          double              tol )
{
   return( HYPRE_LGMRESSetTol( (HYPRE_Solver) solver, tol ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_SStructLGMRESSetAbsoluteTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructLGMRESSetAbsoluteTol( HYPRE_SStructSolver solver,
                          double              atol )
{
   return( HYPRE_LGMRESSetAbsoluteTol( (HYPRE_Solver) solver, atol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructLGMRESSetMinIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructLGMRESSetMinIter( HYPRE_SStructSolver solver,
                              HYPRE_Int           min_iter )
{
   return( HYPRE_LGMRESSetMinIter( (HYPRE_Solver) solver, min_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructLGMRESSetMaxIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructLGMRESSetMaxIter( HYPRE_SStructSolver solver,
                              HYPRE_Int           max_iter )
{
   return( HYPRE_LGMRESSetMaxIter( (HYPRE_Solver) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructLGMRESSetPrecond
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructLGMRESSetPrecond( HYPRE_SStructSolver          solver,
                              HYPRE_PtrToSStructSolverFcn  precond,
                              HYPRE_PtrToSStructSolverFcn  precond_setup,
                              void *          precond_data )
{
   return( HYPRE_LGMRESSetPrecond( (HYPRE_Solver) solver,
                                  (HYPRE_PtrToSolverFcn) precond,
                                  (HYPRE_PtrToSolverFcn) precond_setup,
                                  (HYPRE_Solver) precond_data ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructLGMRESSetLogging
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructLGMRESSetLogging( HYPRE_SStructSolver solver,
                              HYPRE_Int           logging )
{
   return( HYPRE_LGMRESSetLogging( (HYPRE_Solver) solver, logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructLGMRESSetPrintLevel
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructLGMRESSetPrintLevel( HYPRE_SStructSolver solver,
                              HYPRE_Int           level )
{
   return( HYPRE_LGMRESSetPrintLevel( (HYPRE_Solver) solver, level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructLGMRESGetNumIterations
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructLGMRESGetNumIterations( HYPRE_SStructSolver  solver,
                                    HYPRE_Int           *num_iterations )
{
   return( HYPRE_LGMRESGetNumIterations( (HYPRE_Solver) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructLGMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructLGMRESGetFinalRelativeResidualNorm( HYPRE_SStructSolver  solver,
                                                double              *norm )
{
   return( HYPRE_LGMRESGetFinalRelativeResidualNorm( (HYPRE_Solver) solver, norm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructLGMRESGetResidual
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructLGMRESGetResidual( HYPRE_SStructSolver  solver,
                                void              **residual )
{
   return( HYPRE_LGMRESGetResidual( (HYPRE_Solver) solver, residual ) );
}
