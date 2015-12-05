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
 * HYPRE_SStructFlexGMRES interface
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_SStructFlexGMRESCreate
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructFlexGMRESCreate( MPI_Comm             comm,
                          HYPRE_SStructSolver *solver )
{
   hypre_FlexGMRESFunctions * fgmres_functions =
      hypre_FlexGMRESFunctionsCreate(
         hypre_CAlloc, hypre_SStructKrylovFree, hypre_SStructKrylovCommInfo,
         hypre_SStructKrylovCreateVector,
         hypre_SStructKrylovCreateVectorArray,
         hypre_SStructKrylovDestroyVector, hypre_SStructKrylovMatvecCreate,
         hypre_SStructKrylovMatvec, hypre_SStructKrylovMatvecDestroy,
         hypre_SStructKrylovInnerProd, hypre_SStructKrylovCopyVector,
         hypre_SStructKrylovClearVector,
         hypre_SStructKrylovScaleVector, hypre_SStructKrylovAxpy,
         hypre_SStructKrylovIdentitySetup, hypre_SStructKrylovIdentity );

   *solver = ( (HYPRE_SStructSolver) hypre_FlexGMRESCreate( fgmres_functions ) );

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFlexGMRESDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_SStructFlexGMRESDestroy( HYPRE_SStructSolver solver )
{
   return( hypre_FlexGMRESDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFlexGMRESSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_SStructFlexGMRESSetup( HYPRE_SStructSolver solver,
                         HYPRE_SStructMatrix A,
                         HYPRE_SStructVector b,
                         HYPRE_SStructVector x )
{
   return( HYPRE_FlexGMRESSetup( (HYPRE_Solver) solver,
                             (HYPRE_Matrix) A,
                             (HYPRE_Vector) b,
                             (HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFlexGMRESSolve
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_SStructFlexGMRESSolve( HYPRE_SStructSolver solver,
                         HYPRE_SStructMatrix A,
                         HYPRE_SStructVector b,
                         HYPRE_SStructVector x )
{
   return( HYPRE_FlexGMRESSolve( (HYPRE_Solver) solver,
                             (HYPRE_Matrix) A,
                             (HYPRE_Vector) b,
                             (HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFlexGMRESSetKDim
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructFlexGMRESSetKDim( HYPRE_SStructSolver solver,
                           HYPRE_Int           k_dim )
{
   return( HYPRE_FlexGMRESSetKDim( (HYPRE_Solver) solver, k_dim ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFlexGMRESSetTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructFlexGMRESSetTol( HYPRE_SStructSolver solver,
                          double              tol )
{
   return( HYPRE_FlexGMRESSetTol( (HYPRE_Solver) solver, tol ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_SStructFlexGMRESSetAbsoluteTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructFlexGMRESSetAbsoluteTol( HYPRE_SStructSolver solver,
                          double              tol )
{
   return( HYPRE_FlexGMRESSetAbsoluteTol( (HYPRE_Solver) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFlexGMRESSetMinIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructFlexGMRESSetMinIter( HYPRE_SStructSolver solver,
                              HYPRE_Int           min_iter )
{
   return( HYPRE_FlexGMRESSetMinIter( (HYPRE_Solver) solver, min_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFlexGMRESSetMaxIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructFlexGMRESSetMaxIter( HYPRE_SStructSolver solver,
                              HYPRE_Int           max_iter )
{
   return( HYPRE_FlexGMRESSetMaxIter( (HYPRE_Solver) solver, max_iter ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_SStructFlexGMRESSetPrecond
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructFlexGMRESSetPrecond( HYPRE_SStructSolver          solver,
                              HYPRE_PtrToSStructSolverFcn  precond,
                              HYPRE_PtrToSStructSolverFcn  precond_setup,
                              void *          precond_data )
{
   return( HYPRE_FlexGMRESSetPrecond( (HYPRE_Solver) solver,
                                  (HYPRE_PtrToSolverFcn) precond,
                                  (HYPRE_PtrToSolverFcn) precond_setup,
                                  (HYPRE_Solver) precond_data ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFlexGMRESSetLogging
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructFlexGMRESSetLogging( HYPRE_SStructSolver solver,
                              HYPRE_Int           logging )
{
   return( HYPRE_FlexGMRESSetLogging( (HYPRE_Solver) solver, logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFlexGMRESSetPrintLevel
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructFlexGMRESSetPrintLevel( HYPRE_SStructSolver solver,
                              HYPRE_Int           level )
{
   return( HYPRE_FlexGMRESSetPrintLevel( (HYPRE_Solver) solver, level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFlexGMRESGetNumIterations
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructFlexGMRESGetNumIterations( HYPRE_SStructSolver  solver,
                                    HYPRE_Int           *num_iterations )
{
   return( HYPRE_FlexGMRESGetNumIterations( (HYPRE_Solver) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFlexGMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructFlexGMRESGetFinalRelativeResidualNorm( HYPRE_SStructSolver  solver,
                                                double              *norm )
{
   return( HYPRE_FlexGMRESGetFinalRelativeResidualNorm( (HYPRE_Solver) solver, norm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_SStructFlexGMRESGetResidual
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_SStructFlexGMRESGetResidual( HYPRE_SStructSolver  solver,
                                void              **residual )
{
   return( HYPRE_FlexGMRESGetResidual( (HYPRE_Solver) solver, residual ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_SStructFlexGMRESSetModifyPC
 *--------------------------------------------------------------------------*/
 

HYPRE_Int HYPRE_SStructFlexGMRESSetModifyPC( HYPRE_SStructSolver  solver,
                                      HYPRE_PtrToModifyPCFcn modify_pc)

{
   return ( HYPRE_FlexGMRESSetModifyPC( (HYPRE_Solver) solver,  (HYPRE_PtrToModifyPCFcn) modify_pc));
   
}

