/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/

#include "_hypre_struct_ls.h"

/*==========================================================================*/

HYPRE_Int
HYPRE_StructBiCGSTABCreate( MPI_Comm comm, HYPRE_StructSolver *solver )
{
   hypre_BiCGSTABFunctions * bicgstab_functions =
      hypre_BiCGSTABFunctionsCreate(
         hypre_StructKrylovCreateVector,
         hypre_StructKrylovDestroyVector, hypre_StructKrylovMatvecCreate,
         hypre_StructKrylovMatvec, hypre_StructKrylovMatvecDestroy,
         hypre_StructKrylovInnerProd, hypre_StructKrylovCopyVector,
         hypre_StructKrylovClearVector,
         hypre_StructKrylovScaleVector, hypre_StructKrylovAxpy,
         hypre_StructKrylovCommInfo,
         hypre_StructKrylovIdentitySetup, hypre_StructKrylovIdentity );

   *solver = ( (HYPRE_StructSolver) hypre_BiCGSTABCreate( bicgstab_functions ) );

   return hypre_error_flag;
}

/*==========================================================================*/

HYPRE_Int 
HYPRE_StructBiCGSTABDestroy( HYPRE_StructSolver solver )
{
   return( hypre_BiCGSTABDestroy( (void *) solver ) );
}

/*==========================================================================*/

HYPRE_Int 
HYPRE_StructBiCGSTABSetup( HYPRE_StructSolver solver,
                           HYPRE_StructMatrix A,
                           HYPRE_StructVector b,
                           HYPRE_StructVector x      )
{
   return( HYPRE_BiCGSTABSetup( (HYPRE_Solver) solver,
                                (HYPRE_Matrix) A,
                                (HYPRE_Vector) b,
                                (HYPRE_Vector) x ) );
}

/*==========================================================================*/

HYPRE_Int 
HYPRE_StructBiCGSTABSolve( HYPRE_StructSolver solver,
                           HYPRE_StructMatrix A,
                           HYPRE_StructVector b,
                           HYPRE_StructVector x      )
{
   return( HYPRE_BiCGSTABSolve( (HYPRE_Solver) solver,
                                (HYPRE_Matrix) A,
                                (HYPRE_Vector) b,
                                (HYPRE_Vector) x ) );
}

/*==========================================================================*/

HYPRE_Int
HYPRE_StructBiCGSTABSetTol( HYPRE_StructSolver solver,
                            HYPRE_Real         tol    )
{
   return( HYPRE_BiCGSTABSetTol( (HYPRE_Solver) solver, tol ) );
}

/*==========================================================================*/

HYPRE_Int
HYPRE_StructBiCGSTABSetAbsoluteTol( HYPRE_StructSolver solver,
                                    HYPRE_Real         tol    )
{
   return( HYPRE_BiCGSTABSetAbsoluteTol( (HYPRE_Solver) solver, tol ) );
}

/*==========================================================================*/

HYPRE_Int
HYPRE_StructBiCGSTABSetMaxIter( HYPRE_StructSolver solver,
                                HYPRE_Int          max_iter )
{
   return( HYPRE_BiCGSTABSetMaxIter( (HYPRE_Solver) solver, max_iter ) );
}


/*==========================================================================*/

HYPRE_Int
HYPRE_StructBiCGSTABSetPrecond( HYPRE_StructSolver         solver,
                                HYPRE_PtrToStructSolverFcn precond,
                                HYPRE_PtrToStructSolverFcn precond_setup,
                                HYPRE_StructSolver         precond_solver )
{
   return( HYPRE_BiCGSTABSetPrecond( (HYPRE_Solver) solver,
                                     (HYPRE_PtrToSolverFcn) precond,
                                     (HYPRE_PtrToSolverFcn) precond_setup,
                                     (HYPRE_Solver) precond_solver ) );
}

/*==========================================================================*/

HYPRE_Int
HYPRE_StructBiCGSTABSetLogging( HYPRE_StructSolver solver,
                                HYPRE_Int          logging )
{
   return( HYPRE_BiCGSTABSetLogging( (HYPRE_Solver) solver, logging ) );
}

/*==========================================================================*/

HYPRE_Int
HYPRE_StructBiCGSTABSetPrintLevel( HYPRE_StructSolver solver,
                                   HYPRE_Int level)
{
   return( HYPRE_BiCGSTABSetPrintLevel( (HYPRE_Solver) solver, level ) );
}

/*==========================================================================*/

HYPRE_Int
HYPRE_StructBiCGSTABGetNumIterations( HYPRE_StructSolver  solver,
                                      HYPRE_Int          *num_iterations )
{
   return( HYPRE_BiCGSTABGetNumIterations( (HYPRE_Solver) solver,
                                           num_iterations ) );
}

/*==========================================================================*/

HYPRE_Int
HYPRE_StructBiCGSTABGetFinalRelativeResidualNorm( HYPRE_StructSolver  solver,
                                                  HYPRE_Real         *norm   )
{
   return( HYPRE_BiCGSTABGetFinalRelativeResidualNorm( (HYPRE_Solver) solver,
                                                       norm ) );
}

/*==========================================================================*/

HYPRE_Int
HYPRE_StructBiCGSTABGetResidual( HYPRE_StructSolver  solver,
                                 void  **residual)
{
   return( HYPRE_BiCGSTABGetResidual( (HYPRE_Solver) solver, residual ) );
}

