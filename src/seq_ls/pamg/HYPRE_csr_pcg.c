/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.6 $
 ***********************************************************************EHEADER*/





/******************************************************************************
 *
 * HYPRE_CSRPCG interface
 *
 *****************************************************************************/
#include "headers.h"
#include "krylov.h"

/*--------------------------------------------------------------------------
 * HYPRE_CSRPCGCreate
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_CSRPCGCreate( HYPRE_Solver *solver )
{
   /* The function names with a CG in them are in
      seq_ls/pamg/cg_fun.c .  These functions do rather little -
      e.g., cast to the correct type - before calling something else.
   */
   hypre_PCGFunctions * pcg_functions =
      hypre_PCGFunctionsCreate(
         hypre_CAlloc, hypre_CGFree, hypre_CGCommInfo,
         hypre_CGCreateVector,
         hypre_CGDestroyVector, hypre_CGMatvecCreate,
         hypre_CGMatvec, hypre_CGMatvecDestroy,
         hypre_CGInnerProd, hypre_CGCopyVector,
         hypre_CGClearVector,
         hypre_CGScaleVector, hypre_CGAxpy,
         hypre_CGIdentitySetup, hypre_CGIdentity );

   *solver = ( (HYPRE_Solver) hypre_PCGCreate( pcg_functions ) );

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRPCGDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_CSRPCGDestroy( HYPRE_Solver solver )
{
   return( hypre_PCGDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRPCGSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_CSRPCGSetup( HYPRE_Solver solver,
                      HYPRE_CSRMatrix A,
                      HYPRE_Vector b,
                      HYPRE_Vector x      )
{
   return( HYPRE_PCGSetup( solver,
                           (HYPRE_Matrix) A,
                           b, x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRPCGSolve
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_CSRPCGSolve( HYPRE_Solver solver,
                      HYPRE_CSRMatrix A,
                      HYPRE_Vector b,
                      HYPRE_Vector x      )
{
   return( HYPRE_PCGSolve( solver,
                           (HYPRE_Matrix) A,
                           b, x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRPCGSetTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_CSRPCGSetTol( HYPRE_Solver solver,
                       double             tol    )
{
   return( HYPRE_PCGSetTol( solver, tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRPCGSetMaxIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_CSRPCGSetMaxIter( HYPRE_Solver solver,
                           HYPRE_Int                max_iter )
{
   return( HYPRE_PCGSetMaxIter( solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRPCGSetStopCrit
 *--------------------------------------------------------------------------*/

/* not supported by krylov/pcg.c.  was supported by old cg.c.  TO DO:
 decide whether this has any functionality. If so, add the parameter and
 implement this function.  If not, delete this function.
HYPRE_Int
HYPRE_CSRPCGSetStopCrit( HYPRE_Solver solver,
                           HYPRE_Int          stop_crit )
{
   return( hypre_PCGSetStopCrit( (void *) solver, stop_crit ) );
}
*/

/*--------------------------------------------------------------------------
 * HYPRE_CSRPCGSetTwoNorm
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_CSRPCGSetTwoNorm( HYPRE_Solver solver,
                           HYPRE_Int                two_norm )
{
   return( HYPRE_PCGSetTwoNorm( solver, two_norm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRPCGSetRelChange
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_CSRPCGSetRelChange( HYPRE_Solver solver,
                             HYPRE_Int                rel_change )
{
   return( HYPRE_PCGSetRelChange( solver, rel_change ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRPCGSetPrecond
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_CSRPCGSetPrecond( HYPRE_Solver  solver,
                           HYPRE_Int (*precond)      (HYPRE_Solver sol,
						HYPRE_CSRMatrix matrix,
						HYPRE_Vector b,
						HYPRE_Vector x),
                           HYPRE_Int (*precond_setup)(HYPRE_Solver sol,
						HYPRE_CSRMatrix matrix,
						HYPRE_Vector b,
						HYPRE_Vector x),
                           void                *precond_data )
{
   return( HYPRE_PCGSetPrecond( solver,
                                (HYPRE_PtrToSolverFcn) precond,
                                (HYPRE_PtrToSolverFcn) precond_setup,
                                (HYPRE_Solver) precond_data ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRPCGGetPrecond
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_CSRPCGGetPrecond( HYPRE_Solver  solver,
                           HYPRE_Solver *precond_data_ptr )
{
   return( HYPRE_PCGGetPrecond( solver, precond_data_ptr ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRPCGSetLogging
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_CSRPCGSetLogging( HYPRE_Solver solver,
                           HYPRE_Int                logging )
{
   return( HYPRE_PCGSetLogging( solver, logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRPCGGetNumIterations
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_CSRPCGGetNumIterations( HYPRE_Solver  solver,
                                 HYPRE_Int                *num_iterations )
{
   return( HYPRE_PCGGetNumIterations( solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRPCGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_CSRPCGGetFinalRelativeResidualNorm( HYPRE_Solver  solver,
                                             double             *norm   )
{
   return( HYPRE_PCGGetFinalRelativeResidualNorm( solver, norm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRDiagScaleSetup
 *--------------------------------------------------------------------------*/
 
HYPRE_Int 
HYPRE_CSRDiagScaleSetup( HYPRE_Solver solver,
                            HYPRE_CSRMatrix A,
                            HYPRE_Vector y,
                            HYPRE_Vector x      )
{
   return 0;
}
 
/*--------------------------------------------------------------------------
 * HYPRE_CSRDiagScale
 *--------------------------------------------------------------------------*/
 
HYPRE_Int 
HYPRE_CSRDiagScale( HYPRE_Solver solver,
                       HYPRE_CSRMatrix HA,
                       HYPRE_Vector Hy,
                       HYPRE_Vector Hx      )
{
   hypre_CSRMatrix *A = (hypre_CSRMatrix *) HA;
   hypre_Vector    *y = (hypre_Vector *) Hy;
   hypre_Vector    *x = (hypre_Vector *) Hx;
   double *x_data = hypre_VectorData(x);
   double *y_data = hypre_VectorData(y);
   double *A_data = hypre_CSRMatrixData(A);
   HYPRE_Int *A_i = hypre_CSRMatrixI(A);

   HYPRE_Int i, ierr = 0;

   for (i=0; i < hypre_VectorSize(x); i++)
   {
	x_data[i] = y_data[i]/A_data[A_i[i]];
   } 
 
   return ierr;
}
