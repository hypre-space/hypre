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
 * HYPRE_CSRGMRES interface
 *
 *****************************************************************************/
#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_CSRGMRESCreate
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_CSRGMRESCreate( HYPRE_Solver *solver )
{
   hypre_GMRESFunctions * gmres_functions =
      hypre_GMRESFunctionsCreate(
         hypre_CAlloc, hypre_CGFree, hypre_CGCommInfo,
         hypre_CGCreateVector,
         hypre_CGCreateVectorArray,
         hypre_CGDestroyVector, hypre_CGMatvecCreate,
         hypre_CGMatvec, hypre_CGMatvecDestroy,
         hypre_CGInnerProd, hypre_CGCopyVector,
         hypre_CGClearVector,
         hypre_CGScaleVector, hypre_CGAxpy,
         hypre_CGIdentitySetup, hypre_CGIdentity );

   *solver = ( (HYPRE_Solver) hypre_GMRESCreate( gmres_functions ) );

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRGMRESDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_CSRGMRESDestroy( HYPRE_Solver solver )
{
   return( hypre_GMRESDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRGMRESSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_CSRGMRESSetup( HYPRE_Solver solver,
                        HYPRE_CSRMatrix A,
                        HYPRE_Vector b,
                        HYPRE_Vector x      )
{
   return( HYPRE_GMRESSetup( solver,
                             (HYPRE_Matrix) A,
                             b, x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRGMRESSolve
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_CSRGMRESSolve( HYPRE_Solver solver,
                        HYPRE_CSRMatrix A,
                        HYPRE_Vector b,
                        HYPRE_Vector x      )
{
   return( HYPRE_GMRESSolve( solver,
                             (HYPRE_Matrix) A,
                             b, x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRGMRESSetKDim
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_CSRGMRESSetKDim( HYPRE_Solver solver,
                          HYPRE_Int             k_dim    )
{
   return( HYPRE_GMRESSetKDim( solver, k_dim ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRGMRESSetTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_CSRGMRESSetTol( HYPRE_Solver solver,
                         double             tol    )
{
   return( HYPRE_GMRESSetTol( solver, tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRGMRESSetMinIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_CSRGMRESSetMinIter( HYPRE_Solver solver,
                             HYPRE_Int          min_iter )
{
   return( HYPRE_GMRESSetMinIter( solver, min_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRGMRESSetMaxIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_CSRGMRESSetMaxIter( HYPRE_Solver solver,
                             HYPRE_Int          max_iter )
{
   return( HYPRE_GMRESSetMaxIter( solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRGMRESSetStopCrit
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_CSRGMRESSetStopCrit( HYPRE_Solver solver,
                              HYPRE_Int          stop_crit )
{
   return( HYPRE_GMRESSetStopCrit( solver, stop_crit ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRGMRESSetPrecond
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_CSRGMRESSetPrecond( HYPRE_Solver  solver,
                             HYPRE_Int (*precond)      (HYPRE_Solver sol, 
					 	  HYPRE_CSRMatrix matrix,
						  HYPRE_Vector b,
						  HYPRE_Vector x),
                             HYPRE_Int (*precond_setup)(HYPRE_Solver sol, 
					 	  HYPRE_CSRMatrix matrix,
						  HYPRE_Vector b,
						  HYPRE_Vector x),
                             void               *precond_data )
{
   return( HYPRE_GMRESSetPrecond( solver,
                                  (HYPRE_PtrToSolverFcn) precond,
                                  (HYPRE_PtrToSolverFcn) precond_setup,
                                  (HYPRE_Solver) precond_data ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRGMRESGetPrecond
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_CSRGMRESGetPrecond( HYPRE_Solver  solver,
                             HYPRE_Solver *precond_data_ptr )
{
   return( HYPRE_GMRESGetPrecond( solver, precond_data_ptr ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRGMRESSetLogging
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_CSRGMRESSetLogging( HYPRE_Solver solver,
                             HYPRE_Int logging)
{
   return( HYPRE_GMRESSetLogging( solver, logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRGMRESGetNumIterations
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_CSRGMRESGetNumIterations( HYPRE_Solver  solver,
                                   HYPRE_Int                *num_iterations )
{
   return( HYPRE_GMRESGetNumIterations( solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRGMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_CSRGMRESGetFinalRelativeResidualNorm( HYPRE_Solver  solver,
                                               double             *norm   )
{
   return( HYPRE_GMRESGetFinalRelativeResidualNorm( solver, norm ) );
}
