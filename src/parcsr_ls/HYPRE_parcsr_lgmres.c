/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.3 $
 ***********************************************************************EHEADER*/





/******************************************************************************
 *
 * HYPRE_ParCSRLGMRES interface
 *
 *****************************************************************************/
#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLGMRESCreate
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRLGMRESCreate( MPI_Comm comm, HYPRE_Solver *solver )
{
   hypre_LGMRESFunctions * lgmres_functions =
      hypre_LGMRESFunctionsCreate(
         hypre_CAlloc, hypre_ParKrylovFree, hypre_ParKrylovCommInfo,
         hypre_ParKrylovCreateVector,
         hypre_ParKrylovCreateVectorArray,
         hypre_ParKrylovDestroyVector, hypre_ParKrylovMatvecCreate,
         hypre_ParKrylovMatvec, hypre_ParKrylovMatvecDestroy,
         hypre_ParKrylovInnerProd, hypre_ParKrylovCopyVector,
         hypre_ParKrylovClearVector,
         hypre_ParKrylovScaleVector, hypre_ParKrylovAxpy,
         hypre_ParKrylovIdentitySetup, hypre_ParKrylovIdentity );

   *solver = ( (HYPRE_Solver) hypre_LGMRESCreate( lgmres_functions ) );
   if (!solver) hypre_error_in_arg(2);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLGMRESDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_ParCSRLGMRESDestroy( HYPRE_Solver solver )
{
   return( hypre_LGMRESDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLGMRESSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_ParCSRLGMRESSetup( HYPRE_Solver solver,
                        HYPRE_ParCSRMatrix A,
                        HYPRE_ParVector b,
                        HYPRE_ParVector x      )
{
   return( HYPRE_LGMRESSetup( solver,
                             (HYPRE_Matrix) A,
                             (HYPRE_Vector) b,
                             (HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLGMRESSolve
 *--------------------------------------------------------------------------*/

HYPRE_Int 
HYPRE_ParCSRLGMRESSolve( HYPRE_Solver solver,
                        HYPRE_ParCSRMatrix A,
                        HYPRE_ParVector b,
                        HYPRE_ParVector x      )
{
   return( HYPRE_LGMRESSolve( solver,
                             (HYPRE_Matrix) A,
                             (HYPRE_Vector) b,
                             (HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLGMRESSetKDim
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRLGMRESSetKDim( HYPRE_Solver solver,
                          HYPRE_Int             k_dim    )
{
   return( HYPRE_LGMRESSetKDim( solver, k_dim ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLGMRESSetAugDim
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRLGMRESSetAugDim( HYPRE_Solver solver,
                          HYPRE_Int             aug_dim    )
{
   return( HYPRE_LGMRESSetAugDim( solver, aug_dim ) );
}



/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLGMRESSetTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRLGMRESSetTol( HYPRE_Solver solver,
                         double             tol    )
{
   return( HYPRE_LGMRESSetTol( solver, tol ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLGMRESSetAbsoluteTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRLGMRESSetAbsoluteTol( HYPRE_Solver solver,
                         double             a_tol    )
{
   return( HYPRE_LGMRESSetAbsoluteTol( solver, a_tol ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLGMRESSetMinIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRLGMRESSetMinIter( HYPRE_Solver solver,
                             HYPRE_Int          min_iter )
{
   return( HYPRE_LGMRESSetMinIter( solver, min_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLGMRESSetMaxIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRLGMRESSetMaxIter( HYPRE_Solver solver,
                             HYPRE_Int          max_iter )
{
   return( HYPRE_LGMRESSetMaxIter( solver, max_iter ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLGMRESSetPrecond
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRLGMRESSetPrecond( HYPRE_Solver          solver,
                             HYPRE_PtrToParSolverFcn  precond,
                             HYPRE_PtrToParSolverFcn  precond_setup,
                             HYPRE_Solver          precond_solver )
{
   return( HYPRE_LGMRESSetPrecond( solver,
                                  (HYPRE_PtrToSolverFcn) precond,
                                  (HYPRE_PtrToSolverFcn) precond_setup,
                                  precond_solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLGMRESGetPrecond
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRLGMRESGetPrecond( HYPRE_Solver  solver,
                             HYPRE_Solver *precond_data_ptr )
{
   return( HYPRE_LGMRESGetPrecond( solver, precond_data_ptr ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLGMRESSetLogging
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRLGMRESSetLogging( HYPRE_Solver solver,
                             HYPRE_Int logging)
{
   return( HYPRE_LGMRESSetLogging( solver, logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLGMRESSetPrintLevel
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRLGMRESSetPrintLevel( HYPRE_Solver solver,
                             HYPRE_Int print_level)
{
   return( HYPRE_LGMRESSetPrintLevel( solver, print_level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLGMRESGetNumIterations
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRLGMRESGetNumIterations( HYPRE_Solver  solver,
                                   HYPRE_Int                *num_iterations )
{
   return( HYPRE_LGMRESGetNumIterations( solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLGMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRLGMRESGetFinalRelativeResidualNorm( HYPRE_Solver  solver,
                                               double             *norm   )
{
   return( HYPRE_LGMRESGetFinalRelativeResidualNorm( solver, norm ) );
}
