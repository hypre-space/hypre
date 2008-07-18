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





/******************************************************************************
 *
 * HYPRE_ParCSRFlexGMRES interface
 *
 *****************************************************************************/
#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFlexGMRESCreate
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRFlexGMRESCreate( MPI_Comm comm, HYPRE_Solver *solver )
{
   hypre_FlexGMRESFunctions * fgmres_functions =
      hypre_FlexGMRESFunctionsCreate(
         hypre_CAlloc, hypre_ParKrylovFree, hypre_ParKrylovCommInfo,
         hypre_ParKrylovCreateVector,
         hypre_ParKrylovCreateVectorArray,
         hypre_ParKrylovDestroyVector, hypre_ParKrylovMatvecCreate,
         hypre_ParKrylovMatvec, hypre_ParKrylovMatvecDestroy,
         hypre_ParKrylovInnerProd, hypre_ParKrylovCopyVector,
         hypre_ParKrylovClearVector,
         hypre_ParKrylovScaleVector, hypre_ParKrylovAxpy,
         hypre_ParKrylovIdentitySetup, hypre_ParKrylovIdentity );

   *solver = ( (HYPRE_Solver) hypre_FlexGMRESCreate( fgmres_functions ) );
   if (!solver) hypre_error_in_arg(2);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFlexGMRESDestroy
 *--------------------------------------------------------------------------*/

int 
HYPRE_ParCSRFlexGMRESDestroy( HYPRE_Solver solver )
{
   return( hypre_FlexGMRESDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFlexGMRESSetup
 *--------------------------------------------------------------------------*/

int 
HYPRE_ParCSRFlexGMRESSetup( HYPRE_Solver solver,
                        HYPRE_ParCSRMatrix A,
                        HYPRE_ParVector b,
                        HYPRE_ParVector x      )
{
   return( HYPRE_FlexGMRESSetup( solver,
                             (HYPRE_Matrix) A,
                             (HYPRE_Vector) b,
                             (HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFlexGMRESSolve
 *--------------------------------------------------------------------------*/

int 
HYPRE_ParCSRFlexGMRESSolve( HYPRE_Solver solver,
                        HYPRE_ParCSRMatrix A,
                        HYPRE_ParVector b,
                        HYPRE_ParVector x      )
{
   return( HYPRE_FlexGMRESSolve( solver,
                             (HYPRE_Matrix) A,
                             (HYPRE_Vector) b,
                             (HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFlexGMRESSetKDim
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRFlexGMRESSetKDim( HYPRE_Solver solver,
                          int             k_dim    )
{
   return( HYPRE_FlexGMRESSetKDim( solver, k_dim ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFlexGMRESSetTol
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRFlexGMRESSetTol( HYPRE_Solver solver,
                         double             tol    )
{
   return( HYPRE_FlexGMRESSetTol( solver, tol ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFlexGMRESSetAbsoluteTol
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRFlexGMRESSetAbsoluteTol( HYPRE_Solver solver,
                         double             a_tol    )
{
   return( HYPRE_FlexGMRESSetAbsoluteTol( solver, a_tol ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFlexGMRESSetMinIter
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRFlexGMRESSetMinIter( HYPRE_Solver solver,
                             int          min_iter )
{
   return( HYPRE_FlexGMRESSetMinIter( solver, min_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFlexGMRESSetMaxIter
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRFlexGMRESSetMaxIter( HYPRE_Solver solver,
                             int          max_iter )
{
   return( HYPRE_FlexGMRESSetMaxIter( solver, max_iter ) );
}



/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFlexGMRESSetPrecond
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRFlexGMRESSetPrecond( HYPRE_Solver          solver,
                             HYPRE_PtrToParSolverFcn  precond,
                             HYPRE_PtrToParSolverFcn  precond_setup,
                             HYPRE_Solver          precond_solver )
{
   return( HYPRE_FlexGMRESSetPrecond( solver,
                                  (HYPRE_PtrToSolverFcn) precond,
                                  (HYPRE_PtrToSolverFcn) precond_setup,
                                  precond_solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFlexGMRESGetPrecond
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRFlexGMRESGetPrecond( HYPRE_Solver  solver,
                             HYPRE_Solver *precond_data_ptr )
{
   return( HYPRE_FlexGMRESGetPrecond( solver, precond_data_ptr ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFlexGMRESSetLogging
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRFlexGMRESSetLogging( HYPRE_Solver solver,
                             int logging)
{
   return( HYPRE_FlexGMRESSetLogging( solver, logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFlexGMRESSetPrintLevel
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRFlexGMRESSetPrintLevel( HYPRE_Solver solver,
                             int print_level)
{
   return( HYPRE_FlexGMRESSetPrintLevel( solver, print_level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFlexGMRESGetNumIterations
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRFlexGMRESGetNumIterations( HYPRE_Solver  solver,
                                   int                *num_iterations )
{
   return( HYPRE_FlexGMRESGetNumIterations( solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFlexGMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRFlexGMRESGetFinalRelativeResidualNorm( HYPRE_Solver  solver,
                                               double             *norm   )
{
   return( HYPRE_FlexGMRESGetFinalRelativeResidualNorm( solver, norm ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFlexGMRESSetModifyPC
 *--------------------------------------------------------------------------*/
 

int HYPRE_ParCSRFlexGMRESSetModifyPC( HYPRE_Solver  solver,
                                   HYPRE_PtrToModifyPCFcn modify_pc)

{
   return ( HYPRE_FlexGMRESSetModifyPC( solver,  (HYPRE_PtrToModifyPCFcn) modify_pc));
   

   
}



