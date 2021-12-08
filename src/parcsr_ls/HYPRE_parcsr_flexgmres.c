/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFlexGMRESCreate
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRFlexGMRESCreate( MPI_Comm comm, HYPRE_Solver *solver )
{
   hypre_FlexGMRESFunctions * fgmres_functions;

   if (!solver)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   fgmres_functions =
      hypre_FlexGMRESFunctionsCreate(
         hypre_ParKrylovCAlloc, hypre_ParKrylovFree, hypre_ParKrylovCommInfo,
         hypre_ParKrylovCreateVector,
         hypre_ParKrylovCreateVectorArray,
         hypre_ParKrylovDestroyVector, hypre_ParKrylovMatvecCreate,
         hypre_ParKrylovMatvec, hypre_ParKrylovMatvecDestroy,
         hypre_ParKrylovInnerProd, hypre_ParKrylovCopyVector,
         hypre_ParKrylovClearVector,
         hypre_ParKrylovScaleVector, hypre_ParKrylovAxpy,
         hypre_ParKrylovIdentitySetup, hypre_ParKrylovIdentity );
   *solver = ( (HYPRE_Solver) hypre_FlexGMRESCreate( fgmres_functions ) );

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFlexGMRESDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRFlexGMRESDestroy( HYPRE_Solver solver )
{
   return ( hypre_FlexGMRESDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFlexGMRESSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRFlexGMRESSetup( HYPRE_Solver solver,
                            HYPRE_ParCSRMatrix A,
                            HYPRE_ParVector b,
                            HYPRE_ParVector x      )
{
   return ( HYPRE_FlexGMRESSetup( solver,
                                  (HYPRE_Matrix) A,
                                  (HYPRE_Vector) b,
                                  (HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFlexGMRESSolve
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRFlexGMRESSolve( HYPRE_Solver solver,
                            HYPRE_ParCSRMatrix A,
                            HYPRE_ParVector b,
                            HYPRE_ParVector x      )
{
   return ( HYPRE_FlexGMRESSolve( solver,
                                  (HYPRE_Matrix) A,
                                  (HYPRE_Vector) b,
                                  (HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFlexGMRESSetKDim
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRFlexGMRESSetKDim( HYPRE_Solver solver,
                              HYPRE_Int             k_dim    )
{
   return ( HYPRE_FlexGMRESSetKDim( solver, k_dim ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFlexGMRESSetTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRFlexGMRESSetTol( HYPRE_Solver solver,
                             HYPRE_Real         tol    )
{
   return ( HYPRE_FlexGMRESSetTol( solver, tol ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFlexGMRESSetAbsoluteTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRFlexGMRESSetAbsoluteTol( HYPRE_Solver solver,
                                     HYPRE_Real         a_tol    )
{
   return ( HYPRE_FlexGMRESSetAbsoluteTol( solver, a_tol ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFlexGMRESSetMinIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRFlexGMRESSetMinIter( HYPRE_Solver solver,
                                 HYPRE_Int          min_iter )
{
   return ( HYPRE_FlexGMRESSetMinIter( solver, min_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFlexGMRESSetMaxIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRFlexGMRESSetMaxIter( HYPRE_Solver solver,
                                 HYPRE_Int          max_iter )
{
   return ( HYPRE_FlexGMRESSetMaxIter( solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFlexGMRESSetPrecond
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRFlexGMRESSetPrecond( HYPRE_Solver          solver,
                                 HYPRE_PtrToParSolverFcn  precond,
                                 HYPRE_PtrToParSolverFcn  precond_setup,
                                 HYPRE_Solver          precond_solver )
{
   return ( HYPRE_FlexGMRESSetPrecond( solver,
                                       (HYPRE_PtrToSolverFcn) precond,
                                       (HYPRE_PtrToSolverFcn) precond_setup,
                                       precond_solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFlexGMRESGetPrecond
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRFlexGMRESGetPrecond( HYPRE_Solver  solver,
                                 HYPRE_Solver *precond_data_ptr )
{
   return ( HYPRE_FlexGMRESGetPrecond( solver, precond_data_ptr ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFlexGMRESSetLogging
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRFlexGMRESSetLogging( HYPRE_Solver solver,
                                 HYPRE_Int logging)
{
   return ( HYPRE_FlexGMRESSetLogging( solver, logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFlexGMRESSetPrintLevel
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRFlexGMRESSetPrintLevel( HYPRE_Solver solver,
                                    HYPRE_Int print_level)
{
   return ( HYPRE_FlexGMRESSetPrintLevel( solver, print_level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFlexGMRESGetNumIterations
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRFlexGMRESGetNumIterations( HYPRE_Solver  solver,
                                       HYPRE_Int                *num_iterations )
{
   return ( HYPRE_FlexGMRESGetNumIterations( solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFlexGMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRFlexGMRESGetFinalRelativeResidualNorm( HYPRE_Solver  solver,
                                                   HYPRE_Real         *norm   )
{
   return ( HYPRE_FlexGMRESGetFinalRelativeResidualNorm( solver, norm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFlexGMRESGetResidual
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRFlexGMRESGetResidual( HYPRE_Solver  solver,
                                  HYPRE_ParVector *residual)
{
   return ( HYPRE_FlexGMRESGetResidual( solver, (void *) residual ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFlexGMRESSetModifyPC
 *--------------------------------------------------------------------------*/

HYPRE_Int HYPRE_ParCSRFlexGMRESSetModifyPC( HYPRE_Solver  solver,
                                            HYPRE_PtrToModifyPCFcn modify_pc)

{
   return ( HYPRE_FlexGMRESSetModifyPC( solver,
                                        (HYPRE_PtrToModifyPCFcn) modify_pc));
}



