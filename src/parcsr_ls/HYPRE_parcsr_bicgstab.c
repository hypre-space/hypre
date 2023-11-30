/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABCreate
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRBiCGSTABCreate( MPI_Comm comm, HYPRE_Solver *solver )
{
   HYPRE_UNUSED_VAR(comm);

   hypre_BiCGSTABFunctions * bicgstab_functions;

   if (!solver)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   bicgstab_functions =
      hypre_BiCGSTABFunctionsCreate(
         hypre_ParKrylovCreateVector,
         hypre_ParKrylovDestroyVector,
         hypre_ParKrylovMatvecCreate,
         hypre_ParKrylovMatvec,
         hypre_ParKrylovMatvecDestroy,
         hypre_ParKrylovInnerProd,
         hypre_ParKrylovCopyVector,
         hypre_ParKrylovClearVector,
         hypre_ParKrylovScaleVector,
         hypre_ParKrylovAxpy,
         hypre_ParKrylovCommInfo,
         hypre_ParKrylovIdentitySetup,
         hypre_ParKrylovIdentity );
   *solver = ( (HYPRE_Solver) hypre_BiCGSTABCreate( bicgstab_functions) );

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRBiCGSTABDestroy( HYPRE_Solver solver )
{
   return ( hypre_BiCGSTABDestroy( (void *) solver ) );
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
   return ( HYPRE_BiCGSTABSetup( solver,
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
   return ( HYPRE_BiCGSTABSolve( solver,
                                 (HYPRE_Matrix) A,
                                 (HYPRE_Vector) b,
                                 (HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABSetTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRBiCGSTABSetTol( HYPRE_Solver solver,
                            HYPRE_Real         tol    )
{
   return ( HYPRE_BiCGSTABSetTol( solver, tol ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABSetAbsoluteTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRBiCGSTABSetAbsoluteTol( HYPRE_Solver solver,
                                    HYPRE_Real         a_tol    )
{
   return ( HYPRE_BiCGSTABSetAbsoluteTol( solver, a_tol ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABSetMinIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRBiCGSTABSetMinIter( HYPRE_Solver solver,
                                HYPRE_Int          min_iter )
{
   return ( HYPRE_BiCGSTABSetMinIter( solver, min_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABSetMaxIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRBiCGSTABSetMaxIter( HYPRE_Solver solver,
                                HYPRE_Int          max_iter )
{
   return ( HYPRE_BiCGSTABSetMaxIter( solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABSetStopCrit
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRBiCGSTABSetStopCrit( HYPRE_Solver solver,
                                 HYPRE_Int          stop_crit )
{
   return ( HYPRE_BiCGSTABSetStopCrit( solver, stop_crit ) );
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
   return ( HYPRE_BiCGSTABSetPrecond( solver,
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
   return ( HYPRE_BiCGSTABGetPrecond( solver, precond_data_ptr ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABSetLogging
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRBiCGSTABSetLogging( HYPRE_Solver solver,
                                HYPRE_Int logging)
{
   return ( HYPRE_BiCGSTABSetLogging( solver, logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABSetPrintLevel
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRBiCGSTABSetPrintLevel( HYPRE_Solver solver,
                                   HYPRE_Int print_level)
{
   return ( HYPRE_BiCGSTABSetPrintLevel( solver, print_level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABGetNumIterations
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRBiCGSTABGetNumIterations( HYPRE_Solver  solver,
                                      HYPRE_Int                *num_iterations )
{
   return ( HYPRE_BiCGSTABGetNumIterations( solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRBiCGSTABGetFinalRelativeResidualNorm( HYPRE_Solver  solver,
                                                  HYPRE_Real         *norm   )
{
   return ( HYPRE_BiCGSTABGetFinalRelativeResidualNorm( solver, norm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABGetResidual
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRBiCGSTABGetResidual( HYPRE_Solver  solver,
                                 HYPRE_ParVector *residual)
{
   return ( HYPRE_BiCGSTABGetResidual( solver, (void *) residual ) );
}
