/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESCreate
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRGMRESCreate( MPI_Comm comm, HYPRE_Solver *solver )
{
   HYPRE_UNUSED_VAR(comm);

   hypre_GMRESFunctions * gmres_functions;

   if (!solver)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   gmres_functions =
      hypre_GMRESFunctionsCreate(
         hypre_ParKrylovCAlloc,
         hypre_ParKrylovFree,
         hypre_ParKrylovCommInfo,
         hypre_ParKrylovCreateVector,
         hypre_ParKrylovCreateVectorArray,
         hypre_ParKrylovDestroyVector,
         hypre_ParKrylovMatvecCreate,
         hypre_ParKrylovMatvec,
         hypre_ParKrylovMatvecDestroy,
         hypre_ParKrylovInnerProd,
         hypre_ParKrylovCopyVector,
         hypre_ParKrylovClearVector,
         hypre_ParKrylovScaleVector,
         hypre_ParKrylovAxpy,
         hypre_ParKrylovIdentitySetup,
         hypre_ParKrylovIdentity );
   *solver = ( (HYPRE_Solver) hypre_GMRESCreate( gmres_functions ) );

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRGMRESDestroy( HYPRE_Solver solver )
{
   return ( hypre_GMRESDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRGMRESSetup( HYPRE_Solver solver,
                        HYPRE_ParCSRMatrix A,
                        HYPRE_ParVector b,
                        HYPRE_ParVector x      )
{
   return ( HYPRE_GMRESSetup( solver,
                              (HYPRE_Matrix) A,
                              (HYPRE_Vector) b,
                              (HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSolve
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRGMRESSolve( HYPRE_Solver solver,
                        HYPRE_ParCSRMatrix A,
                        HYPRE_ParVector b,
                        HYPRE_ParVector x      )
{
   return ( HYPRE_GMRESSolve( solver,
                              (HYPRE_Matrix) A,
                              (HYPRE_Vector) b,
                              (HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSetKDim
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRGMRESSetKDim( HYPRE_Solver solver,
                          HYPRE_Int             k_dim    )
{
   return ( HYPRE_GMRESSetKDim( solver, k_dim ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSetTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRGMRESSetTol( HYPRE_Solver solver,
                         HYPRE_Real         tol    )
{
   return ( HYPRE_GMRESSetTol( solver, tol ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSetAbsoluteTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRGMRESSetAbsoluteTol( HYPRE_Solver solver,
                                 HYPRE_Real         a_tol    )
{
   return ( HYPRE_GMRESSetAbsoluteTol( solver, a_tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSetMinIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRGMRESSetMinIter( HYPRE_Solver solver,
                             HYPRE_Int          min_iter )
{
   return ( HYPRE_GMRESSetMinIter( solver, min_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSetMaxIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRGMRESSetMaxIter( HYPRE_Solver solver,
                             HYPRE_Int          max_iter )
{
   return ( HYPRE_GMRESSetMaxIter( solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSetStopCrit - OBSOLETE
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRGMRESSetStopCrit( HYPRE_Solver solver,
                              HYPRE_Int          stop_crit )
{
   return ( HYPRE_GMRESSetStopCrit( solver, stop_crit ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSetPrecond
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRGMRESSetPrecond( HYPRE_Solver          solver,
                             HYPRE_PtrToParSolverFcn  precond,
                             HYPRE_PtrToParSolverFcn  precond_setup,
                             HYPRE_Solver          precond_solver )
{
   return ( HYPRE_GMRESSetPrecond( solver,
                                   (HYPRE_PtrToSolverFcn) precond,
                                   (HYPRE_PtrToSolverFcn) precond_setup,
                                   precond_solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESGetPrecond
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRGMRESGetPrecond( HYPRE_Solver  solver,
                             HYPRE_Solver *precond_data_ptr )
{
   return ( HYPRE_GMRESGetPrecond( solver, precond_data_ptr ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSetLogging
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRGMRESSetLogging( HYPRE_Solver solver,
                             HYPRE_Int logging)
{
   return ( HYPRE_GMRESSetLogging( solver, logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESSetPrintLevel
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRGMRESSetPrintLevel( HYPRE_Solver solver,
                                HYPRE_Int print_level)
{
   return ( HYPRE_GMRESSetPrintLevel( solver, print_level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESGetNumIterations
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRGMRESGetNumIterations( HYPRE_Solver  solver,
                                   HYPRE_Int    *num_iterations )
{
   return ( HYPRE_GMRESGetNumIterations( solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRGMRESGetFinalRelativeResidualNorm( HYPRE_Solver  solver,
                                               HYPRE_Real   *norm   )
{
   return ( HYPRE_GMRESGetFinalRelativeResidualNorm( solver, norm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRGMRESGetResidual
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRGMRESGetResidual( HYPRE_Solver solver,
                              HYPRE_ParVector *residual   )
{
   return ( HYPRE_GMRESGetResidual( solver, (void *) residual ) );
}

/*--------------------------------------------------------------------------
 * Setup routine for on-processor triangular solve as preconditioning.
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSROnProcTriSetup(HYPRE_Solver       solver,
                           HYPRE_ParCSRMatrix HA,
                           HYPRE_ParVector    Hy,
                           HYPRE_ParVector    Hx)
{
   HYPRE_UNUSED_VAR(solver);
   HYPRE_UNUSED_VAR(Hy);
   HYPRE_UNUSED_VAR(Hx);

   hypre_ParCSRMatrix *A = (hypre_ParCSRMatrix *) HA;

   /* Check for and get topological ordering of matrix */
   if (!hypre_ParCSRMatrixProcOrdering(A))
   {
      hypre_CSRMatrix *A_diag        = hypre_ParCSRMatrixDiag(A);
      HYPRE_Real      *A_diag_data   = hypre_CSRMatrixData(A_diag);
      HYPRE_Int       *A_diag_i      = hypre_CSRMatrixI(A_diag);
      HYPRE_Int       *A_diag_j      = hypre_CSRMatrixJ(A_diag);
      HYPRE_Int        n             = hypre_CSRMatrixNumRows(A_diag);
      HYPRE_Int       *proc_ordering = hypre_TAlloc(HYPRE_Int, n, HYPRE_MEMORY_HOST);

      hypre_topo_sort(A_diag_i, A_diag_j, A_diag_data, proc_ordering, n);
      hypre_ParCSRMatrixProcOrdering(A) = proc_ordering;
   }

   return 0;
}

/*--------------------------------------------------------------------------
 * Solve routine for on-processor triangular solve as preconditioning.
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSROnProcTriSolve(HYPRE_Solver       solver,
                           HYPRE_ParCSRMatrix HA,
                           HYPRE_ParVector    Hy,
                           HYPRE_ParVector    Hx)
{
   HYPRE_UNUSED_VAR(solver);

   hypre_ParCSRMatrix *A = (hypre_ParCSRMatrix *) HA;
   hypre_ParVector    *y = (hypre_ParVector *) Hy;
   hypre_ParVector    *x = (hypre_ParVector *) Hx;
   HYPRE_Int           ierr = 0;

   ierr = hypre_BoomerAMGRelax(A, y, NULL, 10, 0, 1, 1, NULL, x, NULL, NULL);

   return ierr;
}
