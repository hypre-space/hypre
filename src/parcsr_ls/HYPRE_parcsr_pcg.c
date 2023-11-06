/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_parcsr_ls.h"

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGCreate
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRPCGCreate( MPI_Comm comm, HYPRE_Solver *solver )
{
   HYPRE_UNUSED_VAR(comm);

   hypre_PCGFunctions * pcg_functions;

   if (!solver)
   {
      hypre_error_in_arg(2);
      return hypre_error_flag;
   }
   pcg_functions =
      hypre_PCGFunctionsCreate(
         hypre_ParKrylovCAlloc, hypre_ParKrylovFree, hypre_ParKrylovCommInfo,
         hypre_ParKrylovCreateVector,
         hypre_ParKrylovDestroyVector, hypre_ParKrylovMatvecCreate,
         hypre_ParKrylovMatvec, hypre_ParKrylovMatvecDestroy,
         hypre_ParKrylovInnerProd, hypre_ParKrylovCopyVector,
         hypre_ParKrylovClearVector,
         hypre_ParKrylovScaleVector, hypre_ParKrylovAxpy,
         hypre_ParKrylovIdentitySetup, hypre_ParKrylovIdentity );
   *solver = ( (HYPRE_Solver) hypre_PCGCreate( pcg_functions ) );

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRPCGDestroy( HYPRE_Solver solver )
{
   return ( hypre_PCGDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRPCGSetup( HYPRE_Solver solver,
                      HYPRE_ParCSRMatrix A,
                      HYPRE_ParVector b,
                      HYPRE_ParVector x      )
{
   return ( HYPRE_PCGSetup( solver,
                            (HYPRE_Matrix) A,
                            (HYPRE_Vector) b,
                            (HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSolve
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRPCGSolve( HYPRE_Solver solver,
                      HYPRE_ParCSRMatrix A,
                      HYPRE_ParVector b,
                      HYPRE_ParVector x      )
{
   return ( HYPRE_PCGSolve( solver,
                            (HYPRE_Matrix) A,
                            (HYPRE_Vector) b,
                            (HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRPCGSetTol( HYPRE_Solver solver,
                       HYPRE_Real   tol    )
{
   return ( HYPRE_PCGSetTol( solver, tol ) );
}
/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetAbsoluteTol
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRPCGSetAbsoluteTol( HYPRE_Solver solver,
                               HYPRE_Real   a_tol    )
{
   return ( HYPRE_PCGSetAbsoluteTol( solver, a_tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetMaxIter
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRPCGSetMaxIter( HYPRE_Solver solver,
                           HYPRE_Int    max_iter )
{
   return ( HYPRE_PCGSetMaxIter( solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetStopCrit
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRPCGSetStopCrit( HYPRE_Solver solver,
                            HYPRE_Int    stop_crit )
{
   return ( HYPRE_PCGSetStopCrit( solver, stop_crit ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetTwoNorm
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRPCGSetTwoNorm( HYPRE_Solver solver,
                           HYPRE_Int    two_norm )
{
   return ( HYPRE_PCGSetTwoNorm( solver, two_norm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetRelChange
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRPCGSetRelChange( HYPRE_Solver solver,
                             HYPRE_Int    rel_change )
{
   return ( HYPRE_PCGSetRelChange( solver, rel_change ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetPrecond
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRPCGSetPrecond( HYPRE_Solver            solver,
                           HYPRE_PtrToParSolverFcn precond,
                           HYPRE_PtrToParSolverFcn precond_setup,
                           HYPRE_Solver            precond_solver )
{
   return ( HYPRE_PCGSetPrecond( solver,
                                 (HYPRE_PtrToSolverFcn) precond,
                                 (HYPRE_PtrToSolverFcn) precond_setup,
                                 precond_solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetPreconditioner
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRPCGSetPreconditioner( HYPRE_Solver solver,
                                  HYPRE_Solver precond )
{
   return ( HYPRE_PCGSetPreconditioner( solver, precond ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGGetPrecond
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRPCGGetPrecond( HYPRE_Solver  solver,
                           HYPRE_Solver *precond_data_ptr )
{
   return ( HYPRE_PCGGetPrecond( solver, precond_data_ptr ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetPrintLevel
 * an obsolete function; use HYPRE_PCG* functions instead
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRPCGSetPrintLevel( HYPRE_Solver solver,
                              HYPRE_Int level )
{
   return ( HYPRE_PCGSetPrintLevel( solver, level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetLogging
 * an obsolete function; use HYPRE_PCG* functions instead
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRPCGSetLogging( HYPRE_Solver solver,
                           HYPRE_Int level )
{
   return ( HYPRE_PCGSetLogging( solver, level ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGGetNumIterations
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRPCGGetNumIterations( HYPRE_Solver  solver,
                                 HYPRE_Int    *num_iterations )
{
   return ( HYPRE_PCGGetNumIterations( solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRPCGGetFinalRelativeResidualNorm( HYPRE_Solver  solver,
                                             HYPRE_Real   *norm   )
{
   return ( HYPRE_PCGGetFinalRelativeResidualNorm( solver, norm ) );
}


/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGGetResidual
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRPCGGetResidual( HYPRE_Solver  solver,
                            HYPRE_ParVector *residual   )
{
   return ( HYPRE_PCGGetResidual( solver, (void *) residual ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRDiagScaleSetup
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRDiagScaleSetup( HYPRE_Solver solver,
                            HYPRE_ParCSRMatrix A,
                            HYPRE_ParVector y,
                            HYPRE_ParVector x      )
{
   HYPRE_UNUSED_VAR(solver);
   HYPRE_UNUSED_VAR(A);
   HYPRE_UNUSED_VAR(y);
   HYPRE_UNUSED_VAR(x);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRDiagScale
 *--------------------------------------------------------------------------*/

HYPRE_Int
HYPRE_ParCSRDiagScale( HYPRE_Solver solver,
                       HYPRE_ParCSRMatrix HA,
                       HYPRE_ParVector Hy,
                       HYPRE_ParVector Hx      )
{
   HYPRE_UNUSED_VAR(solver);

   return hypre_ParCSRDiagScaleVector((hypre_ParCSRMatrix *) HA,
                                      (hypre_ParVector *)    Hy,
                                      (hypre_ParVector *)    Hx);
}
