/*BHEADER**********************************************************************
 * (c) 1998   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/

/******************************************************************************
 *
 * HYPRE_ParCSRBiCGSTAB interface
 *
 *****************************************************************************/
#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABCreate
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRBiCGSTABCreate( MPI_Comm comm, HYPRE_Solver *solver )
{
   hypre_BiCGSTABFunctions * bicgstab_functions =
      hypre_BiCGSTABFunctionsCreate(
         hypre_ParKrylovCreateVector,
         hypre_ParKrylovDestroyVector, hypre_ParKrylovMatvecCreate,
         hypre_ParKrylovMatvec, hypre_ParKrylovMatvecDestroy,
         hypre_ParKrylovInnerProd, hypre_ParKrylovCopyVector,
         hypre_ParKrylovScaleVector, hypre_ParKrylovAxpy,
         hypre_ParKrylovCommInfo,
         hypre_ParKrylovIdentitySetup, hypre_ParKrylovIdentity );

   *solver = ( (HYPRE_Solver) hypre_BiCGSTABCreate( bicgstab_functions) );

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABDestroy
 *--------------------------------------------------------------------------*/

int 
HYPRE_ParCSRBiCGSTABDestroy( HYPRE_Solver solver )
{
   return( HYPRE_BiCGSTABDestroy( solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABSetup
 *--------------------------------------------------------------------------*/

int 
HYPRE_ParCSRBiCGSTABSetup( HYPRE_Solver solver,
                        HYPRE_ParCSRMatrix A,
                        HYPRE_ParVector b,
                        HYPRE_ParVector x      )
{
   return( HYPRE_BiCGSTABSetup( solver, A, b, x ));
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABSolve
 *--------------------------------------------------------------------------*/

int 
HYPRE_ParCSRBiCGSTABSolve( HYPRE_Solver solver,
                        HYPRE_ParCSRMatrix A,
                        HYPRE_ParVector b,
                        HYPRE_ParVector x      )
{
   return( HYPRE_BiCGSTABSolve( solver, A, b, x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABSetTol
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRBiCGSTABSetTol( HYPRE_Solver solver,
                         double             tol    )
{
   return( HYPRE_BiCGSTABSetTol( solver, tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABSetMinIter
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRBiCGSTABSetMinIter( HYPRE_Solver solver,
                             int          min_iter )
{
   return( HYPRE_BiCGSTABSetMinIter( solver, min_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABSetMaxIter
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRBiCGSTABSetMaxIter( HYPRE_Solver solver,
                             int          max_iter )
{
   return( HYPRE_BiCGSTABSetMaxIter( solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABSetStopCrit
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRBiCGSTABSetStopCrit( HYPRE_Solver solver,
                              int          stop_crit )
{
   return( HYPRE_BiCGSTABSetStopCrit( solver, stop_crit ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABSetPrecond
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRBiCGSTABSetPrecond( HYPRE_Solver         solver,
                                HYPRE_PtrToParSolverFcn precond,
                                HYPRE_PtrToParSolverFcn precond_setup,
                                HYPRE_Solver         precond_solver )
{
   return( HYPRE_BiCGSTABSetPrecond( solver,
                                     precond, precond_setup,
                                     precond_solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABGetPrecond
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRBiCGSTABGetPrecond( HYPRE_Solver  solver,
                             HYPRE_Solver *precond_data_ptr )
{
   return( HYPRE_BiCGSTABGetPrecond(   solver,
                                    precond_data_ptr ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABSetLogging
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRBiCGSTABSetLogging( HYPRE_Solver solver,
                             int logging)
{
   return( HYPRE_BiCGSTABSetLogging( solver, logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABGetNumIterations
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRBiCGSTABGetNumIterations( HYPRE_Solver  solver,
                                   int                *num_iterations )
{
   return( HYPRE_BiCGSTABGetNumIterations( solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRBiCGSTABGetFinalRelativeResidualNorm( HYPRE_Solver  solver,
                                               double             *norm   )
{
   return( HYPRE_BiCGSTABGetFinalRelativeResidualNorm( solver, norm ) );
}
