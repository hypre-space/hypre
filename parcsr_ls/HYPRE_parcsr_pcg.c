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
 * HYPRE_ParCSRPCG interface
 *
 *****************************************************************************/
#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGCreate
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRPCGCreate( MPI_Comm comm, HYPRE_Solver *solver )
{
   /* The function names with a PCG in them are in
      parcsr_ls/pcg_par.c .  These functions do rather little -
      e.g., cast to the correct type - before calling something else.
      These names should be called, e.g., hypre_struct_Free, to reduce the
      chance of name conflicts. */
   hypre_PCGFunctions * pcg_functions =
      hypre_PCGFunctionsCreate(
         hypre_CAlloc, hypre_ParKrylovFree, hypre_ParKrylovCreateVector,
         hypre_ParKrylovDestroyVector, hypre_ParKrylovMatvecCreate,
         hypre_ParKrylovMatvec, hypre_ParKrylovMatvecDestroy,
         hypre_ParKrylovInnerProd, hypre_ParKrylovCopyVector,
         hypre_ParKrylovClearVector,
         hypre_ParKrylovScaleVector, hypre_ParKrylovAxpy,
         hypre_ParKrylovIdentitySetup, hypre_ParKrylovIdentity );

   *solver = ( (HYPRE_Solver) hypre_PCGCreate( pcg_functions ) );

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGDestroy
 *--------------------------------------------------------------------------*/

int 
HYPRE_ParCSRPCGDestroy( HYPRE_Solver solver )
{
   return( hypre_PCGDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetup
 *--------------------------------------------------------------------------*/

int 
HYPRE_ParCSRPCGSetup( HYPRE_Solver solver,
                      HYPRE_ParCSRMatrix A,
                      HYPRE_ParVector b,
                      HYPRE_ParVector x      )
{
   return( HYPRE_PCGSetup( solver,
                           (HYPRE_Matrix) A,
                           (HYPRE_Vector) b,
                           (HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSolve
 *--------------------------------------------------------------------------*/

int 
HYPRE_ParCSRPCGSolve( HYPRE_Solver solver,
                      HYPRE_ParCSRMatrix A,
                      HYPRE_ParVector b,
                      HYPRE_ParVector x      )
{
   return( HYPRE_PCGSolve( solver,
                           (HYPRE_Matrix) A,
                           (HYPRE_Vector) b,
                           (HYPRE_Vector) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetTol
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRPCGSetTol( HYPRE_Solver solver,
                       double             tol    )
{
   return( HYPRE_PCGSetTol( solver, tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetMaxIter
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRPCGSetMaxIter( HYPRE_Solver solver,
                           int                max_iter )
{
   return( HYPRE_PCGSetMaxIter( solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetStopCrit
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRPCGSetStopCrit( HYPRE_Solver solver,
                            int          stop_crit )
{
   return( HYPRE_PCGSetStopCrit( solver, stop_crit ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetTwoNorm
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRPCGSetTwoNorm( HYPRE_Solver solver,
                           int                two_norm )
{
   return( HYPRE_PCGSetTwoNorm( solver, two_norm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetRelChange
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRPCGSetRelChange( HYPRE_Solver solver,
                             int                rel_change )
{
   return( HYPRE_PCGSetRelChange( solver, rel_change ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetPrecond
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRPCGSetPrecond( HYPRE_Solver         solver,
                           HYPRE_PtrToParSolverFcn precond,
                           HYPRE_PtrToParSolverFcn precond_setup,
                           HYPRE_Solver         precond_solver )
{
   return( HYPRE_PCGSetPrecond( solver,
                                (HYPRE_PtrToSolverFcn) precond,
                                (HYPRE_PtrToSolverFcn) precond_setup,
                                precond_solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGGetPrecond
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRPCGGetPrecond( HYPRE_Solver  solver,
                           HYPRE_Solver *precond_data_ptr )
{
   return( HYPRE_PCGGetPrecond( solver, precond_data_ptr ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGSetLogging
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRPCGSetLogging( HYPRE_Solver solver,
                           int                logging )
{
   return( HYPRE_PCGSetLogging( solver, logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGGetNumIterations
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRPCGGetNumIterations( HYPRE_Solver  solver,
                                 int                *num_iterations )
{
   return( HYPRE_PCGGetNumIterations( solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRPCGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRPCGGetFinalRelativeResidualNorm( HYPRE_Solver  solver,
                                             double             *norm   )
{
   return( HYPRE_PCGGetFinalRelativeResidualNorm( solver, norm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRDiagScaleSetup
 *--------------------------------------------------------------------------*/
 
int 
HYPRE_ParCSRDiagScaleSetup( HYPRE_Solver solver,
                            HYPRE_ParCSRMatrix A,
                            HYPRE_ParVector y,
                            HYPRE_ParVector x      )
{
   return 0;
}
 
/*--------------------------------------------------------------------------
 * HYPRE_ParCSRDiagScale
 *--------------------------------------------------------------------------*/
 
int 
HYPRE_ParCSRDiagScale( HYPRE_Solver solver,
                       HYPRE_ParCSRMatrix HA,
                       HYPRE_ParVector Hy,
                       HYPRE_ParVector Hx      )
{
   hypre_ParCSRMatrix *A = (hypre_ParCSRMatrix *) HA;
   hypre_ParVector    *y = (hypre_ParVector *) Hy;
   hypre_ParVector    *x = (hypre_ParVector *) Hx;
   double *x_data = hypre_VectorData(hypre_ParVectorLocalVector(x));
   double *y_data = hypre_VectorData(hypre_ParVectorLocalVector(y));
   double *A_data = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(A));
   int *A_i = hypre_CSRMatrixI(hypre_ParCSRMatrixDiag(A));
   int local_size = hypre_VectorSize(hypre_ParVectorLocalVector(x));
   int i, ierr = 0;

   for (i=0; i < local_size; i++)
   {
	x_data[i] = y_data[i]/A_data[A_i[i]];
   } 
 
   return ierr;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRSymPrecondSetup
 *--------------------------------------------------------------------------*/
 
/*

int 
HYPRE_ParCSRSymPrecondSetup( HYPRE_Solver solver,
                             HYPRE_ParCSRMatrix A,
                             HYPRE_ParVector b,
                             HYPRE_ParVector x      )
{
   hypre_ParCSRMatrix *A = (hypre_ParCSRMatrix *) A;
   hypre_ParVector    *y = (hypre_ParVector *) b;
   hypre_ParVector    *x = (hypre_ParVector *) x;

   double *x_data = hypre_VectorData(hypre_ParVectorLocalVector(x));
   double *y_data = hypre_VectorData(hypre_ParVectorLocalVector(y));
   double *A_diag = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(A));
   double *A_offd = hypre_CSRMatrixData(hypre_ParCSRMatrixOffD(A));

   int i, ierr = 0;
   hypre_ParCSRMatrix *Asym;
   MPI_Comm comm;
   int global_num_rows;
   int global_num_cols;
   int *row_starts;
   int *col_starts;
   int num_cols_offd;
   int num_nonzeros_diag;
   int num_nonzeros_offd;

   Asym = hypre_ParCSRMatrixCreate(comm, global_num_rows, global_num_cols,
                                   row_starts, col_starts, num_cols_offd,
                                   num_nonzeros_diag, num_nonzeros_offd);

   for (i=0; i < hypre_VectorSize(hypre_ParVectorLocalVector(x)); i++)
   {
	x_data[i] = y_data[i]/A_data[A_i[i]];
   } 
 
   return ierr;
} */
