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
 * HYPRE_CSRPCG interface
 *
 *****************************************************************************/
#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_CSRPCGCreate
 *--------------------------------------------------------------------------*/

int
HYPRE_CSRPCGCreate( HYPRE_Solver *solver )
{
   *solver = ( (HYPRE_Solver) hypre_CGCreate( ) );

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRPCGDestroy
 *--------------------------------------------------------------------------*/

int 
HYPRE_CSRPCGDestroy( HYPRE_Solver solver )
{
   return( hypre_CGDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRPCGSetup
 *--------------------------------------------------------------------------*/

int 
HYPRE_CSRPCGSetup( HYPRE_Solver solver,
                      HYPRE_CSRMatrix A,
                      HYPRE_Vector b,
                      HYPRE_Vector x      )
{
   return( hypre_CGSetup( (void *) solver,
                           (void *) A,
                           (void *) b,
                           (void *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRPCGSolve
 *--------------------------------------------------------------------------*/

int 
HYPRE_CSRPCGSolve( HYPRE_Solver solver,
                      HYPRE_CSRMatrix A,
                      HYPRE_Vector b,
                      HYPRE_Vector x      )
{
   return( hypre_CGSolve( (void *) solver,
                           (void *) A,
                           (void *) b,
                           (void *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRPCGSetTol
 *--------------------------------------------------------------------------*/

int
HYPRE_CSRPCGSetTol( HYPRE_Solver solver,
                       double             tol    )
{
   return( hypre_CGSetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRPCGSetMaxIter
 *--------------------------------------------------------------------------*/

int
HYPRE_CSRPCGSetMaxIter( HYPRE_Solver solver,
                           int                max_iter )
{
   return( hypre_CGSetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRPCGSetStopCrit
 *--------------------------------------------------------------------------*/

int
HYPRE_CSRPCGSetStopCrit( HYPRE_Solver solver,
                           int          stop_crit )
{
   return( hypre_CGSetStopCrit( (void *) solver, stop_crit ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRPCGSetTwoNorm
 *--------------------------------------------------------------------------*/

int
HYPRE_CSRPCGSetTwoNorm( HYPRE_Solver solver,
                           int                two_norm )
{
   return( hypre_CGSetTwoNorm( (void *) solver, two_norm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRPCGSetRelChange
 *--------------------------------------------------------------------------*/

int
HYPRE_CSRPCGSetRelChange( HYPRE_Solver solver,
                             int                rel_change )
{
   return( hypre_CGSetRelChange( (void *) solver, rel_change ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRPCGSetPrecond
 *--------------------------------------------------------------------------*/

int
HYPRE_CSRPCGSetPrecond( HYPRE_Solver  solver,
                           int (*precond)      (HYPRE_Solver sol,
						HYPRE_CSRMatrix matrix,
						HYPRE_Vector b,
						HYPRE_Vector x),
                           int (*precond_setup)(HYPRE_Solver sol,
						HYPRE_CSRMatrix matrix,
						HYPRE_Vector b,
						HYPRE_Vector x),
                           void                *precond_data )
{
   return( hypre_CGSetPrecond( (void *) solver,
                                precond, precond_setup, precond_data ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRPCGGetPrecond
 *--------------------------------------------------------------------------*/

int
HYPRE_CSRPCGGetPrecond( HYPRE_Solver  solver,
                           HYPRE_Solver *precond_data_ptr )
{
   return( hypre_CGGetPrecond( (void *)     solver,
                                   (HYPRE_Solver *) precond_data_ptr ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRPCGSetLogging
 *--------------------------------------------------------------------------*/

int
HYPRE_CSRPCGSetLogging( HYPRE_Solver solver,
                           int                logging )
{
   return( hypre_CGSetLogging( (void *) solver, logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRPCGGetNumIterations
 *--------------------------------------------------------------------------*/

int
HYPRE_CSRPCGGetNumIterations( HYPRE_Solver  solver,
                                 int                *num_iterations )
{
   return( hypre_CGGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRPCGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int
HYPRE_CSRPCGGetFinalRelativeResidualNorm( HYPRE_Solver  solver,
                                             double             *norm   )
{
   return( hypre_CGGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRDiagScaleSetup
 *--------------------------------------------------------------------------*/
 
int 
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
 
int 
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
   int *A_i = hypre_CSRMatrixI(A);

   int i, ierr = 0;

   for (i=0; i < hypre_VectorSize(x); i++)
   {
	x_data[i] = y_data[i]/A_data[A_i[i]];
   } 
 
   return ierr;
}
