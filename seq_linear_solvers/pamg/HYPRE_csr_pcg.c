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
   /* The function names with a CG in them are in
      seq_linear_solvers/pamg/cg_fun.c .  These functions do rather little -
      e.g., cast to the correct type - before calling something else.
   */
   hypre_PCGFunctions * pcg_functions =
      hypre_PCGFunctionsCreate(
         hypre_CAlloc, hypre_CGFree, hypre_CGCreateVector,
         hypre_CGDestroyVector, hypre_CGMatvecCreate,
         hypre_CGMatvec, hypre_CGMatvecDestroy,
         hypre_CGInnerProd, hypre_CGCopyVector,
         hypre_CGClearVector,
         hypre_CGScaleVector, hypre_CGAxpy,
         hypre_CGIdentitySetup, hypre_CGIdentity );

   *solver = ( (HYPRE_Solver) hypre_PCGCreate( pcg_functions ) );

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRPCGDestroy
 *--------------------------------------------------------------------------*/

int 
HYPRE_CSRPCGDestroy( HYPRE_Solver solver )
{
   return( hypre_PCGDestroy( (void *) solver ) );
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
   return( hypre_PCGSetup( (void *) solver,
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
   return( hypre_PCGSolve( (void *) solver,
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
   return( hypre_PCGSetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRPCGSetMaxIter
 *--------------------------------------------------------------------------*/

int
HYPRE_CSRPCGSetMaxIter( HYPRE_Solver solver,
                           int                max_iter )
{
   return( hypre_PCGSetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRPCGSetStopCrit
 *--------------------------------------------------------------------------*/

/* not supported by krylov/pcg.c.  was supported by old cg.c.  TO DO:
 decide whether this has any functionality. If so, add the parameter and
 implement this function.  If not, delete this function.
int
HYPRE_CSRPCGSetStopCrit( HYPRE_Solver solver,
                           int          stop_crit )
{
   return( hypre_PCGSetStopCrit( (void *) solver, stop_crit ) );
}
*/

/*--------------------------------------------------------------------------
 * HYPRE_CSRPCGSetTwoNorm
 *--------------------------------------------------------------------------*/

int
HYPRE_CSRPCGSetTwoNorm( HYPRE_Solver solver,
                           int                two_norm )
{
   return( hypre_PCGSetTwoNorm( (void *) solver, two_norm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRPCGSetRelChange
 *--------------------------------------------------------------------------*/

int
HYPRE_CSRPCGSetRelChange( HYPRE_Solver solver,
                             int                rel_change )
{
   return( hypre_PCGSetRelChange( (void *) solver, rel_change ) );
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
   return( hypre_PCGSetPrecond( (void *) solver,
                                precond, precond_setup, precond_data ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRPCGGetPrecond
 *--------------------------------------------------------------------------*/

int
HYPRE_CSRPCGGetPrecond( HYPRE_Solver  solver,
                           HYPRE_Solver *precond_data_ptr )
{
   return( hypre_PCGGetPrecond( (void *)     solver,
                                   (HYPRE_Solver *) precond_data_ptr ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRPCGSetLogging
 *--------------------------------------------------------------------------*/

int
HYPRE_CSRPCGSetLogging( HYPRE_Solver solver,
                           int                logging )
{
   return( hypre_PCGSetLogging( (void *) solver, logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRPCGGetNumIterations
 *--------------------------------------------------------------------------*/

int
HYPRE_CSRPCGGetNumIterations( HYPRE_Solver  solver,
                                 int                *num_iterations )
{
   return( hypre_PCGGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRPCGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int
HYPRE_CSRPCGGetFinalRelativeResidualNorm( HYPRE_Solver  solver,
                                             double             *norm   )
{
   return( hypre_PCGGetFinalRelativeResidualNorm( (void *) solver, norm ) );
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
