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
 * HYPRE_CSRGMRES interface
 *
 *****************************************************************************/
#include "headers.h"

/*--------------------------------------------------------------------------
 * HYPRE_CSRGMRESCreate
 *--------------------------------------------------------------------------*/

int
HYPRE_CSRGMRESCreate( HYPRE_Solver *solver )
{
   hypre_GMRESFunctions * gmres_functions =
      hypre_GMRESFunctionsCreate(
         hypre_CAlloc, hypre_CGFree, hypre_CGCommInfo,
         hypre_CGCreateVector,
         hypre_CGCreateVectorArray,
         hypre_CGDestroyVector, hypre_CGMatvecCreate,
         hypre_CGMatvec, hypre_CGMatvecDestroy,
         hypre_CGInnerProd, hypre_CGCopyVector,
         hypre_CGClearVector,
         hypre_CGScaleVector, hypre_CGAxpy,
         hypre_CGIdentitySetup, hypre_CGIdentity );

   *solver = ( (HYPRE_Solver) hypre_GMRESCreate( gmres_functions ) );

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRGMRESDestroy
 *--------------------------------------------------------------------------*/

int 
HYPRE_CSRGMRESDestroy( HYPRE_Solver solver )
{
   return( hypre_GMRESDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRGMRESSetup
 *--------------------------------------------------------------------------*/

int 
HYPRE_CSRGMRESSetup( HYPRE_Solver solver,
                        HYPRE_CSRMatrix A,
                        HYPRE_Vector b,
                        HYPRE_Vector x      )
{
   return( hypre_GMRESSetup( (void *) solver,
                             (void *) A,
                             (void *) b,
                             (void *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRGMRESSolve
 *--------------------------------------------------------------------------*/

int 
HYPRE_CSRGMRESSolve( HYPRE_Solver solver,
                        HYPRE_CSRMatrix A,
                        HYPRE_Vector b,
                        HYPRE_Vector x      )
{
   return( hypre_GMRESSolve( (void *) solver,
                             (void *) A,
                             (void *) b,
                             (void *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRGMRESSetKDim
 *--------------------------------------------------------------------------*/

int
HYPRE_CSRGMRESSetKDim( HYPRE_Solver solver,
                          int             k_dim    )
{
   return( hypre_GMRESSetKDim( (void *) solver, k_dim ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRGMRESSetTol
 *--------------------------------------------------------------------------*/

int
HYPRE_CSRGMRESSetTol( HYPRE_Solver solver,
                         double             tol    )
{
   return( hypre_GMRESSetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRGMRESSetMinIter
 *--------------------------------------------------------------------------*/

int
HYPRE_CSRGMRESSetMinIter( HYPRE_Solver solver,
                             int          min_iter )
{
   return( hypre_GMRESSetMinIter( (void *) solver, min_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRGMRESSetMaxIter
 *--------------------------------------------------------------------------*/

int
HYPRE_CSRGMRESSetMaxIter( HYPRE_Solver solver,
                             int          max_iter )
{
   return( hypre_GMRESSetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRGMRESSetStopCrit
 *--------------------------------------------------------------------------*/

int
HYPRE_CSRGMRESSetStopCrit( HYPRE_Solver solver,
                              int          stop_crit )
{
   return( hypre_GMRESSetStopCrit( (void *) solver, stop_crit ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRGMRESSetPrecond
 *--------------------------------------------------------------------------*/

int
HYPRE_CSRGMRESSetPrecond( HYPRE_Solver  solver,
                             int (*precond)      (HYPRE_Solver sol, 
					 	  HYPRE_CSRMatrix matrix,
						  HYPRE_Vector b,
						  HYPRE_Vector x),
                             int (*precond_setup)(HYPRE_Solver sol, 
					 	  HYPRE_CSRMatrix matrix,
						  HYPRE_Vector b,
						  HYPRE_Vector x),
                             void               *precond_data )
{
   return( hypre_GMRESSetPrecond( (void *) solver,
                                precond, precond_setup, precond_data ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRGMRESGetPrecond
 *--------------------------------------------------------------------------*/

int
HYPRE_CSRGMRESGetPrecond( HYPRE_Solver  solver,
                             HYPRE_Solver *precond_data_ptr )
{
   return( hypre_GMRESGetPrecond( (void *)     solver,
                                  (HYPRE_Solver *) precond_data_ptr ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRGMRESSetLogging
 *--------------------------------------------------------------------------*/

int
HYPRE_CSRGMRESSetLogging( HYPRE_Solver solver,
                             int logging)
{
   return( hypre_GMRESSetLogging( (void *) solver, logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRGMRESGetNumIterations
 *--------------------------------------------------------------------------*/

int
HYPRE_CSRGMRESGetNumIterations( HYPRE_Solver  solver,
                                   int                *num_iterations )
{
   return( hypre_GMRESGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_CSRGMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int
HYPRE_CSRGMRESGetFinalRelativeResidualNorm( HYPRE_Solver  solver,
                                               double             *norm   )
{
   return( hypre_GMRESGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}
