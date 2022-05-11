/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include "utilities/_hypre_utilities.h"
#include "HYPRE.h"
#include "IJ_mv/HYPRE_IJ_mv.h"
#include "parcsr_mv/HYPRE_parcsr_mv.h"
#include "parcsr_mv/_hypre_parcsr_mv.h"
#include "parcsr_ls/HYPRE_parcsr_ls.h"

#include "HYPRE_FEI.h"
#include "_hypre_FEI.h"

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSCreate
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSRBiCGSCreate( MPI_Comm comm, HYPRE_Solver *solver )
{
   *solver = (HYPRE_Solver) hypre_BiCGSCreate( );

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSDestroy
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSRBiCGSDestroy( HYPRE_Solver solver )
{
   return( hypre_BiCGSDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSSetup
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSRBiCGSSetup( HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                               HYPRE_ParVector b, HYPRE_ParVector x      )
{
   return( hypre_BiCGSSetup( (void *) solver, (void *) A, (void *) b,
                                 (void *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSSolve
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSRBiCGSSolve( HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                                HYPRE_ParVector b, HYPRE_ParVector x      )
{
   return( hypre_BiCGSSolve( (void *) solver, (void *) A,
                                 (void *) b, (void *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSSetTol
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSRBiCGSSetTol( HYPRE_Solver solver, double tol    )
{
   return( hypre_BiCGSSetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSSetMaxIter
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSRBiCGSSetMaxIter( HYPRE_Solver solver, int max_iter )
{
   return( hypre_BiCGSSetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSetStopCrit
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSRBiCGSSetStopCrit( HYPRE_Solver solver, int stop_crit )
{
   return( hypre_BiCGSSetStopCrit( (void *) solver, stop_crit ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSSetPrecond
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSRBiCGSSetPrecond( HYPRE_Solver  solver,
          int (*precond)      (HYPRE_Solver sol, HYPRE_ParCSRMatrix matrix,
			       HYPRE_ParVector b, HYPRE_ParVector x),
          int (*precond_setup)(HYPRE_Solver sol, HYPRE_ParCSRMatrix matrix,
			       HYPRE_ParVector b, HYPRE_ParVector x),
          void               *precond_data )
{
   return( hypre_BiCGSSetPrecond( (void *) solver,
								  (HYPRE_Int (*)(void*,void*,void*,void*))precond,
								  (HYPRE_Int (*)(void*,void*,void*,void*))precond_setup,
								  precond_data ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSSetLogging
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSRBiCGSSetLogging( HYPRE_Solver solver, int logging)
{
   return( hypre_BiCGSSetLogging( (void *) solver, logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSetNumIterations
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSRBiCGSGetNumIterations(HYPRE_Solver solver,int *num_iterations)
{
   return( hypre_BiCGSGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSRBiCGSGetFinalRelativeResidualNorm( HYPRE_Solver  solver,
                                                       double *norm   )
{
   return( hypre_BiCGSGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}

