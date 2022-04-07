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
/******************************************************************************
 *
 * HYPRE_ParCSRSymQMR interface
 *
 *****************************************************************************/

extern void *hypre_SymQMRCreate();
extern int  hypre_SymQMRDestroy(void *);
extern int  hypre_SymQMRSetup(void *, void *, void *, void *);
extern int  hypre_SymQMRSolve(void *, void *, void *, void *);
extern int  hypre_SymQMRSetTol(void *, double);
extern int  hypre_SymQMRSetMaxIter(void *, int);
extern int  hypre_SymQMRSetStopCrit(void *, double);
extern int  hypre_SymQMRSetPrecond(void *, int (*precond)(void*,void*,void*,void*),
                                   int (*precond_setup)(void*,void*,void*,void*), void *);
extern int  hypre_SymQMRSetLogging(void *, int );
extern int  hypre_SymQMRGetNumIterations(void *, int *);
extern int  hypre_SymQMRGetFinalRelativeResidualNorm(void *, double *);

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRSymQMRCreate
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSRSymQMRCreate( MPI_Comm comm, HYPRE_Solver *solver )
{
   *solver = (HYPRE_Solver) hypre_SymQMRCreate( );

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRSymQMRDestroy
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSRSymQMRDestroy( HYPRE_Solver solver )
{
   return( hypre_SymQMRDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRSymQMRSetup
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSRSymQMRSetup( HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                             HYPRE_ParVector b, HYPRE_ParVector x      )
{
   return( hypre_SymQMRSetup( (void *) solver, (void *) A, (void *) b,
                              (void *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRSymQMRSolve
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSRSymQMRSolve( HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                                HYPRE_ParVector b, HYPRE_ParVector x      )
{
   return( hypre_SymQMRSolve( (void *) solver, (void *) A,
                              (void *) b, (void *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRSymQMRSetTol
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSRSymQMRSetTol( HYPRE_Solver solver, double tol    )
{
   return( hypre_SymQMRSetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRSymQMRSetMaxIter
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSRSymQMRSetMaxIter( HYPRE_Solver solver, int max_iter )
{
   return( hypre_SymQMRSetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRSymQMRSetStopCrit
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSRSymQMRSetStopCrit( HYPRE_Solver solver, int stop_crit )
{
   return( hypre_SymQMRSetStopCrit( (void *) solver, stop_crit ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRSymQMRSetPrecond
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSRSymQMRSetPrecond( HYPRE_Solver  solver,
          int (*precond)      (HYPRE_Solver sol, HYPRE_ParCSRMatrix matrix,
			       HYPRE_ParVector b, HYPRE_ParVector x),
          int (*precond_setup)(HYPRE_Solver sol, HYPRE_ParCSRMatrix matrix,
			       HYPRE_ParVector b, HYPRE_ParVector x),
          void                *precond_data )
{
   return( hypre_SymQMRSetPrecond( (void *) solver,
								   (HYPRE_Int (*)(void*,void*,void*,void*))precond,
								   (HYPRE_Int (*)(void*,void*,void*,void*))precond_setup,
								   precond_data ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRSymQMRSetLogging
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSRSymQMRSetLogging( HYPRE_Solver solver, int logging)
{
   return( hypre_SymQMRSetLogging( (void *) solver, logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRSymQMRetNumIterations
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSRSymQMRGetNumIterations(HYPRE_Solver solver,int *num_iterations)
{
   return( hypre_SymQMRGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRSymQMRGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSRSymQMRGetFinalRelativeResidualNorm( HYPRE_Solver  solver,
                                                       double *norm   )
{
   return( hypre_SymQMRGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}

