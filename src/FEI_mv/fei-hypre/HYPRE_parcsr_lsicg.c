/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.9 $
 ***********************************************************************EHEADER*/





#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#include "utilities/_hypre_utilities.h"
#include "HYPRE.h"
#include "IJ_mv/HYPRE_IJ_mv.h"
#include "parcsr_mv/HYPRE_parcsr_mv.h"
#include "parcsr_mv/_hypre_parcsr_mv.h"
#include "parcsr_ls/HYPRE_parcsr_ls.h"

/******************************************************************************
 *
 * HYPRE_ParCSRLSICG interface
 *
 *****************************************************************************/

extern void *hypre_LSICGCreate();
extern int  hypre_LSICGDestroy(void *);
extern int  hypre_LSICGSetup(void *, void *, void *, void *);
extern int  hypre_LSICGSolve(void *, void  *, void  *, void  *);
extern int  hypre_LSICGSetTol(void *, double);
extern int  hypre_LSICGSetMaxIter(void *, int);
extern int  hypre_LSICGSetStopCrit(void *, double);
extern int  hypre_LSICGSetPrecond(void *, int (*precond)(),
                                  int (*precond_setup)(), void *);
extern int  hypre_LSICGSetLogging(void *, int);
extern int  hypre_LSICGGetNumIterations(void *,int *);
extern int hypre_LSICGGetFinalRelativeResidualNorm(void *, double *);

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLSICGCreate
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSRLSICGCreate( MPI_Comm comm, HYPRE_Solver *solver )
{
   *solver = (HYPRE_Solver) hypre_LSICGCreate( );

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLSICGDestroy
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSRLSICGDestroy( HYPRE_Solver solver )
{
   return( hypre_LSICGDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLSICGSetup
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSRLSICGSetup( HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                            HYPRE_ParVector b, HYPRE_ParVector x      )
{
   return( hypre_LSICGSetup( (void *) solver, (void *) A, (void *) b,
                                 (void *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLSICGSolve
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSRLSICGSolve( HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                                HYPRE_ParVector b, HYPRE_ParVector x      )
{
   return( hypre_LSICGSolve( (void *) solver, (void *) A,
                                 (void *) b, (void *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLSICGSetTol
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSRLSICGSetTol( HYPRE_Solver solver, double tol    )
{
   return( hypre_LSICGSetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLSICGSetMaxIter
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSRLSICGSetMaxIter( HYPRE_Solver solver, int max_iter )
{
   return( hypre_LSICGSetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLSICGetStopCrit
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSRLSICGSetStopCrit( HYPRE_Solver solver, int stop_crit )
{
   return( hypre_LSICGSetStopCrit( (void *) solver, stop_crit ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLSICGSetPrecond
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSRLSICGSetPrecond( HYPRE_Solver  solver,
          int (*precond)      (HYPRE_Solver sol, HYPRE_ParCSRMatrix matrix,
			       HYPRE_ParVector b, HYPRE_ParVector x),
          int (*precond_setup)(HYPRE_Solver sol, HYPRE_ParCSRMatrix matrix,
			       HYPRE_ParVector b, HYPRE_ParVector x),
          void *precond_data )
{
   return( hypre_LSICGSetPrecond( (void *) solver,
                                precond, precond_setup, precond_data ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLSICGSetLogging
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSRLSICGSetLogging( HYPRE_Solver solver, int logging)
{
   return( hypre_LSICGSetLogging( (void *) solver, logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLSICGetNumIterations
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSRLSICGGetNumIterations(HYPRE_Solver solver,int *num_iterations)
{
   return( hypre_LSICGGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRLSICGGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSRLSICGGetFinalRelativeResidualNorm( HYPRE_Solver  solver,
                                                       double *norm   )
{
   return( hypre_LSICGGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}

