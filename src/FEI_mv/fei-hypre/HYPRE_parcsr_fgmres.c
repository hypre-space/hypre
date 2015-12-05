/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.11 $
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

extern void *hypre_FGMRESCreate();
extern int  hypre_FGMRESDestroy(void *);
extern int  hypre_FGMRESSetup(void *, void *, void *, void *);
extern int  hypre_FGMRESSolve(void *, void *, void *, void *);
extern int  hypre_FGMRESSetKDim(void *, int);
extern int  hypre_FGMRESSetTol(void *, double);
extern int  hypre_FGMRESSetMaxIter(void *, int);
extern int  hypre_FGMRESSetStopCrit(void *, double);
extern int  hypre_FGMRESSetPrecond(void *, int (*precond)(), 
                                   int (*precond_setup)(),void *precond_data);
extern int  hypre_FGMRESGetPrecond(void *, HYPRE_Solver *);
extern int  hypre_FGMRESSetLogging(void *, int);
extern int  hypre_FGMRESGetNumIterations(void *, int *);
extern int  hypre_FGMRESGetFinalRelativeResidualNorm(void *,double *);
extern int  hypre_FGMRESUpdatePrecondTolerance(void *, int (*update_tol)());

/******************************************************************************
 *
 * HYPRE_ParCSRFGMRES interface
 *
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFGMRESCreate
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSRFGMRESCreate( MPI_Comm comm, HYPRE_Solver *solver )
{
   *solver = (HYPRE_Solver) hypre_FGMRESCreate( );

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFGMRESDestroy
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSRFGMRESDestroy( HYPRE_Solver solver )
{
   return( hypre_FGMRESDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFGMRESSetup
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSRFGMRESSetup( HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                             HYPRE_ParVector b, HYPRE_ParVector x      )
{
   return( hypre_FGMRESSetup( (void *) solver, (void *) A, (void *) b,
                              (void *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFGMRESSolve
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSRFGMRESSolve( HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                             HYPRE_ParVector b, HYPRE_ParVector x      )
{
   return( hypre_FGMRESSolve( (void *) solver, (void *) A,
                              (void *) b, (void *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFGMRESSetKDim
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSRFGMRESSetKDim( HYPRE_Solver solver, int dim    )
{
   return( hypre_FGMRESSetKDim( (void *) solver, dim ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFGMRESSetTol
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSRFGMRESSetTol( HYPRE_Solver solver, double tol    )
{
   return( hypre_FGMRESSetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFGMRESSetMaxIter
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSRFGMRESSetMaxIter( HYPRE_Solver solver, int max_iter )
{
   return( hypre_FGMRESSetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFGMRESetStopCrit
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSRFGMRESSetStopCrit( HYPRE_Solver solver, int stop_crit )
{
   return( hypre_FGMRESSetStopCrit( (void *) solver, stop_crit ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFGMRESSetPrecond
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSRFGMRESSetPrecond( HYPRE_Solver  solver,
          int (*precond)(HYPRE_Solver sol, HYPRE_ParCSRMatrix matrix,
			HYPRE_ParVector b, HYPRE_ParVector x),
          int (*precond_setup)(HYPRE_Solver sol, HYPRE_ParCSRMatrix matrix,
			       HYPRE_ParVector b, HYPRE_ParVector x),
          void *precond_data )
{
   return( hypre_FGMRESSetPrecond( (void *) solver,
                                   precond, precond_setup, precond_data ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFGMRESSetLogging
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSRFGMRESSetLogging( HYPRE_Solver solver, int logging)
{
   return( hypre_FGMRESSetLogging( (void *) solver, logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFGMRESetNumIterations
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSRFGMRESGetNumIterations(HYPRE_Solver solver,int *num_iterations)
{
   return( hypre_FGMRESGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFGMRESGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSRFGMRESGetFinalRelativeResidualNorm( HYPRE_Solver  solver,
                                                    double *norm   )
{
   return( hypre_FGMRESGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRFGMRESUpdatePrecondTolerance
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSRFGMRESUpdatePrecondTolerance( HYPRE_Solver  solver,
          int (*update_tol)(HYPRE_Solver sol, double ) )
{
   return( hypre_FGMRESUpdatePrecondTolerance(solver, update_tol) );
}

