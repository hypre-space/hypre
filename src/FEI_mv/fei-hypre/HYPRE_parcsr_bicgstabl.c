/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.10 $
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
 * HYPRE_ParCSRBiCGSTABL interface
 *
 *****************************************************************************/

extern void *hypre_BiCGSTABLCreate();
extern int  hypre_BiCGSTABLDestroy(void *);
extern int  hypre_BiCGSTABLSetup(void *, void *, void *, void *);
extern int  hypre_BiCGSTABLSolve(void *, void *, void *, void *);
extern int  hypre_BiCGSTABLSetTol(void *, double);
extern int  hypre_BiCGSTABLSetSize(void *, int);
extern int  hypre_BiCGSTABLSetMaxIter(void *, int);
extern int  hypre_BiCGSTABLSetStopCrit(void *, double);
extern int  hypre_BiCGSTABLSetPrecond(void *, int (*precond)(),
                               int (*precond_setup)(), void *);
extern int  hypre_BiCGSTABLSetLogging(void *, int);
extern int  hypre_BiCGSTABLGetNumIterations(void *,int *);
extern int  hypre_BiCGSTABLGetFinalRelativeResidualNorm(void *, double *);

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABLCreate
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSRBiCGSTABLCreate( MPI_Comm comm, HYPRE_Solver *solver )
{
   *solver = (HYPRE_Solver) hypre_BiCGSTABLCreate( );

   return 0;
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABLDestroy
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSRBiCGSTABLDestroy( HYPRE_Solver solver )
{
   return( hypre_BiCGSTABLDestroy( (void *) solver ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABLSetup
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSRBiCGSTABLSetup( HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                               HYPRE_ParVector b, HYPRE_ParVector x      )
{
   return( hypre_BiCGSTABLSetup( (void *) solver, (void *) A, (void *) b,
                                 (void *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABLSolve
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSRBiCGSTABLSolve( HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                                HYPRE_ParVector b, HYPRE_ParVector x      )
{
   return( hypre_BiCGSTABLSolve( (void *) solver, (void *) A,
                                 (void *) b, (void *) x ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABLSetTol
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSRBiCGSTABLSetTol( HYPRE_Solver solver, double tol    )
{
   return( hypre_BiCGSTABLSetTol( (void *) solver, tol ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABLSetSize
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSRBiCGSTABLSetSize( HYPRE_Solver solver, int size )
{
   return( hypre_BiCGSTABLSetSize( (void *) solver, size ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABLSetMaxIter
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRBiCGSTABLSetMaxIter( HYPRE_Solver solver, int max_iter )
{
   return( hypre_BiCGSTABLSetMaxIter( (void *) solver, max_iter ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABSetStopCrit
 *--------------------------------------------------------------------------*/

int
HYPRE_ParCSRBiCGSTABLSetStopCrit( HYPRE_Solver solver, int stop_crit )
{
   return( hypre_BiCGSTABLSetStopCrit( (void *) solver, stop_crit ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABLSetPrecond
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSRBiCGSTABLSetPrecond( HYPRE_Solver  solver,
          int (*precond)      (HYPRE_Solver sol, HYPRE_ParCSRMatrix matrix,
			       HYPRE_ParVector b, HYPRE_ParVector x),
          int (*precond_setup)(HYPRE_Solver sol, HYPRE_ParCSRMatrix matrix,
			       HYPRE_ParVector b, HYPRE_ParVector x),
          void               *precond_data )
{
   return( hypre_BiCGSTABLSetPrecond( (void *) solver,
                                precond, precond_setup, precond_data ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABLSetLogging
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSRBiCGSTABLSetLogging( HYPRE_Solver solver, int logging)
{
   return( hypre_BiCGSTABLSetLogging( (void *) solver, logging ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABGetNumIterations
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSRBiCGSTABLGetNumIterations(HYPRE_Solver solver,int *num_iterations)
{
   return( hypre_BiCGSTABLGetNumIterations( (void *) solver, num_iterations ) );
}

/*--------------------------------------------------------------------------
 * HYPRE_ParCSRBiCGSTABLGetFinalRelativeResidualNorm
 *--------------------------------------------------------------------------*/

int HYPRE_ParCSRBiCGSTABLGetFinalRelativeResidualNorm( HYPRE_Solver  solver,
                                                       double *norm   )
{
   return( hypre_BiCGSTABLGetFinalRelativeResidualNorm( (void *) solver, norm ) );
}

