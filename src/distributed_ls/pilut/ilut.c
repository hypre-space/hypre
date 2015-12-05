/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.6 $
 ***********************************************************************EHEADER*/




/*
 * ilut.c
 *
 * This file contains the top level code for the parallel hypre_ILUT algorithms
 *
 * Started 11/29/95
 * George
 *
 * $Id: ilut.c,v 2.6 2010/12/20 19:27:34 falgout Exp $
 */

#include <math.h>
#include "./DistributedMatrixPilutSolver.h"

/*************************************************************************
* This function is the entry point of the hypre_ILUT factorization
**************************************************************************/
HYPRE_Int hypre_ILUT(DataDistType *ddist, HYPRE_DistributedMatrix matrix, FactorMatType *ldu, 
          HYPRE_Int maxnz, double tol, hypre_PilutSolverGlobals *globals )
{
  HYPRE_Int i, ierr;
  ReduceMatType rmat;
  HYPRE_Int dummy_row_ptr[2], size;
  double *values;

#ifdef HYPRE_DEBUG
  hypre_printf("hypre_ILUT, maxnz = %d\n ", maxnz);
#endif

  /* Allocate memory for ldu */
  if (ldu->lsrowptr) hypre_TFree(ldu->lsrowptr);
  ldu->lsrowptr = hypre_idx_malloc(ddist->ddist_lnrows, "hypre_ILUT: ldu->lsrowptr");

  if (ldu->lerowptr) hypre_TFree(ldu->lerowptr);
  ldu->lerowptr = hypre_idx_malloc(ddist->ddist_lnrows, "hypre_ILUT: ldu->lerowptr");

  if (ldu->lcolind) hypre_TFree(ldu->lcolind);
  ldu->lcolind  = hypre_idx_malloc_init(maxnz*ddist->ddist_lnrows, 0, "hypre_ILUT: ldu->lcolind");

  if (ldu->lvalues) hypre_TFree(ldu->lvalues);
  ldu->lvalues  =  hypre_fp_malloc_init(maxnz*ddist->ddist_lnrows, 0, "hypre_ILUT: ldu->lvalues");

  if (ldu->usrowptr) hypre_TFree(ldu->usrowptr);
  ldu->usrowptr = hypre_idx_malloc(ddist->ddist_lnrows, "hypre_ILUT: ldu->usrowptr");

  if (ldu->uerowptr) hypre_TFree(ldu->uerowptr);
  ldu->uerowptr = hypre_idx_malloc(ddist->ddist_lnrows, "hypre_ILUT: ldu->uerowptr");

  if (ldu->ucolind) hypre_TFree(ldu->ucolind);
  ldu->ucolind  = hypre_idx_malloc_init(maxnz*ddist->ddist_lnrows, 0, "hypre_ILUT: ldu->ucolind");

  if (ldu->uvalues) hypre_TFree(ldu->uvalues);
  ldu->uvalues  =  hypre_fp_malloc_init(maxnz*ddist->ddist_lnrows, 0.0, "hypre_ILUT: ldu->uvalues");

  if (ldu->dvalues) hypre_TFree(ldu->dvalues);
  ldu->dvalues = hypre_fp_malloc(ddist->ddist_lnrows, "hypre_ILUT: ldu->dvalues");

  if (ldu->nrm2s) hypre_TFree(ldu->nrm2s);
  ldu->nrm2s   = hypre_fp_malloc_init(ddist->ddist_lnrows, 0.0, "hypre_ILUT: ldu->nrm2s");

  if (ldu->perm) hypre_TFree(ldu->perm);
  ldu->perm  = hypre_idx_malloc_init(ddist->ddist_lnrows, 0, "hypre_ILUT: ldu->perm");

  if (ldu->iperm) hypre_TFree(ldu->iperm);
  ldu->iperm = hypre_idx_malloc_init(ddist->ddist_lnrows, 0, "hypre_ILUT: ldu->iperm");

  firstrow = ddist->ddist_rowdist[mype];

  dummy_row_ptr[ 0 ] = 0;

  /* Initialize ldu */
  for (i=0; i<ddist->ddist_lnrows; i++) {
    ldu->lsrowptr[i] =
      ldu->lerowptr[i] =
      ldu->usrowptr[i] =
      ldu->uerowptr[i] = maxnz*i;

    ierr = HYPRE_DistributedMatrixGetRow( matrix, firstrow+i, &size,
               NULL, &values);
    /* if (ierr) return(ierr);*/
    dummy_row_ptr[ 1 ] = size;
    hypre_ComputeAdd2Nrms( 1, dummy_row_ptr, values, &(ldu->nrm2s[i]) );
    ierr = HYPRE_DistributedMatrixRestoreRow( matrix, firstrow+i, &size,
               NULL, &values);
  }

  /* Factor the internal nodes first */
  hypre_MPI_Barrier( pilut_comm );

#ifdef HYPRE_TIMING
  {
   HYPRE_Int SerILUT_timer;

   SerILUT_timer = hypre_InitializeTiming( "Sequential hypre_ILUT done on each proc" );

   hypre_BeginTiming( SerILUT_timer );
#endif

  hypre_SerILUT(ddist, matrix, ldu, &rmat, maxnz, tol, globals);

  hypre_MPI_Barrier( pilut_comm );

#ifdef HYPRE_TIMING
   hypre_EndTiming( SerILUT_timer );
   /* hypre_FinalizeTiming( SerILUT_timer ); */
  }
#endif

  /* Factor the interface nodes */
#ifdef HYPRE_TIMING
  {
   HYPRE_Int ParILUT_timer;

   ParILUT_timer = hypre_InitializeTiming( "Parallel portion of hypre_ILUT factorization" );

   hypre_BeginTiming( ParILUT_timer );
#endif

  hypre_ParILUT(ddist, ldu, &rmat, maxnz, tol, globals);

  hypre_MPI_Barrier( pilut_comm );

#ifdef HYPRE_TIMING
   hypre_EndTiming( ParILUT_timer );
   /* hypre_FinalizeTiming( ParILUT_timer ); */
  }
#endif

  /*hypre_free_multi(rmat.rmat_rnz, rmat.rmat_rrowlen, 
             rmat.rmat_rcolind, rmat.rmat_rvalues, -1);*/
  hypre_TFree(rmat.rmat_rnz);
  hypre_TFree(rmat.rmat_rrowlen);
  hypre_TFree(rmat.rmat_rcolind);
  hypre_TFree(rmat.rmat_rvalues);

  return( ierr );
}


/*************************************************************************
* This function computes the 2 norms of the rows and adds them into the 
* nrm2s array ... Changed to "Add" by AJC, Dec 22 1997.
**************************************************************************/
void hypre_ComputeAdd2Nrms(HYPRE_Int num_rows, HYPRE_Int *rowptr, double *values, double *nrm2s)
{
  HYPRE_Int i, j, n;
  double sum;

  for (i=0; i<num_rows; i++) {
    n = rowptr[i+1]-rowptr[i];
    /* sum = SNRM2(&n, values+rowptr[i], &incx);*/
    sum = 0.0;
    for (j=0; j<n; j++) sum += (values[rowptr[i]+j] * values[rowptr[i]+j]);
    sum = sqrt( sum );
    nrm2s[i] += sum;
  }
}
