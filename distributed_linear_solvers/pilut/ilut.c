/*
 * ilut.c
 *
 * This file contains the top level code for the parallel ILUT algorithms
 *
 * Started 11/29/95
 * George
 *
 * $Id$
 */

#include "./DistributedMatrixPilutSolver.h"

/*************************************************************************
* This function is the entry point of the ILUT factorization
**************************************************************************/
int ILUT(DataDistType *ddist, HYPRE_DistributedMatrix matrix, FactorMatType *ldu, 
          int maxnz, double tol, hypre_PilutSolverGlobals *globals )
{
  int i, j, k, ierr;
  ReduceMatType rmat;
  int dummy_row_ptr[2], *col_ind, size;
  double *values;

#ifdef HYPRE_DEBUG
  printf("ILUT, maxnz = %d\n ", maxnz);
#endif

  /* Allocate memory for ldu */
  if (ldu->lsrowptr) hypre_TFree(ldu->lsrowptr);
  ldu->lsrowptr = idx_malloc(ddist->ddist_lnrows, "ILUT: ldu->lsrowptr");

  if (ldu->lerowptr) hypre_TFree(ldu->lerowptr);
  ldu->lerowptr = idx_malloc(ddist->ddist_lnrows, "ILUT: ldu->lerowptr");

  if (ldu->lcolind) hypre_TFree(ldu->lcolind);
  ldu->lcolind  = idx_malloc_init(maxnz*ddist->ddist_lnrows, 0, "ILUT: ldu->lcolind");

  if (ldu->lvalues) hypre_TFree(ldu->lvalues);
  ldu->lvalues  =  fp_malloc_init(maxnz*ddist->ddist_lnrows, 0, "ILUT: ldu->lvalues");

  if (ldu->usrowptr) hypre_TFree(ldu->usrowptr);
  ldu->usrowptr = idx_malloc(ddist->ddist_lnrows, "ILUT: ldu->usrowptr");

  if (ldu->uerowptr) hypre_TFree(ldu->uerowptr);
  ldu->uerowptr = idx_malloc(ddist->ddist_lnrows, "ILUT: ldu->uerowptr");

  if (ldu->ucolind) hypre_TFree(ldu->ucolind);
  ldu->ucolind  = idx_malloc_init(maxnz*ddist->ddist_lnrows, 0, "ILUT: ldu->ucolind");

  if (ldu->uvalues) hypre_TFree(ldu->uvalues);
  ldu->uvalues  =  fp_malloc_init(maxnz*ddist->ddist_lnrows, 0.0, "ILUT: ldu->uvalues");

  if (ldu->dvalues) hypre_TFree(ldu->dvalues);
  ldu->dvalues = fp_malloc(ddist->ddist_lnrows, "ILUT: ldu->dvalues");

  if (ldu->nrm2s) hypre_TFree(ldu->nrm2s);
  ldu->nrm2s   = fp_malloc_init(ddist->ddist_lnrows, 0.0, "ILUT: ldu->nrm2s");

  if (ldu->perm) hypre_TFree(ldu->perm);
  ldu->perm  = idx_malloc_init(ddist->ddist_lnrows, 0, "ILUT: ldu->perm");

  if (ldu->iperm) hypre_TFree(ldu->iperm);
  ldu->iperm = idx_malloc_init(ddist->ddist_lnrows, 0, "ILUT: ldu->iperm");

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
    if (ierr) return(ierr);
    dummy_row_ptr[ 1 ] = size;
    ComputeAdd2Nrms( 1, dummy_row_ptr, values, &(ldu->nrm2s[i]) );
    ierr = HYPRE_DistributedMatrixRestoreRow( matrix, firstrow+i, &size,
               NULL, &values);
  }

  /* Factor the internal nodes first */
  MPI_Barrier( pilut_comm );

#ifdef HYPRE_TIMING
  {
   int SerILUT_timer;

   SerILUT_timer = hypre_InitializeTiming( "Sequential ILUT done on each proc" );

   hypre_BeginTiming( SerILUT_timer );
#endif

  SerILUT(ddist, matrix, ldu, &rmat, maxnz, tol, globals);

  MPI_Barrier( pilut_comm );

#ifdef HYPRE_TIMING
   hypre_EndTiming( SerILUT_timer );
   /* hypre_FinalizeTiming( SerILUT_timer ); */
  }
#endif

  /* Factor the interface nodes */
#ifdef HYPRE_TIMING
  {
   int ParILUT_timer;

   ParILUT_timer = hypre_InitializeTiming( "Parallel portion of ILUT factorization" );

   hypre_BeginTiming( ParILUT_timer );
#endif

  ParILUT(ddist, ldu, &rmat, maxnz, tol, globals);

  MPI_Barrier( pilut_comm );

#ifdef HYPRE_TIMING
   hypre_EndTiming( ParILUT_timer );
   /* hypre_FinalizeTiming( ParILUT_timer ); */
  }
#endif

  free_multi(rmat.rmat_rnz, rmat.rmat_rrowlen, 
             rmat.rmat_rcolind, rmat.rmat_rvalues, -1);

  return( ierr );
}


/*************************************************************************
* This function computes the 2 norms of the rows and adds them into the 
* nrm2s array ... Changed to "Add" by AJC, Dec 22 1997.
**************************************************************************/
void ComputeAdd2Nrms(int num_rows, int *rowptr, double *values, double *nrm2s)
{
  int i, j, n, incx=1;
  double sum;

  for (i=0; i<num_rows; i++) {
    n = rowptr[i+1]-rowptr[i];
    sum = SNRM2(&n, values+rowptr[i], &incx);
    nrm2s[i] += sum;
  }
}
