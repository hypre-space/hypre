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

  /* Allocate memory for ldu */
  ldu->lsrowptr = idx_malloc(ddist->ddist_lnrows, "ILUT: ldu->lsrowptr");
  ldu->lerowptr = idx_malloc(ddist->ddist_lnrows, "ILUT: ldu->lerowptr");
  ldu->lcolind  = idx_malloc_init(maxnz*ddist->ddist_lnrows, 0, "ILUT: ldu->lcolind");
  ldu->lvalues  =  fp_malloc_init(maxnz*ddist->ddist_lnrows, 0, "ILUT: ldu->lvalues");

  ldu->usrowptr = idx_malloc(ddist->ddist_lnrows, "ILUT: ldu->usrowptr");
  ldu->uerowptr = idx_malloc(ddist->ddist_lnrows, "ILUT: ldu->uerowptr");
  ldu->ucolind  = idx_malloc_init(maxnz*ddist->ddist_lnrows, 0, "ILUT: ldu->ucolind");
  ldu->uvalues  =  fp_malloc_init(maxnz*ddist->ddist_lnrows, 0, "ILUT: ldu->uvalues");

  ldu->dvalues = fp_malloc(ddist->ddist_lnrows, "ILUT: ldu->dvalues");
  ldu->nrm2s   = fp_malloc_init(ddist->ddist_lnrows, 0, "ILUT: ldu->nrm2s");

  ldu->perm  = idx_malloc_init(ddist->ddist_lnrows, 0, "ILUT: ldu->perm");
  ldu->iperm = idx_malloc_init(ddist->ddist_lnrows, 0, "ILUT: ldu->iperm");

  firstrow = ddist->ddist_rowdist[mype];

  dummy_row_ptr[ 0 ] = 0;

  /* Initialize ldu */
  for (i=0; i<ddist->ddist_lnrows; i++) {
    ldu->lsrowptr[i] =
      ldu->lerowptr[i] =
      ldu->usrowptr[i] =
      ldu->uerowptr[i] = maxnz*i;

    ierr = HYPRE_GetDistributedMatrixRow( matrix, firstrow+i, &size,
               NULL, &values);
    if (ierr) return(ierr);
    dummy_row_ptr[ 1 ] = size;
    ComputeAdd2Nrms( 1, dummy_row_ptr, values, &(ldu->nrm2s[i]) );
    ierr = HYPRE_RestoreDistributedMatrixRow( matrix, firstrow+i, &size,
               NULL, &values);
  }

  /* Factor the internal nodes first */
  MPI_Barrier( pilut_comm ); starttimer(SerTmr);

  SerILUT(ddist, matrix, ldu, &rmat, maxnz, tol, globals);

  MPI_Barrier( pilut_comm ); stoptimer(SerTmr);

  /* Factor the interface nodes */
  MPI_Barrier( pilut_comm ); starttimer(ParTmr);

  ParILUT(ddist, ldu, &rmat, maxnz, tol, globals);

  MPI_Barrier( pilut_comm ); stoptimer(ParTmr);

  free_multi(rmat.rmat_rnz, rmat.rmat_rcolind, rmat.rmat_rvalues, -1);
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
