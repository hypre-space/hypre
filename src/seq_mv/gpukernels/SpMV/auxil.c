#include "spmv.h"

double wall_timer() {
  struct timeval tim;
  gettimeofday(&tim, NULL);
  double t = tim.tv_sec + tim.tv_usec/1e6;
  return(t);
}

/*---------------------------------------------*/
void print_header() {
#if DOUBLEPRECISION
    printf("\nTesting SpMV, DOUBLE precision\n");
#else
    printf("\nTesting SpMV, SINGLE precision\n");
#endif
}

/*-----------------------------------------*/
double error_norm(REAL *x, REAL *y, int n) {
  int i;
  double t, normz, normx;
  normx = normz = 0.0;
  for (i=0; i<n; i++) {
    t = x[i]-y[i];
    normz += t*t;
    normx += x[i]*x[i];
  }
  return (sqrt(normz/normx));
}

/*---------------------------*/
void FreeCOO(struct coo_t *coo)
{
  free(coo->ir);
  free(coo->jc);
  free(coo->val);
}

/*---------------------------*/
void FreeCSR(struct csr_t *csr)
{
  free(csr->a);
  free(csr->ia);
  free(csr->ja);
}

/**
 * @brief convert csr to csc
 * Assume input csr is 0-based index
 * output csc 0/1 index specified by OUTINDEX      *
 * @param[in] OUTINDEX specifies if CSC should be 0/1 index
 * @param[in] nrow Number of rows
 * @param[in] ncol Number of columns
 * @param[in] job flag
 * @param[in] a Values of input matrix
 * @param[in] ia Input row pointers
 * @param[in] ja Input column indices
 * @param[out] ao Output values
 * @param[out] iao Output row pointers
 * @param[out] jao Output column indices
 */
void csrcsc(int OUTINDEX, const int nrow, const int ncol, int job,
    double *a, int *ja, int *ia,
    double *ao, int *jao, int *iao) {
  int i,k;
  for (i=0; i<ncol+1; i++) {
    iao[i] = 0;
  }
  // compute nnz of columns of A
  for (i=0; i<nrow; i++) {
    for (k=ia[i]; k<ia[i+1]; k++) {
      iao[ja[k]+1] ++;
    }
  }
  // compute pointers from lengths
  for (i=0; i<ncol; i++) {
    iao[i+1] += iao[i];
  }
  // now do the actual copying
  for (i=0; i<nrow; i++) {
    for (k=ia[i]; k<ia[i+1]; k++) {
      int j = ja[k];
      if (job) {
        ao[iao[j]] = a[k];
      }
      jao[iao[j]++] = i + OUTINDEX;
    }
  }
  /*---- reshift iao and leave */
  for (i=ncol; i>0; i--) {
    iao[i] = iao[i-1] + OUTINDEX;
  }
  iao[0] = OUTINDEX;
}

/**
 * @brief  Sort each row of a csr by increasing column
 * order
 * By double transposition
 * @param[in] A Matrix to sort
 */
void sortrow(struct csr_t *A) {
  /*-------------------------------------------*/
  int nrows = A->nrows;
  int ncols = A->ncols;
  int nnz = A->ia[nrows];
  // work array
  double *b;
  int *jb, *ib;
  b = (double *) malloc(nnz*sizeof(double));
  jb = (int *) malloc(nnz*sizeof(int));
  ib = (int *) malloc((ncols+1)*sizeof(int));
  // double transposition
  csrcsc(0, nrows, ncols, 1, A->a, A->ja, A->ia, b, jb, ib);
  csrcsc(0, ncols, nrows, 1, b, jb, ib, A->a, A->ja, A->ia);
  // free
  free(b);
  free(jb);
  free(ib);
}

