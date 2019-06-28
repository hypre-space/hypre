#include "spmv.h"

hypre_double wall_timer()
{
  struct timeval tim;
  gettimeofday(&tim, NULL);
  hypre_double t = tim.tv_sec + tim.tv_usec/1e6;

  return(t);
}

/*---------------------------------------------*/
void print_header()
{
   if (sizeof(HYPRE_Real) == sizeof(hypre_double))
   {
      printf("\nTesting SpMV, DOUBLE precision\n");
   }
   else if (sizeof(HYPRE_Real) == sizeof(float))
   {
      printf("\nTesting SpMV, SINGLE precision\n");
   }
}

/*-----------------------------------------*/
HYPRE_Real error_norm(HYPRE_Real *x, HYPRE_Real *y, HYPRE_Int n) {
  HYPRE_Int i;
  HYPRE_Real t, normz, normx;
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
void csrcsc(HYPRE_Int OUTINDEX, const HYPRE_Int nrow, const HYPRE_Int ncol, HYPRE_Int job,
            HYPRE_Real *a, HYPRE_Int *ja, HYPRE_Int *ia,
            HYPRE_Real *ao, HYPRE_Int *jao, HYPRE_Int *iao) {
  HYPRE_Int i,k;
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
      HYPRE_Int j = ja[k];
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
 * By Double transposition
 * @param[in] A Matrix to sort
 */
void sortrow(hypre_CSRMatrix *A) {
  /*-------------------------------------------*/
  HYPRE_Int nrows = A->num_rows;
  HYPRE_Int ncols = A->num_cols;
  HYPRE_Int nnz = A->i[nrows];
  // work array
  HYPRE_Real *b;
  HYPRE_Int *jb, *ib;
  b = (HYPRE_Real *) malloc(nnz*sizeof(HYPRE_Real));
  jb = (HYPRE_Int *) malloc(nnz*sizeof(HYPRE_Int));
  ib = (HYPRE_Int *) malloc((ncols+1)*sizeof(HYPRE_Int));
  // Double transposition
  csrcsc(0, nrows, ncols, 1, A->data, A->j, A->i, b, jb, ib);
  csrcsc(0, ncols, nrows, 1, b, jb, ib, A->data, A->j, A->i);
  // free
  free(b);
  free(jb);
  free(ib);
}

void spmv_csr_cpu(hypre_CSRMatrix *csr, HYPRE_Real *x, HYPRE_Real *y)
{
   hypre_Vector *vx = hypre_SeqVectorCreate(csr->num_cols);
   hypre_Vector *vy = hypre_SeqVectorCreate(csr->num_rows);
   hypre_VectorMemoryLocation(vx) = HYPRE_MEMORY_HOST;
   hypre_VectorMemoryLocation(vy) = HYPRE_MEMORY_HOST;
   hypre_VectorOwnsData(vx) = 0;
   hypre_VectorOwnsData(vy) = 0;
   hypre_VectorData(vx) = x;
   hypre_VectorData(vy) = y;

   /*------------- CPU CSR SpMV kernel */
   hypre_double t1, t2;
   t1 = wall_timer();
   for (HYPRE_Int ii=0; ii<REPEAT; ii++)
   {
      hypre_CSRMatrixMatvecOutOfPlaceHost(1.0, csr, vx, 0.0, vy, vy, 0);
   }
   t2 = wall_timer() - t1;
   /*--------------------------------------------------*/
   printf("\n=== [CPU] CSR Kernel ===\n");
   printf("  %.2f ms, %.2f GFLOPS\n",
         t2*1e3/REPEAT, 2*(csr->i[csr->num_rows])/t2/1e9*REPEAT);

   hypre_SeqVectorDestroy(vx);
   hypre_SeqVectorDestroy(vy);
}

