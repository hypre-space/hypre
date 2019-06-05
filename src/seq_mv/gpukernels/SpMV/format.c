#include "spmv.h"

/**
 * @brief convert coo to csr
 * @param[in] cooidx Specify if 0 or 1 indexed
 * @param[in] coo COO matrix
 * @param[out] csr CSR matrix
 */
int coo_to_csr(int cooidx, struct coo_t *coo, struct csr_t *csr) {
  const int nnz = coo->nnz;
  //printf("@@@@ coo2csr, nnz %d\n", nnz);
  /* allocate memory */
  csr->nrows = coo->nrows;
  csr->ncols = coo->ncols;
  csr->ia = (int *) malloc((csr->nrows+1)*sizeof(int));
  csr->ja = (int *) malloc(nnz*sizeof(int));
  csr->a = (REAL *) malloc(nnz*sizeof(REAL));
  const int nrows = coo->nrows;
  /* fill (ia, ja, a) */
  int i;
  for (i=0; i<nrows+1; i++) {
    csr->ia[i] = 0;
  }
  for (i=0; i<nnz; i++) {
    int row = coo->ir[i] - cooidx;
    csr->ia[row+1] ++;
  }
  for (i=0; i<nrows; i++) {
    csr->ia[i+1] += csr->ia[i];
  }
  for (i=0; i<nnz; i++) {
    int row = coo->ir[i] - cooidx;
    int col = coo->jc[i] - cooidx;
    double val = coo->val[i];
    int k = csr->ia[row];
    csr->a[k] = val;
    csr->ja[k] = col;
    csr->ia[row]++;
  }
  for (i=nrows; i>0; i--) {
    csr->ia[i] = csr->ia[i-1];
  }
  csr->ia[0] = 0;

  assert(csr->ia[csr->nrows] == nnz);

  /* sort rows ? */
  sortrow(csr);

  return 0;
}

