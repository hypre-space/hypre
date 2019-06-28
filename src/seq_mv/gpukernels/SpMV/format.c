#include "spmv.h"

/**
 * @brief convert coo to csr
 * @param[in] cooidx Specify if 0 or 1 indexed
 * @param[in] coo COO matrix
 * @param[out] csr CSR matrix
 */
HYPRE_Int coo_to_csr(HYPRE_Int cooidx, struct coo_t *coo, hypre_CSRMatrix **csr_ptr)
{
   const HYPRE_Int nrows = coo->nrows;
   const HYPRE_Int nnz = coo->nnz;
   hypre_CSRMatrix *csr = hypre_CSRMatrixCreate(coo->nrows, coo->ncols, coo->nnz);
   hypre_CSRMatrixInitialize_v2(csr, 0, HYPRE_MEMORY_HOST);

   /* fill (ia, ja, a) */
   HYPRE_Int i;
   for (i=0; i<nrows+1; i++)
   {
      csr->i[i] = 0;
   }
   for (i=0; i<nnz; i++)
   {
      HYPRE_Int row = coo->ir[i] - cooidx;
      csr->i[row+1] ++;
   }
   for (i=0; i<nrows; i++)
   {
      csr->i[i+1] += csr->i[i];
   }
   for (i=0; i<nnz; i++)
   {
      HYPRE_Int row = coo->ir[i] - cooidx;
      HYPRE_Int col = coo->jc[i] - cooidx;
      HYPRE_Real val = coo->val[i];
      HYPRE_Int k = csr->i[row];
      csr->data[k] = val;
      csr->j[k] = col;
      csr->i[row]++;
   }
   for (i=nrows; i>0; i--)
   {
      csr->i[i] = csr->i[i-1];
   }
   csr->i[0] = 0;

   assert(csr->i[csr->num_rows] == nnz);

   csr->num_nonzeros = nnz;

   /* sort rows ? */
   sortrow(csr);

   *csr_ptr = csr;

   return 0;
}

