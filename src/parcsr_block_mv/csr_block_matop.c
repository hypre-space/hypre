/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Matrix operation functions for hypre_CSRMatrix class.
 *
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixAdd:
 * adds two CSR Matrices A and B and returns a CSR Matrix C;
 * Note: The routine does not check for 0-elements which might be generated
 *       through cancellation of elements in A and B or already contained
         in A and B. To remove those, use hypre_CSRMatrixDeleteZeros
 *--------------------------------------------------------------------------*/

hypre_CSRBlockMatrix *
hypre_CSRBlockMatrixAdd(hypre_CSRBlockMatrix *A, hypre_CSRBlockMatrix *B)
{
   HYPRE_Complex    *A_data   = hypre_CSRMatrixData(A);
   HYPRE_Int        *A_i      = hypre_CSRMatrixI(A);
   HYPRE_Int        *A_j      = hypre_CSRMatrixJ(A);
   HYPRE_Int         nrows_A  = hypre_CSRMatrixNumRows(A);
   HYPRE_Int         ncols_A  = hypre_CSRMatrixNumCols(A);
   HYPRE_Complex    *B_data   = hypre_CSRMatrixData(B);
   HYPRE_Int        *B_i      = hypre_CSRMatrixI(B);
   HYPRE_Int        *B_j      = hypre_CSRMatrixJ(B);
   HYPRE_Int         nrows_B  = hypre_CSRMatrixNumRows(B);
   HYPRE_Int         ncols_B  = hypre_CSRMatrixNumCols(B);
   hypre_CSRMatrix  *C;
   HYPRE_Complex    *C_data;
   HYPRE_Int        *C_i;
   HYPRE_Int        *C_j;

   HYPRE_Int         block_size  = hypre_CSRBlockMatrixBlockSize(A);
   HYPRE_Int         block_sizeB = hypre_CSRBlockMatrixBlockSize(B);
   HYPRE_Int         ia, ib, ic, ii, jcol, num_nonzeros, bnnz;
   HYPRE_Int           pos;
   HYPRE_Int         *marker;

   if (nrows_A != nrows_B || ncols_A != ncols_B)
   {
      hypre_printf("Warning! incompatible matrix dimensions!\n");
      return NULL;
   }
   if (block_size != block_sizeB)
   {
      hypre_printf("Warning! incompatible matrix block size!\n");
      return NULL;
   }

   bnnz = block_size * block_size;
   marker = hypre_CTAlloc(HYPRE_Int,  ncols_A, HYPRE_MEMORY_HOST);
   C_i = hypre_CTAlloc(HYPRE_Int,  nrows_A + 1, HYPRE_MEMORY_HOST);

   for (ia = 0; ia < ncols_A; ia++) { marker[ia] = -1; }

   num_nonzeros = 0;
   C_i[0] = 0;
   for (ic = 0; ic < nrows_A; ic++)
   {
      for (ia = A_i[ic]; ia < A_i[ic + 1]; ia++)
      {
         jcol = A_j[ia];
         marker[jcol] = ic;
         num_nonzeros++;
      }
      for (ib = B_i[ic]; ib < B_i[ic + 1]; ib++)
      {
         jcol = B_j[ib];
         if (marker[jcol] != ic)
         {
            marker[jcol] = ic;
            num_nonzeros++;
         }
      }
      C_i[ic + 1] = num_nonzeros;
   }

   C = hypre_CSRBlockMatrixCreate(block_size, nrows_A, ncols_A, num_nonzeros);
   hypre_CSRMatrixI(C) = C_i;
   hypre_CSRMatrixInitialize(C);
   C_j = hypre_CSRMatrixJ(C);
   C_data = hypre_CSRMatrixData(C);

   for (ia = 0; ia < ncols_A; ia++) { marker[ia] = -1; }

   pos = 0;
   for (ic = 0; ic < nrows_A; ic++)
   {
      for (ia = A_i[ic]; ia < A_i[ic + 1]; ia++)
      {
         jcol = A_j[ia];
         C_j[pos] = jcol;
         for (ii = 0; ii < bnnz; ii++)
         {
            C_data[pos * bnnz + ii] = A_data[ia * bnnz + ii];
         }
         marker[jcol] = pos;
         pos++;
      }
      for (ib = B_i[ic]; ib < B_i[ic + 1]; ib++)
      {
         jcol = B_j[ib];
         if (marker[jcol] < C_i[ic])
         {
            C_j[pos] = jcol;
            for (ii = 0; ii < bnnz; ii++)
            {
               C_data[pos * bnnz + ii] = B_data[ib * bnnz + ii];
            }
            marker[jcol] = pos;
            pos++;
         }
         else
         {
            for (ii = 0; ii < bnnz; ii++)
            {
               C_data[marker[jcol]*bnnz + ii] = B_data[ib * bnnz + ii];
            }
         }
      }
   }
   hypre_TFree(marker, HYPRE_MEMORY_HOST);
   return C;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixMultiply
 * multiplies two CSR Matrices A and B and returns a CSR Matrix C;
 * Note: The routine does not check for 0-elements which might be generated
 *       through cancellation of elements in A and B or already contained
         in A and B. To remove those, use hypre_CSRMatrixDeleteZeros
 *--------------------------------------------------------------------------*/

hypre_CSRBlockMatrix *
hypre_CSRBlockMatrixMultiply(hypre_CSRBlockMatrix *A, hypre_CSRBlockMatrix *B)
{
   HYPRE_Complex    *A_data   = hypre_CSRMatrixData(A);
   HYPRE_Int        *A_i      = hypre_CSRMatrixI(A);
   HYPRE_Int        *A_j      = hypre_CSRMatrixJ(A);
   HYPRE_Int         nrows_A  = hypre_CSRMatrixNumRows(A);
   HYPRE_Int         ncols_A  = hypre_CSRMatrixNumCols(A);
   HYPRE_Int         block_size  = hypre_CSRBlockMatrixBlockSize(A);
   HYPRE_Complex    *B_data   = hypre_CSRMatrixData(B);
   HYPRE_Int        *B_i      = hypre_CSRMatrixI(B);
   HYPRE_Int        *B_j      = hypre_CSRMatrixJ(B);
   HYPRE_Int         nrows_B  = hypre_CSRMatrixNumRows(B);
   HYPRE_Int         ncols_B  = hypre_CSRMatrixNumCols(B);
   HYPRE_Int         block_sizeB = hypre_CSRBlockMatrixBlockSize(B);
   hypre_CSRMatrix  *C;
   HYPRE_Complex    *C_data;
   HYPRE_Int        *C_i;
   HYPRE_Int        *C_j;

   HYPRE_Int         ia, ib, ic, ja, jb, num_nonzeros = 0, bnnz;
   HYPRE_Int         row_start, counter;
   HYPRE_Complex    *a_entries, *b_entries, *c_entries, dzero = 0.0, done = 1.0;
   HYPRE_Int        *B_marker;

   if (ncols_A != nrows_B)
   {
      hypre_printf("Warning! incompatible matrix dimensions!\n");
      return NULL;
   }
   if (block_size != block_sizeB)
   {
      hypre_printf("Warning! incompatible matrix block size!\n");
      return NULL;
   }

   bnnz = block_size * block_size;
   B_marker = hypre_CTAlloc(HYPRE_Int,  ncols_B, HYPRE_MEMORY_HOST);
   C_i = hypre_CTAlloc(HYPRE_Int,  nrows_A + 1, HYPRE_MEMORY_HOST);

   for (ib = 0; ib < ncols_B; ib++) { B_marker[ib] = -1; }

   for (ic = 0; ic < nrows_A; ic++)
   {
      for (ia = A_i[ic]; ia < A_i[ic + 1]; ia++)
      {
         ja = A_j[ia];
         for (ib = B_i[ja]; ib < B_i[ja + 1]; ib++)
         {
            jb = B_j[ib];
            if (B_marker[jb] != ic)
            {
               B_marker[jb] = ic;
               num_nonzeros++;
            }
         }
      }
      C_i[ic + 1] = num_nonzeros;
   }

   C = hypre_CSRBlockMatrixCreate(block_size, nrows_A, ncols_B, num_nonzeros);
   hypre_CSRMatrixI(C) = C_i;
   hypre_CSRMatrixInitialize(C);
   C_j = hypre_CSRMatrixJ(C);
   C_data = hypre_CSRMatrixData(C);

   for (ib = 0; ib < ncols_B; ib++) { B_marker[ib] = -1; }

   counter = 0;
   for (ic = 0; ic < nrows_A; ic++)
   {
      row_start = C_i[ic];
      for (ia = A_i[ic]; ia < A_i[ic + 1]; ia++)
      {
         ja = A_j[ia];
         a_entries = &(A_data[ia * bnnz]);
         for (ib = B_i[ja]; ib < B_i[ja + 1]; ib++)
         {
            jb = B_j[ib];
            b_entries = &(B_data[ib * bnnz]);
            if (B_marker[jb] < row_start)
            {
               B_marker[jb] = counter;
               C_j[B_marker[jb]] = jb;
               c_entries = &(C_data[B_marker[jb] * bnnz]);
               hypre_CSRBlockMatrixBlockMultAdd(a_entries, b_entries, dzero,
                                                c_entries, block_size);
               counter++;
            }
            else
            {
               c_entries = &(C_data[B_marker[jb] * bnnz]);
               hypre_CSRBlockMatrixBlockMultAdd(a_entries, b_entries, done,
                                                c_entries, block_size);
            }
         }
      }
   }
   hypre_TFree(B_marker, HYPRE_MEMORY_HOST);
   return C;
}

