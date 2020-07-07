/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Matvec functions for hypre_CSRMatrix class.
 *
 *****************************************************************************/

#include "seq_mv.h"
#include "_hypre_utilities.hpp"

#if defined(HYPRE_USING_DEVICE_OPENMP)

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixMatvec
 *--------------------------------------------------------------------------*/

/* y[offset:end] = alpha*A[offset:end,:]*x + beta*b[offset:end] */
HYPRE_Int
hypre_CSRMatrixMatvecOutOfPlaceOOMP( HYPRE_Int        trans,
                                     HYPRE_Complex    alpha,
                                     hypre_CSRMatrix *A,
                                     hypre_Vector    *x,
                                     HYPRE_Complex    beta,
                                     hypre_Vector    *b,
                                     hypre_Vector    *y,
                                     HYPRE_Int        offset )
{
   HYPRE_Int         A_nrows  = hypre_CSRMatrixNumRows(A);
   HYPRE_Int         A_ncols  = hypre_CSRMatrixNumCols(A);
   HYPRE_Int         A_nnz    = hypre_CSRMatrixNumNonzeros(A);
   HYPRE_Complex    *A_data   = hypre_CSRMatrixData(A);
   HYPRE_Int        *A_i      = hypre_CSRMatrixI(A) + offset;
   HYPRE_Int        *A_j      = hypre_CSRMatrixJ(A);
   HYPRE_Int         y_size = hypre_VectorSize(y) - offset;
   HYPRE_Complex    *x_data = hypre_VectorData(x);
   HYPRE_Complex    *b_data = hypre_VectorData(b) + offset;
   HYPRE_Complex    *y_data = hypre_VectorData(y) + offset;
   HYPRE_Int i;

#ifdef HYPRE_USING_CUSPARSE
   cusparseHandle_t handle = hypre_HandleCusparseHandle(hypre_handle());
   cusparseMatDescr_t descr = hypre_HandleCusparseMatDescr(hypre_handle());
#endif

   //hypre_CSRMatrixPrefetch(A, HYPRE_MEMORY_DEVICE);
   //hypre_SeqVectorPrefetch(x, HYPRE_MEMORY_DEVICE);
   //hypre_SeqVectorPrefetch(b, HYPRE_MEMORY_DEVICE);

   //if (b != y)
   //{
   //   hypre_SeqVectorPrefetch(y, HYPRE_MEMORY_DEVICE);
   //}

   if (b != y)
   {
#pragma omp target teams distribute parallel for private(i) is_device_ptr(y_data, b_data)
      for (i = 0; i < y_size; i++)
      {
         y_data[i] = b_data[i];
      }
   }

   if (x == y)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,"ERROR::x and y are the same pointer in hypre_CSRMatrixMatvecDevice\n");
   }

   // TODO
   if (offset != 0)
   {
      hypre_printf("WARNING:: Offset is not zero in hypre_CSRMatrixMatvecDevice :: \n");
   }

   hypre_assert(offset == 0);

   if (trans)
   {
      HYPRE_Complex *csc_a = hypre_TAlloc(HYPRE_Complex, A->num_nonzeros, HYPRE_MEMORY_DEVICE);
      HYPRE_Int     *csc_j = hypre_TAlloc(HYPRE_Int,     A->num_nonzeros, HYPRE_MEMORY_DEVICE);
      HYPRE_Int     *csc_i = hypre_TAlloc(HYPRE_Int,     A->num_cols+1,   HYPRE_MEMORY_DEVICE);

      HYPRE_CUSPARSE_CALL( cusparseDcsr2csc(handle, A->num_rows, A->num_cols, A->num_nonzeros,
                           A->data, A->i, A->j, csc_a, csc_j, csc_i,
                           CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO) );

#ifdef HYPRE_USING_CUSPARSE
      HYPRE_CUSPARSE_CALL( cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                           A->num_cols, A->num_rows, A->num_nonzeros,
                           &alpha, descr,
                           csc_a, csc_i, csc_j,
                           x->data, &beta, y->data) );
#else
#pragma omp target teams distribute parallel for private(i) is_device_ptr(csc_a, csc_i, csc_j, y_data, x_data)
      for (i = 0; i < A_ncols; i++)
      {
         HYPRE_Complex tempx = 0.0;
         HYPRE_Int j;
         for (j = csc_i[i]; j < csc_i[i+1]; j++)
         {
            tempx += csc_a[j] * x_data[csc_j[j]];
         }
         y_data[i] = alpha*tempx + beta*y_data[i];
      }
#endif

      hypre_TFree(csc_a, HYPRE_MEMORY_DEVICE);
      hypre_TFree(csc_i, HYPRE_MEMORY_DEVICE);
      hypre_TFree(csc_j, HYPRE_MEMORY_DEVICE);
   }
   else
   {
#ifdef HYPRE_USING_CUSPARSE
      HYPRE_CUSPARSE_CALL( cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                           A_nrows, A_ncols, A_nnz,
                           &alpha, descr,
                           A_data, A_i, A_j,
                           x_data, &beta, y_data) );
#else
#pragma omp target teams distribute parallel for private(i) is_device_ptr(A_data, A_i, A_j, y_data, x_data)
      for (i = 0; i < A_num_rows; i++)
      {
         HYPRE_Complex tempx = 0.0;
         HYPRE_Int j;
         for (j = A_i[i]; j < A_i[i+1]; j++)
         {
            tempx += A_data[j] * x_data[A_j[j]];
         }
         y_data[i] = alpha*tempx + beta*y_data[i];
      }
#endif
   }

   return hypre_error_flag;
}

#endif /* #if defined(HYPRE_USING_DEVICE_OPENMP) */

