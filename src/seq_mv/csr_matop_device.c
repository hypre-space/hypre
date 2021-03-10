/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Matrix operation functions for hypre_CSRMatrix class.
 *
 *****************************************************************************/

#include "seq_mv.h"
#include "csr_matrix.h"
#include "_hypre_utilities.hpp"

#if defined(HYPRE_USING_CUSPARSE)
hypre_CsrsvData*
hypre_CsrsvDataCreate()
{
   hypre_CsrsvData *data = hypre_CTAlloc(hypre_CsrsvData, 1, HYPRE_MEMORY_HOST);

   return data;
}

void
hypre_CsrsvDataDestroy(hypre_CsrsvData* data)
{
   if (!data)
   {
      return;
   }

   if ( hypre_CsrsvDataInfoL(data) )
   {
      HYPRE_CUSPARSE_CALL( cusparseDestroyCsrsv2Info( hypre_CsrsvDataInfoL(data) ) );
   }

   if ( hypre_CsrsvDataInfoU(data) )
   {
      HYPRE_CUSPARSE_CALL( cusparseDestroyCsrsv2Info( hypre_CsrsvDataInfoU(data) ) );
   }

   if ( hypre_CsrsvDataBuffer(data) )
   {
      hypre_TFree(hypre_CsrsvDataBuffer(data), HYPRE_MEMORY_DEVICE);
   }

   hypre_TFree(data, HYPRE_MEMORY_HOST);
}
#endif /* #if defined(HYPRE_USING_CUSPARSE) */

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)

hypre_CSRMatrix*
hypre_CSRMatrixAddDevice ( hypre_CSRMatrix *A,
                           hypre_CSRMatrix *B )
{
   HYPRE_Complex    *A_data   = hypre_CSRMatrixData(A);
   HYPRE_Int        *A_i      = hypre_CSRMatrixI(A);
   HYPRE_Int        *A_j      = hypre_CSRMatrixJ(A);
   HYPRE_Int         nrows_A  = hypre_CSRMatrixNumRows(A);
   HYPRE_Int         ncols_A  = hypre_CSRMatrixNumCols(A);
   HYPRE_Int         nnz_A    = hypre_CSRMatrixNumNonzeros(A);
   HYPRE_Complex    *B_data   = hypre_CSRMatrixData(B);
   HYPRE_Int        *B_i      = hypre_CSRMatrixI(B);
   HYPRE_Int        *B_j      = hypre_CSRMatrixJ(B);
   HYPRE_Int         nrows_B  = hypre_CSRMatrixNumRows(B);
   HYPRE_Int         ncols_B  = hypre_CSRMatrixNumCols(B);
   HYPRE_Int         nnz_B    = hypre_CSRMatrixNumNonzeros(B);
   HYPRE_Complex    *C_data;
   HYPRE_Int        *C_i;
   HYPRE_Int        *C_j;
   HYPRE_Int         nnzC;
   hypre_CSRMatrix  *C;

   if (nrows_A != nrows_B || ncols_A != ncols_B)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,"Warning! incompatible matrix dimensions!\n");

      return NULL;
   }

   hypreDevice_CSRSpAdd(nrows_A, nrows_B, ncols_A, nnz_A, nnz_B, A_i, A_j, A_data, B_i, B_j, B_data, NULL,
                        &nnzC, &C_i, &C_j, &C_data);

   C = hypre_CSRMatrixCreate(nrows_A, ncols_B, nnzC);
   hypre_CSRMatrixI(C) = C_i;
   hypre_CSRMatrixJ(C) = C_j;
   hypre_CSRMatrixData(C) = C_data;
   hypre_CSRMatrixMemoryLocation(C) = HYPRE_MEMORY_DEVICE;

   hypre_SyncCudaComputeStream(hypre_handle());

   return C;
}

hypre_CSRMatrix*
hypre_CSRMatrixMultiplyDevice( hypre_CSRMatrix *A,
                               hypre_CSRMatrix *B)
{
   HYPRE_Complex    *A_data   = hypre_CSRMatrixData(A);
   HYPRE_Int        *A_i      = hypre_CSRMatrixI(A);
   HYPRE_Int        *A_j      = hypre_CSRMatrixJ(A);
   HYPRE_Int         nrows_A  = hypre_CSRMatrixNumRows(A);
   HYPRE_Int         ncols_A  = hypre_CSRMatrixNumCols(A);
   HYPRE_Int         nnz_A    = hypre_CSRMatrixNumNonzeros(A);
   HYPRE_Complex    *B_data   = hypre_CSRMatrixData(B);
   HYPRE_Int        *B_i      = hypre_CSRMatrixI(B);
   HYPRE_Int        *B_j      = hypre_CSRMatrixJ(B);
   HYPRE_Int         nrows_B  = hypre_CSRMatrixNumRows(B);
   HYPRE_Int         ncols_B  = hypre_CSRMatrixNumCols(B);
   HYPRE_Int         nnz_B    = hypre_CSRMatrixNumNonzeros(B);
   HYPRE_Complex    *C_data;
   HYPRE_Int        *C_i;
   HYPRE_Int        *C_j;
   HYPRE_Int         nnzC;
   hypre_CSRMatrix  *C;

   /* HYPRE_Int         allsquare = 0; */

   if (ncols_A != nrows_B)
   {
      hypre_printf("Warning! incompatible matrix dimensions!\n");
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,"Warning! incompatible matrix dimensions!\n");

      return NULL;
   }

   /*
   if (nrows_A == ncols_B)
   {
      allsquare = 1;
   }
   */

   hypreDevice_CSRSpGemm(nrows_A, ncols_A, ncols_B, nnz_A, nnz_B, A_i, A_j, A_data, B_i, B_j, B_data,
                         &C_i, &C_j, &C_data, &nnzC);

   C = hypre_CSRMatrixCreate(nrows_A, ncols_B, nnzC);
   hypre_CSRMatrixI(C) = C_i;
   hypre_CSRMatrixJ(C) = C_j;
   hypre_CSRMatrixData(C) = C_data;
   hypre_CSRMatrixMemoryLocation(C) = HYPRE_MEMORY_DEVICE;

   hypre_SyncCudaComputeStream(hypre_handle());

   return C;
}

hypre_CSRMatrix*
hypre_CSRMatrixTripleMultiplyDevice ( hypre_CSRMatrix *A,
                                      hypre_CSRMatrix *B,
                                      hypre_CSRMatrix *C )
{
   hypre_CSRMatrix *BC  = hypre_CSRMatrixMultiplyDevice(B, C);
   hypre_CSRMatrix *ABC = hypre_CSRMatrixMultiplyDevice(A, BC);

   hypre_CSRMatrixDestroy(BC);

   return ABC;
}

/* split CSR matrix B_ext (extended rows of parcsr B) into diag part and offd part
 * corresponding to B.
 * Input  col_map_offd_B:
 * Output col_map_offd_C: union of col_map_offd_B and offd-indices of Bext_offd
 *        map_B_to_C: mapping from col_map_offd_B to col_map_offd_C
 */

HYPRE_Int
hypre_CSRMatrixSplitDevice( hypre_CSRMatrix  *B_ext,
                            HYPRE_BigInt      first_col_diag_B,
                            HYPRE_BigInt      last_col_diag_B,
                            HYPRE_Int         num_cols_offd_B,
                            HYPRE_BigInt     *col_map_offd_B,
                            HYPRE_Int       **map_B_to_C_ptr,
                            HYPRE_Int        *num_cols_offd_C_ptr,
                            HYPRE_BigInt    **col_map_offd_C_ptr,
                            hypre_CSRMatrix **B_ext_diag_ptr,
                            hypre_CSRMatrix **B_ext_offd_ptr )
{
   HYPRE_Int num_rows = hypre_CSRMatrixNumRows(B_ext);
   HYPRE_Int B_ext_nnz = hypre_CSRMatrixNumNonzeros(B_ext);

   HYPRE_Int *B_ext_ii = hypre_TAlloc(HYPRE_Int, B_ext_nnz, HYPRE_MEMORY_DEVICE);
   hypreDevice_CsrRowPtrsToIndices_v2(num_rows, B_ext_nnz, hypre_CSRMatrixI(B_ext), B_ext_ii);

   HYPRE_Int B_ext_diag_nnz;
   HYPRE_Int B_ext_offd_nnz;
   HYPRE_Int ierr;

   ierr = hypre_CSRMatrixSplitDevice_core( 0,
                                           num_rows,
                                           B_ext_nnz,
                                           NULL,
                                           hypre_CSRMatrixBigJ(B_ext),
                                           NULL,
                                           NULL,
                                           first_col_diag_B,
                                           last_col_diag_B,
                                           num_cols_offd_B,
                                           NULL,
                                           NULL,
                                           NULL,
                                           NULL,
                                           &B_ext_diag_nnz,
                                           NULL,
                                           NULL,
                                           NULL,
                                           NULL,
                                           &B_ext_offd_nnz,
                                           NULL,
                                           NULL,
                                           NULL,
                                           NULL );

   HYPRE_Int     *B_ext_diag_ii = hypre_TAlloc(HYPRE_Int,     B_ext_diag_nnz, HYPRE_MEMORY_DEVICE);
   HYPRE_Int     *B_ext_diag_j  = hypre_TAlloc(HYPRE_Int,     B_ext_diag_nnz, HYPRE_MEMORY_DEVICE);
   HYPRE_Complex *B_ext_diag_a  = hypre_TAlloc(HYPRE_Complex, B_ext_diag_nnz, HYPRE_MEMORY_DEVICE);

   HYPRE_Int     *B_ext_offd_ii = hypre_TAlloc(HYPRE_Int,     B_ext_offd_nnz, HYPRE_MEMORY_DEVICE);
   HYPRE_Int     *B_ext_offd_j  = hypre_TAlloc(HYPRE_Int,     B_ext_offd_nnz, HYPRE_MEMORY_DEVICE);
   HYPRE_Complex *B_ext_offd_a  = hypre_TAlloc(HYPRE_Complex, B_ext_offd_nnz, HYPRE_MEMORY_DEVICE);

   ierr = hypre_CSRMatrixSplitDevice_core( 1,
                                           num_rows,
                                           B_ext_nnz,
                                           B_ext_ii,
                                           hypre_CSRMatrixBigJ(B_ext),
                                           hypre_CSRMatrixData(B_ext),
                                           NULL,
                                           first_col_diag_B,
                                           last_col_diag_B,
                                           num_cols_offd_B,
                                           col_map_offd_B,
                                           map_B_to_C_ptr,
                                           num_cols_offd_C_ptr,
                                           col_map_offd_C_ptr,
                                           &B_ext_diag_nnz,
                                           B_ext_diag_ii,
                                           B_ext_diag_j,
                                           B_ext_diag_a,
                                           NULL,
                                           &B_ext_offd_nnz,
                                           B_ext_offd_ii,
                                           B_ext_offd_j,
                                           B_ext_offd_a,
                                           NULL );

   hypre_TFree(B_ext_ii, HYPRE_MEMORY_DEVICE);

   /* convert to row ptrs */
   HYPRE_Int *B_ext_diag_i = hypreDevice_CsrRowIndicesToPtrs(num_rows, B_ext_diag_nnz, B_ext_diag_ii);
   HYPRE_Int *B_ext_offd_i = hypreDevice_CsrRowIndicesToPtrs(num_rows, B_ext_offd_nnz, B_ext_offd_ii);

   hypre_TFree(B_ext_diag_ii, HYPRE_MEMORY_DEVICE);
   hypre_TFree(B_ext_offd_ii, HYPRE_MEMORY_DEVICE);

   /* create diag and offd CSR */
   hypre_CSRMatrix *B_ext_diag = hypre_CSRMatrixCreate(num_rows, last_col_diag_B - first_col_diag_B + 1, B_ext_diag_nnz);
   hypre_CSRMatrix *B_ext_offd = hypre_CSRMatrixCreate(num_rows, *num_cols_offd_C_ptr, B_ext_offd_nnz);

   hypre_CSRMatrixI(B_ext_diag) = B_ext_diag_i;
   hypre_CSRMatrixJ(B_ext_diag) = B_ext_diag_j;
   hypre_CSRMatrixData(B_ext_diag) = B_ext_diag_a;
   hypre_CSRMatrixNumNonzeros(B_ext_diag) = B_ext_diag_nnz;
   hypre_CSRMatrixMemoryLocation(B_ext_diag) = HYPRE_MEMORY_DEVICE;

   hypre_CSRMatrixI(B_ext_offd) = B_ext_offd_i;
   hypre_CSRMatrixJ(B_ext_offd) = B_ext_offd_j;
   hypre_CSRMatrixData(B_ext_offd) = B_ext_offd_a;
   hypre_CSRMatrixNumNonzeros(B_ext_offd) = B_ext_offd_nnz;
   hypre_CSRMatrixMemoryLocation(B_ext_offd) = HYPRE_MEMORY_DEVICE;

   *B_ext_diag_ptr = B_ext_diag;
   *B_ext_offd_ptr = B_ext_offd;

   hypre_SyncCudaComputeStream(hypre_handle());

   return ierr;
}

HYPRE_Int
hypre_CSRMatrixSplitDevice_core( HYPRE_Int         job,                 /* 0: query B_ext_diag_nnz and B_ext_offd_nnz; 1: the real computation */
                                 HYPRE_Int         num_rows,
                                 HYPRE_Int         B_ext_nnz,
                                 HYPRE_Int        *B_ext_ii,            /* Note: this is NOT row pointers as in CSR but row indices as in COO */
                                 HYPRE_BigInt     *B_ext_bigj,          /* Note: [BigInt] global column indices */
                                 HYPRE_Complex    *B_ext_data,
                                 char             *B_ext_xata,          /* companion data with B_ext_data; NULL if none */
                                 HYPRE_BigInt      first_col_diag_B,
                                 HYPRE_BigInt      last_col_diag_B,
                                 HYPRE_Int         num_cols_offd_B,
                                 HYPRE_BigInt     *col_map_offd_B,
                                 HYPRE_Int       **map_B_to_C_ptr,
                                 HYPRE_Int        *num_cols_offd_C_ptr,
                                 HYPRE_BigInt    **col_map_offd_C_ptr,
                                 HYPRE_Int        *B_ext_diag_nnz_ptr,
                                 HYPRE_Int        *B_ext_diag_ii,       /* memory allocated outside */
                                 HYPRE_Int        *B_ext_diag_j,
                                 HYPRE_Complex    *B_ext_diag_data,
                                 char             *B_ext_diag_xata,     /* companion with B_ext_diag_data_ptr; NULL if none */
                                 HYPRE_Int        *B_ext_offd_nnz_ptr,
                                 HYPRE_Int        *B_ext_offd_ii,       /* memory allocated outside */
                                 HYPRE_Int        *B_ext_offd_j,
                                 HYPRE_Complex    *B_ext_offd_data,
                                 char             *B_ext_offd_xata      /* companion with B_ext_offd_data_ptr; NULL if none */ )
{
   HYPRE_Int      B_ext_diag_nnz;
   HYPRE_Int      B_ext_offd_nnz;
   HYPRE_BigInt  *B_ext_diag_bigj = NULL;
   HYPRE_BigInt  *B_ext_offd_bigj = NULL;
   HYPRE_BigInt  *col_map_offd_C;
   HYPRE_Int     *map_B_to_C = NULL;
   HYPRE_Int      num_cols_offd_C;

   in_range<HYPRE_BigInt> pred1(first_col_diag_B, last_col_diag_B);

   /* get diag and offd nnz */
   if (job == 0)
   {
      /* query the nnz's */
      B_ext_diag_nnz = HYPRE_THRUST_CALL( count_if,
                                          B_ext_bigj,
                                          B_ext_bigj + B_ext_nnz,
                                          pred1 );
      B_ext_offd_nnz = B_ext_nnz - B_ext_diag_nnz;

      *B_ext_diag_nnz_ptr = B_ext_diag_nnz;
      *B_ext_offd_nnz_ptr = B_ext_offd_nnz;

      return hypre_error_flag;
   }
   else
   {
      B_ext_diag_nnz = *B_ext_diag_nnz_ptr;
      B_ext_offd_nnz = *B_ext_offd_nnz_ptr;
   }

   /* copy to diag */
   B_ext_diag_bigj = hypre_TAlloc(HYPRE_BigInt, B_ext_diag_nnz, HYPRE_MEMORY_DEVICE);

   if (B_ext_diag_xata)
   {
      auto new_end = HYPRE_THRUST_CALL(
         copy_if,
         thrust::make_zip_iterator(thrust::make_tuple(B_ext_ii,      B_ext_bigj,      B_ext_data,   B_ext_xata)),             /* first */
         thrust::make_zip_iterator(thrust::make_tuple(B_ext_ii,      B_ext_bigj,      B_ext_data,   B_ext_xata)) + B_ext_nnz, /* last */
         B_ext_bigj,                                                                                                          /* stencil */
         thrust::make_zip_iterator(thrust::make_tuple(B_ext_diag_ii, B_ext_diag_bigj, B_ext_diag_data, B_ext_diag_xata)),     /* result */
         pred1 );

      hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) == B_ext_diag_ii + B_ext_diag_nnz );
   }
   else
   {
      auto new_end = HYPRE_THRUST_CALL(
         copy_if,
         thrust::make_zip_iterator(thrust::make_tuple(B_ext_ii,      B_ext_bigj,      B_ext_data)),             /* first */
         thrust::make_zip_iterator(thrust::make_tuple(B_ext_ii,      B_ext_bigj,      B_ext_data)) + B_ext_nnz, /* last */
         B_ext_bigj,                                                                                            /* stencil */
         thrust::make_zip_iterator(thrust::make_tuple(B_ext_diag_ii, B_ext_diag_bigj, B_ext_diag_data)),        /* result */
         pred1 );

      hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) == B_ext_diag_ii + B_ext_diag_nnz );
   }

   HYPRE_THRUST_CALL( transform,
                      B_ext_diag_bigj,
                      B_ext_diag_bigj + B_ext_diag_nnz,
                      thrust::make_constant_iterator(first_col_diag_B),
                      B_ext_diag_j,
                      thrust::minus<HYPRE_BigInt>());

   hypre_TFree(B_ext_diag_bigj, HYPRE_MEMORY_DEVICE);

   /* copy to offd */
   B_ext_offd_bigj = hypre_TAlloc(HYPRE_BigInt, B_ext_offd_nnz, HYPRE_MEMORY_DEVICE);

   if (B_ext_offd_xata)
   {
      auto new_end = HYPRE_THRUST_CALL(
         copy_if,
         thrust::make_zip_iterator(thrust::make_tuple(B_ext_ii,      B_ext_bigj,      B_ext_data,   B_ext_xata)),             /* first */
         thrust::make_zip_iterator(thrust::make_tuple(B_ext_ii,      B_ext_bigj,      B_ext_data,   B_ext_xata)) + B_ext_nnz, /* last */
         B_ext_bigj,                                                                                                          /* stencil */
         thrust::make_zip_iterator(thrust::make_tuple(B_ext_offd_ii, B_ext_offd_bigj, B_ext_offd_data, B_ext_offd_xata)),     /* result */
         thrust::not1(pred1) );

      hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) == B_ext_offd_ii + B_ext_offd_nnz );
   }
   else
   {
      auto new_end = HYPRE_THRUST_CALL(
         copy_if,
         thrust::make_zip_iterator(thrust::make_tuple(B_ext_ii,      B_ext_bigj,      B_ext_data)),             /* first */
         thrust::make_zip_iterator(thrust::make_tuple(B_ext_ii,      B_ext_bigj,      B_ext_data)) + B_ext_nnz, /* last */
         B_ext_bigj,                                                                                            /* stencil */
         thrust::make_zip_iterator(thrust::make_tuple(B_ext_offd_ii, B_ext_offd_bigj, B_ext_offd_data)),        /* result */
         thrust::not1(pred1) );

      hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) == B_ext_offd_ii + B_ext_offd_nnz );
   }

   /* offd map of B_ext_offd Union col_map_offd_B */
   col_map_offd_C = hypre_TAlloc(HYPRE_BigInt, B_ext_offd_nnz + num_cols_offd_B, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(col_map_offd_C,                  B_ext_offd_bigj, HYPRE_BigInt, B_ext_offd_nnz,  HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(col_map_offd_C + B_ext_offd_nnz, col_map_offd_B,  HYPRE_BigInt, num_cols_offd_B, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

   HYPRE_THRUST_CALL( sort,
                      col_map_offd_C,
                      col_map_offd_C + B_ext_offd_nnz + num_cols_offd_B );

   HYPRE_Int *new_end = HYPRE_THRUST_CALL( unique,
                                           col_map_offd_C,
                                           col_map_offd_C + B_ext_offd_nnz + num_cols_offd_B );

   num_cols_offd_C = new_end - col_map_offd_C;

#if 1
   HYPRE_BigInt *tmp = hypre_TAlloc(HYPRE_BigInt, num_cols_offd_C, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(tmp, col_map_offd_C, HYPRE_BigInt, num_cols_offd_C, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   hypre_TFree(col_map_offd_C, HYPRE_MEMORY_DEVICE);
   col_map_offd_C = tmp;
#else
   col_map_offd_C = hypre_TReAlloc_v2(col_map_offd_C, HYPRE_BigInt, B_ext_offd_nnz + num_cols_offd_B, HYPRE_Int, num_cols_offd_C, HYPRE_MEMORY_DEVICE);
#endif

   /* create map from col_map_offd_B */
   if (num_cols_offd_B)
   {
      map_B_to_C = hypre_TAlloc(HYPRE_Int, num_cols_offd_B, HYPRE_MEMORY_DEVICE);
      HYPRE_THRUST_CALL( lower_bound,
                         col_map_offd_C,
                         col_map_offd_C + num_cols_offd_C,
                         col_map_offd_B,
                         col_map_offd_B + num_cols_offd_B,
                         map_B_to_C );
   }

   HYPRE_THRUST_CALL( lower_bound,
                      col_map_offd_C,
                      col_map_offd_C + num_cols_offd_C,
                      B_ext_offd_bigj,
                      B_ext_offd_bigj + B_ext_offd_nnz,
                      B_ext_offd_j );

   hypre_TFree(B_ext_offd_bigj, HYPRE_MEMORY_DEVICE);

   if (map_B_to_C_ptr)
   {
      *map_B_to_C_ptr   = map_B_to_C;
   }
   *num_cols_offd_C_ptr = num_cols_offd_C;
   *col_map_offd_C_ptr  = col_map_offd_C;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixAddPartial:
 * adds matrix rows in the CSR matrix B to the CSR Matrix A, where row_nums[i]
 * defines to which row of A the i-th row of B is added, and returns a CSR Matrix C;
 * Repeated row indices are allowed in row_nums
 * Note: The routine does not check for 0-elements which might be generated
 *       through cancellation of elements in A and B or already contained
 *       in A and B. To remove those, use hypre_CSRMatrixDeleteZeros
 *--------------------------------------------------------------------------*/

hypre_CSRMatrix*
hypre_CSRMatrixAddPartialDevice( hypre_CSRMatrix *A,
                                 hypre_CSRMatrix *B,
                                 HYPRE_Int       *row_nums)
{
   HYPRE_Complex    *A_data   = hypre_CSRMatrixData(A);
   HYPRE_Int        *A_i      = hypre_CSRMatrixI(A);
   HYPRE_Int        *A_j      = hypre_CSRMatrixJ(A);
   HYPRE_Int         nrows_A  = hypre_CSRMatrixNumRows(A);
   HYPRE_Int         ncols_A  = hypre_CSRMatrixNumCols(A);
   HYPRE_Int         nnz_A    = hypre_CSRMatrixNumNonzeros(A);
   HYPRE_Complex    *B_data   = hypre_CSRMatrixData(B);
   HYPRE_Int        *B_i      = hypre_CSRMatrixI(B);
   HYPRE_Int        *B_j      = hypre_CSRMatrixJ(B);
   HYPRE_Int         nrows_B  = hypre_CSRMatrixNumRows(B);
   HYPRE_Int         ncols_B  = hypre_CSRMatrixNumCols(B);
   HYPRE_Int         nnz_B    = hypre_CSRMatrixNumNonzeros(B);
   HYPRE_Complex    *C_data;
   HYPRE_Int        *C_i;
   HYPRE_Int        *C_j;
   HYPRE_Int         nnzC;
   hypre_CSRMatrix  *C;

   if (ncols_A != ncols_B)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,"Warning! incompatible matrix dimensions!\n");

      return NULL;
   }

   hypreDevice_CSRSpAdd(nrows_A, nrows_B, ncols_A, nnz_A, nnz_B, A_i, A_j, A_data, B_i, B_j, B_data, row_nums,
                        &nnzC, &C_i, &C_j, &C_data);

   C = hypre_CSRMatrixCreate(nrows_A, ncols_B, nnzC);
   hypre_CSRMatrixI(C) = C_i;
   hypre_CSRMatrixJ(C) = C_j;
   hypre_CSRMatrixData(C) = C_data;
   hypre_CSRMatrixMemoryLocation(C) = HYPRE_MEMORY_DEVICE;

   hypre_SyncCudaComputeStream(hypre_handle());

   return C;
}

HYPRE_Int
hypre_CSRMatrixColNNzRealDevice( hypre_CSRMatrix  *A,
                                 HYPRE_Real       *colnnz)
{
   HYPRE_Int *A_j      = hypre_CSRMatrixJ(A);
   HYPRE_Int  ncols_A  = hypre_CSRMatrixNumCols(A);
   HYPRE_Int  nnz_A    = hypre_CSRMatrixNumNonzeros(A);
   HYPRE_Int *A_j_sorted;
   HYPRE_Int  num_reduced_col_indices;
   HYPRE_Int *reduced_col_indices;
   HYPRE_Int *reduced_col_nnz;

   A_j_sorted = hypre_TAlloc(HYPRE_Int, nnz_A, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(A_j_sorted, A_j, HYPRE_Int, nnz_A, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   HYPRE_THRUST_CALL(sort, A_j_sorted, A_j_sorted + nnz_A);

   reduced_col_indices = hypre_TAlloc(HYPRE_Int, ncols_A, HYPRE_MEMORY_DEVICE);
   reduced_col_nnz     = hypre_TAlloc(HYPRE_Int, ncols_A, HYPRE_MEMORY_DEVICE);

   thrust::pair<HYPRE_Int*, HYPRE_Int*> new_end =
   HYPRE_THRUST_CALL(reduce_by_key, A_j_sorted, A_j_sorted + nnz_A,
                     thrust::make_constant_iterator(1),
                     reduced_col_indices,
                     reduced_col_nnz);

   hypre_assert(new_end.first - reduced_col_indices == new_end.second - reduced_col_nnz);

   num_reduced_col_indices = new_end.first - reduced_col_indices;

   hypre_Memset(colnnz, 0, ncols_A * sizeof(HYPRE_Real), HYPRE_MEMORY_DEVICE);
   HYPRE_THRUST_CALL(scatter, reduced_col_nnz, reduced_col_nnz + num_reduced_col_indices,
                     reduced_col_indices, colnnz);

   hypre_TFree(A_j_sorted,          HYPRE_MEMORY_DEVICE);
   hypre_TFree(reduced_col_indices, HYPRE_MEMORY_DEVICE);
   hypre_TFree(reduced_col_nnz,     HYPRE_MEMORY_DEVICE);

   hypre_SyncCudaComputeStream(hypre_handle());

   return hypre_error_flag;
}

__global__ void
hypreCUDAKernel_CSRMoveDiagFirst( HYPRE_Int      nrows,
                                  HYPRE_Int     *ia,
                                  HYPRE_Int     *ja,
                                  HYPRE_Complex *aa )
{
   HYPRE_Int row = hypre_cuda_get_grid_warp_id<1,1>();

   if (row >= nrows)
   {
      return;
   }

   HYPRE_Int lane = hypre_cuda_get_lane_id<1>();
   HYPRE_Int p, q;

   if (lane < 2)
   {
      p = read_only_load(ia + row + lane);
   }
   q = __shfl_sync(HYPRE_WARP_FULL_MASK, p, 1);
   p = __shfl_sync(HYPRE_WARP_FULL_MASK, p, 0);

   for (HYPRE_Int j = p + lane + 1; __any_sync(HYPRE_WARP_FULL_MASK, j < q); j += HYPRE_WARP_SIZE)
   {
      hypre_int find_diag = j < q && ja[j] == row;

      if (find_diag)
      {
         ja[j] = ja[p];
         ja[p] = row;
         HYPRE_Complex tmp = aa[p];
         aa[p] = aa[j];
         aa[j] = tmp;
      }

      if ( __any_sync(HYPRE_WARP_FULL_MASK, find_diag) )
      {
         break;
      }
   }
}

HYPRE_Int
hypre_CSRMatrixMoveDiagFirstDevice( hypre_CSRMatrix  *A )
{
   HYPRE_Int      nrows  = hypre_CSRMatrixNumRows(A);
   HYPRE_Complex *A_data = hypre_CSRMatrixData(A);
   HYPRE_Int     *A_i    = hypre_CSRMatrixI(A);
   HYPRE_Int     *A_j    = hypre_CSRMatrixJ(A);
   dim3           bDim, gDim;

   bDim = hypre_GetDefaultCUDABlockDimension();
   gDim = hypre_GetDefaultCUDAGridDimension(nrows, "warp", bDim);

   HYPRE_CUDA_LAUNCH(hypreCUDAKernel_CSRMoveDiagFirst, gDim, bDim,
                     nrows, A_i, A_j, A_data);

   hypre_SyncCudaComputeStream(hypre_handle());

   return hypre_error_flag;
}

/* check if diagonal entry is the first one at each row
 * Return: the number of rows that do not have the first entry as diagonal
 */
__global__ void
hypreCUDAKernel_CSRCheckDiagFirst( HYPRE_Int  nrows,
                                   HYPRE_Int *ia,
                                   HYPRE_Int *ja,
                                   HYPRE_Int *result )
{
   const HYPRE_Int row = hypre_cuda_get_grid_thread_id<1,1>();
   if (row < nrows)
   {
      result[row] = ja[ia[row]] != row;
   }
}

HYPRE_Int
hypre_CSRMatrixCheckDiagFirstDevice( hypre_CSRMatrix *A )
{
   if (hypre_CSRMatrixNumRows(A) != hypre_CSRMatrixNumCols(A))
   {
      return -1;
   }

   dim3 bDim = hypre_GetDefaultCUDABlockDimension();
   dim3 gDim = hypre_GetDefaultCUDAGridDimension(hypre_CSRMatrixNumRows(A), "thread", bDim);

   HYPRE_Int *result = hypre_TAlloc(HYPRE_Int, hypre_CSRMatrixNumRows(A), HYPRE_MEMORY_DEVICE);
   HYPRE_CUDA_LAUNCH( hypreCUDAKernel_CSRCheckDiagFirst, gDim, bDim,
                      hypre_CSRMatrixNumRows(A),
                      hypre_CSRMatrixI(A), hypre_CSRMatrixJ(A), result );

   HYPRE_Int ierr = HYPRE_THRUST_CALL( reduce,
                                       result,
                                       result + hypre_CSRMatrixNumRows(A) );

   hypre_TFree(result, HYPRE_MEMORY_DEVICE);

   hypre_SyncCudaComputeStream(hypre_handle());

   return ierr;
}

/* check if diagonal entry is the first one at each row, and
 * assign numerical zero diag value `v'
 * Return: the number of rows that do not have the first entry as diagonal
 */
__global__ void
hypreCUDAKernel_CSRCheckDiagFirstSetValueZero( HYPRE_Complex  v,
                                               HYPRE_Int      nrows,
                                               HYPRE_Int     *ia,
                                               HYPRE_Int     *ja,
                                               HYPRE_Complex *data,
                                               HYPRE_Int     *result )
{
   const HYPRE_Int row = hypre_cuda_get_grid_thread_id<1,1>();
   if (row < nrows)
   {
      const HYPRE_Int j = ia[row];
      const HYPRE_Int col = ja[j];

      result[row] = col != row;

      if (col == row && data[j] == 0.0)
      {
         data[j] = v;
      }
   }
}

HYPRE_Int
hypre_CSRMatrixCheckDiagFirstSetValueZeroDevice( hypre_CSRMatrix *A,
                                                 HYPRE_Complex    v )
{
   if (hypre_CSRMatrixNumRows(A) != hypre_CSRMatrixNumCols(A))
   {
      return -1;
   }

   dim3 bDim = hypre_GetDefaultCUDABlockDimension();
   dim3 gDim = hypre_GetDefaultCUDAGridDimension(hypre_CSRMatrixNumRows(A), "thread", bDim);

   HYPRE_Int *result = hypre_TAlloc(HYPRE_Int, hypre_CSRMatrixNumRows(A), HYPRE_MEMORY_DEVICE);
   HYPRE_CUDA_LAUNCH( hypreCUDAKernel_CSRCheckDiagFirstSetValueZero, gDim, bDim,
                      v, hypre_CSRMatrixNumRows(A),
                      hypre_CSRMatrixI(A), hypre_CSRMatrixJ(A), hypre_CSRMatrixData(A),
                      result );

   HYPRE_Int ierr = HYPRE_THRUST_CALL( reduce,
                                       result,
                                       result + hypre_CSRMatrixNumRows(A) );

   hypre_TFree(result, HYPRE_MEMORY_DEVICE);

   hypre_SyncCudaComputeStream(hypre_handle());

   return ierr;
}

typedef thrust::tuple<HYPRE_Int, HYPRE_Int> Int2;
struct Int2Unequal : public thrust::unary_function<Int2, bool>
{
   __host__ __device__
   bool operator()(const Int2& t) const
   {
      return (thrust::get<0>(t) != thrust::get<1>(t));
   }
};

HYPRE_Int
hypre_CSRMatrixRemoveDiagonalDevice(hypre_CSRMatrix *A)
{
   HYPRE_Int      nrows  = hypre_CSRMatrixNumRows(A);
   HYPRE_Int      nnz    = hypre_CSRMatrixNumNonzeros(A);
   HYPRE_Int     *A_i    = hypre_CSRMatrixI(A);
   HYPRE_Int     *A_j    = hypre_CSRMatrixJ(A);
   HYPRE_Complex *A_data = hypre_CSRMatrixData(A);
   HYPRE_Int     *A_ii   = hypreDevice_CsrRowPtrsToIndices(nrows, nnz, A_i);
   HYPRE_Int      new_nnz;
   HYPRE_Int     *new_ii;
   HYPRE_Int     *new_j;
   HYPRE_Complex *new_data;

   new_nnz = HYPRE_THRUST_CALL( count_if,
                                thrust::make_zip_iterator(thrust::make_tuple(A_ii, A_j)),
                                thrust::make_zip_iterator(thrust::make_tuple(A_ii, A_j)) + nnz,
                                Int2Unequal() );

   if (new_nnz == nnz)
   {
      /* no diagonal entries found */
      hypre_TFree(A_ii, HYPRE_MEMORY_DEVICE);
      return hypre_error_flag;
   }

   new_ii = hypre_TAlloc(HYPRE_Int, new_nnz, HYPRE_MEMORY_DEVICE);
   new_j = hypre_TAlloc(HYPRE_Int, new_nnz, HYPRE_MEMORY_DEVICE);

   if (A_data)
   {
      new_data = hypre_TAlloc(HYPRE_Complex, new_nnz, HYPRE_MEMORY_DEVICE);

      thrust::zip_iterator< thrust::tuple<HYPRE_Int*, HYPRE_Int*, HYPRE_Complex*> > new_end;

      new_end = HYPRE_THRUST_CALL( copy_if,
                                   thrust::make_zip_iterator(thrust::make_tuple(A_ii, A_j, A_data)),
                                   thrust::make_zip_iterator(thrust::make_tuple(A_ii, A_j, A_data)) + nnz,
                                   thrust::make_zip_iterator(thrust::make_tuple(A_ii, A_j)),
                                   thrust::make_zip_iterator(thrust::make_tuple(new_ii, new_j, new_data)),
                                   Int2Unequal() );

      hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) == new_ii + new_nnz );
   }
   else
   {
      new_data = NULL;

      thrust::zip_iterator< thrust::tuple<HYPRE_Int*, HYPRE_Int*> > new_end;

      new_end = HYPRE_THRUST_CALL( copy_if,
                                   thrust::make_zip_iterator(thrust::make_tuple(A_ii, A_j)),
                                   thrust::make_zip_iterator(thrust::make_tuple(A_ii, A_j)) + nnz,
                                   thrust::make_zip_iterator(thrust::make_tuple(A_ii, A_j)),
                                   thrust::make_zip_iterator(thrust::make_tuple(new_ii, new_j)),
                                   Int2Unequal() );

      hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) == new_ii + new_nnz );
   }

   hypre_TFree(A_ii,   HYPRE_MEMORY_DEVICE);
   hypre_TFree(A_i,    HYPRE_MEMORY_DEVICE);
   hypre_TFree(A_j,    HYPRE_MEMORY_DEVICE);
   hypre_TFree(A_data, HYPRE_MEMORY_DEVICE);

   hypre_CSRMatrixNumNonzeros(A) = new_nnz;
   hypre_CSRMatrixI(A) = hypreDevice_CsrRowIndicesToPtrs(nrows, new_nnz, new_ii);
   hypre_CSRMatrixJ(A) = new_j;
   hypre_CSRMatrixData(A) = new_data;
   hypre_TFree(new_ii, HYPRE_MEMORY_DEVICE);

   return hypre_error_flag;
}

/* type == 0, sum,
 *         1, abs sum (l-1)
 *         2, square sum (l-2)
 */
template<HYPRE_Int type>
__global__ void
hypreCUDAKernel_CSRRowSum( HYPRE_Int      nrows,
                           HYPRE_Int     *ia,
                           HYPRE_Int     *ja,
                           HYPRE_Complex *aa,
                           HYPRE_Int     *CF_i,
                           HYPRE_Int     *CF_j,
                           HYPRE_Complex *row_sum,
                           HYPRE_Complex  scal,
                           HYPRE_Int      set)
{
   HYPRE_Int row_i = hypre_cuda_get_grid_warp_id<1,1>();

   if (row_i >= nrows)
   {
      return;
   }

   HYPRE_Int lane = hypre_cuda_get_lane_id<1>();
   HYPRE_Int p, q;

   if (lane < 2)
   {
      p = read_only_load(ia + row_i + lane);
   }
   q = __shfl_sync(HYPRE_WARP_FULL_MASK, p, 1);
   p = __shfl_sync(HYPRE_WARP_FULL_MASK, p, 0);

   HYPRE_Complex row_sum_i = 0.0;

   for (HYPRE_Int j = p + lane; __any_sync(HYPRE_WARP_FULL_MASK, j < q); j += HYPRE_WARP_SIZE)
   {
      if ( j >= q || (CF_i && CF_j && read_only_load(&CF_i[row_i]) != read_only_load(&CF_j[ja[j]])) )
      {
         continue;
      }

      HYPRE_Complex aii = aa[j];

      if (type == 0)
      {
         row_sum_i += aii;
      }
      else if (type == 1)
      {
         row_sum_i += fabs(aii);
      }
      else if (type == 2)
      {
         row_sum_i += aii * aii;
      }
   }

   row_sum_i = warp_reduce_sum(row_sum_i);

   if (lane == 0)
   {
      if (set)
      {
         row_sum[row_i] = scal * row_sum_i;
      }
      else
      {
         row_sum[row_i] += scal * row_sum_i;
      }
   }
}

void
hypre_CSRMatrixComputeRowSumDevice( hypre_CSRMatrix *A,
                                    HYPRE_Int       *CF_i,
                                    HYPRE_Int       *CF_j,
                                    HYPRE_Complex   *row_sum,
                                    HYPRE_Int        type,
                                    HYPRE_Complex    scal,
                                    const char      *set_or_add)
{
   HYPRE_Int      nrows  = hypre_CSRMatrixNumRows(A);
   HYPRE_Complex *A_data = hypre_CSRMatrixData(A);
   HYPRE_Int     *A_i    = hypre_CSRMatrixI(A);
   HYPRE_Int     *A_j    = hypre_CSRMatrixJ(A);
   dim3           bDim, gDim;

   bDim = hypre_GetDefaultCUDABlockDimension();
   gDim = hypre_GetDefaultCUDAGridDimension(nrows, "warp", bDim);

   if (type == 0)
   {
      HYPRE_CUDA_LAUNCH( hypreCUDAKernel_CSRRowSum<0>, gDim, bDim, nrows, A_i, A_j, A_data, CF_i, CF_j,
                         row_sum, scal, set_or_add[0] == 's' );
   }
   else if (type == 1)
   {
      HYPRE_CUDA_LAUNCH( hypreCUDAKernel_CSRRowSum<1>, gDim, bDim, nrows, A_i, A_j, A_data, CF_i, CF_j,
                         row_sum, scal, set_or_add[0] == 's' );
   }
   else if (type == 2)
   {
      HYPRE_CUDA_LAUNCH( hypreCUDAKernel_CSRRowSum<2>, gDim, bDim, nrows, A_i, A_j, A_data, CF_i, CF_j,
                         row_sum, scal, set_or_add[0] == 's' );
   }

   hypre_SyncCudaComputeStream(hypre_handle());
}

/* type 0: diag
 *      1: abs diag
 *      2: diag inverse
 */
__global__ void
hypreCUDAKernel_CSRExtractDiag( HYPRE_Int      nrows,
                                HYPRE_Int     *ia,
                                HYPRE_Int     *ja,
                                HYPRE_Complex *aa,
                                HYPRE_Complex *d,
                                HYPRE_Int      type)
{
   HYPRE_Int row = hypre_cuda_get_grid_warp_id<1,1>();

   if (row >= nrows)
   {
      return;
   }

   HYPRE_Int lane = hypre_cuda_get_lane_id<1>();
   HYPRE_Int p, q;

   if (lane < 2)
   {
      p = read_only_load(ia + row + lane);
   }
   q = __shfl_sync(HYPRE_WARP_FULL_MASK, p, 1);
   p = __shfl_sync(HYPRE_WARP_FULL_MASK, p, 0);

   HYPRE_Int has_diag = 0;

   for (HYPRE_Int j = p + lane; __any_sync(HYPRE_WARP_FULL_MASK, j < q); j += HYPRE_WARP_SIZE)
   {
      hypre_int find_diag = j < q && ja[j] == row;

      if (find_diag)
      {
         if (type == 0)
         {
            d[row] = aa[j];
         }
         else if (type == 1)
         {
            d[row] = fabs(aa[j]);
         }
         else if (type == 2)
         {
            d[row] = 1.0 / aa[j];
         }
      }

      if ( __any_sync(HYPRE_WARP_FULL_MASK, find_diag) )
      {
         has_diag = 1;
         break;
      }
   }

   if (!has_diag && lane == 0)
   {
      d[row] = 0.0;
   }
}

void
hypre_CSRMatrixExtractDiagonalDevice( hypre_CSRMatrix *A,
                                      HYPRE_Complex   *d,
                                      HYPRE_Int        type)
{
   HYPRE_Int      nrows  = hypre_CSRMatrixNumRows(A);
   HYPRE_Complex *A_data = hypre_CSRMatrixData(A);
   HYPRE_Int     *A_i    = hypre_CSRMatrixI(A);
   HYPRE_Int     *A_j    = hypre_CSRMatrixJ(A);
   dim3           bDim, gDim;

   bDim = hypre_GetDefaultCUDABlockDimension();
   gDim = hypre_GetDefaultCUDAGridDimension(nrows, "warp", bDim);

   HYPRE_CUDA_LAUNCH( hypreCUDAKernel_CSRExtractDiag, gDim, bDim, nrows, A_i, A_j, A_data, d, type );

   hypre_SyncCudaComputeStream(hypre_handle());
}

/* return C = [A; B] */
hypre_CSRMatrix*
hypre_CSRMatrixStack2Device(hypre_CSRMatrix *A, hypre_CSRMatrix *B)
{
   hypre_assert( hypre_CSRMatrixNumCols(A) == hypre_CSRMatrixNumCols(B) );

   hypre_CSRMatrix *C = hypre_CSRMatrixCreate( hypre_CSRMatrixNumRows(A) + hypre_CSRMatrixNumRows(B),
                                               hypre_CSRMatrixNumCols(A),
                                               hypre_CSRMatrixNumNonzeros(A) + hypre_CSRMatrixNumNonzeros(B) );

   HYPRE_Int     *C_i = hypre_TAlloc(HYPRE_Int,     hypre_CSRMatrixNumRows(C) + 1, HYPRE_MEMORY_DEVICE);
   HYPRE_Int     *C_j = hypre_TAlloc(HYPRE_Int,     hypre_CSRMatrixNumNonzeros(C), HYPRE_MEMORY_DEVICE);
   HYPRE_Complex *C_a = hypre_TAlloc(HYPRE_Complex, hypre_CSRMatrixNumNonzeros(C), HYPRE_MEMORY_DEVICE);

   hypre_TMemcpy(C_i, hypre_CSRMatrixI(A), HYPRE_Int, hypre_CSRMatrixNumRows(A) + 1,
                 HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(C_i + hypre_CSRMatrixNumRows(A) + 1, hypre_CSRMatrixI(B) + 1, HYPRE_Int, hypre_CSRMatrixNumRows(B),
                 HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   HYPRE_THRUST_CALL( transform,
                      C_i + hypre_CSRMatrixNumRows(A) + 1,
                      C_i + hypre_CSRMatrixNumRows(C) + 1,
                      thrust::make_constant_iterator(hypre_CSRMatrixNumNonzeros(A)),
                      C_i + hypre_CSRMatrixNumRows(A) + 1,
                      thrust::plus<HYPRE_Int>() );

   hypre_TMemcpy(C_j, hypre_CSRMatrixJ(A), HYPRE_Int, hypre_CSRMatrixNumNonzeros(A),
                 HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(C_j + hypre_CSRMatrixNumNonzeros(A), hypre_CSRMatrixJ(B), HYPRE_Int, hypre_CSRMatrixNumNonzeros(B),
                 HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

   hypre_TMemcpy(C_a, hypre_CSRMatrixData(A), HYPRE_Complex, hypre_CSRMatrixNumNonzeros(A),
                 HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(C_a + hypre_CSRMatrixNumNonzeros(A), hypre_CSRMatrixData(B), HYPRE_Complex, hypre_CSRMatrixNumNonzeros(B),
                 HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

   hypre_CSRMatrixI(C) = C_i;
   hypre_CSRMatrixJ(C) = C_j;
   hypre_CSRMatrixData(C) = C_a;
   hypre_CSRMatrixMemoryLocation(C) = HYPRE_MEMORY_DEVICE;

   return C;
}

/* A = alp * I */
hypre_CSRMatrix *
hypre_CSRMatrixIdentityDevice(HYPRE_Int n, HYPRE_Complex alp)
{
   hypre_CSRMatrix *A = hypre_CSRMatrixCreate(n, n, n);

   hypre_CSRMatrixInitialize_v2(A, 0, HYPRE_MEMORY_DEVICE);

   HYPRE_THRUST_CALL( sequence,
                      hypre_CSRMatrixI(A),
                      hypre_CSRMatrixI(A) + n + 1,
                      0  );

   HYPRE_THRUST_CALL( sequence,
                      hypre_CSRMatrixJ(A),
                      hypre_CSRMatrixJ(A) + n,
                      0  );

   HYPRE_THRUST_CALL( fill,
                      hypre_CSRMatrixData(A),
                      hypre_CSRMatrixData(A) + n,
                      alp );

   return A;
}

/* abs    == 1, use absolute values
 * option == 0, drop all the entries that are smaller than tol
 * TODO more options
 */
HYPRE_Int
hypre_CSRMatrixDropSmallEntriesDevice( hypre_CSRMatrix *A,
                                       HYPRE_Complex    tol,
                                       HYPRE_Int        abs,
                                       HYPRE_Int        option)
{
   HYPRE_Int      nrows  = hypre_CSRMatrixNumRows(A);
   HYPRE_Int      nnz    = hypre_CSRMatrixNumNonzeros(A);
   HYPRE_Int     *A_i    = hypre_CSRMatrixI(A);
   HYPRE_Int     *A_j    = hypre_CSRMatrixJ(A);
   HYPRE_Complex *A_data = hypre_CSRMatrixData(A);
   HYPRE_Int     *A_ii   = NULL;
   HYPRE_Int      new_nnz = 0;
   HYPRE_Int     *new_ii;
   HYPRE_Int     *new_j;
   HYPRE_Complex *new_data;

   if (abs == 0)
   {
      if (option == 0)
      {
         new_nnz = HYPRE_THRUST_CALL( count_if,
                                      A_data,
                                      A_data + nnz,
                                      thrust::not1(less_than<HYPRE_Complex>(tol)) );

      }
   }

   if (new_nnz == nnz)
   {
      hypre_TFree(A_ii, HYPRE_MEMORY_DEVICE);
      return hypre_error_flag;
   }

   if (!A_ii)
   {
      A_ii = hypreDevice_CsrRowPtrsToIndices(nrows, nnz, A_i);
   }
   new_ii = hypre_TAlloc(HYPRE_Int, new_nnz, HYPRE_MEMORY_DEVICE);
   new_j = hypre_TAlloc(HYPRE_Int, new_nnz, HYPRE_MEMORY_DEVICE);
   new_data = hypre_TAlloc(HYPRE_Complex, new_nnz, HYPRE_MEMORY_DEVICE);

   thrust::zip_iterator< thrust::tuple<HYPRE_Int*, HYPRE_Int*, HYPRE_Complex*> > new_end;

   if (abs == 0)
   {
      if (option == 0)
      {
         new_end = HYPRE_THRUST_CALL( copy_if,
                                      thrust::make_zip_iterator(thrust::make_tuple(A_ii, A_j, A_data)),
                                      thrust::make_zip_iterator(thrust::make_tuple(A_ii, A_j, A_data)) + nnz,
                                      A_data,
                                      thrust::make_zip_iterator(thrust::make_tuple(new_ii, new_j, new_data)),
                                      thrust::not1(less_than<HYPRE_Complex>(tol)) );
      }
   }

   hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) == new_ii + new_nnz );

   hypre_TFree(A_ii,   HYPRE_MEMORY_DEVICE);
   hypre_TFree(A_i,    HYPRE_MEMORY_DEVICE);
   hypre_TFree(A_j,    HYPRE_MEMORY_DEVICE);
   hypre_TFree(A_data, HYPRE_MEMORY_DEVICE);

   hypre_CSRMatrixNumNonzeros(A) = new_nnz;
   hypre_CSRMatrixI(A) = hypreDevice_CsrRowIndicesToPtrs(nrows, new_nnz, new_ii);
   hypre_CSRMatrixJ(A) = new_j;
   hypre_CSRMatrixData(A) = new_data;
   hypre_TFree(new_ii, HYPRE_MEMORY_DEVICE);

   return hypre_error_flag;
}

#endif /* HYPRE_USING_CUDA || defined(HYPRE_USING_HIP) */

#if defined(HYPRE_USING_GPU)

HYPRE_Int
hypre_CSRMatrixTransposeDevice(hypre_CSRMatrix  *A,
                               hypre_CSRMatrix **AT_ptr,
                               HYPRE_Int         data)
{
   HYPRE_Complex    *A_data   = hypre_CSRMatrixData(A);
   HYPRE_Int        *A_i      = hypre_CSRMatrixI(A);
   HYPRE_Int        *A_j      = hypre_CSRMatrixJ(A);
   HYPRE_Int         nrows_A  = hypre_CSRMatrixNumRows(A);
   HYPRE_Int         ncols_A  = hypre_CSRMatrixNumCols(A);
   HYPRE_Int         nnz_A    = hypre_CSRMatrixNumNonzeros(A);
   HYPRE_Complex    *C_data;
   HYPRE_Int        *C_i;
   HYPRE_Int        *C_j;
   hypre_CSRMatrix  *C;


   /* trivial case */
   if (nnz_A == 0)
   {
      C_i =    hypre_CTAlloc(HYPRE_Int,     ncols_A + 1, HYPRE_MEMORY_DEVICE);
      C_j =    hypre_CTAlloc(HYPRE_Int,     0,           HYPRE_MEMORY_DEVICE);
      C_data = hypre_CTAlloc(HYPRE_Complex, 0,           HYPRE_MEMORY_DEVICE);
   }
   else
   {
#if defined(HYPRE_USING_CUSPARSE)
     hypreDevice_CSRSpTransCusparse(nrows_A, ncols_A, nnz_A, A_i, A_j, A_data, &C_i, &C_j, &C_data, data);
#elif defined(HYPRE_USING_ROCSPARSE)
     hypreDevice_CSRSpTransRocsparse(nrows_A, ncols_A, nnz_A, A_i, A_j, A_data, &C_i, &C_j, &C_data, data);
#else
     hypreDevice_CSRSpTrans(nrows_A, ncols_A, nnz_A, A_i, A_j, A_data, &C_i, &C_j, &C_data, data);
#endif
   }

   C = hypre_CSRMatrixCreate(ncols_A, nrows_A, nnz_A);
   hypre_CSRMatrixI(C) = C_i;
   hypre_CSRMatrixJ(C) = C_j;
   hypre_CSRMatrixData(C) = C_data;
   hypre_CSRMatrixMemoryLocation(C) = HYPRE_MEMORY_DEVICE;

   *AT_ptr = C;

   hypre_SyncCudaComputeStream(hypre_handle());

   return hypre_error_flag;
}

#endif

HYPRE_Int
hypre_CSRMatrixSortRow(hypre_CSRMatrix *A)
{
#if defined(HYPRE_USING_CUSPARSE)
   hypre_SortCSRCusparse(hypre_CSRMatrixNumRows(A), hypre_CSRMatrixNumCols(A), hypre_CSRMatrixNumNonzeros(A),
                         hypre_CSRMatrixI(A), hypre_CSRMatrixJ(A), hypre_CSRMatrixData(A));
#elif defined(HYPRE_USING_ROCSPARSE)
   hypre_SortCSRRocsparse(hypre_CSRMatrixNumRows(A), hypre_CSRMatrixNumCols(A), hypre_CSRMatrixNumNonzeros(A),
                          hypre_CSRMatrixI(A), hypre_CSRMatrixJ(A), hypre_CSRMatrixData(A));
#else
   hypre_error_w_msg(HYPRE_ERROR_GENERIC,"hypre_CSRMatrixSortRow only implemented for cuSPARSE!\n");
#endif

   return hypre_error_flag;
}

#if defined(HYPRE_USING_CUSPARSE)
/* @brief This functions sorts values and column indices in each row in ascending order INPLACE
 * @param[in] n Number of rows
 * @param[in] m Number of columns
 * @param[in] nnzA Number of nonzeroes
 * @param[in] *d_ia (Unsorted) Row indices
 * @param[in,out] *d_ja_sorted On Start: Unsorted column indices. On return: Sorted column indices
 * @param[in,out] *d_a_sorted On Start: Unsorted values. On Return: Sorted values corresponding with column indices
 */
void
hypre_SortCSRCusparse( HYPRE_Int      n,
                       HYPRE_Int      m,
                       HYPRE_Int      nnzA,
                 const HYPRE_Int     *d_ia,
                       HYPRE_Int     *d_ja_sorted,
                       HYPRE_Complex *d_a_sorted )
{
   cusparseHandle_t cusparsehandle = hypre_HandleCusparseHandle(hypre_handle());
   cusparseMatDescr_t descrA = hypre_HandleCusparseMatDescr(hypre_handle());

   size_t pBufferSizeInBytes = 0;
   void *pBuffer = NULL;

   csru2csrInfo_t sortInfoA;
   HYPRE_CUSPARSE_CALL( cusparseCreateCsru2csrInfo(&sortInfoA) );

   HYPRE_Int isDoublePrecision = sizeof(HYPRE_Complex) == sizeof(hypre_double);
   HYPRE_Int isSinglePrecision = sizeof(HYPRE_Complex) == sizeof(hypre_double) / 2;

   if (isDoublePrecision)
   {
      HYPRE_CUSPARSE_CALL( cusparseDcsru2csr_bufferSizeExt(cusparsehandle,
                                                           n, m, nnzA, d_a_sorted, d_ia, d_ja_sorted,
                                                           sortInfoA, &pBufferSizeInBytes) );

      pBuffer = hypre_TAlloc(char, pBufferSizeInBytes, HYPRE_MEMORY_DEVICE);

      HYPRE_CUSPARSE_CALL( cusparseDcsru2csr(cusparsehandle,
                                             n, m, nnzA, descrA, d_a_sorted, d_ia, d_ja_sorted,
                                             sortInfoA, pBuffer) );
   }
   else if (isSinglePrecision)
   {
      HYPRE_CUSPARSE_CALL( cusparseScsru2csr_bufferSizeExt(cusparsehandle,
                                                           n, m, nnzA, (float *) d_a_sorted, d_ia, d_ja_sorted,
                                                           sortInfoA, &pBufferSizeInBytes));

      pBuffer = hypre_TAlloc(char, pBufferSizeInBytes, HYPRE_MEMORY_DEVICE);

      HYPRE_CUSPARSE_CALL( cusparseScsru2csr(cusparsehandle,
                                             n, m, nnzA, descrA, (float *)d_a_sorted, d_ia, d_ja_sorted,
                                             sortInfoA, pBuffer) );
   }

   hypre_TFree(pBuffer, HYPRE_MEMORY_DEVICE);
   HYPRE_CUSPARSE_CALL(cusparseDestroyCsru2csrInfo(sortInfoA));
}

HYPRE_Int
hypre_CSRMatrixTriLowerUpperSolveCusparse(char             uplo,
                                          hypre_CSRMatrix *A,
                                          hypre_Vector    *f,
                                          hypre_Vector    *u )
{
   HYPRE_Int      nrow   = hypre_CSRMatrixNumRows(A);
   HYPRE_Int      ncol   = hypre_CSRMatrixNumCols(A);
   HYPRE_Int      nnzA   = hypre_CSRMatrixNumNonzeros(A);
   HYPRE_Int     *A_i    = hypre_CSRMatrixI(A);
   HYPRE_Int     *A_j    = hypre_CSRMatrixJ(A);
   HYPRE_Complex *A_a    = hypre_CSRMatrixData(A);
   HYPRE_Int     *A_sj   = hypre_CSRMatrixSortedJ(A);
   HYPRE_Complex *A_sa   = hypre_CSRMatrixSortedData(A);
   HYPRE_Complex *f_data = hypre_VectorData(f);
   HYPRE_Complex *u_data = hypre_VectorData(u);
   HYPRE_Complex  alpha  = 1.0;
   hypre_int      buffer_size;
   hypre_int      structural_zero;

   if (nrow != ncol)
   {
      hypre_assert(0);
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if (nrow <= 0)
   {
      return hypre_error_flag;
   }

   if (nnzA <= 0)
   {
      hypre_assert(0);
      hypre_error_in_arg(1);
      return hypre_error_flag;
   }

   if ( !A_sj && !A_sa )
   {
      hypre_CSRMatrixSortedJ(A) = A_sj = hypre_TAlloc(HYPRE_Int, nnzA, HYPRE_MEMORY_DEVICE);
      hypre_CSRMatrixSortedData(A) = A_sa = hypre_TAlloc(HYPRE_Complex, nnzA, HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(A_sj, A_j, HYPRE_Int, nnzA, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(A_sa, A_a, HYPRE_Complex, nnzA, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
      hypre_CSRMatrixData(A) = A_sa;
      HYPRE_Int err = hypre_CSRMatrixCheckDiagFirstSetValueZeroDevice(A, INFINITY);  hypre_assert(err == 0);
      hypre_CSRMatrixData(A) = A_a;
#endif

      hypre_SortCSRCusparse(nrow, ncol, nnzA, A_i, A_sj, A_sa);
   }

   cusparseHandle_t handle = hypre_HandleCusparseHandle(hypre_handle());
   cusparseMatDescr_t descr = hypre_HandleCusparseMatDescr(hypre_handle());

   HYPRE_CUSPARSE_CALL( cusparseSetMatDiagType(descr, CUSPARSE_DIAG_TYPE_NON_UNIT) );

   if (!hypre_CSRMatrixCsrsvData(A))
   {
      hypre_CSRMatrixCsrsvData(A) = hypre_CsrsvDataCreate();
   }
   hypre_CsrsvData *csrsv_data = hypre_CSRMatrixCsrsvData(A);

   if (uplo == 'L')
   {
      HYPRE_CUSPARSE_CALL( cusparseSetMatFillMode(descr, CUSPARSE_FILL_MODE_LOWER) );

      if (!hypre_CsrsvDataInfoL(csrsv_data))
      {
         HYPRE_CUSPARSE_CALL( cusparseCreateCsrsv2Info(&hypre_CsrsvDataInfoL(csrsv_data)) );

         HYPRE_CUSPARSE_CALL( cusparseDcsrsv2_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                              nrow, nnzA, descr, A_sa, A_i, A_sj, hypre_CsrsvDataInfoL(csrsv_data), &buffer_size) );

         if (hypre_CsrsvDataBufferSize(csrsv_data) < buffer_size)
         {
            hypre_CsrsvDataBuffer(csrsv_data) = hypre_TReAlloc_v2(hypre_CsrsvDataBuffer(csrsv_data),
                                                                  char, hypre_CsrsvDataBufferSize(csrsv_data),
                                                                  char, buffer_size,
                                                                  HYPRE_MEMORY_DEVICE);
         }

         HYPRE_CUSPARSE_CALL( cusparseDcsrsv2_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                       nrow, nnzA, descr, A_sa, A_i, A_sj,
                                                       hypre_CsrsvDataInfoL(csrsv_data), CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                                                       hypre_CsrsvDataBuffer(csrsv_data)) );

         cusparseStatus_t status = cusparseXcsrsv2_zeroPivot(handle, hypre_CsrsvDataInfoL(csrsv_data), &structural_zero);
         if (CUSPARSE_STATUS_ZERO_PIVOT == status)
         {
            hypre_printf("hypre_CSRMatrixTriLowerUpperSolveCusparse A(%d,%d) is missing\n", structural_zero, structural_zero);
            hypre_assert(0);
            hypre_error_in_arg(1);
            return hypre_error_flag;
         }
      }

      HYPRE_CUSPARSE_CALL( cusparseDcsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                 nrow, nnzA, &alpha, descr, A_sa, A_i, A_sj,
                                                 hypre_CsrsvDataInfoL(csrsv_data), f_data, u_data,
                                                 CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                                                 hypre_CsrsvDataBuffer(csrsv_data)) );
   }
   else
   {
      HYPRE_CUSPARSE_CALL( cusparseSetMatFillMode(descr, CUSPARSE_FILL_MODE_UPPER) );

      if (!hypre_CsrsvDataInfoU(csrsv_data))
      {
         HYPRE_CUSPARSE_CALL( cusparseCreateCsrsv2Info(&hypre_CsrsvDataInfoU(csrsv_data)) );

         HYPRE_CUSPARSE_CALL( cusparseDcsrsv2_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                              nrow, nnzA, descr, A_sa, A_i, A_sj, hypre_CsrsvDataInfoU(csrsv_data), &buffer_size) );

         if (hypre_CsrsvDataBufferSize(csrsv_data) < buffer_size)
         {
            hypre_CsrsvDataBuffer(csrsv_data) = hypre_TReAlloc_v2(hypre_CsrsvDataBuffer(csrsv_data),
                                                                  char, hypre_CsrsvDataBufferSize(csrsv_data),
                                                                  char, buffer_size,
                                                                  HYPRE_MEMORY_DEVICE);
         }

         HYPRE_CUSPARSE_CALL( cusparseDcsrsv2_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                       nrow, nnzA, descr, A_sa, A_i, A_sj,
                                                       hypre_CsrsvDataInfoU(csrsv_data), CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                                                       hypre_CsrsvDataBuffer(csrsv_data)) );

         cusparseStatus_t status = cusparseXcsrsv2_zeroPivot(handle, hypre_CsrsvDataInfoU(csrsv_data), &structural_zero);
         if (CUSPARSE_STATUS_ZERO_PIVOT == status)
         {
            hypre_printf("hypre_CSRMatrixTriLowerUpperSolveCusparse A(%d,%d) is missing\n", structural_zero, structural_zero);
            hypre_assert(0);
            hypre_error_in_arg(1);
            return hypre_error_flag;
         }
      }

      HYPRE_CUSPARSE_CALL( cusparseDcsrsv2_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                 nrow, nnzA, &alpha, descr, A_sa, A_i, A_sj,
                                                 hypre_CsrsvDataInfoU(csrsv_data), f_data, u_data,
                                                 CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                                                 hypre_CsrsvDataBuffer(csrsv_data)) );
   }

   return hypre_error_flag;
}

#endif /* #if defined(HYPRE_USING_CUSPARSE) */


#if defined(HYPRE_USING_ROCSPARSE)
/* @brief This functions sorts values and column indices in each row in ascending order OUT-OF-PLACE
 * @param[in] n Number of rows
 * @param[in] m Number of columns
 * @param[in] nnzA Number of nonzeroes
 * @param[in] *d_ia (Unsorted) Row indices
 * @param[in,out] *d_ja_sorted On Start: Unsorted column indices. On return: Sorted column indices
 * @param[in,out] *d_a_sorted On Start: Unsorted values. On Return: Sorted values corresponding with column indices
 */
void
hypre_SortCSRRocsparse( HYPRE_Int      n,
                        HYPRE_Int      m,
                        HYPRE_Int      nnzA,
                  const HYPRE_Int     *d_ia,
                        HYPRE_Int     *d_ja_sorted,
                        HYPRE_Complex *d_a_sorted )
{
   rocsparse_handle handle = hypre_HandleCusparseHandle(hypre_handle());

   // FIXME: This is an abuse. Really, each matrix should have its own
   //        rocsparse_mat_descr and rocsparse_mat_info and these should
   //        not be global variables.
   rocsparse_mat_descr descrA = hypre_HandleCusparseMatDescr(hypre_handle());

   size_t pBufferSizeInBytes = 0;
   void *pBuffer = NULL;
   HYPRE_Int *P = NULL;

   HYPRE_Int isDoublePrecision = sizeof(HYPRE_Complex) == sizeof(hypre_double);
   HYPRE_Int isSinglePrecision = sizeof(HYPRE_Complex) == sizeof(hypre_double) / 2;

   // FIXME: There is not in-place version of csr sort in rocSPARSE currently, so we make
   //        a temporary copy of the data for gthr, sort that, and then copy the sorted values
   //        back to the array being returned. Where there is an in-place version available,
   //        we should use it.
   HYPRE_Complex * d_a_tmp;
   d_a_tmp  = hypre_TAlloc(HYPRE_Complex, nnzA, HYPRE_MEMORY_DEVICE);

   HYPRE_ROCSPARSE_CALL( rocsparse_csrsort_buffer_size(handle, n, m, nnzA, d_ia, d_ja_sorted, &pBufferSizeInBytes) );

   pBuffer = hypre_TAlloc(char, pBufferSizeInBytes, HYPRE_MEMORY_DEVICE);
   P       = hypre_TAlloc(HYPRE_Int, nnzA, HYPRE_MEMORY_DEVICE);

   HYPRE_ROCSPARSE_CALL( rocsparse_create_identity_permutation(handle, nnzA, P) );
   HYPRE_ROCSPARSE_CALL( rocsparse_csrsort(handle, n, m, nnzA, descrA, d_ia, d_ja_sorted, P, pBuffer) );

   if (isDoublePrecision)
   {
     HYPRE_ROCSPARSE_CALL( rocsparse_dgthr(handle, nnzA, d_a_sorted, d_a_tmp, P, rocsparse_index_base_zero) );
   }
   else if (isSinglePrecision)
   {
     HYPRE_ROCSPARSE_CALL( rocsparse_sgthr(handle, nnzA, (float *) d_a_sorted, (float *) d_a_tmp, P, rocsparse_index_base_zero) );
   }

   hypre_TFree(pBuffer, HYPRE_MEMORY_DEVICE);
   hypre_TFree(P, HYPRE_MEMORY_DEVICE);

   hypre_TMemcpy(d_a_sorted, d_a_tmp, HYPRE_Complex, nnzA, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

   hypre_TFree(d_a_tmp, HYPRE_MEMORY_DEVICE);
}
#endif // #if defined(HYPRE_USING_ROCSPARSE)
