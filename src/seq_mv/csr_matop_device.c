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
#include "_hypre_utilities.hpp"
#include "seq_mv.hpp"

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

#if defined(HYPRE_USING_CUSPARSE) || defined(HYPRE_USING_ROCSPARSE)
hypre_GpuMatData *
hypre_GpuMatDataCreate()
{
   hypre_GpuMatData *data = hypre_CTAlloc(hypre_GpuMatData, 1, HYPRE_MEMORY_HOST);

#if defined(HYPRE_USING_CUSPARSE)
   cusparseMatDescr_t mat_descr;
   HYPRE_CUSPARSE_CALL( cusparseCreateMatDescr(&mat_descr) );
   HYPRE_CUSPARSE_CALL( cusparseSetMatType(mat_descr, CUSPARSE_MATRIX_TYPE_GENERAL) );
   HYPRE_CUSPARSE_CALL( cusparseSetMatIndexBase(mat_descr, CUSPARSE_INDEX_BASE_ZERO) );
   hypre_GpuMatDataMatDecsr(data) = mat_descr;
#endif

#if defined(HYPRE_USING_ROCSPARSE)
   rocsparse_mat_descr mat_descr;
   rocsparse_mat_info  info;
   HYPRE_ROCSPARSE_CALL( rocsparse_create_mat_descr(&mat_descr) );
   HYPRE_ROCSPARSE_CALL( rocsparse_set_mat_type(mat_descr, rocsparse_matrix_type_general) );
   HYPRE_ROCSPARSE_CALL( rocsparse_set_mat_index_base(mat_descr, rocsparse_index_base_zero) );
   hypre_GpuMatDataMatDecsr(data) = mat_descr;
   HYPRE_ROCSPARSE_CALL( rocsparse_create_mat_info(&info) );
   hypre_GpuMatDataMatInfo(data) = info;
#endif

   return data;
}

void
hypre_GpuMatDataDestroy(hypre_GpuMatData *data)
{
   if (!data)
   {
      return;
   }

#if defined(HYPRE_USING_CUSPARSE)
   HYPRE_CUSPARSE_CALL( cusparseDestroyMatDescr(hypre_GpuMatDataMatDecsr(data)) );
#endif

#if defined(HYPRE_USING_ROCSPARSE)
   HYPRE_ROCSPARSE_CALL( rocsparse_destroy_mat_descr(hypre_GpuMatDataMatDecsr(data)) );
   HYPRE_ROCSPARSE_CALL( rocsparse_destroy_mat_info(hypre_GpuMatDataMatInfo(data)) );
#endif

  hypre_TFree(data, HYPRE_MEMORY_HOST);
}
#endif

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)

hypre_CSRMatrix*
hypre_CSRMatrixAddDevice ( HYPRE_Complex    alpha,
                           hypre_CSRMatrix *A,
                           HYPRE_Complex    beta,
                           hypre_CSRMatrix *B     )
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
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,"Warning! Incompatible matrix dimensions!\n");

      return NULL;
   }

   hypreDevice_CSRSpAdd(nrows_A, nrows_B, ncols_A, nnz_A, nnz_B,
                        A_i, A_j, alpha, A_data, NULL, B_i, B_j, beta, B_data, NULL, NULL,
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
   HYPRE_Int         ncols_A  = hypre_CSRMatrixNumCols(A);
   HYPRE_Int         nrows_B  = hypre_CSRMatrixNumRows(B);
   hypre_CSRMatrix  *C;

   if (ncols_A != nrows_B)
   {
      hypre_printf("Warning! incompatible matrix dimensions!\n");
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,"Warning! incompatible matrix dimensions!\n");

      return NULL;
   }

   hypreDevice_CSRSpGemm(A, B, &C);

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

   HYPRE_BigInt *new_end = HYPRE_THRUST_CALL( unique,
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

   hypreDevice_CSRSpAdd(nrows_A, nrows_B, ncols_A, nnz_A, nnz_B, A_i, A_j, 1.0, A_data, NULL, B_i, B_j, 1.0, B_data, NULL, row_nums,
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
 * RL: only check if it's a non-empty row
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
      result[row] = (ia[row+1] > ia[row]) && (ja[ia[row]] != row);
   }
}

HYPRE_Int
hypre_CSRMatrixCheckDiagFirstDevice( hypre_CSRMatrix *A )
{
   if (hypre_CSRMatrixNumRows(A) != hypre_CSRMatrixNumCols(A))
   {
      return 0;
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

__global__ void
hypreCUDAKernel_CSRMatrixFixZeroDiagDevice( HYPRE_Complex  v,
                                            HYPRE_Int      nrows,
                                            HYPRE_Int     *ia,
                                            HYPRE_Int     *ja,
                                            HYPRE_Complex *data,
                                            HYPRE_Real     tol,
                                            HYPRE_Int     *result )
{
   const HYPRE_Int row = hypre_cuda_get_grid_warp_id<1,1>();

   if (row >= nrows)
   {
      return;
   }

   HYPRE_Int lane = hypre_cuda_get_lane_id<1>();
   HYPRE_Int p, q;
   bool has_diag = false;

   if (lane < 2)
   {
      p = read_only_load(ia + row + lane);
   }
   q = __shfl_sync(HYPRE_WARP_FULL_MASK, p, 1);
   p = __shfl_sync(HYPRE_WARP_FULL_MASK, p, 0);

   for (HYPRE_Int j = p + lane; __any_sync(HYPRE_WARP_FULL_MASK, j < q); j += HYPRE_WARP_SIZE)
   {
      hypre_int find_diag = j < q && read_only_load(&ja[j]) == row;

      if (find_diag)
      {
         if (fabs(data[j]) <= tol)
         {
            data[j] = v;
         }
      }

      if ( __any_sync(HYPRE_WARP_FULL_MASK, find_diag) )
      {
         has_diag = true;
         break;
      }
   }

   if (result && !has_diag && lane == 0)
   {
      result[row] = 1;
   }
}

/* For square A, find numerical zeros (absolute values <= tol) on its diagonal and replace with v
 * Does NOT assume diagonal is the first entry of each row of A
 * In debug mode:
 *    Returns the number of rows that do not have diag in the pattern
 *    (i.e., structural zeroes on the diagonal)
 */
HYPRE_Int
hypre_CSRMatrixFixZeroDiagDevice( hypre_CSRMatrix *A,
                                  HYPRE_Complex    v,
                                  HYPRE_Real       tol )
{
   HYPRE_Int ierr = 0;

   if (hypre_CSRMatrixNumRows(A) != hypre_CSRMatrixNumCols(A))
   {
      return ierr;
   }

   dim3 bDim = hypre_GetDefaultCUDABlockDimension();
   dim3 gDim = hypre_GetDefaultCUDAGridDimension(hypre_CSRMatrixNumRows(A), "warp", bDim);

#if HYPRE_DEBUG
   HYPRE_Int *result = hypre_CTAlloc(HYPRE_Int, hypre_CSRMatrixNumRows(A), HYPRE_MEMORY_DEVICE);
#else
   HYPRE_Int *result = NULL;
#endif

   HYPRE_CUDA_LAUNCH( hypreCUDAKernel_CSRMatrixFixZeroDiagDevice, gDim, bDim,
                      v, hypre_CSRMatrixNumRows(A),
                      hypre_CSRMatrixI(A), hypre_CSRMatrixJ(A), hypre_CSRMatrixData(A),
                      tol, result );

#if HYPRE_DEBUG
   ierr = HYPRE_THRUST_CALL( reduce,
                             result,
                             result + hypre_CSRMatrixNumRows(A) );

   hypre_TFree(result, HYPRE_MEMORY_DEVICE);
#endif

   hypre_SyncCudaComputeStream(hypre_handle());

   return ierr;
}

__global__ void
hypreCUDAKernel_CSRMatrixReplaceDiagDevice( HYPRE_Complex *new_diag,
                                            HYPRE_Complex  v,
                                            HYPRE_Int      nrows,
                                            HYPRE_Int     *ia,
                                            HYPRE_Int     *ja,
                                            HYPRE_Complex *data,
                                            HYPRE_Real     tol,
                                            HYPRE_Int     *result )
{
   const HYPRE_Int row = hypre_cuda_get_grid_warp_id<1,1>();

   if (row >= nrows)
   {
      return;
   }

   HYPRE_Int lane = hypre_cuda_get_lane_id<1>();
   HYPRE_Int p, q;
   bool has_diag = false;

   if (lane < 2)
   {
      p = read_only_load(ia + row + lane);
   }
   q = __shfl_sync(HYPRE_WARP_FULL_MASK, p, 1);
   p = __shfl_sync(HYPRE_WARP_FULL_MASK, p, 0);

   for (HYPRE_Int j = p + lane; __any_sync(HYPRE_WARP_FULL_MASK, j < q); j += HYPRE_WARP_SIZE)
   {
      hypre_int find_diag = j < q && read_only_load(&ja[j]) == row;

      if (find_diag)
      {
         HYPRE_Complex d = read_only_load(&new_diag[row]);
         if (fabs(d) <= tol)
         {
            d = v;
         }
         data[j] = d;
      }

      if ( __any_sync(HYPRE_WARP_FULL_MASK, find_diag) )
      {
         has_diag = true;
         break;
      }
   }

   if (result && !has_diag && lane == 0)
   {
      result[row] = 1;
   }
}

HYPRE_Int
hypre_CSRMatrixReplaceDiagDevice( hypre_CSRMatrix *A,
                                  HYPRE_Complex   *new_diag,
                                  HYPRE_Complex    v,
                                  HYPRE_Real       tol )
{
   HYPRE_Int ierr = 0;

   if (hypre_CSRMatrixNumRows(A) != hypre_CSRMatrixNumCols(A))
   {
      return ierr;
   }

   dim3 bDim = hypre_GetDefaultCUDABlockDimension();
   dim3 gDim = hypre_GetDefaultCUDAGridDimension(hypre_CSRMatrixNumRows(A), "warp", bDim);

#if HYPRE_DEBUG
   HYPRE_Int *result = hypre_CTAlloc(HYPRE_Int, hypre_CSRMatrixNumRows(A), HYPRE_MEMORY_DEVICE);
#else
   HYPRE_Int *result = NULL;
#endif

   HYPRE_CUDA_LAUNCH( hypreCUDAKernel_CSRMatrixReplaceDiagDevice, gDim, bDim,
                      new_diag, v, hypre_CSRMatrixNumRows(A),
                      hypre_CSRMatrixI(A), hypre_CSRMatrixJ(A), hypre_CSRMatrixData(A),
                      tol, result );

#if HYPRE_DEBUG
   ierr = HYPRE_THRUST_CALL( reduce,
                             result,
                             result + hypre_CSRMatrixNumRows(A) );

   hypre_TFree(result, HYPRE_MEMORY_DEVICE);
#endif

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
 *      3: diag inverse sqrt
 *      4: abs diag inverse sqrt
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
         else if (type == 3)
         {
            d[row] = 1.0 / sqrt(aa[j]);
         }
         else if (type == 4)
         {
            d[row] = 1.0 / sqrt(fabs(aa[j]));
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

/* this predicate compares first and second element in a tuple in absolute value */
/* first is assumed to be complex, second to be real > 0 */
struct cabsfirst_greaterthan_second_pred : public thrust::unary_function<thrust::tuple<HYPRE_Complex, HYPRE_Real>,bool>
{
   __host__ __device__
   bool operator()(const thrust::tuple<HYPRE_Complex, HYPRE_Real>& t) const
   {
      const HYPRE_Complex i = thrust::get<0>(t);
      const HYPRE_Real j = thrust::get<1>(t);

      return hypre_cabs(i) > j;
   }
};

/* drop the entries that are smaller than:
 *    tol if elmt_tols == null,
 *    elmt_tols[j] otherwise where j = 0...NumNonzeros(A) */
HYPRE_Int
hypre_CSRMatrixDropSmallEntriesDevice( hypre_CSRMatrix *A,
                                       HYPRE_Real       tol,
                                       HYPRE_Real      *elmt_tols)
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

   if (elmt_tols == NULL)
   {
      new_nnz = HYPRE_THRUST_CALL( count_if,
                                   A_data,
                                   A_data + nnz,
                                   thrust::not1(less_than<HYPRE_Complex>(tol)) );
   }
   else
   {
      new_nnz = HYPRE_THRUST_CALL( count_if,
                                   thrust::make_zip_iterator(thrust::make_tuple(A_data, elmt_tols)),
                                   thrust::make_zip_iterator(thrust::make_tuple(A_data, elmt_tols)) + nnz,
                                   cabsfirst_greaterthan_second_pred() );
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

   if (elmt_tols == NULL)
   {
      new_end = HYPRE_THRUST_CALL( copy_if,
                                   thrust::make_zip_iterator(thrust::make_tuple(A_ii, A_j, A_data)),
                                   thrust::make_zip_iterator(thrust::make_tuple(A_ii, A_j, A_data)) + nnz,
                                   A_data,
                                   thrust::make_zip_iterator(thrust::make_tuple(new_ii, new_j, new_data)),
                                   thrust::not1(less_than<HYPRE_Complex>(tol)) );
   }
   else
   {
      new_end = HYPRE_THRUST_CALL( copy_if,
                                   thrust::make_zip_iterator(thrust::make_tuple(A_ii, A_j, A_data)),
                                   thrust::make_zip_iterator(thrust::make_tuple(A_ii, A_j, A_data)) + nnz,
                                   thrust::make_zip_iterator(thrust::make_tuple(A_data, elmt_tols)),
                                   thrust::make_zip_iterator(thrust::make_tuple(new_ii, new_j, new_data)),
                                   cabsfirst_greaterthan_second_pred() );
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

/* mark is of size nA
 * diag_option: 1: special treatment for diag entries, mark as -2
 */
__global__ void
hypreCUDAKernel_CSRMatrixIntersectPattern(HYPRE_Int  n,
                                          HYPRE_Int  nA,
                                          HYPRE_Int *rowid,
                                          HYPRE_Int *colid,
                                          HYPRE_Int *idx,
                                          HYPRE_Int *mark,
                                          HYPRE_Int  diag_option)
{
   HYPRE_Int i = hypre_cuda_get_grid_thread_id<1,1>();

   if (i >= n)
   {
      return;
   }

   HYPRE_Int r1 = read_only_load(&rowid[i]);
   HYPRE_Int c1 = read_only_load(&colid[i]);
   HYPRE_Int j = read_only_load(&idx[i]);

   if (0 == diag_option)
   {
      if (j < nA)
      {
         HYPRE_Int r2 = i < n - 1 ? read_only_load(&rowid[i + 1]) : -1;
         HYPRE_Int c2 = i < n - 1 ? read_only_load(&colid[i + 1]) : -1;
         if (r1 == r2 && c1 == c2)
         {
            mark[j] = c1;
         }
         else
         {
            mark[j] = -1;
         }
      }
   }
   else if (1 == diag_option)
   {
      if (j < nA)
      {
         if (r1 == c1)
         {
            mark[j] = -2;
         }
         else
         {
            HYPRE_Int r2 = i < n - 1 ? read_only_load(&rowid[i + 1]) : -1;
            HYPRE_Int c2 = i < n - 1 ? read_only_load(&colid[i + 1]) : -1;
            if (r1 == r2 && c1 == c2)
            {
               mark[j] = c1;
            }
            else
            {
               mark[j] = -1;
            }
         }
      }
   }
}

/* markA: array of size nnz(A), for pattern of (A and B), markA is the column indices as in A_J
 * Otherwise, mark pattern not in A-B as -1 in markA
 * Note the special treatment for diagonal entries of A (marked as -2) */
HYPRE_Int
hypre_CSRMatrixIntersectPattern(hypre_CSRMatrix *A,
                                hypre_CSRMatrix *B,
                                HYPRE_Int       *markA,
                                HYPRE_Int        diag_opt)
{
   HYPRE_Int nrows = hypre_CSRMatrixNumRows(A);
   HYPRE_Int nnzA  = hypre_CSRMatrixNumNonzeros(A);
   HYPRE_Int nnzB  = hypre_CSRMatrixNumNonzeros(B);

   HYPRE_Int *Cii = hypre_TAlloc(HYPRE_Int, nnzA + nnzB, HYPRE_MEMORY_DEVICE);
   HYPRE_Int *Cjj = hypre_TAlloc(HYPRE_Int, nnzA + nnzB, HYPRE_MEMORY_DEVICE);
   HYPRE_Int *idx = hypre_TAlloc(HYPRE_Int, nnzA + nnzB, HYPRE_MEMORY_DEVICE);

   hypreDevice_CsrRowPtrsToIndices_v2(nrows, nnzA, hypre_CSRMatrixI(A), Cii);
   hypreDevice_CsrRowPtrsToIndices_v2(nrows, nnzB, hypre_CSRMatrixI(B), Cii + nnzA);
   hypre_TMemcpy(Cjj,        hypre_CSRMatrixJ(A), HYPRE_Int, nnzA, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(Cjj + nnzA, hypre_CSRMatrixJ(B), HYPRE_Int, nnzB, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   HYPRE_THRUST_CALL( sequence, idx, idx + nnzA + nnzB );

   HYPRE_THRUST_CALL( stable_sort_by_key,
                      thrust::make_zip_iterator(thrust::make_tuple(Cii, Cjj)),
                      thrust::make_zip_iterator(thrust::make_tuple(Cii, Cjj)) + nnzA + nnzB,
                      idx );

   hypre_TMemcpy(markA, hypre_CSRMatrixJ(A), HYPRE_Int, nnzA, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

   dim3 bDim = hypre_GetDefaultCUDABlockDimension();
   dim3 gDim = hypre_GetDefaultCUDAGridDimension(nnzA + nnzB, "thread", bDim);

   HYPRE_CUDA_LAUNCH( hypreCUDAKernel_CSRMatrixIntersectPattern, gDim, bDim,
                      nnzA + nnzB, nnzA, Cii, Cjj, idx, markA, diag_opt );

   hypre_TFree(Cii, HYPRE_MEMORY_DEVICE);
   hypre_TFree(Cjj, HYPRE_MEMORY_DEVICE);
   hypre_TFree(idx, HYPRE_MEMORY_DEVICE);

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
   hypre_SortCSRCusparse(hypre_CSRMatrixNumRows(A), hypre_CSRMatrixNumCols(A), hypre_CSRMatrixNumNonzeros(A), hypre_CSRMatrixGPUMatDescr(A),
                         hypre_CSRMatrixI(A), hypre_CSRMatrixJ(A), hypre_CSRMatrixData(A));
#elif defined(HYPRE_USING_ROCSPARSE)
   hypre_SortCSRRocsparse(hypre_CSRMatrixNumRows(A), hypre_CSRMatrixNumCols(A), hypre_CSRMatrixNumNonzeros(A), hypre_CSRMatrixGPUMatDescr(A),
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
hypre_SortCSRCusparse(       HYPRE_Int      n,
                             HYPRE_Int      m,
                             HYPRE_Int      nnzA,
                             cusparseMatDescr_t descrA,
                       const HYPRE_Int     *d_ia,
                             HYPRE_Int     *d_ja_sorted,
                             HYPRE_Complex *d_a_sorted )
{
   cusparseHandle_t cusparsehandle = hypre_HandleCusparseHandle(hypre_handle());

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
                                          HYPRE_Real      *l1_norms,
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

   cusparseHandle_t handle = hypre_HandleCusparseHandle(hypre_handle());
   cusparseMatDescr_t descr = hypre_CSRMatrixGPUMatDescr(A);

   if ( !A_sj && !A_sa )
   {
      hypre_CSRMatrixSortedJ(A) = A_sj = hypre_TAlloc(HYPRE_Int, nnzA, HYPRE_MEMORY_DEVICE);
      hypre_CSRMatrixSortedData(A) = A_sa = hypre_TAlloc(HYPRE_Complex, nnzA, HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(A_sj, A_j, HYPRE_Int, nnzA, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
      hypre_TMemcpy(A_sa, A_a, HYPRE_Complex, nnzA, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)
      hypre_CSRMatrixData(A) = A_sa;
      HYPRE_Int err = 0;
      if (l1_norms)
      {
         err = hypre_CSRMatrixReplaceDiagDevice(A, l1_norms, INFINITY, 0.0);
      }
      else
      {
         err = hypre_CSRMatrixFixZeroDiagDevice(A, INFINITY, 0.0);
      }
      hypre_CSRMatrixData(A) = A_a;
      if (err)
      {
         hypre_error_w_msg(1, "structural zero in hypre_CSRMatrixTriLowerUpperSolveCusparse");
         //hypre_assert(0);
      }
#endif

      hypre_SortCSRCusparse(nrow, ncol, nnzA, descr, A_i, A_sj, A_sa);
   }

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
            char msg[256];
            hypre_sprintf(msg, "hypre_CSRMatrixTriLowerUpperSolveCusparse A(%d,%d) is missing\n",
                          structural_zero, structural_zero);
            hypre_error_w_msg(1, msg);
            //hypre_assert(0);
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
            char msg[256];
            hypre_sprintf(msg, "hypre_CSRMatrixTriLowerUpperSolveCusparse A(%d,%d) is missing\n",
                          structural_zero, structural_zero);
            hypre_error_w_msg(1, msg);
            //hypre_assert(0);
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
// FIXME: We need a stub for this function until we can implement a rocsparse version
HYPRE_Int
hypre_CSRMatrixTriLowerUpperSolveCusparse(char              /*uplo*/,
                                          hypre_CSRMatrix * /*A*/,
                                          HYPRE_Real      * /*l1_norms*/,
                                          hypre_Vector    * /*f*/,
                                          hypre_Vector    * /*u*/ )
{
   hypre_error_w_msg(HYPRE_ERROR_GENERIC, "hypre_CSRMatrixTriLowerUpperSolveCusparse not implemented for rocSPARSE!\n");

   return 1;
}


/* @brief This functions sorts values and column indices in each row in ascending order OUT-OF-PLACE
 * @param[in] n Number of rows
 * @param[in] m Number of columns
 * @param[in] nnzA Number of nonzeroes
 * @param[in] *d_ia (Unsorted) Row indices
 * @param[in,out] *d_ja_sorted On Start: Unsorted column indices. On return: Sorted column indices
 * @param[in,out] *d_a_sorted On Start: Unsorted values. On Return: Sorted values corresponding with column indices
 */
void
hypre_SortCSRRocsparse(       HYPRE_Int      n,
                              HYPRE_Int      m,
                              HYPRE_Int      nnzA,
                              rocsparse_mat_descr descrA,
                        const HYPRE_Int     *d_ia,
                              HYPRE_Int     *d_ja_sorted,
                              HYPRE_Complex *d_a_sorted )
{
   rocsparse_handle handle = hypre_HandleCusparseHandle(hypre_handle());

   size_t pBufferSizeInBytes = 0;
   void *pBuffer = NULL;
   HYPRE_Int *P = NULL;

   HYPRE_Int isDoublePrecision = sizeof(HYPRE_Complex) == sizeof(hypre_double);
   HYPRE_Int isSinglePrecision = sizeof(HYPRE_Complex) == sizeof(hypre_double) / 2;

   // FIXME: There is not in-place version of csr sort in rocSPARSE currently, so we make
   //        a temporary copy of the data for gthr, sort that, and then copy the sorted values
   //        back to the array being returned. Where there is an in-place version available,
   //        we should use it.
   HYPRE_Complex *d_a_tmp;
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
