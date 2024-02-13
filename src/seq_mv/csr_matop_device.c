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

#include "_hypre_onedpl.hpp"
#include "seq_mv.h"
#include "_hypre_utilities.hpp"
#include "seq_mv.hpp"

#if defined(HYPRE_USING_CUSPARSE)  ||\
    defined(HYPRE_USING_ROCSPARSE) ||\
    defined(HYPRE_USING_ONEMKLSPARSE)

/*--------------------------------------------------------------------------
 * hypre_CsrsvDataCreate
 *--------------------------------------------------------------------------*/

hypre_CsrsvData*
hypre_CsrsvDataCreate()
{
   hypre_CsrsvData *data = hypre_CTAlloc(hypre_CsrsvData, 1, HYPRE_MEMORY_HOST);

#if defined(HYPRE_USING_CUSPARSE)
   HYPRE_CUSPARSE_CALL( hypre_cusparseSpSV_createDescr(&hypre_CsrsvDataInfoL(data)) );
   HYPRE_CUSPARSE_CALL( hypre_cusparseSpSV_createDescr(&hypre_CsrsvDataInfoU(data)) );

#elif defined(HYPRE_USING_ROCSPARSE)
   HYPRE_ROCSPARSE_CALL( rocsparse_create_mat_info(&(hypre_CsrsvDataInfoL(data)) ) );
   HYPRE_ROCSPARSE_CALL( rocsparse_create_mat_info(&(hypre_CsrsvDataInfoU(data)) ) );
#endif

   hypre_CsrsvDataAnalyzedL(data) = 0;
   hypre_CsrsvDataAnalyzedU(data) = 0;

   return data;
}

/*--------------------------------------------------------------------------
 * hypre_CsrsvDataDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CsrsvDataDestroy(hypre_CsrsvData* data)
{
   if (data)
   {
#if !defined(HYPRE_USING_ONEMKLSPARSE)
      /* Lower matrix info */
      if (hypre_CsrsvDataInfoL(data))
      {
#if defined(HYPRE_USING_CUSPARSE)
         HYPRE_CUSPARSE_CALL( hypre_cusparseSpSV_destroyDescr(hypre_CsrsvDataInfoL(data)) );

#elif defined(HYPRE_USING_ROCSPARSE)
         HYPRE_ROCSPARSE_CALL( rocsparse_destroy_mat_info(hypre_CsrsvDataInfoL(data)) );
#endif
      }

      /* Upper matrix info */
      if (hypre_CsrsvDataInfoU(data))
      {
#if defined(HYPRE_USING_CUSPARSE)
         HYPRE_CUSPARSE_CALL( hypre_cusparseSpSV_destroyDescr(hypre_CsrsvDataInfoU(data)) );

#elif defined(HYPRE_USING_ROCSPARSE)
         HYPRE_ROCSPARSE_CALL( rocsparse_destroy_mat_info(hypre_CsrsvDataInfoU(data)) );
#endif
      }

      /* Buffers */
#if defined(HYPRE_USING_CUSPARSE) && (CUSPARSE_VERSION >= CUSPARSE_SPSV_VERSION)
      hypre_TFree(hypre_CsrsvDataBufferL(data), HYPRE_MEMORY_DEVICE);
      hypre_TFree(hypre_CsrsvDataBufferU(data), HYPRE_MEMORY_DEVICE);
#else
      hypre_TFree(hypre_CsrsvDataBuffer(data), HYPRE_MEMORY_DEVICE);
#endif
#endif // #if !defined(HYPRE_USING_ONEMKLSPARSE)
      hypre_TFree(hypre_CsrsvDataMatData(data), HYPRE_MEMORY_DEVICE);

      /* Free data structure pointer */
      hypre_TFree(data, HYPRE_MEMORY_HOST);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_GpuMatDataCreate
 *--------------------------------------------------------------------------*/

hypre_GpuMatData *
hypre_GpuMatDataCreate()
{
   hypre_GpuMatData *data = hypre_CTAlloc(hypre_GpuMatData, 1, HYPRE_MEMORY_HOST);

#if defined(HYPRE_USING_CUSPARSE)
   cusparseMatDescr_t mat_descr;

   HYPRE_CUSPARSE_CALL( cusparseCreateMatDescr(&mat_descr) );
   HYPRE_CUSPARSE_CALL( cusparseSetMatType(mat_descr, CUSPARSE_MATRIX_TYPE_GENERAL) );
   HYPRE_CUSPARSE_CALL( cusparseSetMatIndexBase(mat_descr, CUSPARSE_INDEX_BASE_ZERO) );
   hypre_GpuMatDataMatDescr(data) = mat_descr;

#elif defined(HYPRE_USING_ROCSPARSE)
   rocsparse_mat_descr mat_descr;
   rocsparse_mat_info  info;

   HYPRE_ROCSPARSE_CALL( rocsparse_create_mat_descr(&mat_descr) );
   HYPRE_ROCSPARSE_CALL( rocsparse_set_mat_type(mat_descr, rocsparse_matrix_type_general) );
   HYPRE_ROCSPARSE_CALL( rocsparse_set_mat_index_base(mat_descr, rocsparse_index_base_zero) );
   HYPRE_ROCSPARSE_CALL( rocsparse_create_mat_info(&info) );

   hypre_GpuMatDataMatDescr(data) = mat_descr;
   hypre_GpuMatDataMatInfo(data) = info;

#elif defined(HYPRE_USING_ONEMKLSPARSE)
   oneapi::mkl::sparse::matrix_handle_t mat_handle;
   HYPRE_ONEMKL_CALL( oneapi::mkl::sparse::init_matrix_handle(&mat_handle) );
   hypre_GpuMatDataMatHandle(data) = mat_handle;
#endif

   return data;
}

/*--------------------------------------------------------------------------
 * hypre_GPUMatDataSetCSRData
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_GPUMatDataSetCSRData(hypre_CSRMatrix *matrix)
{

#if defined(HYPRE_USING_ONEMKLSPARSE)
#if defined(HYPRE_BIGINT)
   HYPRE_ONEMKL_CALL( oneapi::mkl::sparse::set_csr_data(hypre_CSRMatrixGPUMatHandle(matrix),
                                                        hypre_CSRMatrixNumRows(matrix),
                                                        hypre_CSRMatrixNumCols(matrix),
                                                        oneapi::mkl::index_base::zero,
                                                        reinterpret_cast<std::int64_t*>(hypre_CSRMatrixI(matrix)),
                                                        reinterpret_cast<std::int64_t*>(hypre_CSRMatrixJ(matrix)),
                                                        hypre_CSRMatrixData(matrix)) );
#else
   HYPRE_ONEMKL_CALL( oneapi::mkl::sparse::set_csr_data(hypre_CSRMatrixGPUMatHandle(matrix),
                                                        hypre_CSRMatrixNumRows(matrix),
                                                        hypre_CSRMatrixNumCols(matrix),
                                                        oneapi::mkl::index_base::zero,
                                                        hypre_CSRMatrixI(matrix),
                                                        hypre_CSRMatrixJ(matrix),
                                                        hypre_CSRMatrixData(matrix)) );
#endif
#endif

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_GpuMatDataDestroy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_GpuMatDataDestroy(hypre_GpuMatData *data)
{
   if (data)
   {
#if defined(HYPRE_USING_CUSPARSE)
      HYPRE_CUSPARSE_CALL( cusparseDestroyMatDescr(hypre_GpuMatDataMatDescr(data)) );
      hypre_TFree(hypre_GpuMatDataSpMVBuffer(data), HYPRE_MEMORY_DEVICE);

#elif defined(HYPRE_USING_ROCSPARSE)
      HYPRE_ROCSPARSE_CALL( rocsparse_destroy_mat_descr(hypre_GpuMatDataMatDescr(data)) );
      HYPRE_ROCSPARSE_CALL( rocsparse_destroy_mat_info(hypre_GpuMatDataMatInfo(data)) );

#elif defined(HYPRE_USING_ONEMKLSPARSE)
      HYPRE_ONEMKL_CALL( oneapi::mkl::sparse::release_matrix_handle(&hypre_GpuMatDataMatHandle(data)) );
#endif
   }

   hypre_TFree(data, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

#endif /* #if defined(HYPRE_USING_CUSPARSE) || defined(HYPRE_USING_ROCSPARSE) || defined(HYPRE_USING_ONEMKLSPARSE) */

#if defined(HYPRE_USING_GPU)

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixAddDevice
 *--------------------------------------------------------------------------*/

hypre_CSRMatrix*
hypre_CSRMatrixAddDevice( HYPRE_Complex    alpha,
                          hypre_CSRMatrix *A,
                          HYPRE_Complex    beta,
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
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Warning! Incompatible matrix dimensions!\n");

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

   hypre_SyncComputeStream(hypre_handle());

   return C;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixMultiplyDevice
 *--------------------------------------------------------------------------*/

hypre_CSRMatrix*
hypre_CSRMatrixMultiplyDevice( hypre_CSRMatrix *A,
                               hypre_CSRMatrix *B )
{
   HYPRE_Int         ncols_A  = hypre_CSRMatrixNumCols(A);
   HYPRE_Int         nrows_B  = hypre_CSRMatrixNumRows(B);
   hypre_CSRMatrix  *C;

   if (ncols_A != nrows_B)
   {
      hypre_printf("Warning! incompatible matrix dimensions!\n");
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Warning! incompatible matrix dimensions!\n");

      return NULL;
   }

   hypre_GpuProfilingPushRange("CSRMatrixMultiply");

   hypreDevice_CSRSpGemm(A, B, &C);

   hypre_SyncComputeStream(hypre_handle());

   hypre_GpuProfilingPopRange();

   return C;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixTripleMultiplyDevice
 *--------------------------------------------------------------------------*/

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

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixSplitDevice
 *
 * Split CSR matrix B_ext (extended rows of parcsr B) into diag part and
 * offd part corresponding to B.
 *
 * Input  - col_map_offd_B:
 * Output - col_map_offd_C: union of col_map_offd_B and offd-indices of
 *                          Bext_offd
 *          map_B_to_C: mapping from col_map_offd_B to col_map_offd_C
 *--------------------------------------------------------------------------*/

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
   hypre_CSRMatrix *B_ext_diag = hypre_CSRMatrixCreate(num_rows,
                                                       last_col_diag_B - first_col_diag_B + 1, B_ext_diag_nnz);
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

   hypre_SyncComputeStream(hypre_handle());

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixMergeColMapOffd
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixMergeColMapOffd( HYPRE_Int      num_cols_offd_B,
                                HYPRE_BigInt  *col_map_offd_B,
                                HYPRE_Int      B_ext_offd_nnz,
                                HYPRE_BigInt  *B_ext_offd_bigj,
                                HYPRE_Int     *num_cols_offd_C_ptr,
                                HYPRE_BigInt **col_map_offd_C_ptr,
                                HYPRE_Int    **map_B_to_C_ptr )
{
   /* offd map of B_ext_offd Union col_map_offd_B */
   HYPRE_BigInt *col_map_offd_C = hypre_TAlloc(HYPRE_BigInt, B_ext_offd_nnz + num_cols_offd_B,
                                               HYPRE_MEMORY_DEVICE);

   hypre_TMemcpy(col_map_offd_C, B_ext_offd_bigj, HYPRE_BigInt, B_ext_offd_nnz,
                 HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

   hypre_TMemcpy(col_map_offd_C + B_ext_offd_nnz, col_map_offd_B, HYPRE_BigInt, num_cols_offd_B,
                 HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

#if defined(HYPRE_USING_SYCL)
   HYPRE_ONEDPL_CALL( std::sort,
                      col_map_offd_C,
                      col_map_offd_C + B_ext_offd_nnz + num_cols_offd_B );

   HYPRE_BigInt *new_end = HYPRE_ONEDPL_CALL( std::unique,
                                              col_map_offd_C,
                                              col_map_offd_C + B_ext_offd_nnz + num_cols_offd_B );
#else
   HYPRE_THRUST_CALL( sort,
                      col_map_offd_C,
                      col_map_offd_C + B_ext_offd_nnz + num_cols_offd_B );

   HYPRE_BigInt *new_end = HYPRE_THRUST_CALL( unique,
                                              col_map_offd_C,
                                              col_map_offd_C + B_ext_offd_nnz + num_cols_offd_B );
#endif

   HYPRE_Int num_cols_offd_C = new_end - col_map_offd_C;

#if 1
   HYPRE_BigInt *tmp = hypre_TAlloc(HYPRE_BigInt, num_cols_offd_C, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(tmp, col_map_offd_C, HYPRE_BigInt, num_cols_offd_C, HYPRE_MEMORY_DEVICE,
                 HYPRE_MEMORY_DEVICE);
   hypre_TFree(col_map_offd_C, HYPRE_MEMORY_DEVICE);
   col_map_offd_C = tmp;
#else
   col_map_offd_C = hypre_TReAlloc_v2(col_map_offd_C, HYPRE_BigInt, B_ext_offd_nnz + num_cols_offd_B,
                                      HYPRE_Int, num_cols_offd_C, HYPRE_MEMORY_DEVICE);
#endif

   /* create map from col_map_offd_B */
   HYPRE_Int *map_B_to_C = hypre_TAlloc(HYPRE_Int, num_cols_offd_B, HYPRE_MEMORY_DEVICE);

   if (num_cols_offd_B)
   {
#if defined(HYPRE_USING_SYCL)
      HYPRE_ONEDPL_CALL( oneapi::dpl::lower_bound,
                         col_map_offd_C,
                         col_map_offd_C + num_cols_offd_C,
                         col_map_offd_B,
                         col_map_offd_B + num_cols_offd_B,
                         map_B_to_C );
#else
      HYPRE_THRUST_CALL( lower_bound,
                         col_map_offd_C,
                         col_map_offd_C + num_cols_offd_C,
                         col_map_offd_B,
                         col_map_offd_B + num_cols_offd_B,
                         map_B_to_C );
#endif
   }

   *map_B_to_C_ptr = map_B_to_C;
   *num_cols_offd_C_ptr = num_cols_offd_C;
   *col_map_offd_C_ptr  = col_map_offd_C;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixSplitDevice_core
 *
 * job = 0: query B_ext_diag/offd_nnz; 1: real computation
 *
 * NOTES:
 *   B_ext_ii: NOT row pointers of CSR but row indices of COO
 *   B_ext_bigj: [BigInt] global column indices
 *   B_ext_xata: companion data with B_ext_data; NULL if none
 *   B_ext_diag_ii: memory allocated outside
 *   B_ext_diag_xata: companion with B_ext_diag_data_ptr; NULL if none
 *   B_ext_offd_ii: memory allocated outside
 *   B_ext_offd_xata: companion with B_ext_offd_data_ptr; NULL if none
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixSplitDevice_core( HYPRE_Int      job,
                                 HYPRE_Int      num_rows,
                                 HYPRE_Int      B_ext_nnz,
                                 HYPRE_Int     *B_ext_ii,
                                 HYPRE_BigInt  *B_ext_bigj,
                                 HYPRE_Complex *B_ext_data,
                                 char          *B_ext_xata,
                                 HYPRE_BigInt   first_col_diag_B,
                                 HYPRE_BigInt   last_col_diag_B,
                                 HYPRE_Int      num_cols_offd_B,
                                 HYPRE_BigInt  *col_map_offd_B,
                                 HYPRE_Int    **map_B_to_C_ptr,
                                 HYPRE_Int     *num_cols_offd_C_ptr,
                                 HYPRE_BigInt **col_map_offd_C_ptr,
                                 HYPRE_Int     *B_ext_diag_nnz_ptr,
                                 HYPRE_Int     *B_ext_diag_ii,
                                 HYPRE_Int     *B_ext_diag_j,
                                 HYPRE_Complex *B_ext_diag_data,
                                 char          *B_ext_diag_xata,
                                 HYPRE_Int     *B_ext_offd_nnz_ptr,
                                 HYPRE_Int     *B_ext_offd_ii,
                                 HYPRE_Int     *B_ext_offd_j,
                                 HYPRE_Complex *B_ext_offd_data,
                                 char          *B_ext_offd_xata )
{
   HYPRE_Int      B_ext_diag_nnz;
   HYPRE_Int      B_ext_offd_nnz;
   HYPRE_BigInt  *B_ext_diag_bigj = NULL;
   HYPRE_BigInt  *B_ext_offd_bigj = NULL;
   HYPRE_BigInt  *col_map_offd_C;
   HYPRE_Int     *map_B_to_C = NULL;
   HYPRE_Int      num_cols_offd_C;

   hypre_GpuProfilingPushRange("CSRMatrixSplitDevice_core");

   in_range<HYPRE_BigInt> pred1(first_col_diag_B, last_col_diag_B);

   /* get diag and offd nnz */
   if (job == 0)
   {
      /* query the nnz's */
#if defined(HYPRE_USING_SYCL)
      B_ext_diag_nnz = HYPRE_ONEDPL_CALL( std::count_if,
                                          B_ext_bigj,
                                          B_ext_bigj + B_ext_nnz,
                                          pred1 );
#else
      B_ext_diag_nnz = HYPRE_THRUST_CALL( count_if,
                                          B_ext_bigj,
                                          B_ext_bigj + B_ext_nnz,
                                          pred1 );
#endif
      B_ext_offd_nnz = B_ext_nnz - B_ext_diag_nnz;

      *B_ext_diag_nnz_ptr = B_ext_diag_nnz;
      *B_ext_offd_nnz_ptr = B_ext_offd_nnz;

      hypre_GpuProfilingPopRange();

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
#if defined(HYPRE_USING_SYCL)
      auto first = oneapi::dpl::make_zip_iterator(B_ext_ii, B_ext_bigj, B_ext_data, B_ext_xata);
      auto new_end = hypreSycl_copy_if(
                        first,                                                             /* first   */
                        first + B_ext_nnz,                                                 /* last    */
                        B_ext_bigj,                                                        /* stencil */
                        oneapi::dpl::make_zip_iterator(B_ext_diag_ii, B_ext_diag_bigj, B_ext_diag_data,
                                                       B_ext_diag_xata),                   /* result  */
                        pred1 );
      hypre_assert( std::get<0>(new_end.base()) == B_ext_diag_ii + B_ext_diag_nnz );
#else
      auto new_end = HYPRE_THRUST_CALL(
                        copy_if,
                        thrust::make_zip_iterator(thrust::make_tuple(B_ext_ii,      B_ext_bigj,      B_ext_data,
                                                                     B_ext_xata)),             /* first */
                        thrust::make_zip_iterator(thrust::make_tuple(B_ext_ii,      B_ext_bigj,      B_ext_data,
                                                                     B_ext_xata)) + B_ext_nnz, /* last */
                        B_ext_bigj,                                                            /* stencil */
                        thrust::make_zip_iterator(thrust::make_tuple(B_ext_diag_ii, B_ext_diag_bigj, B_ext_diag_data,
                                                                     B_ext_diag_xata)),        /* result */
                        pred1 );

      hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) == B_ext_diag_ii + B_ext_diag_nnz );
#endif
   }
   else
   {
#if defined(HYPRE_USING_SYCL)
      auto first = oneapi::dpl::make_zip_iterator(B_ext_ii, B_ext_bigj, B_ext_data);
      auto new_end = hypreSycl_copy_if(
                        first,                                                                /* first   */
                        first + B_ext_nnz,                                                    /* last    */
                        B_ext_bigj,                                                           /* stencil */
                        oneapi::dpl::make_zip_iterator(B_ext_diag_ii, B_ext_diag_bigj, B_ext_diag_data),   /* result  */
                        pred1 );
      hypre_assert( std::get<0>(new_end.base()) == B_ext_diag_ii + B_ext_diag_nnz );
#else
      auto new_end = HYPRE_THRUST_CALL(
                        copy_if,
                        thrust::make_zip_iterator(thrust::make_tuple(B_ext_ii,      B_ext_bigj,
                                                                     B_ext_data)),             /* first */
                        thrust::make_zip_iterator(thrust::make_tuple(B_ext_ii,      B_ext_bigj,
                                                                     B_ext_data)) + B_ext_nnz, /* last */
                        B_ext_bigj,                                                            /* stencil */
                        thrust::make_zip_iterator(thrust::make_tuple(B_ext_diag_ii, B_ext_diag_bigj,
                                                                     B_ext_diag_data)),        /* result */
                        pred1 );

      hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) == B_ext_diag_ii + B_ext_diag_nnz );
#endif
   }

#if defined(HYPRE_USING_SYCL)
   HYPRE_ONEDPL_CALL( std::transform,
                      B_ext_diag_bigj,
                      B_ext_diag_bigj + B_ext_diag_nnz,
                      B_ext_diag_j,
   [const_val = first_col_diag_B](const auto & x) {return x - const_val;} );
#else
   HYPRE_THRUST_CALL( transform,
                      B_ext_diag_bigj,
                      B_ext_diag_bigj + B_ext_diag_nnz,
                      thrust::make_constant_iterator(first_col_diag_B),
                      B_ext_diag_j,
                      thrust::minus<HYPRE_BigInt>());
#endif
   hypre_TFree(B_ext_diag_bigj, HYPRE_MEMORY_DEVICE);

   /* copy to offd */
   B_ext_offd_bigj = hypre_TAlloc(HYPRE_BigInt, B_ext_offd_nnz, HYPRE_MEMORY_DEVICE);

   if (B_ext_offd_xata)
   {
#if defined(HYPRE_USING_SYCL)
      auto first = oneapi::dpl::make_zip_iterator(B_ext_ii, B_ext_bigj, B_ext_data, B_ext_xata);
      auto new_end = hypreSycl_copy_if(
                        first,                                           /* first */
                        first + B_ext_nnz,                               /* last */
                        B_ext_bigj,                                      /* stencil */
                        oneapi::dpl::make_zip_iterator(B_ext_offd_ii, B_ext_offd_bigj, B_ext_offd_data,
                                                       B_ext_offd_xata), /* result */
                        std::not_fn(pred1) );
      hypre_assert( std::get<0>(new_end.base()) == B_ext_offd_ii + B_ext_offd_nnz );
#else
      auto new_end = HYPRE_THRUST_CALL(
                        copy_if,
                        thrust::make_zip_iterator(thrust::make_tuple(B_ext_ii,      B_ext_bigj,      B_ext_data,
                                                                     B_ext_xata)),             /* first */
                        thrust::make_zip_iterator(thrust::make_tuple(B_ext_ii,      B_ext_bigj,      B_ext_data,
                                                                     B_ext_xata)) + B_ext_nnz, /* last */
                        B_ext_bigj,                                                            /* stencil */
                        thrust::make_zip_iterator(thrust::make_tuple(B_ext_offd_ii, B_ext_offd_bigj, B_ext_offd_data,
                                                                     B_ext_offd_xata)),        /* result */
                        thrust::not1(pred1) );

      hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) == B_ext_offd_ii + B_ext_offd_nnz );
#endif
   }
   else
   {
#if defined(HYPRE_USING_SYCL)
      auto first = oneapi::dpl::make_zip_iterator(B_ext_ii, B_ext_bigj, B_ext_data);
      auto new_end = hypreSycl_copy_if(
                        first,                                                              /* first   */
                        first + B_ext_nnz,                                                  /* last    */
                        B_ext_bigj,                                                         /* stencil */
                        oneapi::dpl::make_zip_iterator(B_ext_offd_ii, B_ext_offd_bigj, B_ext_offd_data), /* result  */
                        std::not_fn(pred1) );
      hypre_assert( std::get<0>(new_end.base()) == B_ext_offd_ii + B_ext_offd_nnz );
#else
      auto new_end = HYPRE_THRUST_CALL(
                        copy_if,
                        thrust::make_zip_iterator(thrust::make_tuple(B_ext_ii,      B_ext_bigj,
                                                                     B_ext_data)),             /* first */
                        thrust::make_zip_iterator(thrust::make_tuple(B_ext_ii,      B_ext_bigj,
                                                                     B_ext_data)) + B_ext_nnz, /* last */
                        B_ext_bigj,                                                            /* stencil */
                        thrust::make_zip_iterator(thrust::make_tuple(B_ext_offd_ii, B_ext_offd_bigj,
                                                                     B_ext_offd_data)),        /* result */
                        thrust::not1(pred1) );

      hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) == B_ext_offd_ii + B_ext_offd_nnz );
#endif
   }

   hypre_CSRMatrixMergeColMapOffd(num_cols_offd_B, col_map_offd_B, B_ext_offd_nnz, B_ext_offd_bigj,
                                  &num_cols_offd_C, &col_map_offd_C, &map_B_to_C);

#if defined(HYPRE_USING_SYCL)
   if (num_cols_offd_C > 0 && B_ext_offd_nnz > 0)
   {
      HYPRE_ONEDPL_CALL( oneapi::dpl::lower_bound,
                         col_map_offd_C,
                         col_map_offd_C + num_cols_offd_C,
                         B_ext_offd_bigj,
                         B_ext_offd_bigj + B_ext_offd_nnz,
                         B_ext_offd_j );
   }
#else
   HYPRE_THRUST_CALL( lower_bound,
                      col_map_offd_C,
                      col_map_offd_C + num_cols_offd_C,
                      B_ext_offd_bigj,
                      B_ext_offd_bigj + B_ext_offd_nnz,
                      B_ext_offd_j );
#endif

   hypre_TFree(B_ext_offd_bigj, HYPRE_MEMORY_DEVICE);

   *map_B_to_C_ptr = map_B_to_C;
   *num_cols_offd_C_ptr = num_cols_offd_C;
   *col_map_offd_C_ptr  = col_map_offd_C;

   hypre_GpuProfilingPopRange();

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixCompressColumnsDevice
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixCompressColumnsDevice(hypre_CSRMatrix  *A,
                                     HYPRE_BigInt     *col_map,
                                     HYPRE_Int       **col_idx_new_ptr,
                                     HYPRE_BigInt    **col_map_new_ptr)
{
   HYPRE_Int  num_cols = hypre_CSRMatrixNumCols(A);
   HYPRE_Int  nnz      = hypre_CSRMatrixNumNonzeros(A);
   HYPRE_Int *A_j      = hypre_CSRMatrixJ(A);
   HYPRE_Int *tmp_j    = hypre_TAlloc(HYPRE_Int, nnz, HYPRE_MEMORY_DEVICE);
   HYPRE_Int *tmp_end;
   HYPRE_Int  num_cols_new;

   hypre_TMemcpy(tmp_j, A_j, HYPRE_Int, nnz, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
#if defined(HYPRE_USING_SYCL)
   HYPRE_ONEDPL_CALL(std::sort, tmp_j, tmp_j + nnz);
   tmp_end = HYPRE_ONEDPL_CALL(std::unique, tmp_j, tmp_j + nnz);
#else
   HYPRE_THRUST_CALL(sort, tmp_j, tmp_j + nnz);
   tmp_end = HYPRE_THRUST_CALL(unique, tmp_j, tmp_j + nnz);
#endif
   num_cols_new = tmp_end - tmp_j;

   hypre_assert(num_cols_new <= num_cols);

   if (num_cols_new < num_cols)
   {
      HYPRE_Int    *offd_mark = NULL;
      HYPRE_BigInt *col_map_new;

      if (num_cols_new)
      {
         offd_mark = hypre_TAlloc(HYPRE_Int, num_cols, HYPRE_MEMORY_DEVICE);
      }

      if (col_map_new_ptr)
      {
         col_map_new = hypre_TAlloc(HYPRE_BigInt, num_cols_new, HYPRE_MEMORY_DEVICE);
      }

#if defined(HYPRE_USING_SYCL)
      oneapi::dpl::counting_iterator count(0);
      hypreSycl_scatter( count,
                         count + num_cols_new,
                         tmp_j,
                         offd_mark );

      hypreSycl_gather(A_j, A_j + nnz, offd_mark, A_j);

      if (col_map_new_ptr)
      {
         hypreSycl_gather(tmp_j, tmp_j + num_cols_new, col_map, col_map_new);
      }
#else
      HYPRE_THRUST_CALL( scatter,
                         thrust::counting_iterator<HYPRE_Int>(0),
                         thrust::counting_iterator<HYPRE_Int>(num_cols_new),
                         tmp_j,
                         offd_mark );

      HYPRE_THRUST_CALL(gather, A_j, A_j + nnz, offd_mark, A_j);

      if (col_map_new_ptr)
      {
         HYPRE_THRUST_CALL(gather, tmp_j, tmp_j + num_cols_new, col_map, col_map_new);
      }
#endif

      hypre_TFree(offd_mark, HYPRE_MEMORY_DEVICE);

      hypre_CSRMatrixNumCols(A) = num_cols_new;

      if (col_idx_new_ptr)
      {
         *col_idx_new_ptr = tmp_j;
      }
      else
      {
         hypre_TFree(tmp_j, HYPRE_MEMORY_DEVICE);
      }

      if (col_map_new_ptr)
      {
         *col_map_new_ptr = col_map_new;
      }
   }
   else
   {
      if (col_idx_new_ptr)
      {
         *col_idx_new_ptr = NULL;
      }

      if (col_map_new_ptr)
      {
         *col_map_new_ptr = NULL;
      }

      hypre_TFree(tmp_j, HYPRE_MEMORY_DEVICE);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixTriLowerUpperSolveDevice_core
 *
 * TODO (VPM): The analysis portion (setup phase) of the triangular solve
 *             is embedded into the vendor libraries wrappers.
 *             Should we create a separate function "Setup" function?
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixTriLowerUpperSolveDevice_core(char             uplo,
                                             HYPRE_Int        unit_diag,
                                             hypre_CSRMatrix *A,
                                             HYPRE_Real      *l1_norms,
                                             hypre_Vector    *f,
                                             HYPRE_Int        offset_f,
                                             hypre_Vector    *u,
                                             HYPRE_Int        offset_u)
{
   /* Trivial case: no rows */
   if (hypre_CSRMatrixNumRows(A) <= 0)
   {
      return hypre_error_flag;
   }

   /* Trivial case: empty rows */
   if (hypre_CSRMatrixNumNonzeros(A) <= 0)
   {
      return hypre_error_flag;
   }

   /* Sanity check */
   if (hypre_CSRMatrixNumRows(A) != hypre_CSRMatrixNumCols(A))
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                        "Triangular matrix solver works only for square matrices!");
      return hypre_error_flag;
   }

   /* Call vendor specific implementations */
#if defined(HYPRE_USING_CUSPARSE)
   hypre_CSRMatrixTriLowerUpperSolveCusparse(uplo, unit_diag, A,
                                             l1_norms,
                                             hypre_VectorData(f) + offset_f,
                                             hypre_VectorData(u) + offset_u);
#elif defined(HYPRE_USING_ROCSPARSE)
   hypre_CSRMatrixTriLowerUpperSolveRocsparse(uplo, unit_diag, A,
                                              l1_norms,
                                              hypre_VectorData(f) + offset_f,
                                              hypre_VectorData(u) + offset_u);
#elif defined(HYPRE_USING_ONEMKLSPARSE)
   hypre_CSRMatrixTriLowerUpperSolveOnemklsparse(uplo, unit_diag, A,
                                                 l1_norms,
                                                 hypre_VectorData(f) + offset_f,
                                                 hypre_VectorData(u) + offset_u);
#else
   hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                     "hypre_CSRMatrixTriLowerUpperSolveDevice requires configuration with either cuSPARSE or rocSPARSE\n");
#endif

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixTriLowerUpperSolveDevice
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixTriLowerUpperSolveDevice(char             uplo,
                                        HYPRE_Int        unit_diag,
                                        hypre_CSRMatrix *A,
                                        HYPRE_Real      *l1_norms,
                                        hypre_Vector    *f,
                                        hypre_Vector    *u )
{
   return hypre_CSRMatrixTriLowerUpperSolveDevice_core(uplo, unit_diag, A, l1_norms, f, 0, u, 0);
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixAddPartial
 *
 * Adds matrix rows in the CSR matrix B to the CSR Matrix A, where row_nums[i]
 * defines to which row of A the i-th row of B is added, and returns a CSR
 * Matrix C. Repeated row indices are allowed in row_nums
 *
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
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Warning! incompatible matrix dimensions!\n");

      return NULL;
   }

   hypreDevice_CSRSpAdd(nrows_A, nrows_B, ncols_A, nnz_A, nnz_B,
                        A_i, A_j, 1.0, A_data, NULL, B_i, B_j,
                        1.0, B_data, NULL, row_nums,
                        &nnzC, &C_i, &C_j, &C_data);

   C = hypre_CSRMatrixCreate(nrows_A, ncols_B, nnzC);
   hypre_CSRMatrixI(C) = C_i;
   hypre_CSRMatrixJ(C) = C_j;
   hypre_CSRMatrixData(C) = C_data;
   hypre_CSRMatrixMemoryLocation(C) = HYPRE_MEMORY_DEVICE;

   hypre_SyncComputeStream(hypre_handle());

   return C;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixColNNzRealDevice
 *--------------------------------------------------------------------------*/

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

#if defined(HYPRE_USING_SYCL)
   HYPRE_ONEDPL_CALL(std::sort, A_j_sorted, A_j_sorted + nnz_A);
#else
   HYPRE_THRUST_CALL(sort, A_j_sorted, A_j_sorted + nnz_A);
#endif

   reduced_col_indices = hypre_TAlloc(HYPRE_Int, ncols_A, HYPRE_MEMORY_DEVICE);
   reduced_col_nnz     = hypre_TAlloc(HYPRE_Int, ncols_A, HYPRE_MEMORY_DEVICE);

#if defined(HYPRE_USING_SYCL)

   /* WM: todo - better way to get around lack of constant iterator in DPL? */
   HYPRE_Int *ones = hypre_TAlloc(HYPRE_Int, nnz_A, HYPRE_MEMORY_DEVICE);
   HYPRE_ONEDPL_CALL( std::fill_n, ones, nnz_A, 1 );
   auto new_end = HYPRE_ONEDPL_CALL( oneapi::dpl::reduce_by_segment,
                                     A_j_sorted,
                                     A_j_sorted + nnz_A,
                                     ones,
                                     reduced_col_indices,
                                     reduced_col_nnz);

   hypre_TFree(ones, HYPRE_MEMORY_DEVICE);
   hypre_assert(new_end.first - reduced_col_indices == new_end.second - reduced_col_nnz);
   num_reduced_col_indices = new_end.first - reduced_col_indices;
#else
   thrust::pair<HYPRE_Int*, HYPRE_Int*> new_end =
      HYPRE_THRUST_CALL(reduce_by_key, A_j_sorted, A_j_sorted + nnz_A,
                        thrust::make_constant_iterator(1),
                        reduced_col_indices,
                        reduced_col_nnz);
   hypre_assert(new_end.first - reduced_col_indices == new_end.second - reduced_col_nnz);
   num_reduced_col_indices = new_end.first - reduced_col_indices;
#endif


   hypre_Memset(colnnz, 0, ncols_A * sizeof(HYPRE_Real), HYPRE_MEMORY_DEVICE);
#if defined(HYPRE_USING_SYCL)
   HYPRE_ONEDPL_CALL( std::copy, reduced_col_nnz, reduced_col_nnz + num_reduced_col_indices,
                      oneapi::dpl::make_permutation_iterator(colnnz, reduced_col_indices) );
#else
   HYPRE_THRUST_CALL(scatter, reduced_col_nnz, reduced_col_nnz + num_reduced_col_indices,
                     reduced_col_indices, colnnz);
#endif

   hypre_TFree(A_j_sorted,          HYPRE_MEMORY_DEVICE);
   hypre_TFree(reduced_col_indices, HYPRE_MEMORY_DEVICE);
   hypre_TFree(reduced_col_nnz,     HYPRE_MEMORY_DEVICE);

   hypre_SyncComputeStream(hypre_handle());

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypreGPUKernel_CSRMoveDiagFirst
 *--------------------------------------------------------------------------*/

__global__ void
hypreGPUKernel_CSRMoveDiagFirst( hypre_DeviceItem    &item,
                                 HYPRE_Int      nrows,
                                 HYPRE_Int     *ia,
                                 HYPRE_Int     *ja,
                                 HYPRE_Complex *aa )
{
   HYPRE_Int row  = hypre_gpu_get_grid_warp_id<1, 1>(item);
   HYPRE_Int lane = hypre_gpu_get_lane_id<1>(item);

   if (row >= nrows)
   {
      return;
   }

   HYPRE_Int p = 0, q = 0;

   if (lane < 2)
   {
      p = read_only_load(ia + row + lane);
   }

   q = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p, 1);
   p = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p, 0);

   for (HYPRE_Int j = p + lane + 1; warp_any_sync(item, HYPRE_WARP_FULL_MASK, j < q);
        j += HYPRE_WARP_SIZE)
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

      if ( warp_any_sync(item, HYPRE_WARP_FULL_MASK, find_diag) )
      {
         break;
      }
   }
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixMoveDiagFirstDevice
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixMoveDiagFirstDevice( hypre_CSRMatrix  *A )
{
   HYPRE_Int      nrows  = hypre_CSRMatrixNumRows(A);
   HYPRE_Complex *A_data = hypre_CSRMatrixData(A);
   HYPRE_Int     *A_i    = hypre_CSRMatrixI(A);
   HYPRE_Int     *A_j    = hypre_CSRMatrixJ(A);

   dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = hypre_GetDefaultDeviceGridDimension(nrows, "warp", bDim);

   HYPRE_GPU_LAUNCH(hypreGPUKernel_CSRMoveDiagFirst, gDim, bDim,
                    nrows, A_i, A_j, A_data);

   hypre_SyncComputeStream(hypre_handle());

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixStack2Device
 *
 * return C = [A; B]
 *--------------------------------------------------------------------------*/

hypre_CSRMatrix*
hypre_CSRMatrixStack2Device(hypre_CSRMatrix *A, hypre_CSRMatrix *B)
{
   hypre_GpuProfilingPushRange("CSRMatrixStack2");

   hypre_assert( hypre_CSRMatrixNumCols(A) == hypre_CSRMatrixNumCols(B) );

   hypre_CSRMatrix *C = hypre_CSRMatrixCreate( hypre_CSRMatrixNumRows(A) + hypre_CSRMatrixNumRows(B),
                                               hypre_CSRMatrixNumCols(A),
                                               hypre_CSRMatrixNumNonzeros(A) + hypre_CSRMatrixNumNonzeros(B) );

   HYPRE_Int     *C_i = hypre_TAlloc(HYPRE_Int,     hypre_CSRMatrixNumRows(C) + 1,
                                     HYPRE_MEMORY_DEVICE);
   HYPRE_Int     *C_j = hypre_TAlloc(HYPRE_Int,     hypre_CSRMatrixNumNonzeros(C),
                                     HYPRE_MEMORY_DEVICE);
   HYPRE_Complex *C_a = hypre_TAlloc(HYPRE_Complex, hypre_CSRMatrixNumNonzeros(C),
                                     HYPRE_MEMORY_DEVICE);

   hypre_TMemcpy(C_i, hypre_CSRMatrixI(A), HYPRE_Int, hypre_CSRMatrixNumRows(A) + 1,
                 HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(C_i + hypre_CSRMatrixNumRows(A) + 1, hypre_CSRMatrixI(B) + 1, HYPRE_Int,
                 hypre_CSRMatrixNumRows(B),
                 HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
#if defined(HYPRE_USING_SYCL)
   HYPRE_ONEDPL_CALL( std::transform,
                      C_i + hypre_CSRMatrixNumRows(A) + 1,
                      C_i + hypre_CSRMatrixNumRows(C) + 1,
                      C_i + hypre_CSRMatrixNumRows(A) + 1,
   [const_val = hypre_CSRMatrixNumNonzeros(A)] (const auto & x) {return x + const_val;} );
#else
   HYPRE_THRUST_CALL( transform,
                      C_i + hypre_CSRMatrixNumRows(A) + 1,
                      C_i + hypre_CSRMatrixNumRows(C) + 1,
                      thrust::make_constant_iterator(hypre_CSRMatrixNumNonzeros(A)),
                      C_i + hypre_CSRMatrixNumRows(A) + 1,
                      thrust::plus<HYPRE_Int>() );
#endif

   hypre_TMemcpy(C_j, hypre_CSRMatrixJ(A), HYPRE_Int, hypre_CSRMatrixNumNonzeros(A),
                 HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(C_j + hypre_CSRMatrixNumNonzeros(A), hypre_CSRMatrixJ(B), HYPRE_Int,
                 hypre_CSRMatrixNumNonzeros(B),
                 HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

   hypre_TMemcpy(C_a, hypre_CSRMatrixData(A), HYPRE_Complex, hypre_CSRMatrixNumNonzeros(A),
                 HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(C_a + hypre_CSRMatrixNumNonzeros(A), hypre_CSRMatrixData(B), HYPRE_Complex,
                 hypre_CSRMatrixNumNonzeros(B),
                 HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

   hypre_CSRMatrixI(C) = C_i;
   hypre_CSRMatrixJ(C) = C_j;
   hypre_CSRMatrixData(C) = C_a;
   hypre_CSRMatrixMemoryLocation(C) = HYPRE_MEMORY_DEVICE;

   hypre_GpuProfilingPopRange();

   return C;
}

/*--------------------------------------------------------------------------
 * hypreGPUKernel_CSRRowSum
 *
 * type == 0, sum,
 *         1, abs sum (l-1)
 *         2, square sum (l-2)
 *--------------------------------------------------------------------------*/

template<HYPRE_Int type>
__global__ void
hypreGPUKernel_CSRRowSum( hypre_DeviceItem    &item,
                          HYPRE_Int      nrows,
                          HYPRE_Int     *ia,
                          HYPRE_Int     *ja,
                          HYPRE_Complex *aa,
                          HYPRE_Int     *CF_i,
                          HYPRE_Int     *CF_j,
                          HYPRE_Complex *row_sum,
                          HYPRE_Complex  scal,
                          HYPRE_Int      set)
{
   HYPRE_Int row_i = hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (row_i >= nrows)
   {
      return;
   }

   HYPRE_Int lane = hypre_gpu_get_lane_id<1>(item);
   HYPRE_Int p = 0, q = 0;

   if (lane < 2)
   {
      p = read_only_load(ia + row_i + lane);
   }
   q = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p, 1);
   p = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p, 0);

   HYPRE_Complex row_sum_i = 0.0;

   for (HYPRE_Int j = p + lane; j < q; j += HYPRE_WARP_SIZE)
   {
      if ( CF_i && CF_j && read_only_load(&CF_i[row_i]) != read_only_load(&CF_j[ja[j]]) )
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
         row_sum_i += hypre_abs(aii);
      }
      else if (type == 2)
      {
         row_sum_i += aii * aii;
      }
   }

   row_sum_i = warp_reduce_sum(item, row_sum_i);

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

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixComputeRowSumDevice
 *--------------------------------------------------------------------------*/

HYPRE_Int
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

   dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = hypre_GetDefaultDeviceGridDimension(nrows, "warp", bDim);

   HYPRE_Int set = set_or_add[0] == 's';
   if (type == 0)
   {
      HYPRE_GPU_LAUNCH( hypreGPUKernel_CSRRowSum<0>, gDim, bDim,
                        nrows, A_i, A_j, A_data,
                        CF_i, CF_j, row_sum, scal, set );
   }
   else if (type == 1)
   {
      HYPRE_GPU_LAUNCH( hypreGPUKernel_CSRRowSum<1>, gDim, bDim,
                        nrows, A_i, A_j, A_data,
                        CF_i, CF_j, row_sum, scal, set );
   }
   else if (type == 2)
   {
      HYPRE_GPU_LAUNCH( hypreGPUKernel_CSRRowSum<2>, gDim, bDim,
                        nrows, A_i, A_j, A_data, CF_i, CF_j,
                        row_sum, scal, set );
   }

   hypre_SyncComputeStream(hypre_handle());

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypreGPUKernel_CSRMatrixIntersectPattern
 *
 * mark is of size nA
 * diag_option: 1: special treatment for diag entries, mark as -2
 *--------------------------------------------------------------------------*/

__global__ void
hypreGPUKernel_CSRMatrixIntersectPattern(hypre_DeviceItem &item,
                                         HYPRE_Int  n,
                                         HYPRE_Int  nA,
                                         HYPRE_Int *rowid,
                                         HYPRE_Int *colid,
                                         HYPRE_Int *idx,
                                         HYPRE_Int *mark,
                                         HYPRE_Int  diag_option)
{
   HYPRE_Int i = hypre_gpu_get_grid_thread_id<1, 1>(item);

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

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixIntersectPattern
 *
 * markA: array of size nnz(A), for pattern of (A and B), markA is the
 * column indices as in A_J. Otherwise, mark pattern not in A-B as -1 in markA
 *
 * Note the special treatment for diagonal entries of A (marked as -2)
 *--------------------------------------------------------------------------*/

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
   hypre_TMemcpy(Cjj,        hypre_CSRMatrixJ(A), HYPRE_Int, nnzA, HYPRE_MEMORY_DEVICE,
                 HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(Cjj + nnzA, hypre_CSRMatrixJ(B), HYPRE_Int, nnzB, HYPRE_MEMORY_DEVICE,
                 HYPRE_MEMORY_DEVICE);

#if defined(HYPRE_USING_SYCL)
   hypreSycl_sequence(idx, idx + nnzA + nnzB, 0);

   auto zipped_begin = oneapi::dpl::make_zip_iterator(Cii, Cjj, idx);
   HYPRE_ONEDPL_CALL( std::stable_sort, zipped_begin, zipped_begin + nnzA + nnzB,
                      [](auto lhs, auto rhs)
   {
      if (std::get<0>(lhs) == std::get<0>(rhs))
      {
         return std::get<1>(lhs) < std::get<1>(rhs);
      }
      else
      {
         return std::get<0>(lhs) < std::get<0>(rhs);
      }
   } );
#else
   HYPRE_THRUST_CALL( sequence, idx, idx + nnzA + nnzB );

   HYPRE_THRUST_CALL( stable_sort_by_key,
                      thrust::make_zip_iterator(thrust::make_tuple(Cii, Cjj)),
                      thrust::make_zip_iterator(thrust::make_tuple(Cii, Cjj)) + nnzA + nnzB,
                      idx );
#endif

   hypre_TMemcpy(markA, hypre_CSRMatrixJ(A), HYPRE_Int, nnzA, HYPRE_MEMORY_DEVICE,
                 HYPRE_MEMORY_DEVICE);

   dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = hypre_GetDefaultDeviceGridDimension(nnzA + nnzB, "thread", bDim);

   HYPRE_GPU_LAUNCH( hypreGPUKernel_CSRMatrixIntersectPattern, gDim, bDim,
                     nnzA + nnzB, nnzA, Cii, Cjj, idx, markA, diag_opt );

   hypre_TFree(Cii, HYPRE_MEMORY_DEVICE);
   hypre_TFree(Cjj, HYPRE_MEMORY_DEVICE);
   hypre_TFree(idx, HYPRE_MEMORY_DEVICE);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypreGPUKernel_CSRExtractDiag
 *
 * type 0: diag
 *      1: abs diag
 *      2: diag inverse
 *      3: diag inverse sqrt
 *      4: abs diag inverse sqrt
 *--------------------------------------------------------------------------*/

__global__ void
hypreGPUKernel_CSRExtractDiag( hypre_DeviceItem    &item,
                               HYPRE_Int      nrows,
                               HYPRE_Int     *ia,
                               HYPRE_Int     *ja,
                               HYPRE_Complex *aa,
                               HYPRE_Complex *d,
                               HYPRE_Int      type)
{
   HYPRE_Int row = hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (row >= nrows)
   {
      return;
   }

   HYPRE_Int lane = hypre_gpu_get_lane_id<1>(item);
   HYPRE_Int p = 0, q = 0;

   if (lane < 2)
   {
      p = read_only_load(ia + row + lane);
   }
   q = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p, 1);
   p = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p, 0);

   HYPRE_Int has_diag = 0;

   for (HYPRE_Int j = p + lane;
        warp_any_sync(item, HYPRE_WARP_FULL_MASK, j < q);
        j += HYPRE_WARP_SIZE)
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
            d[row] = hypre_abs(aa[j]);
         }
         else
         {
            if (aa[j] == 0.0)
            {
               d[row] = 0.0;
            }
            else if (type == 2)
            {
               d[row] = 1.0 / aa[j];
            }
            else if (type == 3)
            {
               d[row] = 1.0 / hypre_sqrt(aa[j]);
            }
            else if (type == 4)
            {
               d[row] = 1.0 / hypre_sqrt(hypre_abs(aa[j]));
            }
         }
      }

      if (warp_any_sync(item, HYPRE_WARP_FULL_MASK, find_diag))
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

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixExtractDiagonalDevice
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixExtractDiagonalDevice( hypre_CSRMatrix *A,
                                      HYPRE_Complex   *d,
                                      HYPRE_Int        type)
{
   HYPRE_Int      nrows  = hypre_CSRMatrixNumRows(A);
   HYPRE_Complex *A_data = hypre_CSRMatrixData(A);
   HYPRE_Int     *A_i    = hypre_CSRMatrixI(A);
   HYPRE_Int     *A_j    = hypre_CSRMatrixJ(A);

   dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = hypre_GetDefaultDeviceGridDimension(nrows, "warp", bDim);

   HYPRE_GPU_LAUNCH( hypreGPUKernel_CSRExtractDiag, gDim, bDim, nrows,
                     A_i, A_j, A_data, d, type );

   hypre_SyncComputeStream(hypre_handle());

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypreGPUKernel_CSRCheckDiagFirst
 *
 * check if diagonal entry is the first one at each row
 * Return: the number of rows that do not have the first entry as diagonal
 * RL: only check if it's a non-empty row
 *--------------------------------------------------------------------------*/

__global__ void
hypreGPUKernel_CSRCheckDiagFirst( hypre_DeviceItem &item,
                                  HYPRE_Int  nrows,
                                  HYPRE_Int *ia,
                                  HYPRE_Int *ja,
                                  HYPRE_Int *result )
{
   const HYPRE_Int row = hypre_gpu_get_grid_thread_id<1, 1>(item);
   if (row < nrows)
   {
      result[row] = (ia[row + 1] > ia[row]) && (ja[ia[row]] != row);
   }
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixCheckDiagFirstDevice
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixCheckDiagFirstDevice( hypre_CSRMatrix *A )
{
   HYPRE_Int  *A_i      = hypre_CSRMatrixI(A);
   HYPRE_Int  *A_j      = hypre_CSRMatrixJ(A);
   HYPRE_Int   num_rows = hypre_CSRMatrixNumRows(A);
   HYPRE_Int  *result;
   HYPRE_Int   ierr;

   /* Sanity check */
   if (hypre_CSRMatrixNumRows(A) != hypre_CSRMatrixNumCols(A))
   {
      return 0;
   }

   dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = hypre_GetDefaultDeviceGridDimension(hypre_CSRMatrixNumRows(A), "thread", bDim);

   result = hypre_TAlloc(HYPRE_Int, num_rows, HYPRE_MEMORY_DEVICE);
   HYPRE_GPU_LAUNCH( hypreGPUKernel_CSRCheckDiagFirst, gDim, bDim,
                     num_rows, A_i, A_j, result );

   /* Compute number of rows in which the diagonal is not the first entry */
#if defined(HYPRE_USING_SYCL)
   ierr = HYPRE_ONEDPL_CALL( std::reduce,
                             result,
                             result + hypre_CSRMatrixNumRows(A) );
#else
   ierr = HYPRE_THRUST_CALL( reduce,
                             result,
                             result + hypre_CSRMatrixNumRows(A) );
#endif

   hypre_TFree(result, HYPRE_MEMORY_DEVICE);
   hypre_SyncComputeStream(hypre_handle());

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypreGPUKernel_CSRMatrixCheckForMissingDiagonal
 *--------------------------------------------------------------------------*/

__global__ void
hypreGPUKernel_CSRMatrixCheckForMissingDiagonal( hypre_DeviceItem    &item,
                                                 HYPRE_Int      nrows,
                                                 HYPRE_Int     *ia,
                                                 HYPRE_Int     *ja,
                                                 HYPRE_Int     *result )
{
   const HYPRE_Int row = hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (row >= nrows)
   {
      return;
   }

   HYPRE_Int lane = hypre_gpu_get_lane_id<1>(item);
   HYPRE_Int p = 0, q = 0;
   bool missing_diagonal = true;

   if (lane < 2)
   {
      p = read_only_load(ia + row + lane);
   }
   q = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p, 1);
   p = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p, 0);

   for (HYPRE_Int j = p + lane; warp_any_sync(item, HYPRE_WARP_FULL_MASK, j < q); j += HYPRE_WARP_SIZE)
   {
      hypre_int find_diag = j < q && read_only_load(&ja[j]) == row;

      if ( warp_any_sync(item, HYPRE_WARP_FULL_MASK, find_diag) )
      {
         missing_diagonal = false;
         break;
      }
   }

   if (missing_diagonal && lane == 0)
   {
      result[row] = 1;
   }
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixCheckForMissingDiagonal
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixCheckForMissingDiagonal( hypre_CSRMatrix *A )
{
   HYPRE_Int  *A_i      = hypre_CSRMatrixI(A);
   HYPRE_Int  *A_j      = hypre_CSRMatrixJ(A);
   HYPRE_Int   num_rows = hypre_CSRMatrixNumRows(A);
   HYPRE_Int  *result;
   HYPRE_Int   ierr;

   /* This test is only for square matrices */
   if (hypre_CSRMatrixNumRows(A) != hypre_CSRMatrixNumCols(A))
   {
      return 0;
   }

   dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = hypre_GetDefaultDeviceGridDimension(hypre_CSRMatrixNumRows(A), "thread", bDim);

   result = hypre_CTAlloc(HYPRE_Int, num_rows, HYPRE_MEMORY_DEVICE);
   HYPRE_GPU_LAUNCH( hypreGPUKernel_CSRMatrixCheckForMissingDiagonal, gDim, bDim,
                     num_rows, A_i, A_j, result );

   /* Compute number of rows in which the diagonal is not the first entry */
#if defined(HYPRE_USING_SYCL)
   ierr = HYPRE_ONEDPL_CALL( std::reduce,
                             result,
                             result + hypre_CSRMatrixNumRows(A) );
#else
   ierr = HYPRE_THRUST_CALL( reduce,
                             result,
                             result + hypre_CSRMatrixNumRows(A) );
#endif

   hypre_TFree(result, HYPRE_MEMORY_DEVICE);

   return ierr;
}

/*--------------------------------------------------------------------------
 * hypreGPUKernel_CSRMatrixReplaceDiagDevice
 *--------------------------------------------------------------------------*/

__global__ void
hypreGPUKernel_CSRMatrixReplaceDiagDevice( hypre_DeviceItem    &item,
                                           HYPRE_Complex *new_diag,
                                           HYPRE_Complex  v,
                                           HYPRE_Int      nrows,
                                           HYPRE_Int     *ia,
                                           HYPRE_Int     *ja,
                                           HYPRE_Complex *data,
                                           HYPRE_Real     tol,
                                           HYPRE_Int     *result )
{
   const HYPRE_Int row = hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (row >= nrows)
   {
      return;
   }

   HYPRE_Int lane = hypre_gpu_get_lane_id<1>(item);
   HYPRE_Int p = 0, q = 0;
   bool has_diag = false;

   if (lane < 2)
   {
      p = read_only_load(ia + row + lane);
   }
   q = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p, 1);
   p = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p, 0);

   for (HYPRE_Int j = p + lane; warp_any_sync(item, HYPRE_WARP_FULL_MASK, j < q); j += HYPRE_WARP_SIZE)
   {
      hypre_int find_diag = j < q && read_only_load(&ja[j]) == row;

      if (find_diag)
      {
         if (new_diag)
         {
            HYPRE_Complex d = read_only_load(&new_diag[row]);
            data[j] = hypre_abs(d) <= tol ? v : d;
         }
         else
         {
            if (hypre_abs(data[j]) <= tol)
            {
               data[j] = v;
            }
         }
      }

      if ( warp_any_sync(item, HYPRE_WARP_FULL_MASK, find_diag) )
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

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixReplaceDiagDevice
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixReplaceDiagDevice( hypre_CSRMatrix *A,
                                  HYPRE_Complex   *new_diag,
                                  HYPRE_Complex    v,
                                  HYPRE_Real       tol )
{
   if (hypre_CSRMatrixNumRows(A) != hypre_CSRMatrixNumCols(A))
   {
      return hypre_error_flag;
   }

   dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = hypre_GetDefaultDeviceGridDimension(hypre_CSRMatrixNumRows(A), "warp", bDim);

#if HYPRE_DEBUG
   HYPRE_Int *result = hypre_CTAlloc(HYPRE_Int, hypre_CSRMatrixNumRows(A), HYPRE_MEMORY_DEVICE);
#else
   HYPRE_Int *result = NULL;
#endif

   HYPRE_Int    num_rows = hypre_CSRMatrixNumRows(A);
   HYPRE_Int        *A_i = hypre_CSRMatrixI(A);
   HYPRE_Int        *A_j = hypre_CSRMatrixJ(A);
   HYPRE_Complex *A_data = hypre_CSRMatrixData(A);
   HYPRE_GPU_LAUNCH( hypreGPUKernel_CSRMatrixReplaceDiagDevice, gDim, bDim,
                     new_diag, v, num_rows,
                     A_i, A_j, A_data,
                     tol, result );

#if HYPRE_DEBUG
   /* the number of structural zero in A */
#if defined(HYPRE_USING_SYCL)
   HYPRE_Int num_zeros = HYPRE_ONEDPL_CALL( std::reduce,
                                            result,
                                            result + hypre_CSRMatrixNumRows(A) );
#else
   HYPRE_Int num_zeros = HYPRE_THRUST_CALL( reduce,
                                            result,
                                            result + hypre_CSRMatrixNumRows(A) );
#endif

   hypre_TFree(result, HYPRE_MEMORY_DEVICE);

   if (num_zeros)
   {
      hypre_error_w_msg(num_zeros, "structural zero in hypre_CSRMatrixReplaceDiagDevice");
   }
#endif

   hypre_SyncComputeStream(hypre_handle());

   return hypre_error_flag;
}

#if defined(HYPRE_USING_SYCL)
typedef std::tuple<HYPRE_Int, HYPRE_Int> Int2;
struct Int2Unequal
{
   bool operator()(const Int2& t) const
   {
      return (std::get<0>(t) != std::get<1>(t));
   }
};
#else
typedef thrust::tuple<HYPRE_Int, HYPRE_Int> Int2;
struct Int2Unequal : public thrust::unary_function<Int2, bool>
{
   __host__ __device__
   bool operator()(const Int2& t) const
   {
      return (thrust::get<0>(t) != thrust::get<1>(t));
   }
};
#endif

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixRemoveDiagonalDevice
 *--------------------------------------------------------------------------*/

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

#if defined(HYPRE_USING_SYCL)
   auto zip_ij = oneapi::dpl::make_zip_iterator(A_ii, A_j);
   new_nnz = HYPRE_ONEDPL_CALL( std::count_if,
                                zip_ij,
                                zip_ij + nnz,
                                Int2Unequal() );
#else
   new_nnz = HYPRE_THRUST_CALL( count_if,
                                thrust::make_zip_iterator(thrust::make_tuple(A_ii, A_j)),
                                thrust::make_zip_iterator(thrust::make_tuple(A_ii, A_j)) + nnz,
                                Int2Unequal() );
#endif

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

#if defined(HYPRE_USING_SYCL)
      auto zip_ija = oneapi::dpl::make_zip_iterator(A_ii, A_j, A_data);
      auto zip_new_ija = oneapi::dpl::make_zip_iterator(new_ii, new_j, new_data);
      auto new_end = hypreSycl_copy_if(
                        zip_ija,
                        zip_ija + nnz,
                        zip_ij,
                        zip_new_ija,
                        Int2Unequal() );

      hypre_assert( std::get<0>(new_end.base()) == new_ii + new_nnz );
#else
      auto new_end = HYPRE_THRUST_CALL( copy_if,
                                        thrust::make_zip_iterator(thrust::make_tuple(A_ii, A_j, A_data)),
                                        thrust::make_zip_iterator(thrust::make_tuple(A_ii, A_j, A_data)) + nnz,
                                        thrust::make_zip_iterator(thrust::make_tuple(A_ii, A_j)),
                                        thrust::make_zip_iterator(thrust::make_tuple(new_ii, new_j, new_data)),
                                        Int2Unequal() );

      hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) == new_ii + new_nnz );
#endif
   }
   else
   {
      new_data = NULL;

#if defined(HYPRE_USING_SYCL)
      auto zip_new_ij = oneapi::dpl::make_zip_iterator(new_ii, new_j);
      auto new_end = hypreSycl_copy_if( zip_ij,
                                        zip_ij + nnz,
                                        zip_ij,
                                        zip_new_ij,
                                        Int2Unequal() );

      hypre_assert( std::get<0>(new_end.base()) == new_ii + new_nnz );
#else
      auto new_end = HYPRE_THRUST_CALL( copy_if,
                                        thrust::make_zip_iterator(thrust::make_tuple(A_ii, A_j)),
                                        thrust::make_zip_iterator(thrust::make_tuple(A_ii, A_j)) + nnz,
                                        thrust::make_zip_iterator(thrust::make_tuple(A_ii, A_j)),
                                        thrust::make_zip_iterator(thrust::make_tuple(new_ii, new_j)),
                                        Int2Unequal() );

      hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) == new_ii + new_nnz );
#endif
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

/*--------------------------------------------------------------------------
 * hypreGPUKernel_CSRDiagScale
 *--------------------------------------------------------------------------*/

__global__ void
hypreGPUKernel_CSRDiagScale( hypre_DeviceItem    &item,
                             HYPRE_Int      nrows,
                             HYPRE_Int     *ia,
                             HYPRE_Int     *ja,
                             HYPRE_Complex *aa,
                             HYPRE_Complex *ld,
                             HYPRE_Complex *rd)
{
   HYPRE_Int row = hypre_gpu_get_grid_warp_id<1, 1>(item);

   if (row >= nrows)
   {
      return;
   }

   HYPRE_Int lane = hypre_gpu_get_lane_id<1>(item);
   HYPRE_Int p = 0, q = 0;

   if (lane < 2)
   {
      p = read_only_load(ia + row + lane);
   }
   q = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p, 1);
   p = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, p, 0);

   HYPRE_Complex sl = 1.0;

   if (ld)
   {
      if (!lane)
      {
         sl = read_only_load(ld + row);
      }
      sl = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, sl, 0);
   }

   if (rd)
   {
      for (HYPRE_Int i = p + lane; i < q; i += HYPRE_WARP_SIZE)
      {
         const HYPRE_Int col = read_only_load(ja + i);
         const HYPRE_Complex sr = read_only_load(rd + col);
         aa[i] = sl * aa[i] * sr;
      }
   }
   else if (sl != 1.0)
   {
      for (HYPRE_Int i = p + lane; i < q; i += HYPRE_WARP_SIZE)
      {
         aa[i] = sl * aa[i];
      }
   }
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixDiagScaleDevice
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixDiagScaleDevice( hypre_CSRMatrix *A,
                                hypre_Vector    *ld,
                                hypre_Vector    *rd)
{
   HYPRE_Int      nrows  = hypre_CSRMatrixNumRows(A);
   HYPRE_Complex *A_data = hypre_CSRMatrixData(A);
   HYPRE_Int     *A_i    = hypre_CSRMatrixI(A);
   HYPRE_Int     *A_j    = hypre_CSRMatrixJ(A);
   HYPRE_Complex *ldata  = ld ? hypre_VectorData(ld) : NULL;
   HYPRE_Complex *rdata  = rd ? hypre_VectorData(rd) : NULL;

   dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
   dim3 gDim = hypre_GetDefaultDeviceGridDimension(nrows, "warp", bDim);

   HYPRE_GPU_LAUNCH(hypreGPUKernel_CSRDiagScale, gDim, bDim,
                    nrows, A_i, A_j, A_data, ldata, rdata);

   hypre_SyncComputeStream(hypre_handle());

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * cabsfirst_greaterthan_second_pred
 *
 * This predicate compares first and second element in a tuple in absolute
 * value first is assumed to be complex, second to be real > 0
 *--------------------------------------------------------------------------*/

#if defined(HYPRE_USING_SYCL)
struct cabsfirst_greaterthan_second_pred
{
   bool operator()(const std::tuple<HYPRE_Complex, HYPRE_Real>& t) const
   {
      const HYPRE_Complex i = std::get<0>(t);
      const HYPRE_Real j = std::get<1>(t);

      return hypre_cabs(i) > j;
   }
};
#else
struct cabsfirst_greaterthan_second_pred : public
   thrust::unary_function<thrust::tuple<HYPRE_Complex, HYPRE_Real>, bool>
{
   __host__ __device__
   bool operator()(const thrust::tuple<HYPRE_Complex, HYPRE_Real>& t) const
   {
      const HYPRE_Complex i = thrust::get<0>(t);
      const HYPRE_Real j = thrust::get<1>(t);

      return hypre_cabs(i) > j;
   }
};
#endif

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixDropSmallEntriesDevice
 *
 * drop the entries that are smaller than:
 *    tol if elmt_tols == null,
 *    elmt_tols[j] otherwise where j = 0...NumNonzeros(A)
 *--------------------------------------------------------------------------*/

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

#if defined(HYPRE_USING_SYCL)
   if (elmt_tols == NULL)
   {
      new_nnz = HYPRE_ONEDPL_CALL( std::count_if,
                                   A_data,
                                   A_data + nnz,
      [tol] (const auto & x) {return !(x < tol);} );
   }
   else
   {
      new_nnz = HYPRE_ONEDPL_CALL( std::count_if,
                                   oneapi::dpl::make_zip_iterator(A_data, elmt_tols),
                                   oneapi::dpl::make_zip_iterator(A_data, elmt_tols) + nnz,
                                   cabsfirst_greaterthan_second_pred() );
   }
#else
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
#endif

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

#if defined(HYPRE_USING_SYCL)
   if (elmt_tols == NULL)
   {
      auto new_end = hypreSycl_copy_if( oneapi::dpl::make_zip_iterator(A_ii, A_j, A_data),
                                        oneapi::dpl::make_zip_iterator(A_ii, A_j, A_data) + nnz,
                                        A_data,
                                        oneapi::dpl::make_zip_iterator(new_ii, new_j, new_data),
      [tol] (const auto & x) {return !(x < tol);} );

      hypre_assert( std::get<0>(new_end.base()) == new_ii + new_nnz );
   }
   else
   {
      auto new_end = hypreSycl_copy_if( oneapi::dpl::make_zip_iterator(A_ii, A_j, A_data),
                                        oneapi::dpl::make_zip_iterator(A_ii, A_j, A_data) + nnz,
                                        oneapi::dpl::make_zip_iterator(A_data, elmt_tols),
                                        oneapi::dpl::make_zip_iterator(new_ii, new_j, new_data),
                                        cabsfirst_greaterthan_second_pred() );

      hypre_assert( std::get<0>(new_end.base()) == new_ii + new_nnz );
   }
#else
   if (elmt_tols == NULL)
   {
      auto new_end = HYPRE_THRUST_CALL( copy_if,
                                        thrust::make_zip_iterator(thrust::make_tuple(A_ii, A_j, A_data)),
                                        thrust::make_zip_iterator(thrust::make_tuple(A_ii, A_j, A_data)) + nnz,
                                        A_data,
                                        thrust::make_zip_iterator(thrust::make_tuple(new_ii, new_j, new_data)),
                                        thrust::not1(less_than<HYPRE_Complex>(tol)) );

      hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) == new_ii + new_nnz );
   }
   else
   {
      auto new_end = HYPRE_THRUST_CALL( copy_if,
                                        thrust::make_zip_iterator(thrust::make_tuple(A_ii, A_j, A_data)),
                                        thrust::make_zip_iterator(thrust::make_tuple(A_ii, A_j, A_data)) + nnz,
                                        thrust::make_zip_iterator(thrust::make_tuple(A_data, elmt_tols)),
                                        thrust::make_zip_iterator(thrust::make_tuple(new_ii, new_j, new_data)),
                                        cabsfirst_greaterthan_second_pred() );

      hypre_assert( thrust::get<0>(new_end.get_iterator_tuple()) == new_ii + new_nnz );
   }
#endif


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

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixIdentityDevice
 *
 * A = alp * I
 *--------------------------------------------------------------------------*/

hypre_CSRMatrix *
hypre_CSRMatrixIdentityDevice(HYPRE_Int n, HYPRE_Complex alp)
{
   hypre_CSRMatrix *A = hypre_CSRMatrixCreate(n, n, n);

   hypre_CSRMatrixInitialize_v2(A, 0, HYPRE_MEMORY_DEVICE);

#if defined(HYPRE_USING_SYCL)
   hypreSycl_sequence( hypre_CSRMatrixI(A),
                       hypre_CSRMatrixI(A) + n + 1,
                       0  );

   hypreSycl_sequence( hypre_CSRMatrixJ(A),
                       hypre_CSRMatrixJ(A) + n,
                       0  );

   HYPRE_ONEDPL_CALL( std::fill,
                      hypre_CSRMatrixData(A),
                      hypre_CSRMatrixData(A) + n,
                      alp );
#else
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
#endif

   return A;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixDiagMatrixFromVectorDevice
 *
 * A = diag(v)
 *--------------------------------------------------------------------------*/

hypre_CSRMatrix *
hypre_CSRMatrixDiagMatrixFromVectorDevice(HYPRE_Int n, HYPRE_Complex *v)
{
   hypre_CSRMatrix *A = hypre_CSRMatrixCreate(n, n, n);

   hypre_CSRMatrixInitialize_v2(A, 0, HYPRE_MEMORY_DEVICE);

#if defined(HYPRE_USING_SYCL)
   hypreSycl_sequence( hypre_CSRMatrixI(A),
                       hypre_CSRMatrixI(A) + n + 1,
                       0  );

   hypreSycl_sequence( hypre_CSRMatrixJ(A),
                       hypre_CSRMatrixJ(A) + n,
                       0  );

   HYPRE_ONEDPL_CALL( std::copy,
                      v,
                      v + n,
                      hypre_CSRMatrixData(A) );
#else
   HYPRE_THRUST_CALL( sequence,
                      hypre_CSRMatrixI(A),
                      hypre_CSRMatrixI(A) + n + 1,
                      0  );

   HYPRE_THRUST_CALL( sequence,
                      hypre_CSRMatrixJ(A),
                      hypre_CSRMatrixJ(A) + n,
                      0  );

   HYPRE_THRUST_CALL( copy,
                      v,
                      v + n,
                      hypre_CSRMatrixData(A) );
#endif

   return A;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixDiagMatrixFromMatrixDevice
 *
 * B = diagm(A)
 *--------------------------------------------------------------------------*/

hypre_CSRMatrix *
hypre_CSRMatrixDiagMatrixFromMatrixDevice(hypre_CSRMatrix *A, HYPRE_Int type)
{
   HYPRE_Int      nrows  = hypre_CSRMatrixNumRows(A);
   HYPRE_Complex  *diag = hypre_CTAlloc(HYPRE_Complex, nrows, HYPRE_MEMORY_DEVICE);
   hypre_CSRMatrixExtractDiagonalDevice(A, diag, type);

   hypre_CSRMatrix *diag_mat = hypre_CSRMatrixDiagMatrixFromVectorDevice(nrows, diag);

   hypre_TFree(diag, HYPRE_MEMORY_DEVICE);
   return diag_mat;
}

/*--------------------------------------------------------------------------
 * adj_functor (Used in hypre_CSRMatrixPermuteDevice)
 *--------------------------------------------------------------------------*/

#if defined(HYPRE_USING_SYCL)
struct adj_functor
#else
struct adj_functor : public thrust::unary_function<HYPRE_Int, HYPRE_Int>
#endif
{
   HYPRE_Int *ia_;

   adj_functor(HYPRE_Int *ia)
   {
      ia_ = ia;
   }

   __host__ __device__ HYPRE_Int operator()(HYPRE_Int i) const
   {
      return ia_[i + 1] - ia_[i];
   }
};

/*--------------------------------------------------------------------------
 * bii_functor (Used in hypre_CSRMatrixPermuteDevice)
 *--------------------------------------------------------------------------*/

struct bii_functor
{
   HYPRE_Int *p_, *ia_, *ib_, *rb_;

   bii_functor(HYPRE_Int *p, HYPRE_Int *ia, HYPRE_Int *ib, HYPRE_Int *rb)
   {
      p_ = p;
      ia_ = ia;
      ib_ = ib;
      rb_ = rb;
   }

   __host__ __device__ void operator()(HYPRE_Int i) const
   {
      const HYPRE_Int r = rb_[i];
      rb_[i] = ia_[p_[r]] + i - ib_[r];
   }
};

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixPermuteDevice
 *
 * See hypre_CSRMatrixPermute.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixPermuteDevice( hypre_CSRMatrix  *A,
                              HYPRE_Int        *perm,
                              HYPRE_Int        *rqperm,
                              hypre_CSRMatrix  *B )
{
   /* Input matrix */
   HYPRE_Int         num_rows     = hypre_CSRMatrixNumRows(A);
   HYPRE_Int         num_nonzeros = hypre_CSRMatrixNumNonzeros(A);
   HYPRE_Int        *A_i          = hypre_CSRMatrixI(A);
   HYPRE_Int        *A_j          = hypre_CSRMatrixJ(A);
   HYPRE_Complex    *A_a          = hypre_CSRMatrixData(A);
   HYPRE_Int        *B_i          = hypre_CSRMatrixI(B);
   HYPRE_Int        *B_j          = hypre_CSRMatrixJ(B);
   HYPRE_Complex    *B_a          = hypre_CSRMatrixData(B);

   HYPRE_Int        *B_ii;

   /* Build B_i */
#if defined(HYPRE_USING_SYCL)
   oneapi::dpl::counting_iterator count(0);
   hypreSycl_gather(perm,
                    perm + num_rows,
                    oneapi::dpl::make_transform_iterator(count, adj_functor(A_i)),
                    B_i);
#else
   HYPRE_THRUST_CALL(gather,
                     perm,
                     perm + num_rows,
                     thrust::make_transform_iterator(thrust::make_counting_iterator(0), adj_functor(A_i)),
                     B_i);
#endif
   hypreDevice_IntegerExclusiveScan(num_rows + 1, B_i);

   /* Build B_ii (row indices array) */
   B_ii = hypre_TAlloc(HYPRE_Int, num_nonzeros, HYPRE_MEMORY_DEVICE);
   hypreDevice_CsrRowPtrsToIndices_v2(num_rows, num_nonzeros, B_i, B_ii);
#if defined(HYPRE_USING_SYCL)
   HYPRE_ONEDPL_CALL(std::for_each,
                     count,
                     count + num_nonzeros,
                     bii_functor(perm, A_i, B_i, B_ii));

   /* Build B_j and B_a */
   hypreSycl_gather( B_ii,
                     B_ii + num_nonzeros,
                     oneapi::dpl::make_zip_iterator(oneapi::dpl::make_permutation_iterator(rqperm, A_j), A_a),
                     oneapi::dpl::make_zip_iterator(B_j, B_a));
#else
   HYPRE_THRUST_CALL(for_each,
                     thrust::make_counting_iterator(0),
                     thrust::make_counting_iterator(num_nonzeros),
                     bii_functor(perm, A_i, B_i, B_ii));

   /* Build B_j and B_a */
   HYPRE_THRUST_CALL(gather,
                     B_ii,
                     B_ii + num_nonzeros,
                     thrust::make_zip_iterator(thrust::make_tuple(
                                                  thrust::make_permutation_iterator(rqperm, A_j), A_a)),
                     thrust::make_zip_iterator(thrust::make_tuple(B_j, B_a)));
#endif

   /* Free memory */
   hypre_TFree(B_ii, HYPRE_MEMORY_DEVICE);

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixTransposeDevice
 *--------------------------------------------------------------------------*/

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

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("CSRMatrixTranspose");

   /* trivial case */
   if (nnz_A == 0)
   {
      C_i =    hypre_CTAlloc(HYPRE_Int,     ncols_A + 1, HYPRE_MEMORY_DEVICE);
      C_j =    hypre_CTAlloc(HYPRE_Int,     0,           HYPRE_MEMORY_DEVICE);
      C_data = hypre_CTAlloc(HYPRE_Complex, 0,           HYPRE_MEMORY_DEVICE);
   }
   else
   {
      if ( !hypre_HandleSpTransUseVendor(hypre_handle()) )
      {
#if defined(HYPRE_USING_GPU)
         hypreDevice_CSRSpTrans(nrows_A, ncols_A, nnz_A, A_i, A_j, A_data, &C_i, &C_j, &C_data, data);
#endif
      }
      else
      {
#if defined(HYPRE_USING_CUSPARSE)
         hypreDevice_CSRSpTransCusparse(nrows_A, ncols_A, nnz_A, A_i, A_j, A_data, &C_i, &C_j, &C_data,
                                        data);
#elif defined(HYPRE_USING_ROCSPARSE)
         hypreDevice_CSRSpTransRocsparse(nrows_A, ncols_A, nnz_A, A_i, A_j, A_data, &C_i, &C_j, &C_data,
                                         data);
#elif defined(HYPRE_USING_GPU)
         hypreDevice_CSRSpTrans(nrows_A, ncols_A, nnz_A, A_i, A_j, A_data, &C_i, &C_j, &C_data, data);
#endif
      }
   }

   C = hypre_CSRMatrixCreate(ncols_A, nrows_A, nnz_A);
   hypre_CSRMatrixI(C) = C_i;
   hypre_CSRMatrixJ(C) = C_j;
   hypre_CSRMatrixData(C) = C_data;
   hypre_CSRMatrixMemoryLocation(C) = HYPRE_MEMORY_DEVICE;

   *AT_ptr = C;

   hypre_SyncComputeStream(hypre_handle());

   hypre_GpuProfilingPopRange();
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixSortRow
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixSortRow(hypre_CSRMatrix *A)
{
   hypre_GpuProfilingPushRange("CSRMatrixSort");

#if defined(HYPRE_USING_CUSPARSE)
   hypre_SortCSRCusparse(hypre_CSRMatrixNumRows(A), hypre_CSRMatrixNumCols(A),
                         hypre_CSRMatrixNumNonzeros(A), hypre_CSRMatrixGPUMatDescr(A),
                         hypre_CSRMatrixI(A), hypre_CSRMatrixJ(A), hypre_CSRMatrixData(A));

#elif defined(HYPRE_USING_ROCSPARSE)
   hypre_SortCSRRocsparse(hypre_CSRMatrixNumRows(A), hypre_CSRMatrixNumCols(A),
                          hypre_CSRMatrixNumNonzeros(A), hypre_CSRMatrixGPUMatDescr(A),
                          hypre_CSRMatrixI(A), hypre_CSRMatrixJ(A), hypre_CSRMatrixData(A));
#elif defined(HYPRE_USING_ONEMKLSPARSE)
   HYPRE_ONEMKL_CALL( oneapi::mkl::sparse::sort_matrix(*hypre_HandleComputeStream(hypre_handle()),
                                                       hypre_CSRMatrixGPUMatHandle(A), {}).wait() );
#else
   HYPRE_UNUSED_VAR(A);
   hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                     "hypre_CSRMatrixSortRow only implemented for cuSPARSE/rocSPARSE/oneMKLSparse!\n");
#endif

   hypre_GpuProfilingPopRange();

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixSortRowOutOfPlace
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixSortRowOutOfPlace(hypre_CSRMatrix *A)
{
   HYPRE_Int     *A_j  = hypre_CSRMatrixJ(A);
   HYPRE_Complex *A_a  = hypre_CSRMatrixData(A);
   HYPRE_Int      nnzA = hypre_CSRMatrixNumNonzeros(A);

   /* if both exist, we assume A has been sorted */
   if (hypre_CSRMatrixSortedJ(A) && hypre_CSRMatrixSortedData(A))
   {
      return hypre_error_flag;
   }

   hypre_TFree(hypre_CSRMatrixSortedJ(A), HYPRE_MEMORY_DEVICE);
   hypre_TFree(hypre_CSRMatrixSortedData(A), HYPRE_MEMORY_DEVICE);

   hypre_CSRMatrixSortedJ(A)    = hypre_TAlloc(HYPRE_Int,     nnzA, HYPRE_MEMORY_DEVICE);
   hypre_CSRMatrixSortedData(A) = hypre_TAlloc(HYPRE_Complex, nnzA, HYPRE_MEMORY_DEVICE);

   hypre_TMemcpy(hypre_CSRMatrixSortedJ(A), A_j, HYPRE_Int, nnzA,
                 HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
   hypre_TMemcpy(hypre_CSRMatrixSortedData(A), A_a, HYPRE_Complex, nnzA,
                 HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

   hypre_CSRMatrixJ(A) = hypre_CSRMatrixSortedJ(A);
   hypre_CSRMatrixData(A) = hypre_CSRMatrixSortedData(A);

   hypre_CSRMatrixSortRow(A);

   hypre_CSRMatrixJ(A)    = A_j;
   hypre_CSRMatrixData(A) = A_a;

   return hypre_error_flag;
}

#if defined(HYPRE_USING_CUSPARSE)

/*--------------------------------------------------------------------------
 * hypre_SortCSRCusparse
 *
 * Sorts values and column indices in each row in ascending order INPLACE
 *
 * Parameters:
 *   n: Number of rows [in]
 *   m: Number of columns [in]
 *   nnzA: Number of nonzeros [in]
 *   d_ia: row pointers [in/out]
 *   d_ja_sorted: column indices [in/out]
 *   d_a_sorted: coefficients [in/out]
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SortCSRCusparse( HYPRE_Int            n,
                       HYPRE_Int            m,
                       HYPRE_Int            nnzA,
                       cusparseMatDescr_t   descrA,
                       const HYPRE_Int     *d_ia,
                       HYPRE_Int           *d_ja_sorted,
                       HYPRE_Complex       *d_a_sorted )
{
   cusparseHandle_t  cusparsehandle = hypre_HandleCusparseHandle(hypre_handle());
   size_t            pBufferSizeInBytes = 0;
   void             *pBuffer = NULL;
   csru2csrInfo_t    sortInfoA;

   hypre_GpuProfilingPushRange("SortCSRCusparse");

   HYPRE_CUSPARSE_CALL( cusparseCreateCsru2csrInfo(&sortInfoA) );
   HYPRE_CUSPARSE_CALL( hypre_cusparse_csru2csr_bufferSizeExt(cusparsehandle,
                                                              n, m, nnzA,
                                                              d_a_sorted, d_ia, d_ja_sorted,
                                                              sortInfoA, &pBufferSizeInBytes) );

   pBuffer = hypre_TAlloc(char, pBufferSizeInBytes, HYPRE_MEMORY_DEVICE);
   HYPRE_CUSPARSE_CALL( hypre_cusparse_csru2csr(cusparsehandle,
                                                n, m, nnzA, descrA,
                                                d_a_sorted, d_ia, d_ja_sorted,
                                                sortInfoA, pBuffer) );

   hypre_TFree(pBuffer, HYPRE_MEMORY_DEVICE);
   HYPRE_CUSPARSE_CALL(cusparseDestroyCsru2csrInfo(sortInfoA));

   hypre_GpuProfilingPopRange();

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixTriLowerUpperSolveCusparse
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixTriLowerUpperSolveCusparse(char             uplo,
                                          HYPRE_Int        unit_diag,
                                          hypre_CSRMatrix *A,
                                          HYPRE_Real      *l1_norms,
                                          HYPRE_Complex   *f_data,
                                          HYPRE_Complex   *u_data )
{
   HYPRE_Int              num_rows     = hypre_CSRMatrixNumRows(A);
   HYPRE_Int              num_nonzeros = hypre_CSRMatrixNumNonzeros(A);
   HYPRE_Int             *A_i          = hypre_CSRMatrixI(A);
   HYPRE_Int             *A_j          = hypre_CSRMatrixJ(A);
   HYPRE_Complex         *A_a          = hypre_CSRMatrixData(A);
   hypre_CsrsvData       *csrsv_data   = hypre_CSRMatrixCsrsvData(A);
   HYPRE_Complex         *A_ma;

   cusparseHandle_t       handle       = hypre_HandleCusparseHandle(hypre_handle());
   cusparseDiagType_t     diag_type    = unit_diag ? CUSPARSE_DIAG_TYPE_UNIT :
                                         CUSPARSE_DIAG_TYPE_NON_UNIT;
   cusparseFillMode_t     fill_mode_L  = CUSPARSE_FILL_MODE_LOWER;
   cusparseFillMode_t     fill_mode_U  = CUSPARSE_FILL_MODE_UPPER;
   cusparseOperation_t    operation    = CUSPARSE_OPERATION_NON_TRANSPOSE;

   HYPRE_Complex          alpha        = 1.0;

#if CUSPARSE_VERSION >= CUSPARSE_SPSV_VERSION
   HYPRE_Int              num_cols     = hypre_CSRMatrixNumCols(A);
   cusparseSpMatDescr_t   matA;
   cusparseDnVecDescr_t   vecF;
   cusparseDnVecDescr_t   vecU;

   cudaDataType           data_type    = hypre_HYPREComplexToCudaDataType();
   size_t                 buffer_size;
   char*                  buffer_L;
   char*                  buffer_U;
#else
   cusparseSolvePolicy_t  policy = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
   cusparseMatDescr_t     descr;
   cusparseStatus_t       status;
   HYPRE_Int             *A_sj;
   hypre_int              buffer_size;
   char*                  buffer;
   hypre_int              structural_zero;
   char                   msg[256];

   /* cuSPARSE's legacy API requires sorted rows. Sort and save in CSR's (sj, sa) */
   hypre_CSRMatrixSortRowOutOfPlace(A);
#endif

   /* setup csrsvdata in CSR: modify the diagonal (once) */
   if (!csrsv_data)
   {
      hypre_CSRMatrixCsrsvData(A) = hypre_CsrsvDataCreate();
      csrsv_data = hypre_CSRMatrixCsrsvData(A);

      hypre_CsrsvDataMatData(csrsv_data) = hypre_TAlloc(HYPRE_Complex, num_nonzeros,
                                                        HYPRE_MEMORY_DEVICE);
      hypre_CSRMatrixData(A) = hypre_CsrsvDataMatData(csrsv_data);

#if CUSPARSE_VERSION >= CUSPARSE_SPSV_VERSION
      hypre_TMemcpy(hypre_CsrsvDataMatData(csrsv_data), A_a, HYPRE_Complex, num_nonzeros,
                    HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
#else
      hypre_TMemcpy(hypre_CsrsvDataMatData(csrsv_data), hypre_CSRMatrixSortedData(A),
                    HYPRE_Complex, num_nonzeros,
                    HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
      hypre_CSRMatrixJ(A) = hypre_CSRMatrixSortedJ(A);
#endif

      /* if (l1_norms), replace A's diag with l1_norm, and
       * replace zero diag with inf. so as to skip relaxation for this unknown */
      hypre_CSRMatrixReplaceDiagDevice(A, l1_norms, INFINITY, 0.0);

      hypre_CSRMatrixData(A) = A_a;
#if CUSPARSE_VERSION < CUSPARSE_SPSV_VERSION
      hypre_CSRMatrixJ(A) = A_j;
#endif
   }

   /* Analysis and Solve */
   A_ma = hypre_CsrsvDataMatData(csrsv_data);

#if CUSPARSE_VERSION >= CUSPARSE_SPSV_VERSION
   matA = hypre_CSRMatrixToCusparseSpMat_core(num_rows, num_cols, 0,
                                              num_nonzeros, A_i, A_j, A_ma);
   vecF = hypre_VectorToCusparseDnVec_core(f_data, num_rows);
   vecU = hypre_VectorToCusparseDnVec_core(u_data, num_cols);

   HYPRE_CUSPARSE_CALL( cusparseSpMatSetAttribute(matA, CUSPARSE_SPMAT_DIAG_TYPE,
                                                  &diag_type, sizeof(cusparseDiagType_t)) );
#else
   A_sj  = hypre_CSRMatrixSortedJ(A);
   descr = hypre_CSRMatrixGPUMatDescr(A);
   HYPRE_CUSPARSE_CALL( cusparseSetMatDiagType(descr, diag_type) );
#endif

   if (uplo == 'L')
   {
#if CUSPARSE_VERSION >= CUSPARSE_SPSV_VERSION
      HYPRE_CUSPARSE_CALL( cusparseSpMatSetAttribute(matA,
                                                     CUSPARSE_SPMAT_FILL_MODE,
                                                     &fill_mode_L,
                                                     sizeof(cusparseFillMode_t)) );
#else
      HYPRE_CUSPARSE_CALL( cusparseSetMatFillMode(descr, fill_mode_L) );
#endif

      /* TODO (VPM): move the following block to hypre_CSRMatrixTriSetupCusparse */
      if (!hypre_CsrsvDataAnalyzedL(csrsv_data))
      {
#if CUSPARSE_VERSION >= CUSPARSE_SPSV_VERSION
         HYPRE_CUSPARSE_CALL( cusparseSpSV_bufferSize(handle, operation,
                                                      &alpha, matA, vecF, vecU, data_type,
                                                      CUSPARSE_SPSV_ALG_DEFAULT,
                                                      hypre_CsrsvDataInfoL(csrsv_data),
                                                      &buffer_size) );
#else
         HYPRE_CUSPARSE_CALL( hypre_cusparse_csrsv2_bufferSize(handle,
                                                               operation,
                                                               num_rows, num_nonzeros, descr,
                                                               A_ma, A_i, A_sj,
                                                               hypre_CsrsvDataInfoL(csrsv_data),
                                                               &buffer_size) );
#endif

#if CUSPARSE_VERSION >= CUSPARSE_SPSV_VERSION
         if (hypre_CsrsvDataBufferSizeL(csrsv_data) < buffer_size)
         {
            buffer_L = hypre_TReAlloc_v2(hypre_CsrsvDataBufferL(csrsv_data),
                                         char,
                                         hypre_CsrsvDataBufferSizeL(csrsv_data),
                                         char,
                                         buffer_size,
                                         HYPRE_MEMORY_DEVICE);

            hypre_CsrsvDataBufferL(csrsv_data)     = buffer_L;
            hypre_CsrsvDataBufferSizeL(csrsv_data) = buffer_size;
         }
#else
         if (hypre_CsrsvDataBufferSize(csrsv_data) < buffer_size)
         {
            buffer = hypre_TReAlloc_v2(hypre_CsrsvDataBuffer(csrsv_data),
                                       char,
                                       hypre_CsrsvDataBufferSize(csrsv_data),
                                       char,
                                       buffer_size,
                                       HYPRE_MEMORY_DEVICE);

            hypre_CsrsvDataBuffer(csrsv_data)     = buffer;
            hypre_CsrsvDataBufferSize(csrsv_data) = buffer_size;
         }
#endif

#if CUSPARSE_VERSION >= CUSPARSE_SPSV_VERSION
         HYPRE_CUSPARSE_CALL( cusparseSpSV_analysis(handle, operation,
                                                    &alpha, matA, vecF, vecU, data_type,
                                                    CUSPARSE_SPSV_ALG_DEFAULT,
                                                    hypre_CsrsvDataInfoL(csrsv_data),
                                                    hypre_CsrsvDataBufferL(csrsv_data)) );
#else
         HYPRE_CUSPARSE_CALL( hypre_cusparse_csrsv2_analysis(handle,
                                                             operation,
                                                             num_rows, num_nonzeros, descr,
                                                             A_ma, A_i, A_sj,
                                                             hypre_CsrsvDataInfoL(csrsv_data),
                                                             policy,
                                                             hypre_CsrsvDataBuffer(csrsv_data)) );

         status = cusparseXcsrsv2_zeroPivot(handle,
                                            hypre_CsrsvDataInfoL(csrsv_data),
                                            &structural_zero);
         if (CUSPARSE_STATUS_ZERO_PIVOT == status)
         {
            hypre_sprintf(msg, "A(%d,%d) is missing\n",
                          structural_zero, structural_zero);
            hypre_error_w_msg(1, msg);
         }
#endif
         hypre_CsrsvDataAnalyzedL(csrsv_data) = 1;
      }

#if CUSPARSE_VERSION >= CUSPARSE_SPSV_VERSION
      HYPRE_CUSPARSE_CALL( cusparseSpSV_solve(handle, operation,
                                              &alpha, matA, vecF, vecU, data_type,
                                              CUSPARSE_SPSV_ALG_DEFAULT,
                                              hypre_CsrsvDataInfoL(csrsv_data)) );
#else
      HYPRE_CUSPARSE_CALL( hypre_cusparse_csrsv2_solve(handle, operation,
                                                       num_rows, num_nonzeros, &alpha, descr,
                                                       A_ma, A_i, A_sj,
                                                       hypre_CsrsvDataInfoL(csrsv_data),
                                                       f_data, u_data, policy,
                                                       hypre_CsrsvDataBuffer(csrsv_data)) );
#endif
   }
   else
   {
#if CUSPARSE_VERSION >= CUSPARSE_SPSV_VERSION
      HYPRE_CUSPARSE_CALL( cusparseSpMatSetAttribute(matA,
                                                     CUSPARSE_SPMAT_FILL_MODE,
                                                     &fill_mode_U,
                                                     sizeof(cusparseFillMode_t)) );
#else
      HYPRE_CUSPARSE_CALL( cusparseSetMatFillMode(descr, fill_mode_U) );
#endif

      /* TODO (VPM): move the following block to hypre_CSRMatrixTriSetupCusparse */
      if (!hypre_CsrsvDataAnalyzedU(csrsv_data))
      {
#if CUSPARSE_VERSION >= CUSPARSE_SPSV_VERSION
         HYPRE_CUSPARSE_CALL( cusparseSpSV_bufferSize(handle, operation,
                                                      &alpha, matA, vecF, vecU, data_type,
                                                      CUSPARSE_SPSV_ALG_DEFAULT,
                                                      hypre_CsrsvDataInfoU(csrsv_data),
                                                      &buffer_size) );
#else
         HYPRE_CUSPARSE_CALL( hypre_cusparse_csrsv2_bufferSize(handle,
                                                               operation,
                                                               num_rows, num_nonzeros, descr,
                                                               A_ma, A_i, A_sj,
                                                               hypre_CsrsvDataInfoU(csrsv_data),
                                                               &buffer_size) );
#endif

#if CUSPARSE_VERSION >= CUSPARSE_SPSV_VERSION
         if (hypre_CsrsvDataBufferSizeU(csrsv_data) < buffer_size)
         {
            buffer_U = hypre_TReAlloc_v2(hypre_CsrsvDataBufferU(csrsv_data),
                                         char,
                                         hypre_CsrsvDataBufferSizeU(csrsv_data),
                                         char,
                                         buffer_size,
                                         HYPRE_MEMORY_DEVICE);

            hypre_CsrsvDataBufferU(csrsv_data)     = buffer_U;
            hypre_CsrsvDataBufferSizeU(csrsv_data) = buffer_size;
         }
#else
         if (hypre_CsrsvDataBufferSize(csrsv_data) < buffer_size)
         {
            buffer = hypre_TReAlloc_v2(hypre_CsrsvDataBuffer(csrsv_data),
                                       char,
                                       hypre_CsrsvDataBufferSize(csrsv_data),
                                       char,
                                       buffer_size,
                                       HYPRE_MEMORY_DEVICE);

            hypre_CsrsvDataBuffer(csrsv_data)     = buffer;
            hypre_CsrsvDataBufferSize(csrsv_data) = buffer_size;
         }
#endif

#if CUSPARSE_VERSION >= CUSPARSE_SPSV_VERSION
         HYPRE_CUSPARSE_CALL( cusparseSpSV_analysis(handle, operation,
                                                    &alpha, matA, vecF, vecU, data_type,
                                                    CUSPARSE_SPSV_ALG_DEFAULT,
                                                    hypre_CsrsvDataInfoU(csrsv_data),
                                                    hypre_CsrsvDataBufferU(csrsv_data)) );
#else
         HYPRE_CUSPARSE_CALL( hypre_cusparse_csrsv2_analysis(handle,
                                                             operation,
                                                             num_rows, num_nonzeros, descr,
                                                             A_ma, A_i, A_sj,
                                                             hypre_CsrsvDataInfoU(csrsv_data),
                                                             policy,
                                                             hypre_CsrsvDataBuffer(csrsv_data)) );

         status = cusparseXcsrsv2_zeroPivot(handle,
                                            hypre_CsrsvDataInfoU(csrsv_data),
                                            &structural_zero);
         if (CUSPARSE_STATUS_ZERO_PIVOT == status)
         {
            hypre_sprintf(msg, "A(%d,%d) is missing\n",
                          structural_zero, structural_zero);
            hypre_error_w_msg(1, msg);
         }
#endif
         hypre_CsrsvDataAnalyzedU(csrsv_data) = 1;
      }

#if CUSPARSE_VERSION >= CUSPARSE_SPSV_VERSION
      HYPRE_CUSPARSE_CALL( cusparseSpSV_solve(handle, operation,
                                              &alpha, matA, vecF, vecU, data_type,
                                              CUSPARSE_SPSV_ALG_DEFAULT,
                                              hypre_CsrsvDataInfoU(csrsv_data)) );
#else
      HYPRE_CUSPARSE_CALL( hypre_cusparse_csrsv2_solve(handle,
                                                       operation,
                                                       num_rows, num_nonzeros, &alpha,
                                                       descr, A_ma, A_i, A_sj,
                                                       hypre_CsrsvDataInfoU(csrsv_data),
                                                       f_data, u_data, policy,
                                                       hypre_CsrsvDataBuffer(csrsv_data)) );
#endif
   }

   /* Free memory */
#if CUSPARSE_VERSION >= CUSPARSE_SPSV_VERSION
   HYPRE_CUSPARSE_CALL( cusparseDestroySpMat(matA) );
   HYPRE_CUSPARSE_CALL( cusparseDestroyDnVec(vecF) );
   HYPRE_CUSPARSE_CALL( cusparseDestroyDnVec(vecU) );
#endif

   return hypre_error_flag;
}

#elif defined(HYPRE_USING_ROCSPARSE)

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixTriLowerUpperSolveRocsparse
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixTriLowerUpperSolveRocsparse(char              uplo,
                                           HYPRE_Int         unit_diag,
                                           hypre_CSRMatrix  *A,
                                           HYPRE_Real       *l1_norms,
                                           HYPRE_Complex    *f_data,
                                           HYPRE_Complex    *u_data )
{
   HYPRE_Int            num_rows      = hypre_CSRMatrixNumRows(A);
   HYPRE_Int            num_nonzeros  = hypre_CSRMatrixNumNonzeros(A);
   HYPRE_Int           *A_i           = hypre_CSRMatrixI(A);
   HYPRE_Int           *A_j           = hypre_CSRMatrixJ(A);
   HYPRE_Complex       *A_a           = hypre_CSRMatrixData(A);
   hypre_CsrsvData     *csrsv_data    = hypre_CSRMatrixCsrsvData(A);

   rocsparse_handle     handle        = hypre_HandleCusparseHandle(hypre_handle());
   rocsparse_mat_descr  descr         = hypre_CSRMatrixGPUMatDescr(A);
   HYPRE_Int           *A_sj;
   HYPRE_Complex       *A_ma;

   rocsparse_status     status;
   rocsparse_diag_type  diag_type     = unit_diag ? rocsparse_diag_type_unit :
                                        rocsparse_diag_type_non_unit;
   HYPRE_Complex        alpha         = 1.0;
   hypre_int            structural_zero;
   size_t               buffer_size;
   char                *buffer;
   char                 msg[256];

   /* rocSPARSE requires sorted rows. Sort and save in CSR's (sj, sa) */
   hypre_CSRMatrixSortRowOutOfPlace(A);

   /* Setup csrsvdata in CSR: modify the diagonal (once) */
   if (!csrsv_data)
   {
      hypre_CSRMatrixCsrsvData(A) = hypre_CsrsvDataCreate();
      csrsv_data = hypre_CSRMatrixCsrsvData(A);

      hypre_CsrsvDataMatData(csrsv_data) = hypre_TAlloc(HYPRE_Complex,
                                                        num_nonzeros,
                                                        HYPRE_MEMORY_DEVICE);

      hypre_CSRMatrixData(A) = hypre_CsrsvDataMatData(csrsv_data);
      hypre_TMemcpy(hypre_CsrsvDataMatData(csrsv_data), hypre_CSRMatrixSortedData(A),
                    HYPRE_Complex, num_nonzeros,
                    HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);
      hypre_CSRMatrixJ(A) = hypre_CSRMatrixSortedJ(A);

      /* if (l1_norms), replace A's diag with l1_norm, and
       * replace zero diag with inf. so as to skip relaxation for this unknown */
      hypre_CSRMatrixReplaceDiagDevice(A, l1_norms, INFINITY, 0.0);

      hypre_CSRMatrixData(A) = A_a;
      hypre_CSRMatrixJ(A)    = A_j;
   }

   /* Analysis and Solve */
   buffer = hypre_CsrsvDataBuffer(csrsv_data);
   A_ma   = hypre_CsrsvDataMatData(csrsv_data);
   A_sj   = hypre_CSRMatrixSortedJ(A);

   /* Set matrix diagonal type */
   HYPRE_ROCSPARSE_CALL( rocsparse_set_mat_diag_type(descr, diag_type) );

   if (uplo == 'L')
   {
      HYPRE_ROCSPARSE_CALL( rocsparse_set_mat_fill_mode(descr, rocsparse_fill_mode_lower) );

      /* TODO (VPM): move the following block to hypre_CSRMatrixTriSetupRocsparse */
      if (!hypre_CsrsvDataAnalyzedL(csrsv_data))
      {
         HYPRE_ROCSPARSE_CALL( hypre_rocsparse_csrsv_buffer_size(handle,
                                                                 rocsparse_operation_none,
                                                                 num_rows, num_nonzeros, descr,
                                                                 A_ma, A_i, A_sj,
                                                                 hypre_CsrsvDataInfoL(csrsv_data),
                                                                 &buffer_size) );

         if (hypre_CsrsvDataBufferSize(csrsv_data) < buffer_size)
         {
            buffer = hypre_TReAlloc_v2(hypre_CsrsvDataBuffer(csrsv_data),
                                       char, hypre_CsrsvDataBufferSize(csrsv_data),
                                       char, buffer_size,
                                       HYPRE_MEMORY_DEVICE);

            hypre_CsrsvDataBuffer(csrsv_data)     = buffer;
            hypre_CsrsvDataBufferSize(csrsv_data) = buffer_size;
         }

         HYPRE_ROCSPARSE_CALL( hypre_rocsparse_csrsv_analysis(handle, rocsparse_operation_none,
                                                              num_rows, num_nonzeros, descr,
                                                              A_ma, A_i, A_sj,
                                                              hypre_CsrsvDataInfoL(csrsv_data),
                                                              rocsparse_analysis_policy_reuse,
                                                              rocsparse_solve_policy_auto,
                                                              buffer) );

         status = rocsparse_csrsv_zero_pivot(handle, descr,
                                             hypre_CsrsvDataInfoL(csrsv_data),
                                             &structural_zero);
         if (rocsparse_status_zero_pivot == status)
         {
            hypre_sprintf(msg,
                          "hypre_CSRMatrixTriLowerUpperSolveRocsparse A(%d,%d) is missing\n",
                          structural_zero, structural_zero);
            hypre_error_w_msg(1, msg);
         }
         hypre_CsrsvDataAnalyzedL(csrsv_data) = 1;
      }

      HYPRE_ROCSPARSE_CALL( hypre_rocsparse_csrsv_solve(handle, rocsparse_operation_none,
                                                        num_rows, num_nonzeros, &alpha,
                                                        descr, A_ma, A_i, A_sj,
                                                        hypre_CsrsvDataInfoL(csrsv_data),
                                                        f_data, u_data,
                                                        rocsparse_solve_policy_auto,
                                                        buffer) );
   }
   else
   {
      HYPRE_ROCSPARSE_CALL( rocsparse_set_mat_fill_mode(descr, rocsparse_fill_mode_upper) );

      /* TODO (VPM): move the following block to hypre_CSRMatrixTriSetupRocsparse */
      if (!hypre_CsrsvDataAnalyzedU(csrsv_data))
      {
         HYPRE_ROCSPARSE_CALL( hypre_rocsparse_csrsv_buffer_size(handle,
                                                                 rocsparse_operation_none,
                                                                 num_rows, num_nonzeros, descr,
                                                                 A_ma, A_i, A_sj,
                                                                 hypre_CsrsvDataInfoU(csrsv_data),
                                                                 &buffer_size) );

         if (hypre_CsrsvDataBufferSize(csrsv_data) < buffer_size)
         {
            buffer = hypre_TReAlloc_v2(hypre_CsrsvDataBuffer(csrsv_data),
                                       char, hypre_CsrsvDataBufferSize(csrsv_data),
                                       char, buffer_size,
                                       HYPRE_MEMORY_DEVICE);

            hypre_CsrsvDataBuffer(csrsv_data) = buffer;
            hypre_CsrsvDataBufferSize(csrsv_data) = buffer_size;
         }

         HYPRE_ROCSPARSE_CALL( hypre_rocsparse_csrsv_analysis(handle, rocsparse_operation_none,
                                                              num_rows, num_nonzeros, descr,
                                                              A_ma, A_i, A_sj,
                                                              hypre_CsrsvDataInfoU(csrsv_data),
                                                              rocsparse_analysis_policy_reuse,
                                                              rocsparse_solve_policy_auto,
                                                              buffer) );

         status = rocsparse_csrsv_zero_pivot(handle, descr,
                                             hypre_CsrsvDataInfoU(csrsv_data),
                                             &structural_zero);
         if (rocsparse_status_zero_pivot == status)
         {
            hypre_sprintf(msg,
                          "hypre_CSRMatrixTriLowerUpperSolveRocsparse A(%d,%d) is missing\n",
                          structural_zero, structural_zero);
            hypre_error_w_msg(1, msg);
         }
         hypre_CsrsvDataAnalyzedU(csrsv_data) = 1;
      }

      HYPRE_ROCSPARSE_CALL( hypre_rocsparse_csrsv_solve(handle, rocsparse_operation_none,
                                                        num_rows, num_nonzeros, &alpha, descr,
                                                        A_ma, A_i, A_sj,
                                                        hypre_CsrsvDataInfoU(csrsv_data),
                                                        f_data, u_data,
                                                        rocsparse_solve_policy_auto,
                                                        buffer) );
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_SortCSRRocsparse
 *
 * @brief This functions sorts values and column indices in each row in
 *        ascending order OUT-OF-PLACE
 *
 * @param[in] n Number of rows
 * @param[in] m Number of columns
 * @param[in] nnzA Number of nonzeroes
 * @param[in] *d_ia (Unsorted) Row indices
 * @param[in,out] *d_ja_sorted On Start: Unsorted column indices.
 *                             On return: Sorted column indices
 * @param[in,out] *d_a_sorted On Start: Unsorted values.
 *                            On Return: Sorted values corresponding with
 *                                       column indices
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SortCSRRocsparse( HYPRE_Int            n,
                        HYPRE_Int            m,
                        HYPRE_Int            num_nonzeros,
                        rocsparse_mat_descr  descrA,
                        const HYPRE_Int     *d_ia,
                        HYPRE_Int           *d_ja_sorted,
                        HYPRE_Complex       *d_a_sorted )
{
   rocsparse_handle  handle = hypre_HandleCusparseHandle(hypre_handle());
   size_t            pBufferSizeInBytes = 0;
   void             *pBuffer = NULL;
   HYPRE_Int        *P = NULL;
   HYPRE_Complex    *d_a_tmp;

   // FIXME: There is not in-place version of csr sort in rocSPARSE currently, so we make
   //        a temporary copy of the data for gthr, sort that, and then copy the sorted values
   //        back to the array being returned. Where there is an in-place version available,
   //        we should use it.
   d_a_tmp  = hypre_TAlloc(HYPRE_Complex, num_nonzeros, HYPRE_MEMORY_DEVICE);

   HYPRE_ROCSPARSE_CALL( rocsparse_csrsort_buffer_size(handle, n, m, num_nonzeros,
                                                       d_ia, d_ja_sorted,
                                                       &pBufferSizeInBytes) );

   pBuffer = hypre_TAlloc(char, pBufferSizeInBytes, HYPRE_MEMORY_DEVICE);
   P       = hypre_TAlloc(HYPRE_Int, num_nonzeros, HYPRE_MEMORY_DEVICE);

   HYPRE_ROCSPARSE_CALL( rocsparse_create_identity_permutation(handle, num_nonzeros, P) );
   HYPRE_ROCSPARSE_CALL( rocsparse_csrsort(handle, n, m, num_nonzeros, descrA, d_ia,
                                           d_ja_sorted, P, pBuffer) );
   HYPRE_ROCSPARSE_CALL( hypre_rocsparse_gthr(handle, num_nonzeros, d_a_sorted, d_a_tmp, P,
                                              rocsparse_index_base_zero) );

   hypre_TFree(pBuffer, HYPRE_MEMORY_DEVICE);
   hypre_TFree(P, HYPRE_MEMORY_DEVICE);

   hypre_TMemcpy(d_a_sorted, d_a_tmp, HYPRE_Complex, num_nonzeros,
                 HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

   hypre_TFree(d_a_tmp, HYPRE_MEMORY_DEVICE);

   return hypre_error_flag;
}

#elif defined(HYPRE_USING_ONEMKLSPARSE)

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixTriLowerUpperSolveOnemklsparse
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixTriLowerUpperSolveOnemklsparse(char              uplo,
                                              HYPRE_Int         unit_diag,
                                              hypre_CSRMatrix  *A,
                                              HYPRE_Real       *l1_norms,
                                              HYPRE_Complex    *f_data,
                                              HYPRE_Complex    *u_data )
{
   HYPRE_Int                                *A_j = hypre_CSRMatrixJ(A);
   HYPRE_Complex                            *A_a = hypre_CSRMatrixData(A);
   oneapi::mkl::sparse::matrix_handle_t handle_A = hypre_CSRMatrixGPUMatHandle(A);
   hypre_CsrsvData                   *csrsv_data = hypre_CSRMatrixCsrsvData(A);

   /* Generate sorted matrix */
   hypre_CSRMatrixSortRowOutOfPlace(A);

   /* Generate csrsv data if necessary */
   if (!csrsv_data)
   {
      hypre_CSRMatrixCsrsvData(A) = hypre_CsrsvDataCreate();
      csrsv_data = hypre_CSRMatrixCsrsvData(A);

      hypre_CsrsvDataMatData(csrsv_data) = hypre_TAlloc(HYPRE_Complex,
                                                        hypre_CSRMatrixNumNonzeros(A),
                                                        HYPRE_MEMORY_DEVICE);

      /* Copy the sorted data to csrsv mat data */
      hypre_TMemcpy(hypre_CsrsvDataMatData(csrsv_data), hypre_CSRMatrixSortedData(A),
                    HYPRE_Complex, hypre_CSRMatrixNumNonzeros(A),
                    HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_DEVICE);

      /* if (l1_norms), replace A's diag with l1_norm, and
       * replace zero diag with inf. so as to skip relaxation for this unknown */
      hypre_CSRMatrixData(A) = hypre_CsrsvDataMatData(csrsv_data);
      hypre_CSRMatrixJ(A) = hypre_CSRMatrixSortedJ(A);
      hypre_CSRMatrixReplaceDiagDevice(A, l1_norms, INFINITY, 0.0);
   }

   /* Use sorted column indices and sorted matrix data with modified diagonal */
   hypre_CSRMatrixJ(A) = hypre_CSRMatrixSortedJ(A);
   hypre_CSRMatrixData(A) = hypre_CsrsvDataMatData(csrsv_data);
   hypre_GPUMatDataSetCSRData(A);

   /* Do optimization the first time */
   if ( (!hypre_CsrsvDataAnalyzedL(csrsv_data) && uplo == 'L') ||
        (!hypre_CsrsvDataAnalyzedU(csrsv_data) && uplo == 'U') )
   {
      HYPRE_ONEMKL_CALL( oneapi::mkl::sparse::optimize_trsv( *hypre_HandleComputeStream(hypre_handle()),
                                                             (uplo == 'L') ? oneapi::mkl::uplo::L : oneapi::mkl::uplo::U,
                                                             oneapi::mkl::transpose::N,
                                                             unit_diag ? oneapi::mkl::diag::U : oneapi::mkl::diag::N,
                                                             handle_A,
                                                             {} ).wait() );
   }

   /* Do the triangular solve */
   HYPRE_ONEMKL_CALL( oneapi::mkl::sparse::trsv( *hypre_HandleComputeStream(hypre_handle()),
                                                 (uplo == 'L') ? oneapi::mkl::uplo::L : oneapi::mkl::uplo::U,
                                                 oneapi::mkl::transpose::N,
                                                 unit_diag ? oneapi::mkl::diag::U : oneapi::mkl::diag::N,
                                                 handle_A,
                                                 f_data,
                                                 u_data,
                                                 {} ).wait() );

   /* Restore the original matrix data */
   hypre_CSRMatrixJ(A) = A_j;
   hypre_CSRMatrixData(A) = A_a;
   hypre_GPUMatDataSetCSRData(A);

   return hypre_error_flag;
}
#endif // #if defined(HYPRE_USING_CUSPARSE) #elif defined(HYPRE_USING_ROCSPARSE)

#if defined(HYPRE_USING_CUSPARSE) || defined(HYPRE_USING_ROCSPARSE)

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixILU0
 *
 * TODO (VPM): Change this function's name to hypre_ILU0SetupDevice?
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixILU0(hypre_CSRMatrix *A)
{
   /* Input matrix data */
   HYPRE_Int                 num_rows          = hypre_CSRMatrixNumRows(A);
   HYPRE_Int                 num_cols          = hypre_CSRMatrixNumCols(A);
   HYPRE_Int                 num_nonzeros      = hypre_CSRMatrixNumNonzeros(A);
   HYPRE_Int                *A_i               = hypre_CSRMatrixI(A);
   HYPRE_Int                *A_j               = hypre_CSRMatrixJ(A);
   HYPRE_Complex            *A_data            = hypre_CSRMatrixData(A);

   /* Vendor math sparse libraries data */
#if defined(HYPRE_USING_CUSPARSE)
   csrilu02Info_t            matA_info        = NULL;
   cusparseHandle_t          handle           = hypre_HandleCusparseHandle(hypre_handle());
   cusparseMatDescr_t        descr            = hypre_CSRMatrixGPUMatDescr(A);
   cusparseSolvePolicy_t     analysis_policy  = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
   cusparseSolvePolicy_t     solve_policy     = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
   cusparseStatus_t          status;
   HYPRE_Int                 buffer_size;

#elif defined(HYPRE_USING_ROCSPARSE)
   rocsparse_mat_info        matA_info        = NULL;
   rocsparse_handle          handle           = hypre_HandleCusparseHandle(hypre_handle());
   rocsparse_mat_descr       descr            = hypre_CSRMatrixGPUMatDescr(A);
   rocsparse_analysis_policy analysis_policy  = rocsparse_analysis_policy_reuse;
   rocsparse_solve_policy    solve_policy     = rocsparse_solve_policy_auto;
   rocsparse_status          status;
   size_t                    buffer_size;
#endif

   void                     *buffer           = NULL;
   HYPRE_Int                 zero_pivot;
   char                      errmsg[1024];

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("CSRMatrixILU0");

   /* Sanity check */
   if (num_rows != num_cols)
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "Not a square matrix!");
      return hypre_error_flag;
   }

   /*-------------------------------------------------------------------------------------
    * 1. Sort columns inside each row first, we can't assume that's sorted
    *-------------------------------------------------------------------------------------*/

   hypre_CSRMatrixSortRow(A);

   /*-------------------------------------------------------------------------------------
    * 2. Create info for ilu setup and solve
    *-------------------------------------------------------------------------------------*/

#if defined(HYPRE_USING_CUSPARSE)
   HYPRE_CUSPARSE_CALL(cusparseCreateCsrilu02Info(&matA_info));

#elif defined(HYPRE_USING_ROCSPARSE)
   HYPRE_ROCSPARSE_CALL(rocsparse_create_mat_info(&matA_info));

#endif

   /*-------------------------------------------------------------------------------------
    * 3. Get work array size
    *-------------------------------------------------------------------------------------*/

#if defined(HYPRE_USING_CUSPARSE)
   HYPRE_CUSPARSE_CALL(hypre_cusparse_csrilu02_bufferSize(handle, num_rows, num_nonzeros,
                                                          descr, A_data, A_i, A_j,
                                                          matA_info, &buffer_size));
#elif defined(HYPRE_USING_ROCSPARSE)
   HYPRE_ROCSPARSE_CALL(hypre_rocsparse_csrilu0_buffer_size(handle, num_rows, num_nonzeros,
                                                            descr, A_data, A_i, A_j,
                                                            matA_info, &buffer_size));
#endif

   /*-------------------------------------------------------------------------------------
    * 4. Create work array on the device
    *-------------------------------------------------------------------------------------*/

   buffer = hypre_TAlloc(char, buffer_size, HYPRE_MEMORY_DEVICE);

   /*-------------------------------------------------------------------------------------
    * 5.1 Perform the analysis
    *-------------------------------------------------------------------------------------*/

   hypre_GpuProfilingPushRange("Analysis");
#if defined(HYPRE_USING_CUSPARSE)
   HYPRE_CUSPARSE_CALL(hypre_cusparse_csrilu02_analysis(handle, num_rows, num_nonzeros,
                                                        descr, A_data, A_i, A_j,
                                                        matA_info, analysis_policy, buffer));

#elif defined(HYPRE_USING_ROCSPARSE)
   HYPRE_ROCSPARSE_CALL(hypre_rocsparse_csrilu0_analysis(handle, num_rows, num_nonzeros,
                                                         descr, A_data, A_i, A_j,
                                                         matA_info, analysis_policy,
                                                         solve_policy, buffer));
#endif
   hypre_GpuProfilingPopRange();

   /*-------------------------------------------------------------------------------------
    * 5.2. Check for zero pivots
    *-------------------------------------------------------------------------------------*/

#if defined(HYPRE_USING_CUSPARSE)
   status = cusparseXcsrilu02_zeroPivot(handle, matA_info, &zero_pivot);
   if (status == CUSPARSE_STATUS_ZERO_PIVOT)
   {
      hypre_sprintf(errmsg, "hypre_ILU: found zero pivot at A(%d, %d) after analysis\n",
                    zero_pivot, zero_pivot);
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, errmsg);
      return hypre_error_flag;
   }
   else if (status != CUSPARSE_STATUS_SUCCESS)
   {
      hypre_sprintf(errmsg, "cuSPARSE ERROR (code = %d, %s) at %s:%d\n",
                    status, cusparseGetErrorString(status), __FILE__, __LINE__);
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, errmsg);
      return hypre_error_flag;
   }

#elif defined(HYPRE_USING_ROCSPARSE)
   status = rocsparse_csrsv_zero_pivot(handle, descr, matA_info, &zero_pivot);
   if (status == rocsparse_status_zero_pivot)
   {
      hypre_sprintf(errmsg, "hypre_ILU: found zero pivot at A(%d, %d) after analysis\n",
                    zero_pivot, zero_pivot);
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, errmsg);
      return hypre_error_flag;
   }
   else if (status != rocsparse_status_success)
   {
      hypre_sprintf(errmsg, "rocSPARSE ERROR (code = %d) at %s:%d\n",
                    status, __FILE__, __LINE__);
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, errmsg);
      return hypre_error_flag;
   }
#endif

   /*-------------------------------------------------------------------------------------
    * 6.1 Compute the numerical factorization
    *-------------------------------------------------------------------------------------*/

   hypre_GpuProfilingPushRange("Factorization");
#if defined(HYPRE_USING_CUSPARSE)
   HYPRE_CUSPARSE_CALL(hypre_cusparse_csrilu02(handle, num_rows, num_nonzeros,
                                               descr, A_data, A_i, A_j,
                                               matA_info, solve_policy, buffer));
#elif defined(HYPRE_USING_ROCSPARSE)
   HYPRE_ROCSPARSE_CALL(hypre_rocsparse_csrilu0(handle, num_rows, num_nonzeros,
                                                descr, A_data, A_i, A_j,
                                                matA_info, solve_policy, buffer));
#endif
   hypre_GpuProfilingPopRange();

   /*-------------------------------------------------------------------------------------
    * 6.2 Check for zero pivots
    *-------------------------------------------------------------------------------------*/

#if defined(HYPRE_USING_CUSPARSE)
   status = cusparseXcsrilu02_zeroPivot(handle, matA_info, &zero_pivot);
   if (status == CUSPARSE_STATUS_ZERO_PIVOT)
   {
      hypre_sprintf(errmsg, "hypre_ILU: found zero pivot at A(%d, %d) after factorization\n",
                    zero_pivot, zero_pivot);
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, errmsg);
      return hypre_error_flag;
   }
   else if (status != CUSPARSE_STATUS_SUCCESS)
   {
      hypre_sprintf(errmsg, "cuSPARSE ERROR (code = %d, %s) at %s:%d\n",
                    status, cusparseGetErrorString(status), __FILE__, __LINE__);
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, errmsg);
      return hypre_error_flag;
   }

#elif defined(HYPRE_USING_ROCSPARSE)
   status = rocsparse_csrsv_zero_pivot(handle, descr, matA_info, &zero_pivot);
   if (status == rocsparse_status_zero_pivot)
   {
      hypre_sprintf(errmsg, "hypre_ILU: found zero pivot at A(%d, %d) after factorization\n",
                    zero_pivot, zero_pivot);
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, errmsg);
      return hypre_error_flag;
   }
   else if (status != rocsparse_status_success)
   {
      hypre_sprintf(errmsg, "rocSPARSE ERROR (code = %d) at %s:%d\n",
                    status, __FILE__, __LINE__);
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, errmsg);
      return hypre_error_flag;
   }
#endif

   /*-------------------------------------------------------------------------------------
    * 7. Free memory
    *-------------------------------------------------------------------------------------*/

#if defined(HYPRE_USING_CUSPARSE)
   HYPRE_CUSPARSE_CALL(cusparseDestroyCsrilu02Info(matA_info));

#elif defined(HYPRE_USING_ROCSPARSE)
   HYPRE_ROCSPARSE_CALL(rocsparse_destroy_mat_info(matA_info));
#endif

   /* Free buffer */
   hypre_TFree(buffer, HYPRE_MEMORY_DEVICE);

   hypre_GpuProfilingPopRange();
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

#endif /* #if defined(HYPRE_USING_CUSPARSE) || defined(HYPRE_USING_ROCSPARSE) */

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixSpMVAnalysisDevice
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixSpMVAnalysisDevice(hypre_CSRMatrix *matrix)
{
#if defined(HYPRE_USING_ROCSPARSE)
   HYPRE_ExecutionPolicy  exec = hypre_GetExecPolicy1( hypre_CSRMatrixMemoryLocation(matrix) );
   rocsparse_handle       handle = hypre_HandleCusparseHandle(hypre_handle());

   if (exec == HYPRE_EXEC_DEVICE)
   {
      HYPRE_ROCSPARSE_CALL( hypre_rocsparse_csrmv_analysis(handle,
                                                           rocsparse_operation_none,
                                                           hypre_CSRMatrixNumRows(matrix),
                                                           hypre_CSRMatrixNumCols(matrix),
                                                           hypre_CSRMatrixNumNonzeros(matrix),
                                                           hypre_CSRMatrixGPUMatDescr(matrix),
                                                           hypre_CSRMatrixData(matrix),
                                                           hypre_CSRMatrixI(matrix),
                                                           hypre_CSRMatrixJ(matrix),
                                                           hypre_CSRMatrixGPUMatInfo(matrix)) );
   }
#else
   HYPRE_UNUSED_VAR(matrix);
#endif /* #if defined(HYPRE_USING_ROCSPARSE) */

   return hypre_error_flag;
}

#endif /* #if defined(HYPRE_USING_GPU) */
