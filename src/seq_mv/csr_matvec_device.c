/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Matvec functions for hypre_CSRMatrix class.
 *
 *****************************************************************************/

#include "_hypre_seq_mv.h"
#include "_hypre_utilities.hpp"
#include "seq_mv.hpp"

#if defined(HYPRE_USING_GPU) || defined(HYPRE_USING_DEVICE_OPENMP)

#if defined(HYPRE_USING_CUSPARSE)
static HYPRE_Int
hypre_CSRMatrixMatvecCusparse( HYPRE_Int        trans,
                               HYPRE_Complex    alpha,
                               hypre_CSRMatrix *A,
                               hypre_Vector    *x,
                               HYPRE_Complex    beta,
                               hypre_Vector    *y,
                               HYPRE_Int        offset );
#endif

#if defined(HYPRE_USING_ROCSPARSE)
static HYPRE_Int
hypre_CSRMatrixMatvecRocsparse( HYPRE_Int        trans,
                                HYPRE_Complex    alpha,
                                hypre_CSRMatrix *A,
                                hypre_Vector    *x,
                                HYPRE_Complex    beta,
                                hypre_Vector    *y,
                                HYPRE_Int        offset );
#endif

#if defined(HYPRE_USING_ONEMKLSPARSE)
static HYPRE_Int
hypre_CSRMatrixMatvecOnemklsparse( HYPRE_Int        trans,
                                   HYPRE_Complex    alpha,
                                   hypre_CSRMatrix *A,
                                   hypre_Vector    *x,
                                   HYPRE_Complex    beta,
                                   hypre_Vector    *y,
                                   HYPRE_Int        offset );
#endif

#if CUSPARSE_VERSION >= CUSPARSE_NEWSPMM_VERSION
#define HYPRE_CUSPARSE_SPMM_ALG CUSPARSE_SPMM_CSR_ALG3

#elif CUSPARSE_VERSION >= CUSPARSE_NEWAPI_VERSION
#define HYPRE_CUSPARSE_SPMM_ALG CUSPARSE_SPMM_CSR_ALG1

#else
#define HYPRE_CUSPARSE_SPMM_ALG CUSPARSE_CSRMM_ALG1
#endif

/* y = alpha * A * x + beta * y
 * This function is supposed to be only used inside the other functions in this file
 */
static inline HYPRE_Int
hypre_CSRMatrixMatvecDevice2( HYPRE_Int        trans,
                              HYPRE_Complex    alpha,
                              hypre_CSRMatrix *A,
                              hypre_Vector    *x,
                              HYPRE_Complex    beta,
                              hypre_Vector    *y,
                              HYPRE_Int        offset )
{
   /* Sanity check */
   if (hypre_VectorData(x) == hypre_VectorData(y))
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                        "ERROR::x and y are the same pointer in hypre_CSRMatrixMatvecDevice2");
   }

#if defined(HYPRE_USING_CUSPARSE)  || \
    defined(HYPRE_USING_ROCSPARSE) || \
    defined(HYPRE_USING_ONEMKLSPARSE)

   /* Input variables */
   HYPRE_Int  num_vectors_x      = hypre_VectorNumVectors(x);
   HYPRE_Int  num_vectors_y      = hypre_VectorNumVectors(y);

   /* Local variables */
   HYPRE_Int  use_vendor = hypre_HandleSpMVUseVendor(hypre_handle());

#if defined(HYPRE_USING_CUSPARSE) && CUSPARSE_VERSION >= CUSPARSE_NEWAPI_VERSION
   HYPRE_Int  multivec_storage_x = hypre_VectorMultiVecStorageMethod(x);
   HYPRE_Int  multivec_storage_y = hypre_VectorMultiVecStorageMethod(y);

   /* Force use of hypre's SpMV for row-wise multivectors */
   if ((num_vectors_x > 1 && multivec_storage_x == 1) ||
       (num_vectors_y > 1 && multivec_storage_y == 1))
   {
      use_vendor = 0;
   }
#else
   /* TODO - enable cuda 10, rocsparse, and onemkle sparse support for multi-vectors */
   if (num_vectors_x > 1 || num_vectors_y > 1)
   {
      use_vendor = 0;
   }
#endif

   if (use_vendor)
   {
      hypre_CSRMatrixMatvecVendor(trans, alpha, A, x, beta, y, offset);
   }
   else
#endif // defined(HYPRE_USING_CUSPARSE) || defined(HYPRE_USING_ROCSPARSE) ...
   {
#if defined(HYPRE_USING_GPU)
      hypre_CSRMatrixSpMVDevice(trans, alpha, A, x, beta, y, 0);

#elif defined(HYPRE_USING_DEVICE_OPENMP)
      hypre_CSRMatrixMatvecOMPOffload(trans, alpha, A, x, beta, y, offset);
#endif
   }

   return hypre_error_flag;
}

/* y = alpha * A * x + beta * b */
HYPRE_Int
hypre_CSRMatrixMatvecDevice( HYPRE_Int        trans,
                             HYPRE_Complex    alpha,
                             hypre_CSRMatrix *A,
                             hypre_Vector    *x,
                             HYPRE_Complex    beta,
                             hypre_Vector    *b,
                             hypre_Vector    *y,
                             HYPRE_Int        offset )
{
   //hypre_GpuProfilingPushRange("CSRMatrixMatvec");
   HYPRE_Int   num_vectors = hypre_VectorNumVectors(x);

   // TODO: RL: do we need offset > 0 at all?
   hypre_assert(offset == 0);

   // VPM: offset > 0 does not work with multivectors. Remove offset? See comment above
   hypre_assert(!(offset != 0 && num_vectors > 1));
   hypre_assert(num_vectors > 0);

   HYPRE_Int nx = trans ? hypre_CSRMatrixNumRows(A) : hypre_CSRMatrixNumCols(A);
   HYPRE_Int ny = trans ? hypre_CSRMatrixNumCols(A) : hypre_CSRMatrixNumRows(A);

   //RL: Note the "<=", since the vectors sometimes can be temporary work spaces that have
   //    large sizes than the needed (such as in par_cheby.c)
   hypre_assert(ny <= hypre_VectorSize(y));
   hypre_assert(nx <= hypre_VectorSize(x));
   hypre_assert(ny <= hypre_VectorSize(b));

   //hypre_CSRMatrixPrefetch(A, HYPRE_MEMORY_DEVICE);
   //hypre_SeqVectorPrefetch(x, HYPRE_MEMORY_DEVICE);
   //hypre_SeqVectorPrefetch(b, HYPRE_MEMORY_DEVICE);
   //if (hypre_VectorData(b) != hypre_VectorData(y))
   //{
   //   hypre_SeqVectorPrefetch(y, HYPRE_MEMORY_DEVICE);
   //}

   if (hypre_VectorData(b) != hypre_VectorData(y))
   {
      hypre_TMemcpy( hypre_VectorData(y) + offset,
                     hypre_VectorData(b) + offset,
                     HYPRE_Complex,
                     (ny - offset) * num_vectors,
                     hypre_VectorMemoryLocation(y),
                     hypre_VectorMemoryLocation(b) );

   }

   if (hypre_CSRMatrixNumNonzeros(A) <= 0 || alpha == 0.0)
   {
      hypre_SeqVectorScale(beta, y);
   }
   else
   {
      hypre_CSRMatrixMatvecDevice2(trans, alpha, A, x, beta, y, offset);
   }

#if defined(HYPRE_USING_GPU)
   hypre_SyncComputeStream();
#endif

   //hypre_GpuProfilingPopRange();

   return hypre_error_flag;
}

HYPRE_Int
hypre_CSRMatrixMatvecVendor( HYPRE_Int        trans,
                             HYPRE_Complex    alpha,
                             hypre_CSRMatrix *A,
                             hypre_Vector    *x,
                             HYPRE_Complex    beta,
                             hypre_Vector    *y,
                             HYPRE_Int        offset )
{
#if defined(HYPRE_USING_CUSPARSE)
   return hypre_CSRMatrixMatvecCusparse(trans, alpha, A, x, beta, y, offset);
#elif defined(HYPRE_USING_ROCSPARSE)
   return hypre_CSRMatrixMatvecRocsparse(trans, alpha, A, x, beta, y, offset);
#elif defined(HYPRE_USING_ONEMKLSPARSE)
   return hypre_CSRMatrixMatvecOnemklsparse(trans, alpha, A, x, beta, y, offset);
#else
   HYPRE_UNUSED_VAR(trans);
   HYPRE_UNUSED_VAR(alpha);
   HYPRE_UNUSED_VAR(A);
   HYPRE_UNUSED_VAR(x);
   HYPRE_UNUSED_VAR(beta);
   HYPRE_UNUSED_VAR(y);
   HYPRE_UNUSED_VAR(offset);
   hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                     "Attempting to use device sparse matrix library for SpMV without having compiled support for it!\n");
   return hypre_error_flag;
#endif
}

#if defined(HYPRE_USING_CUSPARSE)
#if CUSPARSE_VERSION >= CUSPARSE_NEWAPI_VERSION

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixMatvecCusparseNewAPI
 *
 * Sparse Matrix/(Multi)Vector interface to cusparse's API 11
 *
 * Note: The descriptor variables are not saved to allow for generic input
 *--------------------------------------------------------------------------*/

static HYPRE_Int
hypre_CSRMatrixMatvecCusparseNewAPI( HYPRE_Int        trans,
                                     HYPRE_Complex    alpha,
                                     hypre_CSRMatrix *A,
                                     hypre_Vector    *x,
                                     HYPRE_Complex    beta,
                                     hypre_Vector    *y,
                                     HYPRE_Int        offset )
{
   /* Input variables */
   HYPRE_Int         num_vectors = hypre_VectorNumVectors(x);
   HYPRE_Int         num_cols    = trans ? hypre_CSRMatrixNumRows(A) : hypre_CSRMatrixNumCols(A);
   HYPRE_Int         num_rows    = trans ? hypre_CSRMatrixNumCols(A) : hypre_CSRMatrixNumRows(A);
   hypre_CSRMatrix  *AT;
   hypre_CSRMatrix  *B;

   /* SpMV data */
   size_t                     bufferSize = 0;
   hypre_GpuMatData          *gpu_mat = hypre_CSRMatrixGetGPUMatData(A);
   char                      *dBuffer = hypre_GpuMatDataSpMVBuffer(gpu_mat);
   cusparseHandle_t           handle = hypre_HandleCusparseHandle(hypre_handle());
   const cudaDataType         data_type = hypre_HYPREComplexToCudaDataType();
   const cusparseIndexType_t  index_type = hypre_HYPREIntToCusparseIndexType();
   const cusparseSpMVAlg_t    spmv_alg =
      (cusparseSpMVAlg_t) hypre_HandleSpMVAlgorithm(hypre_handle());
   const HYPRE_Int            buffer_is_current =
      dBuffer && hypre_GpuMatDataSpMVBufferNumVectors(gpu_mat) == num_vectors &&
      (num_vectors > 1 || hypre_GpuMatDataSpMVBufferAlg(gpu_mat) == (HYPRE_Int) spmv_alg);

   /* Local cusparse descriptor variables */
   cusparseSpMatDescr_t      matA;
   cusparseDnVecDescr_t      vecX, vecY;
   cusparseDnMatDescr_t      matX, matY;

   /* We handle the transpose explicitly to ensure the same output each run
    * and for potential performance improvement memory for AT */
   if (trans)
   {
      hypre_CSRMatrixTransposeDevice(A, &AT, 1);
      B = AT;
   }
   else
   {
      B = A;
   }

   /* Create cuSPARSE vector data structures */
   matA = hypre_CSRMatrixToCusparseSpMat(B, offset);
   if (num_vectors == 1)
   {
      vecX = hypre_VectorToCusparseDnVec(x, 0, num_cols);
      vecY = hypre_VectorToCusparseDnVec(y, offset, num_rows - offset);
   }
   else
   {
      matX = hypre_VectorToCusparseDnMat(x);
      matY = hypre_VectorToCusparseDnMat(y);
   }

   if (!buffer_is_current)
   {
      hypre_TFree(dBuffer, HYPRE_MEMORY_DEVICE);
      dBuffer = NULL;
      hypre_GpuMatDataSpMVBuffer(gpu_mat) = NULL;
      hypre_GpuMatDataSpMVBufferAlg(gpu_mat) = -1;
      hypre_GpuMatDataSpMVBufferNumVectors(gpu_mat) = -1;

      if (num_vectors == 1)
      {
         HYPRE_CUSPARSE_CALL( cusparseSpMV_bufferSize(handle,
                                                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                      &alpha,
                                                      matA,
                                                      vecX,
                                                      &beta,
                                                      vecY,
                                                      data_type,
                                                      spmv_alg,
                                                      &bufferSize) );
      }
      else
      {
         HYPRE_CUSPARSE_CALL( cusparseSpMM_bufferSize(handle,
                                                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                      &alpha,
                                                      matA,
                                                      matX,
                                                      &beta,
                                                      matY,
                                                      data_type,
                                                      HYPRE_CUSPARSE_SPMM_ALG,
                                                      &bufferSize) );
      }

      dBuffer = hypre_TAlloc(char, bufferSize, HYPRE_MEMORY_DEVICE);
      hypre_GpuMatDataSpMVBuffer(gpu_mat) = dBuffer;
      hypre_GpuMatDataSpMVBufferAlg(gpu_mat) = (HYPRE_Int) spmv_alg;
      hypre_GpuMatDataSpMVBufferNumVectors(gpu_mat) = num_vectors;

#if CUSPARSE_VERSION >= CUSPARSE_NEWSPMM_VERSION
      if (num_vectors > 1)
      {
         HYPRE_CUSPARSE_CALL( cusparseSpMM_preprocess(handle,
                                                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                      &alpha,
                                                      matA,
                                                      matX,
                                                      &beta,
                                                      matY,
                                                      data_type,
                                                      HYPRE_CUSPARSE_SPMM_ALG,
                                                      dBuffer) );
      }
#endif
   }

   if (num_vectors == 1)
   {
      HYPRE_CUSPARSE_CALL( cusparseSpMV(handle,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha,
                                        matA,
                                        vecX,
                                        &beta,
                                        vecY,
                                        data_type,
                                        spmv_alg,
                                        dBuffer) );
   }
   else
   {
      HYPRE_CUSPARSE_CALL( cusparseSpMM(handle,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha,
                                        matA,
                                        matX,
                                        &beta,
                                        matY,
                                        data_type,
                                        HYPRE_CUSPARSE_SPMM_ALG,
                                        dBuffer) );
   }

#if defined(HYPRE_USING_GPU)
   hypre_SyncComputeStream();
#endif

   /* Free memory */
   HYPRE_CUSPARSE_CALL( cusparseDestroySpMat(matA) );
   /* For SpMV, vecX and vecY are cached on their hypre_Vector and owned by hypre_GpuVecData. */
   if (num_vectors > 1)
   {
      HYPRE_CUSPARSE_CALL( cusparseDestroyDnMat(matX) );
      HYPRE_CUSPARSE_CALL( cusparseDestroyDnMat(matY) );
   }
   if (trans)
   {
      hypre_CSRMatrixDestroy(AT);
   }

   return hypre_error_flag;
}

#else // #if CUSPARSE_VERSION >= CUSPARSE_NEWAPI_VERSION

static HYPRE_Int
hypre_CSRMatrixMatvecCusparseOldAPI( HYPRE_Int        trans,
                                     HYPRE_Complex    alpha,
                                     hypre_CSRMatrix *A,
                                     hypre_Vector    *x,
                                     HYPRE_Complex    beta,
                                     hypre_Vector    *y,
                                     HYPRE_Int        offset )
{
#ifdef HYPRE_BIGINT
#error "ERROR: cusparse old API should not be used when bigint is enabled!"
#endif
   cusparseHandle_t handle = hypre_HandleCusparseHandle(hypre_handle());
   cusparseMatDescr_t descr = hypre_CSRMatrixGPUMatDescr(A);
   hypre_CSRMatrix *B;

   if (trans)
   {
      hypre_CSRMatrixTransposeDevice(A, &B, 1);
   }
   else
   {
      B = A;
   }

   HYPRE_CUSPARSE_CALL( hypre_cusparse_csrmv(handle,
                                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                                             hypre_CSRMatrixNumRows(B) - offset,
                                             hypre_CSRMatrixNumCols(B),
                                             hypre_CSRMatrixNumNonzeros(B),
                                             &alpha,
                                             descr,
                                             hypre_CSRMatrixData(B),
                                             hypre_CSRMatrixI(B) + offset,
                                             hypre_CSRMatrixJ(B),
                                             hypre_VectorData(x),
                                             &beta,
                                             hypre_VectorData(y) + offset) );

   if (trans)
   {
      hypre_CSRMatrixDestroy(B);
   }

   return hypre_error_flag;
}

#endif // #if CUSPARSE_VERSION >= CUSPARSE_NEWAPI_VERSION

static HYPRE_Int
hypre_CSRMatrixMatvecCusparse( HYPRE_Int        trans,
                               HYPRE_Complex    alpha,
                               hypre_CSRMatrix *A,
                               hypre_Vector    *x,
                               HYPRE_Complex    beta,
                               hypre_Vector    *y,
                               HYPRE_Int        offset )
{
#if CUSPARSE_VERSION >= CUSPARSE_NEWAPI_VERSION
   /* Luke E: The generic API is techinically supported on 10.1,10.2 as a preview,
    * with Dscrmv being deprecated. However, there are limitations.
    * While in Cuda < 11, there are specific mentions of using csr2csc involving
    * transposed matrix products with dcsrm*,
    * they are not present in SpMV interface.
    */
   hypre_CSRMatrixMatvecCusparseNewAPI(trans, alpha, A, x, beta, y, offset);

#else
   hypre_CSRMatrixMatvecCusparseOldAPI(trans, alpha, A, x, beta, y, offset);
#endif

   return hypre_error_flag;
}

#endif // #if defined(HYPRE_USING_CUSPARSE)

#if defined(HYPRE_USING_ROCSPARSE)

#if (ROCSPARSE_VERSION >= 200000)

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixSpMVCreateRocsparseSpMatDescr
 *--------------------------------------------------------------------------*/

static HYPRE_Int
hypre_CSRMatrixSpMVCreateRocsparseSpMatDescr( hypre_CSRMatrix     *matrix,
                                              HYPRE_Int            offset,
                                              rocsparse_datatype   compute_type,
                                              rocsparse_indextype  idx_type )
{
   hypre_GpuMatData       *gpu_mat = hypre_CSRMatrixGetGPUMatData(matrix);
   rocsparse_spmat_descr   cached_mat = hypre_GpuMatDataSpMVSpMatDescr(gpu_mat);

   if (cached_mat)
   {
      return hypre_error_flag;
   }

#if (ROCSPARSE_VERSION >= 300000)
   {
      rocsparse_spmat_descr new_mat = NULL;

      HYPRE_ROCSPARSE_CALL( rocsparse_create_const_csr_descr(
                               (rocsparse_const_spmat_descr *) &new_mat,
                               (int64_t)(hypre_CSRMatrixNumRows(matrix) - offset),
                               (int64_t) hypre_CSRMatrixNumCols(matrix),
                               (int64_t) hypre_CSRMatrixNumNonzeros(matrix),
                               (const void *)(hypre_CSRMatrixI(matrix) + offset),
                               (const void *) hypre_CSRMatrixJ(matrix),
                               (const void *) hypre_CSRMatrixData(matrix),
                               idx_type,
                               idx_type,
                               rocsparse_index_base_zero,
                               compute_type) );
      hypre_GpuMatDataSpMVSpMatDescr(gpu_mat) = new_mat;
   }
#else
   {
      rocsparse_spmat_descr new_mat = NULL;

      HYPRE_ROCSPARSE_CALL( rocsparse_create_csr_descr(
                               &new_mat,
                               (int64_t)(hypre_CSRMatrixNumRows(matrix) - offset),
                               (int64_t) hypre_CSRMatrixNumCols(matrix),
                               (int64_t) hypre_CSRMatrixNumNonzeros(matrix),
                               hypre_CSRMatrixI(matrix) + offset,
                               hypre_CSRMatrixJ(matrix),
                               hypre_CSRMatrixData(matrix),
                               idx_type,
                               idx_type,
                               rocsparse_index_base_zero,
                               compute_type) );
      hypre_GpuMatDataSpMVSpMatDescr(gpu_mat) = new_mat;
   }
#endif

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixSpMVAnalysisRocsparseDevice
 *
 * Perform one-time generic SpMV analysis for a matrix. Per rocSPARSE, the
 * analysis stage depends only on the sparse matrix and algorithm, not on the
 * x/y data pointers.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_CSRMatrixSpMVAnalysisRocsparseDevice( hypre_CSRMatrix *matrix,
                                            HYPRE_Int        offset )
{
   HYPRE_ExecutionPolicy  exec = hypre_GetExecPolicy1( hypre_CSRMatrixMemoryLocation(matrix) );

   if (exec != HYPRE_EXEC_DEVICE)
   {
      return hypre_error_flag;
   }

   rocsparse_handle       handle = hypre_HandleCusparseHandle(hypre_handle());
   hypre_GpuMatData      *gpu_mat = hypre_CSRMatrixGetGPUMatData(matrix);
   rocsparse_spmv_alg     alg = (rocsparse_spmv_alg) hypre_HandleSpMVAlgorithm(hypre_handle());
   rocsparse_datatype     compute_type;
   rocsparse_indextype    idx_type;
   HYPRE_Int              num_rows = hypre_CSRMatrixNumRows(matrix) - offset;
   HYPRE_Int              num_cols = hypre_CSRMatrixNumCols(matrix);
   HYPRE_Complex         *scratch_x = NULL;
   HYPRE_Complex         *scratch_y = NULL;
   rocsparse_const_dnvec_descr vecX = NULL;
   rocsparse_dnvec_descr vecY = NULL;

#if defined(HYPRE_SINGLE)
   hypre_float alpha_v = (hypre_float) 1.0;
   hypre_float beta_v = (hypre_float) 0.0;
   compute_type = rocsparse_datatype_f32_r;
#else
   hypre_double alpha_v = (hypre_double) 1.0;
   hypre_double beta_v = (hypre_double) 0.0;
   compute_type = rocsparse_datatype_f64_r;
#endif

#if defined(HYPRE_BIGINT)
   idx_type = rocsparse_indextype_i64;
#else
   idx_type = rocsparse_indextype_i32;
#endif

#if (ROCSPARSE_VERSION >= 400002)
   if (hypre_GpuMatDataSpMVPreprocessAlg(gpu_mat) == (HYPRE_Int) alg &&
       hypre_GpuMatDataSpMVDescr(gpu_mat))
#else
   if (hypre_GpuMatDataSpMVPreprocessAlg(gpu_mat) == (HYPRE_Int) alg &&
       hypre_GpuMatDataSpMVSpMatDescr(gpu_mat))
#endif
   {
      return hypre_error_flag;
   }

   hypre_CSRMatrixSpMVCreateRocsparseSpMatDescr(matrix, offset, compute_type, idx_type);

   scratch_x = hypre_TAlloc(HYPRE_Complex, num_cols, HYPRE_MEMORY_DEVICE);
   scratch_y = hypre_TAlloc(HYPRE_Complex, num_rows, HYPRE_MEMORY_DEVICE);
   hypre_Memset(scratch_x, 0, (size_t) num_cols * sizeof(HYPRE_Complex), HYPRE_MEMORY_DEVICE);
   hypre_Memset(scratch_y, 0, (size_t) num_rows * sizeof(HYPRE_Complex), HYPRE_MEMORY_DEVICE);

    /* Create dense-vector descriptors. Older rocSPARSE releases (< 3.0.0)
       do not provide rocsparse_create_const_dnvec_descr, so fall back to
       creating a non-const dnvec and cast it. */
#if (ROCSPARSE_VERSION >= 300000)
    HYPRE_ROCSPARSE_CALL( rocsparse_create_const_dnvec_descr(&vecX,
                                                             (int64_t) num_cols,
                                                             (const void *) scratch_x,
                                                             compute_type) );
#else
    {
       rocsparse_dnvec_descr tmp_vecX = NULL;
       HYPRE_ROCSPARSE_CALL( rocsparse_create_dnvec_descr(&tmp_vecX,
                                                          (int64_t) num_cols,
                                                          (void *) scratch_x,
                                                          compute_type) );
       vecX = (rocsparse_const_dnvec_descr) tmp_vecX;
    }
#endif
    HYPRE_ROCSPARSE_CALL( rocsparse_create_dnvec_descr(&vecY,
                                                       (int64_t) num_rows,
                                                       (void *) scratch_y,
                                                       compute_type) );

#if (ROCSPARSE_VERSION >= 400002)
   {
      rocsparse_spmv_descr spmv_descr = hypre_GpuMatDataSpMVDescr(gpu_mat);
      rocsparse_const_spmat_descr cached_mat =
         (rocsparse_const_spmat_descr) hypre_GpuMatDataSpMVSpMatDescr(gpu_mat);
      const rocsparse_const_dnvec_descr vecY_const =
         (rocsparse_const_dnvec_descr) vecY;
      const rocsparse_operation spmv_operation = rocsparse_operation_none;
      size_t analysis_buffer_size = 0;
      size_t compute_buffer_size = 0;
      size_t buffer_size;

      if (spmv_descr)
      {
         HYPRE_ROCSPARSE_CALL( rocsparse_destroy_spmv_descr(spmv_descr) );
         spmv_descr = NULL;
      }

      HYPRE_ROCSPARSE_CALL( rocsparse_create_spmv_descr(&spmv_descr) );
      HYPRE_ROCSPARSE_CALL( rocsparse_spmv_set_input(handle, spmv_descr,
                                                     rocsparse_spmv_input_alg,
                                                     &alg, sizeof(alg), NULL) );
      HYPRE_ROCSPARSE_CALL( rocsparse_spmv_set_input(handle, spmv_descr,
                                                     rocsparse_spmv_input_operation,
                                                     &spmv_operation,
                                                     sizeof(spmv_operation), NULL) );
      HYPRE_ROCSPARSE_CALL( rocsparse_spmv_set_input(handle, spmv_descr,
                                                     rocsparse_spmv_input_scalar_datatype,
                                                     &compute_type,
                                                     sizeof(compute_type), NULL) );
      HYPRE_ROCSPARSE_CALL( rocsparse_spmv_set_input(handle, spmv_descr,
                                                     rocsparse_spmv_input_compute_datatype,
                                                     &compute_type,
                                                     sizeof(compute_type), NULL) );

      HYPRE_ROCSPARSE_CALL( rocsparse_v2_spmv_buffer_size(handle, spmv_descr,
                                                          cached_mat, vecX, vecY_const,
                                                          rocsparse_v2_spmv_stage_analysis,
                                                          &analysis_buffer_size, NULL) );
      HYPRE_ROCSPARSE_CALL( rocsparse_v2_spmv_buffer_size(handle, spmv_descr,
                                                          cached_mat, vecX, vecY_const,
                                                          rocsparse_v2_spmv_stage_compute,
                                                          &compute_buffer_size, NULL) );

      buffer_size = analysis_buffer_size > compute_buffer_size ?
                    analysis_buffer_size : compute_buffer_size;

      if (buffer_size > hypre_GpuMatDataSpMVBufferSize(gpu_mat))
      {
         hypre_TFree(hypre_GpuMatDataSpMVBuffer(gpu_mat), HYPRE_MEMORY_DEVICE);
         hypre_GpuMatDataSpMVBuffer(gpu_mat) = NULL;
         hypre_GpuMatDataSpMVBufferSize(gpu_mat) = 0;
         if (buffer_size > 0)
         {
            hypre_GpuMatDataSpMVBuffer(gpu_mat) = hypre_TAlloc(char, buffer_size,
                                                               HYPRE_MEMORY_DEVICE);
            hypre_GpuMatDataSpMVBufferSize(gpu_mat) = buffer_size;
         }
      }

      HYPRE_ROCSPARSE_CALL( rocsparse_v2_spmv(handle, spmv_descr,
                                              (const void *) &alpha_v,
                                              cached_mat, vecX,
                                              (const void *) &beta_v,
                                              vecY,
                                              rocsparse_v2_spmv_stage_analysis,
                                              buffer_size,
                                              hypre_GpuMatDataSpMVBuffer(gpu_mat),
                                              NULL) );

      hypre_GpuMatDataSpMVDescr(gpu_mat) = spmv_descr;
   }
#else
   {
      rocsparse_spmat_descr cached_mat = hypre_GpuMatDataSpMVSpMatDescr(gpu_mat);
      size_t needed_buffer_size = 0;

       /* rocsparse_spmv has different signatures across releases. Newer
          versions accept an explicit stage argument; older ones do not. */
#if (ROCSPARSE_VERSION >= 300000)
       HYPRE_ROCSPARSE_CALL( rocsparse_spmv(handle,
                                            rocsparse_operation_none,
                                            (const void *) &alpha_v,
                                            cached_mat,
                                            vecX,
                                            (const void *) &beta_v,
                                            vecY,
                                            compute_type,
                                            alg,
                                            rocsparse_spmv_stage_buffer_size,
                                            &needed_buffer_size,
                                            NULL) );
#else
       HYPRE_ROCSPARSE_CALL( rocsparse_spmv(handle,
                                            rocsparse_operation_none,
                                            (const void *) &alpha_v,
                                            cached_mat,
                                            vecX,
                                            (const void *) &beta_v,
                                            vecY,
                                            compute_type,
                                            alg,
                                            &needed_buffer_size,
                                            NULL) );
#endif

      if (needed_buffer_size > hypre_GpuMatDataSpMVBufferSize(gpu_mat))
      {
         hypre_TFree(hypre_GpuMatDataSpMVBuffer(gpu_mat), HYPRE_MEMORY_DEVICE);
         hypre_GpuMatDataSpMVBuffer(gpu_mat) = NULL;
         hypre_GpuMatDataSpMVBufferSize(gpu_mat) = 0;
         if (needed_buffer_size > 0)
         {
            hypre_GpuMatDataSpMVBuffer(gpu_mat) = hypre_TAlloc(char, needed_buffer_size,
                                                               HYPRE_MEMORY_DEVICE);
            hypre_GpuMatDataSpMVBufferSize(gpu_mat) = needed_buffer_size;
         }
      }

#if (ROCSPARSE_VERSION >= 300000)
       HYPRE_ROCSPARSE_CALL( rocsparse_spmv(handle,
                                            rocsparse_operation_none,
                                            (const void *) &alpha_v,
                                            cached_mat,
                                            vecX,
                                            (const void *) &beta_v,
                                            vecY,
                                            compute_type,
                                            alg,
                                            rocsparse_spmv_stage_preprocess,
                                            &needed_buffer_size,
                                            hypre_GpuMatDataSpMVBuffer(gpu_mat)) );
#endif
   }
#endif

   hypre_GpuMatDataSpMVPreprocessAlg(gpu_mat) = (HYPRE_Int) alg;

   HYPRE_ROCSPARSE_CALL( rocsparse_destroy_dnvec_descr(vecX) );
   HYPRE_ROCSPARSE_CALL( rocsparse_destroy_dnvec_descr(vecY) );
   hypre_TFree(scratch_x, HYPRE_MEMORY_DEVICE);
   hypre_TFree(scratch_y, HYPRE_MEMORY_DEVICE);

   return hypre_error_flag;
}

#endif /* ROCSPARSE_VERSION >= 200000 */

static HYPRE_Int
hypre_CSRMatrixMatvecRocsparse( HYPRE_Int        trans,
                                HYPRE_Complex    alpha,
                                hypre_CSRMatrix *A,
                                hypre_Vector    *x,
                                HYPRE_Complex    beta,
                                hypre_Vector    *y,
                                HYPRE_Int        offset )
{
   rocsparse_handle handle = hypre_HandleCusparseHandle(hypre_handle());

   hypre_CSRMatrix *B;

   if (trans)
   {
      hypre_CSRMatrixTransposeDevice(A, &B, 1);
   }
   else
   {
      B = A;
   }

#if (ROCSPARSE_VERSION >= 200000)
   {
      rocsparse_const_dnvec_descr vecX = NULL;
      rocsparse_dnvec_descr vecY = NULL;
      rocsparse_datatype compute_type;
      hypre_GpuMatData *gpu_mat = hypre_CSRMatrixGetGPUMatData(B);
#if defined(HYPRE_SINGLE)
      hypre_float alpha_v = (hypre_float) alpha;
      hypre_float beta_v = (hypre_float) beta;
      compute_type = rocsparse_datatype_f32_r;
#else
      hypre_double alpha_v = (hypre_double) alpha;
      hypre_double beta_v = (hypre_double) beta;
      compute_type = rocsparse_datatype_f64_r;
#endif

      hypre_CSRMatrixSpMVAnalysisRocsparseDevice(B, offset);

      vecX = (rocsparse_const_dnvec_descr) hypre_VectorGetRocsparseDnVecDescr(
                x,
                (int64_t) hypre_CSRMatrixNumCols(B),
                (void *) hypre_VectorData(x),
                compute_type);
      vecY = hypre_VectorGetRocsparseDnVecDescr(y,
                                                (int64_t)(hypre_CSRMatrixNumRows(B) - offset),
                                                hypre_VectorData(y) + offset,
                                                compute_type);

#if (ROCSPARSE_VERSION >= 400002)
      {
         rocsparse_spmv_descr spmv_descr = hypre_GpuMatDataSpMVDescr(gpu_mat);
         rocsparse_const_spmat_descr cached_mat =
            (rocsparse_const_spmat_descr) hypre_GpuMatDataSpMVSpMatDescr(gpu_mat);
         size_t buffer_size = hypre_GpuMatDataSpMVBufferSize(gpu_mat);

         HYPRE_ROCSPARSE_CALL( rocsparse_v2_spmv(handle, spmv_descr,
                                                 (const void *) &alpha_v,
                                                 cached_mat, vecX,
                                                 (const void *) &beta_v,
                                                 vecY,
                                                 rocsparse_v2_spmv_stage_compute,
                                                 buffer_size,
                                                 hypre_GpuMatDataSpMVBuffer(gpu_mat),
                                                 NULL) );
      }
#else
      {
         rocsparse_spmat_descr cached_mat = hypre_GpuMatDataSpMVSpMatDescr(gpu_mat);
         size_t needed_buffer_size = hypre_GpuMatDataSpMVBufferSize(gpu_mat);

#if (ROCSPARSE_VERSION >= 300000)
          HYPRE_ROCSPARSE_CALL( rocsparse_spmv(handle,
                                               rocsparse_operation_none,
                                               (const void *) &alpha_v,
                                               cached_mat,
                                               vecX,
                                               (const void *) &beta_v,
                                               vecY,
                                               compute_type,
                                               (rocsparse_spmv_alg) hypre_GpuMatDataSpMVPreprocessAlg(gpu_mat),
                                               rocsparse_spmv_stage_compute,
                                               &needed_buffer_size,
                                               hypre_GpuMatDataSpMVBuffer(gpu_mat)) );
#else
          HYPRE_ROCSPARSE_CALL( rocsparse_spmv(handle,
                                               rocsparse_operation_none,
                                               (const void *) &alpha_v,
                                               cached_mat,
                                               vecX,
                                               (const void *) &beta_v,
                                               vecY,
                                               compute_type,
                                               (rocsparse_spmv_alg) hypre_GpuMatDataSpMVPreprocessAlg(gpu_mat),
                                               &needed_buffer_size,
                                               hypre_GpuMatDataSpMVBuffer(gpu_mat)) );
#endif
      }
#endif
   }
#else
   /* Legacy rocSPARSE SpMV path */
   rocsparse_mat_descr descr = hypre_CSRMatrixGPUMatDescr(B);
   rocsparse_mat_info info = hypre_CSRMatrixGPUMatInfo(B);

   HYPRE_ROCSPARSE_CALL( hypre_rocsparse_csrmv(handle,
                                               rocsparse_operation_none,
                                               hypre_CSRMatrixNumRows(B) - offset,
                                               hypre_CSRMatrixNumCols(B),
                                               hypre_CSRMatrixNumNonzeros(B),
                                               &alpha,
                                               descr,
                                               hypre_CSRMatrixData(B),
                                               hypre_CSRMatrixI(B) + offset,
                                               hypre_CSRMatrixJ(B),
                                               info,
                                               hypre_VectorData(x),
                                               &beta,
                                               hypre_VectorData(y) + offset) );
#endif /* ROCSPARSE_VERSION >= 200000 */

   if (trans)
   {
      hypre_CSRMatrixDestroy(B);
   }

   return hypre_error_flag;
}
#endif // #if defined(HYPRE_USING_ROCSPARSE)

#if defined(HYPRE_USING_ONEMKLSPARSE)
static HYPRE_Int
hypre_CSRMatrixMatvecOnemklsparse( HYPRE_Int        trans,
                                   HYPRE_Complex    alpha,
                                   hypre_CSRMatrix *A,
                                   hypre_Vector    *x,
                                   HYPRE_Complex    beta,
                                   hypre_Vector    *y,
                                   HYPRE_Int        offset )
{
   sycl::queue *compute_queue = hypre_HandleComputeStream(hypre_handle());
   hypre_CSRMatrix *AT;
   oneapi::mkl::sparse::matrix_handle_t matA_handle = hypre_CSRMatrixGPUMatHandle(A);
   hypre_GPUMatDataSetCSRData(A);

   if (trans)
   {
      hypre_CSRMatrixTransposeDevice(A, &AT, 1);
      hypre_GPUMatDataSetCSRData(AT);
      matA_handle = hypre_CSRMatrixGPUMatHandle(AT);
   }

   HYPRE_ONEMKL_CALL( oneapi::mkl::sparse::gemv(*compute_queue,
                                                oneapi::mkl::transpose::nontrans,
                                                alpha,
                                                matA_handle,
                                                hypre_VectorData(x),
                                                beta,
                                                hypre_VectorData(y) + offset).wait() );

   if (trans)
   {
      hypre_CSRMatrixDestroy(AT);
   }

   return hypre_error_flag;
}
#endif // #if defined(HYPRE_USING_ROCSPARSE)

#endif // #if defined(HYPRE_USING_GPU) || defined(HYPRE_USING_DEVICE_OPENMP)
