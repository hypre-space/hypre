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
#define HYPRE_CUSPARSE_SPMV_ALG CUSPARSE_SPMV_CSR_ALG2
#define HYPRE_CUSPARSE_SPMM_ALG CUSPARSE_SPMM_CSR_ALG3

#elif CUSPARSE_VERSION >= CUSPARSE_NEWAPI_VERSION
#define HYPRE_CUSPARSE_SPMV_ALG CUSPARSE_CSRMV_ALG2
#define HYPRE_CUSPARSE_SPMM_ALG CUSPARSE_SPMM_CSR_ALG1

#else
#define HYPRE_CUSPARSE_SPMV_ALG CUSPARSE_CSRMV_ALG2
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
   size_t                    bufferSize = 0;
   char                     *dBuffer    = hypre_CSRMatrixGPUMatSpMVBuffer(A);
   cusparseHandle_t          handle     = hypre_HandleCusparseHandle(hypre_handle());
   const cudaDataType        data_type  = hypre_HYPREComplexToCudaDataType();
   const cusparseIndexType_t index_type = hypre_HYPREIntToCusparseIndexType();

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

   if (!dBuffer)
   {
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
                                                      HYPRE_CUSPARSE_SPMV_ALG,
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
      hypre_CSRMatrixGPUMatSpMVBuffer(A) = dBuffer;

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
                                        HYPRE_CUSPARSE_SPMV_ALG,
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

#if ROCSPARSE_VERSION >= 200000
   {
      rocsparse_spmat_descr cached_mat;
      rocsparse_const_dnvec_descr vecX = NULL;
      rocsparse_dnvec_descr vecY = NULL;
      rocsparse_datatype compute_type;
      hypre_GpuMatData *gpu_mat;
      const void *x_ptr = hypre_VectorData(x);
      void       *y_ptr = hypre_VectorData(y) + offset;

#if defined(HYPRE_SINGLE)
      hypre_float alpha_v = (hypre_float) alpha;
      hypre_float beta_v = (hypre_float) beta;
      compute_type = rocsparse_datatype_f32_r;
#else
      hypre_double alpha_v = (hypre_double) alpha;
      hypre_double beta_v = (hypre_double) beta;
      compute_type = rocsparse_datatype_f64_r;
#endif

#if defined(HYPRE_BIGINT)
      rocsparse_indextype idx_type = rocsparse_indextype_i64;
#else
      rocsparse_indextype idx_type = rocsparse_indextype_i32;
#endif

      gpu_mat = hypre_CSRMatrixGetGPUMatData(B);
      cached_mat = hypre_GpuMatDataSpMVSpMatDescr(gpu_mat);

      vecX = (rocsparse_const_dnvec_descr) hypre_VectorGetRocsparseDnVecDescr(
                x,
                (int64_t) hypre_CSRMatrixNumCols(B),
                (void *) x_ptr,
                compute_type);
      vecY = hypre_VectorGetRocsparseDnVecDescr(y,
                                                (int64_t)(hypre_CSRMatrixNumRows(B) - offset),
                                                y_ptr,
                                                compute_type);

      if (!cached_mat)
      {
         rocsparse_spmat_descr new_mat = NULL;

#if (ROCSPARSE_VERSION >= 300000)
         HYPRE_ROCSPARSE_CALL( rocsparse_create_const_csr_descr(
                                  (rocsparse_const_spmat_descr *) &new_mat,
                                  (int64_t)(hypre_CSRMatrixNumRows(B) - offset),
                                  (int64_t) hypre_CSRMatrixNumCols(B),
                                  (int64_t) hypre_CSRMatrixNumNonzeros(B),
                                  (const void *)(hypre_CSRMatrixI(B) + offset),
                                  (const void *) hypre_CSRMatrixJ(B),
                                  (const void *) hypre_CSRMatrixData(B),
                                  idx_type,
                                  idx_type,
                                  rocsparse_index_base_zero,
                                  compute_type) );
#else
         HYPRE_ROCSPARSE_CALL( rocsparse_create_csr_descr(
                                  &new_mat,
                                  (int64_t)(hypre_CSRMatrixNumRows(B) - offset),
                                  (int64_t) hypre_CSRMatrixNumCols(B),
                                  (int64_t) hypre_CSRMatrixNumNonzeros(B),
                                  hypre_CSRMatrixI(B) + offset,
                                  hypre_CSRMatrixJ(B),
                                  hypre_CSRMatrixData(B),
                                  idx_type,
                                  idx_type,
                                  rocsparse_index_base_zero,
                                  compute_type) );
#endif
         hypre_GpuMatDataSpMVSpMatDescr(gpu_mat) = new_mat;
         cached_mat = new_mat;
      }

      rocsparse_spmv_alg alg = (rocsparse_spmv_alg) hypre_HandleSpMVAlgorithm(hypre_handle());

#if ROCSPARSE_VERSION >= 400002
      {
         /* rocsparse v2_spmv (rocSPARSE >= 4.0.2) */
         rocsparse_spmv_descr spmv_descr = hypre_GpuMatDataSpMVDescr(gpu_mat);
         HYPRE_Int analysis_alg = hypre_GpuMatDataSpMVPreprocessAlg(gpu_mat);
         size_t buffer_size;
         const rocsparse_const_dnvec_descr vecY_const =
            (rocsparse_const_dnvec_descr) vecY;

         if (analysis_alg < 0 || analysis_alg != (HYPRE_Int) alg ||
             hypre_GpuMatDataSpMVPreprocessXPtr(gpu_mat) != x_ptr ||
             hypre_GpuMatDataSpMVPreprocessYPtr(gpu_mat) != y_ptr)
         {
            size_t analysis_buffer_size = 0;
            size_t compute_buffer_size = 0;
            const rocsparse_operation spmv_operation = rocsparse_operation_none;

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
            hypre_GpuMatDataSpMVPreprocessAlg(gpu_mat) = (HYPRE_Int) alg;
            hypre_GpuMatDataSpMVPreprocessXPtr(gpu_mat) = x_ptr;
            hypre_GpuMatDataSpMVPreprocessYPtr(gpu_mat) = y_ptr;
         }
         else
         {
            spmv_descr = hypre_GpuMatDataSpMVDescr(gpu_mat);
            buffer_size = hypre_GpuMatDataSpMVBufferSize(gpu_mat);
         }

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
         /* rocsparse_spmv (rocSPARSE >= 2.0.0, < 4.0.2) */
         size_t needed_buffer_size = 0;
         HYPRE_Int preprocess_alg = hypre_GpuMatDataSpMVPreprocessAlg(gpu_mat);

         if (preprocess_alg < 0 || preprocess_alg != (HYPRE_Int) alg ||
             hypre_GpuMatDataSpMVPreprocessXPtr(gpu_mat) != x_ptr ||
             hypre_GpuMatDataSpMVPreprocessYPtr(gpu_mat) != y_ptr)
         {
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
            hypre_GpuMatDataSpMVPreprocessAlg(gpu_mat) = (HYPRE_Int) alg;
            hypre_GpuMatDataSpMVPreprocessXPtr(gpu_mat) = x_ptr;
            hypre_GpuMatDataSpMVPreprocessYPtr(gpu_mat) = y_ptr;
         }
         else
         {
            needed_buffer_size = hypre_GpuMatDataSpMVBufferSize(gpu_mat);
         }

         HYPRE_ROCSPARSE_CALL( rocsparse_spmv(handle,
                                              rocsparse_operation_none,
                                              (const void *) &alpha_v,
                                              cached_mat,
                                              vecX,
                                              (const void *) &beta_v,
                                              vecY,
                                              compute_type,
                                              alg,
                                              rocsparse_spmv_stage_compute,
                                              &needed_buffer_size,
                                              hypre_GpuMatDataSpMVBuffer(gpu_mat)) );
      }
#endif /* ROCSPARSE_VERSION >= 400002 */
   }
#else
   /* Legacy rocSPARSE SpMV path */
   rocsparse_mat_descr descr = hypre_CSRMatrixGPUMatDescr(A);
   rocsparse_mat_info info = hypre_CSRMatrixGPUMatInfo(A);

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
