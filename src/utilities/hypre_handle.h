/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * General structures and values
 *
 *****************************************************************************/

#ifndef HYPRE_HANDLE_H
#define HYPRE_HANDLE_H

#ifdef __cplusplus
extern "C++" {
#endif

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
#include <vector>
#endif

typedef struct
{
   HYPRE_Int hypre_error;
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
   HYPRE_Int default_exec_policy;
   HYPRE_Int cuda_device;
   /* by default, hypre puts GPU computations in this stream
    * Do not be confused with the default (null) CUDA stream */
   HYPRE_Int cuda_compute_stream_num;
   HYPRE_Int cuda_prefetch_stream_num;
   HYPRE_Int cuda_compute_stream_sync_default;
   std::vector<HYPRE_Int> cuda_compute_stream_sync;
   curandGenerator_t curand_gen;
   cublasHandle_t cublas_handle;
   cusparseHandle_t cusparse_handle;
   cusparseMatDescr_t cusparse_mat_descr;
   cudaStream_t cuda_streams[HYPRE_MAX_NUM_STREAMS];
   /* work space for hypre's CUDA reducer */
   void* cuda_reduce_buffer;
   /* device spgemm options */
   HYPRE_Int spgemm_use_cusparse;
   HYPRE_Int spgemm_num_passes;
   HYPRE_Int spgemm_rownnz_estimate_method;
   HYPRE_Int spgemm_rownnz_estimate_nsamples;
   float     spgemm_rownnz_estimate_mult_factor;
   char      spgemm_hash_type;
#endif
} hypre_Handle;

extern hypre_Handle *hypre_handle;

hypre_Handle* hypre_HandleCreate();
HYPRE_Int hypre_HandleDestroy(hypre_Handle *hypre_handle_);

/* accessor inline function to hypre_device_csr_handle */

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
static inline HYPRE_Int &
hypre_HandleDefaultExecPolicy(hypre_Handle *hypre_handle_)
{
   return hypre_handle_->default_exec_policy;
}

static inline HYPRE_Int &
hypre_HandleCudaDevice(hypre_Handle *hypre_handle_)
{
   return hypre_handle_->cuda_device;
}

static inline HYPRE_Int &
hypre_HandleCudaComputeStreamNum(hypre_Handle *hypre_handle_)
{
   return hypre_handle_->cuda_compute_stream_num;
}

static inline HYPRE_Int &
hypre_HandleCudaPrefetchStreamNum(hypre_Handle *hypre_handle_)
{
   return hypre_handle_->cuda_prefetch_stream_num;
}

static inline HYPRE_Int &
hypre_HandleCudaComputeStreamSyncDefault(hypre_Handle *hypre_handle_)
{
   return hypre_handle_->cuda_compute_stream_sync_default;
}

static inline std::vector<HYPRE_Int> &
hypre_HandleCudaComputeStreamSync(hypre_Handle *hypre_handle_)
{
   return hypre_handle_->cuda_compute_stream_sync;
}

static inline cudaStream_t
hypre_HandleCudaStream(hypre_Handle *hypre_handle_, HYPRE_Int i)
{
   cudaStream_t stream = 0;
#if defined(HYPRE_USING_CUDA_STREAMS)
   if (i >= HYPRE_MAX_NUM_STREAMS)
   {
      /* return the default stream, i.e., the NULL stream */
      /*
      hypre_printf("CUDA stream %d exceeds the max number %d\n",
                   i, HYPRE_MAX_NUM_STREAMS);
      */
      return NULL;
   }

   if (hypre_handle_->cuda_streams[i])
   {
      return hypre_handle_->cuda_streams[i];
   }

   //HYPRE_CUDA_CALL(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
   HYPRE_CUDA_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamDefault));

   hypre_handle_->cuda_streams[i] = stream;
#endif

   return stream;
}

static inline cudaStream_t
hypre_HandleCudaComputeStream(hypre_Handle *hypre_handle_)
{
   return hypre_HandleCudaStream(hypre_handle_,
                                 hypre_HandleCudaComputeStreamNum(hypre_handle_));
}

static inline cudaStream_t
hypre_HandleCudaPrefetchStream(hypre_Handle *hypre_handle_)
{
   return hypre_HandleCudaStream(hypre_handle_,
                                 hypre_HandleCudaPrefetchStreamNum(hypre_handle_));
}

static inline curandGenerator_t
hypre_HandleCurandGenerator(hypre_Handle *hypre_handle_)
{
   if (hypre_handle_->curand_gen)
   {
      return hypre_handle_->curand_gen;
   }

   curandGenerator_t gen;
   HYPRE_CURAND_CALL( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
   HYPRE_CURAND_CALL( curandSetPseudoRandomGeneratorSeed(gen, 1234ULL) );
   HYPRE_CURAND_CALL( curandSetStream(gen, hypre_HandleCudaComputeStream(hypre_handle_)) );

   hypre_handle_->curand_gen = gen;

   return gen;
}

static inline cublasHandle_t
hypre_HandleCublasHandle(hypre_Handle *hypre_handle_)
{
   if (hypre_handle_->cublas_handle)
   {
      return hypre_handle_->cublas_handle;
   }

   cublasHandle_t handle;
   HYPRE_CUBLAS_CALL( cublasCreate(&handle) );

   HYPRE_CUBLAS_CALL( cublasSetStream(handle, hypre_HandleCudaComputeStream(hypre_handle_)) );

   hypre_handle_->cublas_handle = handle;

   return handle;
}

static inline cusparseHandle_t
hypre_HandleCusparseHandle(hypre_Handle *hypre_handle_)
{
   if (hypre_handle_->cusparse_handle)
   {
      return hypre_handle_->cusparse_handle;
   }

   cusparseHandle_t handle;
   HYPRE_CUSPARSE_CALL( cusparseCreate(&handle) );

   HYPRE_CUSPARSE_CALL( cusparseSetStream(handle, hypre_HandleCudaComputeStream(hypre_handle_)) );

   hypre_handle_->cusparse_handle = handle;

   return handle;
}

static inline cusparseMatDescr_t
hypre_HandleCusparseMatDescr(hypre_Handle *hypre_handle_)
{
   if (hypre_handle_->cusparse_mat_descr)
   {
      return hypre_handle_->cusparse_mat_descr;
   }

   cusparseMatDescr_t mat_descr;
   HYPRE_CUSPARSE_CALL( cusparseCreateMatDescr(&mat_descr) );
   HYPRE_CUSPARSE_CALL( cusparseSetMatType(mat_descr, CUSPARSE_MATRIX_TYPE_GENERAL) );
   HYPRE_CUSPARSE_CALL( cusparseSetMatIndexBase(mat_descr, CUSPARSE_INDEX_BASE_ZERO) );

   hypre_handle_->cusparse_mat_descr = mat_descr;

   return mat_descr;
}

#endif /* defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP) */

static inline void
hypre_HandleCudaComputeStreamSyncPush(hypre_Handle *hypre_handle_, HYPRE_Int sync)
{
#if defined(HYPRE_USING_CUDA) && defined(HYPRE_USING_UNIFIED_MEMORY)
   hypre_HandleCudaComputeStreamSync(hypre_handle_).push_back(sync);
#endif
}

static inline void
hypre_HandleCudaComputeStreamSyncPop(hypre_Handle *hypre_handle_)
{
#if defined(HYPRE_USING_CUDA) && defined(HYPRE_USING_UNIFIED_MEMORY)
   hypre_HandleCudaComputeStreamSync(hypre_handle_).pop_back();
#endif
}

/* synchronize the default stream */
static inline HYPRE_Int
hypre_SyncCudaComputeStream(hypre_Handle *hypre_handle_)
{
#if defined(HYPRE_USING_UNIFIED_MEMORY)
#if defined(HYPRE_USING_CUDA)
   assert(!hypre_HandleCudaComputeStreamSync(hypre_handle_).empty());

   if ( hypre_HandleCudaComputeStreamSync(hypre_handle_).back() )
   {
      HYPRE_CUDA_CALL( cudaStreamSynchronize(hypre_HandleCudaComputeStream(hypre_handle_)) );
   }
#endif
#if defined(HYPRE_USING_DEVICE_OPENMP)
   HYPRE_CUDA_CALL( cudaDeviceSynchronize() );
#endif
#endif /* #if defined(HYPRE_USING_UNIFIED_MEMORY) */
   return hypre_error_flag;
}

#ifdef __cplusplus
}
#endif

#endif

