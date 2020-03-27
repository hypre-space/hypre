/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_utilities.h"

/*
#if defined(HYPRE_USING_KOKKOS)
#include <Kokkos_Core.hpp>
#endif
*/

/* global variable _hypre_handle:
 * Outside this file, do NOT access it directly,
 * but use hypre_handle() instead (see hypre_handle.h) */
hypre_Handle *_hypre_handle = NULL;

/* accessor to the global ``_hypre_handle'' */
hypre_Handle*
hypre_handle()
{
   if (!_hypre_handle)
   {
      _hypre_handle = hypre_HandleCreate();
   }

   return _hypre_handle;
}

hypre_Handle*
hypre_HandleCreate()
{
   hypre_Handle *hypre_handle_ = hypre_CTAlloc(hypre_Handle, 1, HYPRE_MEMORY_HOST);

   hypre_HandleMemoryLocation(hypre_handle_) = HYPRE_MEMORY_DEVICE;

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)

   /* default CUDA options */
   hypre_HandleDefaultExecPolicy(hypre_handle_)            = HYPRE_EXEC_HOST;
   hypre_HandleCudaDevice(hypre_handle_)                   = 0;
   hypre_HandleCudaComputeStreamNum(hypre_handle_)         = 0;
#if defined(HYPRE_USING_UNIFIED_MEMORY)
   hypre_HandleCudaComputeStreamSync(hypre_handle_)        = 1;
#else
   hypre_HandleCudaComputeStreamSync(hypre_handle_)        = 0;
#endif

   /* SpGeMM */
#ifdef HYPRE_USING_CUSPARSE
   hypre_HandleSpgemmUseCusparse(hypre_handle_)            = 1;
#else
   hypre_HandleSpgemmUseCusparse(hypre_handle_)            = 0;
#endif
   hypre_handle_->spgemm_num_passes                        = 3;
   /* 1: naive overestimate, 2: naive underestimate, 3: Cohen's algorithm */
   hypre_handle_->spgemm_rownnz_estimate_method            = 3;
   hypre_handle_->spgemm_rownnz_estimate_nsamples          = 32;
   hypre_handle_->spgemm_rownnz_estimate_mult_factor       = 1.5;
   hypre_handle_->spgemm_hash_type                         = 'L';

   /* cub */
#ifdef HYPRE_USING_CUB_ALLOCATOR
   hypre_handle_->cub_bin_growth                           = 8u;
   hypre_handle_->cub_min_bin                              = 1u;
   hypre_handle_->cub_max_bin                              = (hypre_uint) -1;
   hypre_handle_->cub_max_cached_bytes                     = (size_t) -1;
   hypre_handle_->cub_dev_allocator                        = NULL;
   hypre_handle_->cub_um_allocator                         = NULL;
#endif

#endif // #if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)

   return hypre_handle_;
}

HYPRE_Int
hypre_HandleDestroy(hypre_Handle *hypre_handle_)
{
   if (!hypre_handle_)
   {
      return hypre_error_flag;
   }

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
   HYPRE_Int i;

   hypre_TFree(hypre_HandleCudaReduceBuffer(hypre_handle_),     HYPRE_MEMORY_DEVICE);
   hypre_TFree(hypre_HandleStructCommRecvBuffer(hypre_handle_), HYPRE_MEMORY_DEVICE);
   hypre_TFree(hypre_HandleStructCommSendBuffer(hypre_handle_), HYPRE_MEMORY_DEVICE);

#if defined(HYPRE_USING_CURAND)
   if (hypre_handle_->curand_gen)
   {
      HYPRE_CURAND_CALL( curandDestroyGenerator(hypre_handle_->curand_gen) );
   }
#endif

#if defined(HYPRE_USING_CUBLAS)
   if (hypre_handle_->cublas_handle)
   {
      HYPRE_CUBLAS_CALL( cublasDestroy(hypre_handle_->cublas_handle) );
   }
#endif

#if defined(HYPRE_USING_CUSPARSE)
   if (hypre_handle_->cusparse_handle)
   {
      HYPRE_CUSPARSE_CALL( cusparseDestroy(hypre_handle_->cusparse_handle) );
   }

   if (hypre_handle_->cusparse_mat_descr)
   {
      HYPRE_CUSPARSE_CALL( cusparseDestroyMatDescr(hypre_handle_->cusparse_mat_descr) );
   }
#endif

   for (i = 0; i < HYPRE_MAX_NUM_STREAMS; i++)
   {
      if (hypre_handle_->cuda_streams[i])
      {
         HYPRE_CUDA_CALL( cudaStreamDestroy(hypre_handle_->cuda_streams[i]) );
      }
   }
#endif

#ifdef HYPRE_USING_CUB_ALLOCATOR
   delete hypre_handle_->cub_dev_allocator;
   delete hypre_handle_->cub_um_allocator;
#endif

   hypre_TFree(hypre_handle_, HYPRE_MEMORY_HOST);

   return hypre_error_flag;
}

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
/* use_device == -1 to let Hypre decide on which device to use */
HYPRE_Int
hypre_SetDevice(HYPRE_Int use_device, hypre_Handle *hypre_handle_)
{
   HYPRE_Int myid, nproc, myNodeid, NodeSize;
   HYPRE_Int device_id;
   hypre_MPI_Comm node_comm;

   // TODO should not use COMM_WORLD
   hypre_MPI_Comm_rank(hypre_MPI_COMM_WORLD, &myid);
   hypre_MPI_Comm_size(hypre_MPI_COMM_WORLD, &nproc);

   hypre_MPI_Comm_split_type(hypre_MPI_COMM_WORLD, hypre_MPI_COMM_TYPE_SHARED,
                             myid, hypre_MPI_INFO_NULL, &node_comm);
   hypre_MPI_Comm_rank(node_comm, &myNodeid);
   hypre_MPI_Comm_size(node_comm, &NodeSize);
   hypre_MPI_Comm_free(&node_comm);

   HYPRE_Int nDevices;
#if defined(HYPRE_USING_CUDA)
   HYPRE_CUDA_CALL( cudaGetDeviceCount(&nDevices) );
#else
   nDevices = omp_get_num_devices();
#endif

   if (use_device < 0)
   {
      device_id = myNodeid % nDevices;
   }
   else
   {
      device_id = use_device;
   }

#if defined(HYPRE_USING_CUDA)
   HYPRE_CUDA_CALL( cudaSetDevice(device_id) );
#else
   omp_set_default_device(device_id);
#endif

   hypre_HandleCudaDevice(hypre_handle_) = device_id;

   /*
   hypre_printf("Proc [global %d/%d, local %d/%d] can see %d GPUs and is running on %d\n",
                 myid, nproc, myNodeid, NodeSize, nDevices, device_id);
   */

   return hypre_error_flag;
}

#endif //#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)

/******************************************************************************
 *
 * hypre initialization
 *
 *****************************************************************************/

HYPRE_Int
HYPRE_Init()
{
   if (!_hypre_handle)
   {
      _hypre_handle = hypre_HandleCreate();
   }

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)

   HYPRE_CUDA_CALL( cudaGetLastError() );

   hypre_SetDevice(-1, _hypre_handle);

   /* To include the cost of creating streams/cudahandles in HYPRE_Init */
   /* If not here, will be done at the first use */
   hypre_HandleCudaComputeStream(_hypre_handle);
   //hypre_HandleCudaPrefetchStream(_hypre_handle);
#endif

#if defined(HYPRE_USING_CUBLAS)
   hypre_HandleCublasHandle(_hypre_handle);
#endif

#if defined(HYPRE_USING_CUSPARSE)
   hypre_HandleCusparseHandle(_hypre_handle);
   hypre_HandleCusparseMatDescr(_hypre_handle);
#endif

#if defined(HYPRE_USING_CURAND)
   hypre_HandleCurandGenerator(_hypre_handle);
#endif

   /*
#if defined(HYPRE_USING_KOKKOS)
   Kokkos::initialize (argc, argv);
#endif
   */

   /* Check if cuda arch flags in compiling match the device */
#if defined(HYPRE_USING_CUDA)
   hypre_CudaCompileFlagCheck();
#endif

#if defined(HYPRE_USING_DEVICE_OPENMP)
   HYPRE_OMPOffloadOn();
#endif

#ifdef HYPRE_USING_CUB_ALLOCATOR
   /* Keep this check here at the end of HYPRE_Init()
    * Make sure that CUB Allocator has not been setup in HYPRE_Init,
    * otherwise users are not able to set the parameters of CUB
    * Note: hypre_HandleCubCachingDeviceAllocator and
    *       hypre_HandleCubCachingManagedAllocator
    *       are not used, since allocation would happen therein,
    *       which is not wanted here */
   if (_hypre_handle->cub_dev_allocator || _hypre_handle->cub_um_allocator)
   {
      hypre_printf("ERROR: CUB Allocators have been setup ... \n");
   }
#endif

   return hypre_error_flag;
}

/******************************************************************************
 *
 * hypre finalization
 *
 *****************************************************************************/

HYPRE_Int
HYPRE_Finalize()
{
   hypre_HandleDestroy(_hypre_handle);

   /*
#if defined(HYPRE_USING_KOKKOS)
   Kokkos::finalize ();
#endif
   */

   //if (cudaSuccess == cudaPeekAtLastError() ) hypre_printf("OK...\n");

#if defined(HYPRE_USING_CUDA)
   HYPRE_CUDA_CALL( cudaGetLastError() );
#endif

   return hypre_error_flag;
}

