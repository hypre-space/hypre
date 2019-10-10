/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_utilities.h"

#if defined(HYPRE_USING_KOKKOS)
#include <Kokkos_Core.hpp>
#endif

void hypre_SetExecPolicy( HYPRE_Int policy )
{
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
   if ( policy == HYPRE_EXEC_HOST || policy == HYPRE_EXEC_DEVICE)
   {
      hypre_HandleDefaultExecPolicy(hypre_handle) = policy;
   }
#endif
}

/*---------------------------------------------------
 * hypre_GetExecPolicy
 * Return execution policy based on memory locations
 *---------------------------------------------------*/
/* for unary operation */
HYPRE_Int
hypre_GetExecPolicy1(HYPRE_Int location)
{
   HYPRE_Int exec = HYPRE_EXEC_UNSET;

   location = hypre_GetActualMemLocation(location);

   switch (location)
   {
      case HYPRE_MEMORY_HOST :
         exec = HYPRE_EXEC_HOST;
         break;
      case HYPRE_MEMORY_HOST_PINNED :
         exec = HYPRE_EXEC_HOST;
         break;
      case HYPRE_MEMORY_DEVICE :
         exec = HYPRE_EXEC_DEVICE;
         break;
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
      case HYPRE_MEMORY_SHARED :
         exec = hypre_HandleDefaultExecPolicy(hypre_handle);
         break;
#endif
   }

   return exec;
}

/* for binary operation */
HYPRE_Int
hypre_GetExecPolicy2(HYPRE_Int location1,
                     HYPRE_Int location2)
{
   location1 = hypre_GetActualMemLocation(location1);
   location2 = hypre_GetActualMemLocation(location2);

   /* HOST_PINNED has the same exec policy as HOST */
   if (location1 == HYPRE_MEMORY_HOST_PINNED)
   {
      location1 = HYPRE_MEMORY_HOST;
   }

   if (location2 == HYPRE_MEMORY_HOST_PINNED)
   {
      location2 = HYPRE_MEMORY_HOST;
   }

   /* no policy for these combinations */
   if ( (location1 == HYPRE_MEMORY_HOST && location2 == HYPRE_MEMORY_DEVICE) ||
        (location2 == HYPRE_MEMORY_HOST && location1 == HYPRE_MEMORY_DEVICE) )
   {
      return HYPRE_EXEC_UNSET;
   }

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
   /* policy for S-S can be HOST or DEVICE. Choose HOST by default */
   if (location1 == HYPRE_MEMORY_SHARED && location2 == HYPRE_MEMORY_SHARED)
   {
      return hypre_HandleDefaultExecPolicy(hypre_handle);
   }
#endif

   if (location1 == HYPRE_MEMORY_HOST || location2 == HYPRE_MEMORY_HOST)
   {
      return HYPRE_EXEC_HOST;
   }

   if (location1 == HYPRE_MEMORY_DEVICE || location2 == HYPRE_MEMORY_DEVICE)
   {
      return HYPRE_EXEC_DEVICE;
   }

   return HYPRE_EXEC_UNSET;
}

hypre_Handle *hypre_handle = NULL;

hypre_Handle*
hypre_HandleCreate()
{
   hypre_Handle *handle = hypre_CTAlloc(hypre_Handle, 1, HYPRE_MEMORY_HOST);

   /* set default options */
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
   hypre_HandleDefaultExecPolicy(handle)            = HYPRE_EXEC_HOST;
   hypre_HandleCudaDevice(handle)                   = 0;
   hypre_HandleCudaComputeStreamNum(handle)         = 0;
   hypre_HandleCudaPrefetchStreamNum(handle)        = 1;
   hypre_HandleCudaComputeStreamSyncDefault(handle) = 1;
   handle->spgemm_use_cusparse                      = 0; // TODO: accessor func #ifdef
   handle->spgemm_num_passes                        = 3;
   /* 1: naive overestimate, 2: naive underestimate, 3: Cohen's algorithm */
   handle->spgemm_rownnz_estimate_method            = 3;
   handle->spgemm_rownnz_estimate_nsamples          = 32;
   handle->spgemm_rownnz_estimate_mult_factor       = 1.5;
   handle->spgemm_hash_type                         = 'L';

   hypre_HandleCudaComputeStreamSync(handle).clear();
   hypre_HandleCudaComputeStreamSyncPush( handle, hypre_HandleCudaComputeStreamSyncDefault(handle) );
#endif // #if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)

   return handle;
}

HYPRE_Int
hypre_HandleDestroy(hypre_Handle *hypre_handle_)
{
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
   HYPRE_Int i;

   hypre_TFree(hypre_handle_->cuda_reduce_buffer, HYPRE_MEMORY_DEVICE);

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

   hypre_printf("Proc [global %d/%d, local %d/%d] can see %d GPUs and is running on %d\n",
                 myid, nproc, myNodeid, NodeSize, nDevices, device_id);

   return hypre_error_flag;
}

#endif //#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)

/******************************************************************************
 *
 * hypre initialization
 *
 *****************************************************************************/

HYPRE_Int
HYPRE_Init( hypre_int argc, char *argv[] )
{
   hypre_handle = hypre_HandleCreate();

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
   hypre_SetDevice(-1, hypre_handle);

   /* To include the cost of creating streams/cudahandles in HYPRE_Init */
   /* If not here, will be done at the first use */
   hypre_HandleCudaComputeStream(hypre_handle);
   hypre_HandleCudaPrefetchStream(hypre_handle);
#endif

#if defined(HYPRE_USING_CUBLAS)
   hypre_HandleCublasHandle(hypre_handle);
#endif

#if defined(HYPRE_USING_CUSPARSE)
   hypre_HandleCusparseHandle(hypre_handle);
   hypre_HandleCusparseMatDescr(hypre_handle);
#endif

#if defined(HYPRE_USING_CURAND)
   hypre_HandleCurandGenerator(hypre_handle);
#endif

#if defined(HYPRE_USING_KOKKOS)
   Kokkos::initialize (argc, argv);
#endif

   /* Check if cuda arch flags in compiling match the device */
#if defined(HYPRE_USING_CUDA)
   hypre_CudaCompileFlagCheck();
#endif

#if defined(HYPRE_USING_DEVICE_OPENMP)
   HYPRE_OMPOffloadOn();
#endif

   return hypre_error_flag;
}

/******************************************************************************
 *
 * hypre finalization
 *
 *****************************************************************************/

/* declared in "struct_communication.c" */
extern HYPRE_Complex *global_recv_buffer, *global_send_buffer;
extern HYPRE_Int      global_recv_size, global_send_size;

HYPRE_Int
HYPRE_Finalize()
{
   hypre_HandleDestroy(hypre_handle);

#if defined(HYPRE_USING_KOKKOS)
   Kokkos::finalize ();
#endif

   hypre_TFree(global_send_buffer, HYPRE_MEMORY_DEVICE);
   hypre_TFree(global_recv_buffer, HYPRE_MEMORY_DEVICE);

   //if (cudaSuccess == cudaPeekAtLastError() ) hypre_printf("OK...\n");

#if defined(HYPRE_USING_CUDA)
   HYPRE_CUDA_CALL( cudaGetLastError() );
#endif

   return hypre_error_flag;
}

