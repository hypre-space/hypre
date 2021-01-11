/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_utilities.h"
#include "_hypre_utilities.hpp"

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
   hypre_HandleDefaultExecPolicy(hypre_handle_) = HYPRE_EXEC_HOST;
   hypre_HandleStructExecPolicy(hypre_handle_) = HYPRE_EXEC_DEVICE;
   hypre_HandleCudaData(hypre_handle_) = hypre_CudaDataCreate();
#endif

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
   hypre_CudaDataDestroy(hypre_HandleCudaData(hypre_handle_));
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
   HYPRE_CUDA_CALL( cudaSetDevice(device_id) );
   omp_set_default_device(device_id);
#endif

   hypre_HandleCudaDevice(hypre_handle_) = device_id;

#if defined(HYPRE_DEBUG) && defined(HYPRE_PRINT_ERRORS)
   hypre_printf("Proc [global %d/%d, local %d/%d] can see %d GPUs and is running on %d\n",
                 myid, nproc, myNodeid, NodeSize, nDevices, device_id);
#endif

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

#if defined(HYPRE_USING_GPU)
   HYPRE_CUDA_CALL( cudaGetLastError() );

   /* Notice: the cudaStream created is specific to the device
    * that was in effect when you created the stream.
    * So, we should first set the device and create the streams
    */
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
    */
   if ( hypre_HandleCubDevAllocator(_hypre_handle) ||
        hypre_HandleCubUvmAllocator(_hypre_handle) )
   {
      hypre_printf("ERROR: CUB Allocators have been setup ... \n");
   }
#endif

#if defined(HYPRE_USING_UMPIRE)
   hypre_UmpireInit(_hypre_handle);
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
#if defined(HYPRE_USING_UMPIRE)
   //hypre_UmpireFinalize(_hypre_handle);
#endif

   hypre_HandleDestroy(_hypre_handle);

   _hypre_handle = NULL;

   /*
#if defined(HYPRE_USING_KOKKOS)
   Kokkos::finalize ();
#endif
   */

#if defined(HYPRE_USING_CUDA)
/*
#if defined(HYPRE_DEBUG)
   if (cudaSuccess == cudaPeekAtLastError() )
   {
      hypre_printf("OK...\n");
   }
#endif
*/
   HYPRE_CUDA_CALL( cudaGetLastError() );
#endif

   return hypre_error_flag;
}

/******************************************************************************
 *
 * hypre Umpire
 *
 *****************************************************************************/

#if defined(HYPRE_USING_UMPIRE)
HYPRE_Int
hypre_UmpireInit(hypre_Handle *hypre_handle_)
{
   umpire_resourcemanager_get_instance(&hypre_HandleUmpireResourceMan(hypre_handle_));

   hypre_HandleUmpireDevicePoolSize(hypre_handle_) = 4LL * 1024 * 1024 * 1024;
   hypre_HandleUmpireUMPoolSize(hypre_handle_)     = 4LL * 1024 * 1024 * 1024;
   hypre_HandleUmpireHostPoolSize(hypre_handle_)   = 4LL * 1024 * 1024 * 1024;
   hypre_HandleUmpirePinnedPoolSize(hypre_handle_) = 4LL * 1024 * 1024 * 1024;

   strcpy(hypre_HandleUmpireDevicePoolName(hypre_handle_), "HYPRE_DEVICE_POOL");
   strcpy(hypre_HandleUmpireUMPoolName(hypre_handle_),     "HYPRE_UM_POOL");
   strcpy(hypre_HandleUmpireHostPoolName(hypre_handle_),   "HYPRE_HOST_POOL");
   strcpy(hypre_HandleUmpirePinnedPoolName(hypre_handle_), "HYPRE_PINNED_POOL");

   return hypre_error_flag;
}


HYPRE_Int
hypre_UmpireFinalize(hypre_Handle *hypre_handle_)
{
   umpire_resourcemanager *rm_ptr = &hypre_HandleUmpireResourceMan(hypre_handle_);
   umpire_allocator allocator;

#if defined(HYPRE_USING_UMPIRE_HOST)
   {
      const char *pool_name = hypre_HandleUmpireHostPoolName(hypre_handle_);
      umpire_resourcemanager_get_allocator_by_name(rm_ptr, pool_name, &allocator);
      umpire_allocator_release(&allocator);
   }
#endif

#if defined(HYPRE_USING_UMPIRE_DEVICE)
   {
      const char *pool_name = hypre_HandleUmpireDevicePoolName(hypre_handle_);
      umpire_resourcemanager_get_allocator_by_name(rm_ptr, pool_name, &allocator);
      umpire_allocator_release(&allocator);
   }
#endif

#if defined(HYPRE_USING_UMPIRE_UM)
   {
      const char *pool_name = hypre_HandleUmpireUMPoolName(hypre_handle_);
      umpire_resourcemanager_get_allocator_by_name(rm_ptr, pool_name, &allocator);
      umpire_allocator_release(&allocator);
   }
#endif

#if defined(HYPRE_USING_UMPIRE_PINNED)
   {
      const char *pool_name = hypre_HandleUmpirePinnedPoolName(hypre_handle_);
      umpire_resourcemanager_get_allocator_by_name(rm_ptr, pool_name, &allocator);
      umpire_allocator_release(&allocator);
   }
#endif

   return hypre_error_flag;
}

HYPRE_Int
HYPRE_SetUmpireDevicePoolSize(size_t nbytes)
{
   hypre_HandleUmpireDevicePoolSize(hypre_handle()) = nbytes;

   return hypre_error_flag;
}

HYPRE_Int
HYPRE_SetUmpireUMPoolSize(size_t nbytes)
{
   hypre_HandleUmpireUMPoolSize(hypre_handle()) = nbytes;

   return hypre_error_flag;
}

HYPRE_Int
HYPRE_SetUmpireHostPoolSize(size_t nbytes)
{
   hypre_HandleUmpireHostPoolSize(hypre_handle()) = nbytes;

   return hypre_error_flag;
}

HYPRE_Int
HYPRE_SetUmpirePinnedPoolSize(size_t nbytes)
{
   hypre_HandleUmpirePinnedPoolSize(hypre_handle()) = nbytes;

   return hypre_error_flag;
}

HYPRE_Int
HYPRE_SetUmpireDevicePoolName(const char *pool_name)
{
   if (strlen(pool_name) > HYPRE_UMPIRE_POOL_NAME_MAX_LEN)
   {
      hypre_error_in_arg(1);

      return hypre_error_flag;
   }

   strcpy(hypre_HandleUmpireDevicePoolName(hypre_handle()), pool_name);

   return hypre_error_flag;
}

HYPRE_Int
HYPRE_SetUmpireUMPoolName(const char *pool_name)
{
   if (strlen(pool_name) > HYPRE_UMPIRE_POOL_NAME_MAX_LEN)
   {
      hypre_error_in_arg(1);

      return hypre_error_flag;
   }

   strcpy(hypre_HandleUmpireUMPoolName(hypre_handle()), pool_name);

   return hypre_error_flag;
}

HYPRE_Int
HYPRE_SetUmpireHostPoolName(const char *pool_name)
{
   if (strlen(pool_name) > HYPRE_UMPIRE_POOL_NAME_MAX_LEN)
   {
      hypre_error_in_arg(1);

      return hypre_error_flag;
   }

   strcpy(hypre_HandleUmpireHostPoolName(hypre_handle()), pool_name);

   return hypre_error_flag;
}

HYPRE_Int
HYPRE_SetUmpirePinnedPoolName(const char *pool_name)
{
   if (strlen(pool_name) > HYPRE_UMPIRE_POOL_NAME_MAX_LEN)
   {
      hypre_error_in_arg(1);

      return hypre_error_flag;
   }

   strcpy(hypre_HandleUmpirePinnedPoolName(hypre_handle()), pool_name);

   return hypre_error_flag;
}

#endif /* #if defined(HYPRE_USING_UMPIRE) */

/******************************************************************************
 *
 * HYPRE memory location
 *
 *****************************************************************************/

HYPRE_Int
HYPRE_SetMemoryLocation(HYPRE_MemoryLocation memory_location)
{
   hypre_HandleMemoryLocation(hypre_handle()) = memory_location;

   return hypre_error_flag;
}

HYPRE_Int
HYPRE_GetMemoryLocation(HYPRE_MemoryLocation *memory_location)
{
   *memory_location = hypre_HandleMemoryLocation(hypre_handle());

   return hypre_error_flag;
}

/******************************************************************************
 *
 * HYPRE execution policy
 *
 *****************************************************************************/

HYPRE_Int
HYPRE_SetExecutionPolicy(HYPRE_ExecutionPolicy exec_policy)
{
   hypre_HandleDefaultExecPolicy(hypre_handle()) = exec_policy;

   return hypre_error_flag;
}

HYPRE_Int
HYPRE_GetExecutionPolicy(HYPRE_ExecutionPolicy *exec_policy)
{
   *exec_policy = hypre_HandleDefaultExecPolicy(hypre_handle());

   return hypre_error_flag;
}

