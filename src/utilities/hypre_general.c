/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_utilities.h"
#include "_hypre_utilities.hpp"

#if defined(HYPRE_USING_CUDA) && defined(HYPRE_USING_UMPIRE)
#include "umpire/interface/umpire.h"

#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"

#include "umpire/strategy/DynamicPool.hpp"
#include "umpire/strategy/AllocationAdvisor.hpp"
#include "umpire/strategy/MonotonicAllocationStrategy.hpp"
#include "umpire/util/Macros.hpp"
int umpire_pool_exists(const char *pool);
#endif
void hypre_umpire_init();

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
  hypre_umpire_init();
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
    */
   if ( hypre_HandleCubDevAllocator(_hypre_handle) ||
        hypre_HandleCubUvmAllocator(_hypre_handle) )
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
#if defined(HYPRE_USING_UMPIRE)
   /* umpire_resourcemanager rm; */
   /* umpire_resourcemanager_get_instance(&rm); */
   /* umpire_allocator dev_allocator; */

   /* umpire_resourcemanager_get_allocator_by_name(&rm, "UM_POOL", &dev_allocator); */
   /* umpire_allocator_release(&dev_allocator); */

   /* umpire_resourcemanager_get_allocator_by_name(&rm, "UM", &dev_allocator); */
   /* umpire_allocator_release(&dev_allocator); */

   /* umpire_resourcemanager_get_allocator_by_name(&rm, "DEVICE", &dev_allocator); */
   /* umpire_allocator_release(&dev_allocator); */

   /* umpire_resourcemanager_get_allocator_by_name(&rm, "DEVICE_POOL", &dev_allocator); */
   /* umpire_allocator_release(&dev_allocator); */
#endif

   hypre_HandleDestroy(_hypre_handle);

   _hypre_handle = NULL;

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

int umpire_pool_exists(const char *pool){
#if defined(HYPRE_USING_UMPIRE)
  umpire_resourcemanager rm;
  umpire_resourcemanager_get_instance(&rm);
  if (umpire_resourcemanager_is_allocator(&rm, pool)) return 1;
  return 0;
#endif
  return 0;
}
void hypre_umpire_init(){
#if defined(HYPRE_USING_UMPIRE)
   printf("WARNING :: EXPERIMENTAL UMPIRE ALLOCATORS IN USE\n");

   /* Need to define pools here unless they are already available */
   if (1){


     size_t pool_size = 1024*1024*1024;
     pool_size*=4;
     
     umpire_resourcemanager rm;
     umpire_resourcemanager_get_instance(&rm);

     /* THE UM POOL */
     umpire_allocator um_allocator,um_pool;
     umpire_resourcemanager_get_allocator_by_name(&rm, "UM", &um_allocator);
   
     if (umpire_pool_exists("HYPRE_UM_POOL")){
       //umpire_resourcemanager_get_allocator_by_name(&rm, "HYPRE_UM_POOL", &um_pool);
       printf("Using extant HYPRE_UM_POOL \n");
     } else{
       umpire_resourcemanager_make_allocator_pool(&rm, "HYPRE_UM_POOL", um_allocator, pool_size , 512, &um_pool);
       printf("Creating new HYPRE_UM_POOL \n");
     }

     /* THE DEVICE POOL */
     umpire_allocator dev_pool, dev_allocator;
     umpire_resourcemanager_get_allocator_by_name(&rm, "DEVICE", &dev_allocator);
     
     if (umpire_pool_exists("HYPRE_DEVICE_POOL")){
       //umpire_resourcemanager_get_allocator_by_name(&rm, "HYPRE_DEVICE_POOL", &dev_pool);
       printf("Using extant HYPRE_DEVICE_POOL \n");
     } else{
       umpire_resourcemanager_make_allocator_pool(&rm, "HYPRE_DEVICE_POOL", dev_allocator, pool_size , 512, &dev_pool);
       printf("Creating new HYPRE_DEVICE_POOL \n");
     }

     
   }
   
#endif

#if defined(HYPRE_USING_UMPIRE_HOST)
   {
     /* THE HOSTPOOL */
     umpire_resourcemanager rm;
     umpire_resourcemanager_get_instance(&rm);
     umpire_allocator host_pool, host_allocator;
     umpire_resourcemanager_get_allocator_by_name(&rm, "HOST", &host_allocator);
     
     if (umpire_pool_exists("HYPRE_HOST_POOL")){
       //umpire_resourcemanager_get_allocator_by_name(&rm, "HYPRE_HOST_POOL", &host_pool);
       printf("Using extant HYPRE_HOST_POOL \n");
     } else{
       size_t pool_size = 1024*1024*1024;
       pool_size*=4;
       umpire_resourcemanager_make_allocator_pool(&rm, "HYPRE_HOST_POOL", host_allocator, pool_size , 512, &host_pool);
       printf("Creating new HYPRE_HOST_POOL \n");
     }
   }
#endif
}
