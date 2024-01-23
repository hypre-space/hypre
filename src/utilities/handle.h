/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
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

struct hypre_DeviceData;
typedef struct hypre_DeviceData hypre_DeviceData;

typedef struct
{
   HYPRE_Int              hypre_error;
   HYPRE_MemoryLocation   memory_location;
   HYPRE_ExecutionPolicy  default_exec_policy;

   /* the device buffers needed to do MPI communication for struct comm */
   HYPRE_Complex         *struct_comm_recv_buffer;
   HYPRE_Complex         *struct_comm_send_buffer;
   HYPRE_Int              struct_comm_recv_buffer_size;
   HYPRE_Int              struct_comm_send_buffer_size;

   /* GPU MPI */
#if defined(HYPRE_USING_GPU) || defined(HYPRE_USING_DEVICE_OPENMP)
   HYPRE_Int              use_gpu_aware_mpi;
#endif

#if defined(HYPRE_USING_GPU)
   hypre_DeviceData      *device_data;
   HYPRE_Int              device_gs_method; /* device G-S options */
#endif

   /* user malloc/free function pointers */
   GPUMallocFunc          user_device_malloc;
   GPUMfreeFunc           user_device_free;

#if defined(HYPRE_USING_UMPIRE)
   char                   umpire_device_pool_name[HYPRE_UMPIRE_POOL_NAME_MAX_LEN];
   char                   umpire_um_pool_name[HYPRE_UMPIRE_POOL_NAME_MAX_LEN];
   char                   umpire_host_pool_name[HYPRE_UMPIRE_POOL_NAME_MAX_LEN];
   char                   umpire_pinned_pool_name[HYPRE_UMPIRE_POOL_NAME_MAX_LEN];
   size_t                 umpire_device_pool_size;
   size_t                 umpire_um_pool_size;
   size_t                 umpire_host_pool_size;
   size_t                 umpire_pinned_pool_size;
   size_t                 umpire_block_size;
   HYPRE_Int              own_umpire_device_pool;
   HYPRE_Int              own_umpire_um_pool;
   HYPRE_Int              own_umpire_host_pool;
   HYPRE_Int              own_umpire_pinned_pool;
   umpire_resourcemanager umpire_rm;
#endif

#if defined(HYPRE_USING_MAGMA)
   magma_queue_t          magma_queue;
#endif
} hypre_Handle;

/* accessor macros to hypre_Handle */
#define hypre_HandleMemoryLocation(hypre_handle)                 ((hypre_handle) -> memory_location)
#define hypre_HandleDefaultExecPolicy(hypre_handle)              ((hypre_handle) -> default_exec_policy)

#define hypre_HandleStructCommRecvBuffer(hypre_handle)           ((hypre_handle) -> struct_comm_recv_buffer)
#define hypre_HandleStructCommSendBuffer(hypre_handle)           ((hypre_handle) -> struct_comm_send_buffer)
#define hypre_HandleStructCommRecvBufferSize(hypre_handle)       ((hypre_handle) -> struct_comm_recv_buffer_size)
#define hypre_HandleStructCommSendBufferSize(hypre_handle)       ((hypre_handle) -> struct_comm_send_buffer_size)

#define hypre_HandleDeviceData(hypre_handle)                     ((hypre_handle) -> device_data)
#define hypre_HandleDeviceGSMethod(hypre_handle)                 ((hypre_handle) -> device_gs_method)
#define hypre_HandleUseGpuAwareMPI(hypre_handle)                 ((hypre_handle) -> use_gpu_aware_mpi)

#define hypre_HandleCurandGenerator(hypre_handle)                hypre_DeviceDataCurandGenerator(hypre_HandleDeviceData(hypre_handle))
#define hypre_HandleCublasHandle(hypre_handle)                   hypre_DeviceDataCublasHandle(hypre_HandleDeviceData(hypre_handle))
#define hypre_HandleCusparseHandle(hypre_handle)                 hypre_DeviceDataCusparseHandle(hypre_HandleDeviceData(hypre_handle))
#define hypre_HandleVendorSolverHandle(hypre_handle)             hypre_DeviceDataVendorSolverHandle(hypre_HandleDeviceData(hypre_handle))
#define hypre_HandleComputeStream(hypre_handle)                  hypre_DeviceDataComputeStream(hypre_HandleDeviceData(hypre_handle))
#define hypre_HandleCubBinGrowth(hypre_handle)                   hypre_DeviceDataCubBinGrowth(hypre_HandleDeviceData(hypre_handle))
#define hypre_HandleCubMinBin(hypre_handle)                      hypre_DeviceDataCubMinBin(hypre_HandleDeviceData(hypre_handle))
#define hypre_HandleCubMaxBin(hypre_handle)                      hypre_DeviceDataCubMaxBin(hypre_HandleDeviceData(hypre_handle))
#define hypre_HandleCubMaxCachedBytes(hypre_handle)              hypre_DeviceDataCubMaxCachedBytes(hypre_HandleDeviceData(hypre_handle))
#define hypre_HandleCubDevAllocator(hypre_handle)                hypre_DeviceDataCubDevAllocator(hypre_HandleDeviceData(hypre_handle))
#define hypre_HandleCubUvmAllocator(hypre_handle)                hypre_DeviceDataCubUvmAllocator(hypre_HandleDeviceData(hypre_handle))
#define hypre_HandleDevice(hypre_handle)                         hypre_DeviceDataDevice(hypre_HandleDeviceData(hypre_handle))
#define hypre_HandleDeviceMaxWorkGroupSize(hypre_handle)         hypre_DeviceDataDeviceMaxWorkGroupSize(hypre_HandleDeviceData(hypre_handle))
#define hypre_HandleDeviceMaxShmemPerBlock(hypre_handle)         hypre_DeviceDataDeviceMaxShmemPerBlock(hypre_HandleDeviceData(hypre_handle))
#define hypre_HandleDeviceMaxShmemPerBlockInited(hypre_handle)   hypre_DeviceDataDeviceMaxShmemPerBlockInited(hypre_HandleDeviceData(hypre_handle))
#define hypre_HandleComputeStreamNum(hypre_handle)               hypre_DeviceDataComputeStreamNum(hypre_HandleDeviceData(hypre_handle))
#define hypre_HandleReduceBuffer(hypre_handle)                   hypre_DeviceDataReduceBuffer(hypre_HandleDeviceData(hypre_handle))
#define hypre_HandleSpgemmUseVendor(hypre_handle)                hypre_DeviceDataSpgemmUseVendor(hypre_HandleDeviceData(hypre_handle))
#define hypre_HandleSpMVUseVendor(hypre_handle)                  hypre_DeviceDataSpMVUseVendor(hypre_HandleDeviceData(hypre_handle))
#define hypre_HandleSpTransUseVendor(hypre_handle)               hypre_DeviceDataSpTransUseVendor(hypre_HandleDeviceData(hypre_handle))
#define hypre_HandleSpgemmAlgorithm(hypre_handle)                hypre_DeviceDataSpgemmAlgorithm(hypre_HandleDeviceData(hypre_handle))
#define hypre_HandleSpgemmBinned(hypre_handle)                   hypre_DeviceDataSpgemmBinned(hypre_HandleDeviceData(hypre_handle))
#define hypre_HandleSpgemmNumBin(hypre_handle)                   hypre_DeviceDataSpgemmNumBin(hypre_HandleDeviceData(hypre_handle))
#define hypre_HandleSpgemmHighestBin(hypre_handle)               hypre_DeviceDataSpgemmHighestBin(hypre_HandleDeviceData(hypre_handle))
#define hypre_HandleSpgemmBlockNumDim(hypre_handle)              hypre_DeviceDataSpgemmBlockNumDim(hypre_HandleDeviceData(hypre_handle))
#define hypre_HandleSpgemmRownnzEstimateMethod(hypre_handle)     hypre_DeviceDataSpgemmRownnzEstimateMethod(hypre_HandleDeviceData(hypre_handle))
#define hypre_HandleSpgemmRownnzEstimateNsamples(hypre_handle)   hypre_DeviceDataSpgemmRownnzEstimateNsamples(hypre_HandleDeviceData(hypre_handle))
#define hypre_HandleSpgemmRownnzEstimateMultFactor(hypre_handle) hypre_DeviceDataSpgemmRownnzEstimateMultFactor(hypre_HandleDeviceData(hypre_handle))
#define hypre_HandleDeviceAllocator(hypre_handle)                hypre_DeviceDataDeviceAllocator(hypre_HandleDeviceData(hypre_handle))
#define hypre_HandleUseGpuRand(hypre_handle)                     hypre_DeviceDataUseGpuRand(hypre_HandleDeviceData(hypre_handle))

#define hypre_HandleUserDeviceMalloc(hypre_handle)               ((hypre_handle) -> user_device_malloc)
#define hypre_HandleUserDeviceMfree(hypre_handle)                ((hypre_handle) -> user_device_free)

#define hypre_HandleUmpireResourceMan(hypre_handle)              ((hypre_handle) -> umpire_rm)
#define hypre_HandleUmpireDevicePoolSize(hypre_handle)           ((hypre_handle) -> umpire_device_pool_size)
#define hypre_HandleUmpireUMPoolSize(hypre_handle)               ((hypre_handle) -> umpire_um_pool_size)
#define hypre_HandleUmpireHostPoolSize(hypre_handle)             ((hypre_handle) -> umpire_host_pool_size)
#define hypre_HandleUmpirePinnedPoolSize(hypre_handle)           ((hypre_handle) -> umpire_pinned_pool_size)
#define hypre_HandleUmpireBlockSize(hypre_handle)                ((hypre_handle) -> umpire_block_size)
#define hypre_HandleUmpireDevicePoolName(hypre_handle)           ((hypre_handle) -> umpire_device_pool_name)
#define hypre_HandleUmpireUMPoolName(hypre_handle)               ((hypre_handle) -> umpire_um_pool_name)
#define hypre_HandleUmpireHostPoolName(hypre_handle)             ((hypre_handle) -> umpire_host_pool_name)
#define hypre_HandleUmpirePinnedPoolName(hypre_handle)           ((hypre_handle) -> umpire_pinned_pool_name)
#define hypre_HandleOwnUmpireDevicePool(hypre_handle)            ((hypre_handle) -> own_umpire_device_pool)
#define hypre_HandleOwnUmpireUMPool(hypre_handle)                ((hypre_handle) -> own_umpire_um_pool)
#define hypre_HandleOwnUmpireHostPool(hypre_handle)              ((hypre_handle) -> own_umpire_host_pool)
#define hypre_HandleOwnUmpirePinnedPool(hypre_handle)            ((hypre_handle) -> own_umpire_pinned_pool)

#define hypre_HandleMagmaQueue(hypre_handle)                     ((hypre_handle) -> magma_queue)

#endif
