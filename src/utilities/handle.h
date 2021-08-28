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

struct hypre_CudaData;
typedef struct hypre_CudaData hypre_CudaData;

typedef struct
{
   HYPRE_Int              hypre_error;
   HYPRE_MemoryLocation   memory_location;
   HYPRE_ExecutionPolicy  default_exec_policy;
   HYPRE_ExecutionPolicy  struct_exec_policy;
#if defined(HYPRE_USING_GPU)
   hypre_CudaData        *cuda_data;
   /* device G-S options */
   HYPRE_Int              device_gs_method;
#endif
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
} hypre_Handle;

/* accessor macros to hypre_Handle */
#define hypre_HandleMemoryLocation(hypre_handle)                 ((hypre_handle) -> memory_location)
#define hypre_HandleDefaultExecPolicy(hypre_handle)              ((hypre_handle) -> default_exec_policy)
#define hypre_HandleStructExecPolicy(hypre_handle)               ((hypre_handle) -> struct_exec_policy)
#define hypre_HandleCudaData(hypre_handle)                       ((hypre_handle) -> cuda_data)
#define hypre_HandleDeviceGSMethod(hypre_handle)                 ((hypre_handle) -> device_gs_method)

#define hypre_HandleCurandGenerator(hypre_handle)                hypre_CudaDataCurandGenerator(hypre_HandleCudaData(hypre_handle))
#define hypre_HandleCublasHandle(hypre_handle)                   hypre_CudaDataCublasHandle(hypre_HandleCudaData(hypre_handle))
#define hypre_HandleCusparseHandle(hypre_handle)                 hypre_CudaDataCusparseHandle(hypre_HandleCudaData(hypre_handle))
#define hypre_HandleCudaComputeStream(hypre_handle)              hypre_CudaDataCudaComputeStream(hypre_HandleCudaData(hypre_handle))
#define hypre_HandleCubBinGrowth(hypre_handle)                   hypre_CudaDataCubBinGrowth(hypre_HandleCudaData(hypre_handle))
#define hypre_HandleCubMinBin(hypre_handle)                      hypre_CudaDataCubMinBin(hypre_HandleCudaData(hypre_handle))
#define hypre_HandleCubMaxBin(hypre_handle)                      hypre_CudaDataCubMaxBin(hypre_HandleCudaData(hypre_handle))
#define hypre_HandleCubMaxCachedBytes(hypre_handle)              hypre_CudaDataCubMaxCachedBytes(hypre_HandleCudaData(hypre_handle))
#define hypre_HandleCubDevAllocator(hypre_handle)                hypre_CudaDataCubDevAllocator(hypre_HandleCudaData(hypre_handle))
#define hypre_HandleCubUvmAllocator(hypre_handle)                hypre_CudaDataCubUvmAllocator(hypre_HandleCudaData(hypre_handle))
#define hypre_HandleCudaDevice(hypre_handle)                     hypre_CudaDataCudaDevice(hypre_HandleCudaData(hypre_handle))
#define hypre_HandleCudaComputeStreamNum(hypre_handle)           hypre_CudaDataCudaComputeStreamNum(hypre_HandleCudaData(hypre_handle))
#define hypre_HandleCudaReduceBuffer(hypre_handle)               hypre_CudaDataCudaReduceBuffer(hypre_HandleCudaData(hypre_handle))
#define hypre_HandleStructCommRecvBuffer(hypre_handle)           hypre_CudaDataStructCommRecvBuffer(hypre_HandleCudaData(hypre_handle))
#define hypre_HandleStructCommSendBuffer(hypre_handle)           hypre_CudaDataStructCommSendBuffer(hypre_HandleCudaData(hypre_handle))
#define hypre_HandleStructCommRecvBufferSize(hypre_handle)       hypre_CudaDataStructCommRecvBufferSize(hypre_HandleCudaData(hypre_handle))
#define hypre_HandleStructCommSendBufferSize(hypre_handle)       hypre_CudaDataStructCommSendBufferSize(hypre_HandleCudaData(hypre_handle))
#define hypre_HandleSpgemmUseCusparse(hypre_handle)              hypre_CudaDataSpgemmUseCusparse(hypre_HandleCudaData(hypre_handle))
#define hypre_HandleSpgemmNumPasses(hypre_handle)                hypre_CudaDataSpgemmNumPasses(hypre_HandleCudaData(hypre_handle))
#define hypre_HandleSpgemmRownnzEstimateMethod(hypre_handle)     hypre_CudaDataSpgemmRownnzEstimateMethod(hypre_HandleCudaData(hypre_handle))
#define hypre_HandleSpgemmRownnzEstimateNsamples(hypre_handle)   hypre_CudaDataSpgemmRownnzEstimateNsamples(hypre_HandleCudaData(hypre_handle))
#define hypre_HandleSpgemmRownnzEstimateMultFactor(hypre_handle) hypre_CudaDataSpgemmRownnzEstimateMultFactor(hypre_HandleCudaData(hypre_handle))
#define hypre_HandleSpgemmHashType(hypre_handle)                 hypre_CudaDataSpgemmHashType(hypre_HandleCudaData(hypre_handle))
#define hypre_HandleDeviceAllocator(hypre_handle)                hypre_CudaDataDeviceAllocator(hypre_HandleCudaData(hypre_handle))
#define hypre_HandleUseGpuRand(hypre_handle)                     hypre_CudaDataUseGpuRand(hypre_HandleCudaData(hypre_handle))

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

#endif
