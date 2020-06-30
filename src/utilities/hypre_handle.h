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

//#ifdef __cplusplus
//extern "C++" {
//#endif

struct hypre_CudaData;
typedef struct hypre_CudaData hypre_CudaData;

typedef struct
{
   HYPRE_Int              hypre_error;
   HYPRE_MemoryLocation   memory_location;
   HYPRE_ExecutionPolicy  default_exec_policy;
   HYPRE_ExecutionPolicy  struct_exec_policy;
#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)
   hypre_CudaData        *cuda_data;
#endif
} hypre_Handle;

/* accessor macros to hypre_Handle */
#define hypre_HandleMemoryLocation(hypre_handle)                 ((hypre_handle) -> memory_location)
#define hypre_HandleDefaultExecPolicy(hypre_handle)              ((hypre_handle) -> default_exec_policy)
#define hypre_HandleStructExecPolicy(hypre_handle)               ((hypre_handle) -> struct_exec_policy)
#define hypre_HandleCudaData(hypre_handle)                       ((hypre_handle) -> cuda_data)

#define hypre_HandleCurandGenerator(hypre_handle)                hypre_CudaDataCurandGenerator(hypre_HandleCudaData(hypre_handle))
#define hypre_HandleCublasHandle(hypre_handle)                   hypre_CudaDataCublasHandle(hypre_HandleCudaData(hypre_handle))
#define hypre_HandleCusparseHandle(hypre_handle)                 hypre_CudaDataCusparseHandle(hypre_HandleCudaData(hypre_handle))
#define hypre_HandleCusparseMatDescr(hypre_handle)               hypre_CudaDataCusparseMatDescr(hypre_HandleCudaData(hypre_handle))
#define hypre_HandleCudaComputeStream(hypre_handle)              hypre_CudaDataCudaComputeStream(hypre_HandleCudaData(hypre_handle))
#define hypre_HandleCubBinGrowth(hypre_handle)                   hypre_CudaDataCubBinGrowth(hypre_HandleCudaData(hypre_handle))
#define hypre_HandleCubMinBin(hypre_handle)                      hypre_CudaDataCubMinBin(hypre_HandleCudaData(hypre_handle))
#define hypre_HandleCubMaxBin(hypre_handle)                      hypre_CudaDataCubMaxBin(hypre_HandleCudaData(hypre_handle))
#define hypre_HandleCubMaxCachedBytes(hypre_handle)              hypre_CudaDataCubMaxCachedBytes(hypre_HandleCudaData(hypre_handle))
#define hypre_HandleCubDevAllocator(hypre_handle)                hypre_CudaDataCubDevAllocator(hypre_HandleCudaData(hypre_handle))
#define hypre_HandleCubUvmAllocator(hypre_handle)                hypre_CudaDataCubUvmAllocator(hypre_HandleCudaData(hypre_handle))
#define hypre_HandleCudaDevice(hypre_handle)                     hypre_CudaDataCudaDevice(hypre_HandleCudaData(hypre_handle))
#define hypre_HandleCudaComputeStreamNum(hypre_handle)           hypre_CudaDataCudaComputeStreamNum(hypre_HandleCudaData(hypre_handle))
#define hypre_HandleCudaComputeStreamSync(hypre_handle)          hypre_CudaDataCudaComputeStreamSync(hypre_HandleCudaData(hypre_handle))
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

//#ifdef __cplusplus
//}
//#endif

#endif

