/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/

#if defined(HYPRE_USE_GPU) && defined(HYPRE_USE_MANAGED)
#ifndef __GPUMEM_H__
#define  __GPUMEM_H__
#ifdef HYPRE_USE_GPU
#include <cuda_runtime_api.h>
void hypre_GPUInit(hypre_int use_device);
void hypre_GPUFinalize();
int VecScaleScalar(double *u, const double alpha,  int num_rows,cudaStream_t s);
void VecCopy(double* tgt, const double* src, int size,cudaStream_t s);
void VecSet(double* tgt, int size, double value, cudaStream_t s);
void VecScale(double *u, double *v, double *l1_norm, int num_rows,cudaStream_t s);
void VecScaleSplit(double *u, double *v, double *l1_norm, int num_rows,cudaStream_t s);
void CudaCompileFlagCheck();
#endif

cudaStream_t getstreamOlde(hypre_int i);
nvtxDomainHandle_t getdomain(hypre_int i);
cudaEvent_t getevent(hypre_int i);
void MemAdviseReadOnly(const void *ptr, hypre_int device);
void MemAdviseUnSetReadOnly(const void *ptr, hypre_int device);
void MemAdviseSetPrefLocDevice(const void *ptr, hypre_int device);
void MemAdviseSetPrefLocHost(const void *ptr);
void MemPrefetch(const void *ptr,hypre_int device,cudaStream_t stream);
void MemPrefetchSized(const void *ptr,size_t size,hypre_int device,cudaStream_t stream);
void MemPrefetchForce(const void *ptr,hypre_int device,cudaStream_t stream);
cublasHandle_t getCublasHandle();
cusparseHandle_t getCusparseHandle();
typedef struct node {
  const void *ptr;
  size_t size;
  struct node *next;
} node;
size_t mempush(const void *ptr, size_t size, hypre_int action);
node *memfind(node *head, const void *ptr);
void memdel(node **head, node *found);
void meminsert(node **head, const void *ptr,size_t size);
void printlist(node *head,hypre_int nc);
//#define MEM_PAD_LEN 1
size_t memsize(const void *ptr);
hypre_int getsetasyncmode(hypre_int mode, hypre_int action);
void SetAsyncMode(hypre_int mode);
hypre_int GetAsyncMode();
void branchStream(hypre_int i, hypre_int j);
void joinStreams(hypre_int i, hypre_int j, hypre_int k);
void affs(hypre_int myid);
hypre_int getcore();
hypre_int getnuma();
hypre_int checkDeviceProps();
hypre_int pointerIsManaged(const void *ptr);
/*
 * Global struct for keeping HYPRE GPU Init state
 */

#define MAX_HGS_ELEMENTS 10
struct hypre__global_struct{
  hypre_int initd;
  hypre_int device;
  hypre_int device_count;
  cublasHandle_t cublas_handle;
  cusparseHandle_t cusparse_handle;
  cusparseMatDescr_t cusparse_mat_descr;
  cudaStream_t streams[MAX_HGS_ELEMENTS];
  nvtxDomainHandle_t nvtx_domain;
  hypre_int concurrent_managed_access;
  size_t memoryHWM;
};

extern struct hypre__global_struct hypre__global_handle ;

/*
 * Macros for accessing elements of the global handle
 */
#define HYPRE_GPU_HANDLE hypre__global_handle.initd
#define HYPRE_CUBLAS_HANDLE hypre__global_handle.cublas_handle
#define HYPRE_CUSPARSE_HANDLE hypre__global_handle.cusparse_handle
#define HYPRE_DEVICE hypre__global_handle.device
#define HYPRE_DEVICE_COUNT hypre__global_handle.device_count
#define HYPRE_CUSPARSE_MAT_DESCR hypre__global_handle.cusparse_mat_descr
#define HYPRE_STREAM(index) (hypre__global_handle.streams[index])
#define HYPRE_DOMAIN  hypre__global_handle.nvtx_domain
#define HYPRE_GPU_CMA hypre__global_handle.concurrent_managed_access
#define HYPRE_GPU_HWM hypre__global_handle.memoryHWM

#endif

#else

#define hypre_GPUInit(use_device)
#define hypre_GPUFinalize()

#endif

