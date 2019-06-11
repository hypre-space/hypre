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
#ifndef __GPUMEM_H__
#define  __GPUMEM_H__

#if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP)

#include <cuda_runtime_api.h>
nvtxDomainHandle_t getdomain(hypre_int i);
cusparseHandle_t getCusparseHandle();

/*
 * Global struct for keeping HYPRE GPU Init state
 */

#define MAX_HGS_ELEMENTS 10
struct hypre__global_struct
{
   hypre_int initd;
   hypre_int device;
   hypre_int device_count;
   cusparseHandle_t cusparse_handle;
   cusparseMatDescr_t cusparse_mat_descr;
   cudaStream_t streams[MAX_HGS_ELEMENTS];
   nvtxDomainHandle_t nvtx_domain;
   hypre_int concurrent_managed_access;
};

extern struct hypre__global_struct hypre__global_handle ;

/*
 * Macros for accessing elements of the global handle
 */

#define HYPRE_DOMAIN  hypre__global_handle.nvtx_domain
#define HYPRE_STREAM(index) (hypre__global_handle.streams[index])
#define HYPRE_GPU_HANDLE hypre__global_handle.initd
#define HYPRE_CUSPARSE_HANDLE hypre__global_handle.cusparse_handle
#define HYPRE_DEVICE hypre__global_handle.device
#define HYPRE_DEVICE_COUNT hypre__global_handle.device_count
#define HYPRE_CUSPARSE_MAT_DESCR hypre__global_handle.cusparse_mat_descr

#endif /* #if defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_DEVICE_OPENMP) */


#if defined(HYPRE_USING_DEVICE_OPENMP)

#endif /* #if defined(HYPRE_USING_DEVICE_OPENMP) */

#endif/* __GPUMEM_H__ */

