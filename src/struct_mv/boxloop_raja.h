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

/******************************************************************************
 *
 * Header info for the BoxLoop
 *
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * BoxLoop macros:
 *--------------------------------------------------------------------------*/

#ifndef HYPRE_NEWBOXLOOP_HEADER
#define HYPRE_NEWBOXLOOP_HEADER

#include <cuda.h>
#include <cuda_runtime.h>

extern "C++" {
#include <curand.h>
#include <curand_kernel.h>
#include "RAJA/RAJA.hxx"
}
using namespace RAJA;

__device__ __managed__ HYPRE_Int loop_size_cuda[3];
__device__ __managed__ HYPRE_Int cdim;
__device__ __managed__ HYPRE_Int stride_cuda1[3],start_cuda1[3],dboxmin1[3],dboxmax1[3];
__device__ __managed__ HYPRE_Int stride_cuda2[3],start_cuda2[3],dboxmin2[3],dboxmax2[3];
__device__ __managed__ HYPRE_Int stride_cuda3[3],start_cuda3[3],dboxmin3[3],dboxmax3[3];
__device__ __managed__ HYPRE_Int stride_cuda4[3],start_cuda4[3],dboxmin4[3],dboxmax4[3];

/*--------------------------------------------------------------------------
 * hypre_Index:
 *   This is used to define indices in index space, or dimension
 *   sizes of boxes.
 *
 *   The spatial dimensions x, y, and z may be specified by the
 *   integers 0, 1, and 2, respectively (see the hypre_IndexD macro below).
 *   This simplifies the code in the hypre_Box class by reducing code
 *   replication.
 *--------------------------------------------------------------------------*/


#define hypre_CallocIndex(indexVar, dim)									\
	cudaMallocManaged((void**)&indexVar,sizeof(HYPRE_Int)*dim, cudaMemAttachGlobal);

#define hypre_CallocReal(indexVar, dim)								\
	cudaMallocManaged((void**)&indexVar,sizeof(HYPRE_Real)*dim, cudaMemAttachGlobal);

#define hypre_rand(val) \
{\
    curandState_t state;\
    curand_init(0,0,0,&state);\
    val = curand(&state);\
}

#define zypre_BoxLoopCUDAInit(ndim,loop_size)				\
	HYPRE_Int hypre__tot = 1.0;								\
	const size_t block_size = 256;				  			\
	cdim = ndim;											\
	for (HYPRE_Int d = 0;d < ndim;d ++)						\
	{														\
		loop_size_cuda[d] = loop_size[d];					\
		hypre__tot *= loop_size[d];							\
    }														

#define zypre_BoxLoopCUDADeclare()										\
	HYPRE_Int local_idx;												\
	HYPRE_Int d,idx_local = i;


#define zypre_newBoxLoopInitK(ndim,dbox1,loop_size,start1,stride1,k)	\
	for (HYPRE_Int d = 0;d < ndim;d ++)									\
	{																	\
		stride_cuda##k[d] = stride1[d];									\
		start_cuda##k[d]  =  start1[d];								\
		dboxmin##k[d]     = dbox1->imin[d];						\
		dboxmax##k[d]     = dbox1->imax[d];					\
    }



#define zypre_newBoxLoop0Begin(ndim, loop_size)				\
{    														\
    zypre_BoxLoopCUDAInit(ndim,loop_size)													\
	forall< cuda_exec<block_size> >(0, hypre__tot, [=] __device__ (int i) \
	{


#define zypre_newBoxLoop0End()					\
	});											\
	cudaDeviceSynchronize();					\
}

#define zypre_newBoxLoop1Begin(ndim, loop_size,				\
							   dbox1, start1, stride1, i1) 	\
{												\
	zypre_BoxLoopCUDAInit(ndim,loop_size)								\
	zypre_newBoxLoopInitK(ndim,dbox1,loop_size,start1,stride1,1)		\
	forall< cuda_exec<block_size> >(0, hypre__tot, [=] __device__ (int i) \
	{																	\
	    zypre_BoxLoopCUDADeclare()										\
	    HYPRE_Int hypre_boxD1 = 1.0;									\
	    HYPRE_Int i1 = 0;												\
		for (d = 0;d < cdim;d ++)										\
		{																\
			local_idx  = idx_local % loop_size_cuda[d];					\
			idx_local  = idx_local / loop_size_cuda[d];					\
			i1 += (local_idx*stride_cuda1[d] + start_cuda1[d] - dboxmin1[d]) * hypre_boxD1; \
			hypre_boxD1 *= hypre_max(0, dboxmax1[d] - dboxmin1[d] + 1);         \
        }																		\



#define zypre_newBoxLoop1End(i1)				\
	});											\
    cudaDeviceSynchronize();					\
}

#define zypre_newBoxLoop2Begin(ndim, loop_size,				\
                                dbox1, start1, stride1, i1,	\
                                dbox2, start2, stride2, i2)	\
{\
    zypre_BoxLoopCUDAInit(ndim,loop_size)								\
	zypre_newBoxLoopInitK(ndim,dbox1,loop_size,start1,stride1,1)		\
	zypre_newBoxLoopInitK(ndim,dbox2,loop_size,start2,stride2,2)		\
    forall< cuda_exec<block_size> >(0, hypre__tot, [=] __device__ (int i) \
	{	\
	    zypre_BoxLoopCUDADeclare()											\
	    HYPRE_Int hypre_boxD1 = 1.0,hypre_boxD2 = 1.0;						\
	    HYPRE_Int i1 = 0, i2 = 0;											\
		for (d = 0;d < cdim;d ++)										\
		{																\
			local_idx  = idx_local % loop_size_cuda[d];						\
			idx_local  = idx_local / loop_size_cuda[d];						\
			i1 += (local_idx*stride_cuda1[d] + start_cuda1[d] - dboxmin1[d]) * hypre_boxD1; \
			hypre_boxD1 *= hypre_max(0, dboxmax1[d] - dboxmin1[d] + 1);         \
			i2 += (local_idx*stride_cuda2[d] + start_cuda2[d] - dboxmin2[d]) * hypre_boxD2; \
			hypre_boxD2 *= hypre_max(0, dboxmax2[d] - dboxmin2[d] + 1);         \
		}


#define zypre_newBoxLoop2End(i1, i2)			\
	});											\
	cudaDeviceSynchronize();					\
}


#define zypre_newBoxLoop3Begin(ndim, loop_size,\
							   dbox1, start1, stride1, i1,	\
							   dbox2, start2, stride2, i2,	\
							   dbox3, start3, stride3, i3)	\
{																	\
    zypre_BoxLoopCUDAInit(ndim,loop_size);										\
	zypre_newBoxLoopInitK(ndim,dbox1,loop_size,start1,stride1,1);		\
	zypre_newBoxLoopInitK(ndim,dbox2,loop_size,start2,stride2,2);		\
	zypre_newBoxLoopInitK(ndim,dbox3,loop_size,start3,stride3,3);		\
	forall< cuda_exec<block_size> >(0, hypre__tot, [=] __device__ (int i) \
	{																	\
	    zypre_BoxLoopCUDADeclare()											\
	    HYPRE_Int hypre_boxD1 = 1.0,hypre_boxD2 = 1.0,hypre_boxD3 = 1.0;	\
	    HYPRE_Int i1 = 0, i2 = 0, i3 = 0;											\
		for (d = 0;d < cdim;d ++)										\
		{																\
			local_idx  = idx_local % loop_size_cuda[d];						\
			idx_local  = idx_local / loop_size_cuda[d];						\
			i1 += (local_idx*stride_cuda1[d] + start_cuda1[d] - dboxmin1[d]) * hypre_boxD1; \
			hypre_boxD1 *= hypre_max(0, dboxmax1[d] - dboxmin1[d] + 1);         \
			i2 += (local_idx*stride_cuda2[d] + start_cuda2[d] - dboxmin2[d]) * hypre_boxD2; \
			hypre_boxD2 *= hypre_max(0, dboxmax2[d] - dboxmin2[d] + 1);         \
			i3 += (local_idx*stride_cuda3[d] + start_cuda3[d] - dboxmin3[d]) * hypre_boxD3; \
			hypre_boxD3 *= hypre_max(0, dboxmax3[d] - dboxmin3[d] + 1);         \
		}		

#define zypre_newBoxLoop3End(i1, i2,i3)			\
	});											\
	cudaDeviceSynchronize();					\
}

#define zypre_newBoxLoop4Begin(ndim, loop_size,				\
							   dbox1, start1, stride1, i1,	\
							   dbox2, start2, stride2, i2,	\
							   dbox3, start3, stride3, i3,			\
							   dbox4, start4, stride4, i4)			\
{																	\
	zypre_BoxLoopCUDAInit(ndim,loop_size);										\
	zypre_newBoxLoopInitK(ndim,dbox1,loop_size,start1,stride1,1);		\
	zypre_newBoxLoopInitK(ndim,dbox2,loop_size,start2,stride2,2);		\
	zypre_newBoxLoopInitK(ndim,dbox3,loop_size,start3,stride3,3);		\
	zypre_newBoxLoopInitK(ndim,dbox4,loop_size,start4,stride4,4);		\
	forall< cuda_exec<block_size> >(0, hypre__tot, [=] __device__ (int i) \
	{																	\
	    zypre_BoxLoopCUDADeclare()											\
	    HYPRE_Int hypre_boxD1 = 1.0,hypre_boxD2 = 1.0,hypre_boxD3 = 1.0,hypre_boxD4 = 1.0; \
	    HYPRE_Int i1 = 0, i2 = 0, i3 = 0,i4 = 0;								\
		for (d = 0;d < cdim;d ++)										\
		{																\
			local_idx  = idx_local % loop_size_cuda[d];						\
			idx_local  = idx_local / loop_size_cuda[d];						\
			i1 += (local_idx*stride_cuda1[d] + start_cuda1[d] - dboxmin1[d]) * hypre_boxD1; \
			hypre_boxD1 *= hypre_max(0, dboxmax1[d] - dboxmin1[d] + 1);         \
			i2 += (local_idx*stride_cuda2[d] + start_cuda2[d] - dboxmin2[d]) * hypre_boxD2; \
			hypre_boxD2 *= hypre_max(0, dboxmax2[d] - dboxmin2[d] + 1);         \
			i3 += (local_idx*stride_cuda3[d] + start_cuda3[d] - dboxmin3[d]) * hypre_boxD3; \
			hypre_boxD3 *= hypre_max(0, dboxmax3[d] - dboxmin3[d] + 1);         \
			i4 += (local_idx*stride_cuda4[d] + start_cuda4[d] - dboxmin4[d]) * hypre_boxD4; \
			hypre_boxD4 *= hypre_max(0, dboxmax4[d] - dboxmin4[d] + 1);         \
		}		

#define zypre_newBoxLoop4End(i1, i2, i3, i4)	\
	});											\
	cudaDeviceSynchronize();					\
}

#ifdef HYPRE_USE_RAJA
#define zypre_Reductioninit(local_result) \
const size_t block_size = 256;\
ReduceSum< cuda_reduce<block_size>, HYPRE_Real> local_result(0.0);
#else
#define zypre_Reductioninit(local_result) \
HYPRE_Real       local_result;\
local_result = 0.0;
#endif

#define zypre_newBoxLoop2ReductionBegin(ndim, loop_size,				\
										dbox1, start1, stride1, i1,		\
										dbox2, start2, stride2, i2,sum)	\
{    																	\
	zypre_newBoxLoopDeclareK(1);		\
	zypre_newBoxLoopDeclareK(2);										\
	zypre_BoxLoopCUDAInit(ndim,loop_size);										\
	zypre_newBoxLoopInitK(ndim,dbox1,loop_size,start1,stride1,1);		\
	zypre_newBoxLoopInitK(ndim,dbox2,loop_size,start2,stride2,2);		\
	forall< cuda_exec<block_size> >(0, hypre__tot, [=] __device__ (int i) \
	{																	\
	    zypre_BoxLoopCUDADeclare()											\
	    HYPRE_Int hypre_boxD1 = 1.0,hypre_boxD2 = 1.0;						\
	    HYPRE_Int i1 = 0, i2 = 0;											\
		for (d = 0;d < cdim;d ++)										\
		{																\
			local_idx  = idx_local % loop_size_cuda[d];						\
			idx_local  = idx_local / loop_size_cuda[d];						\
			i1 += (local_idx*stride_cuda1[d] + start_cuda1[d] - dboxmin1[d]) * hypre_boxD1; \
			hypre_boxD1 *= hypre_max(0, dboxmax1[d] - dboxmin1[d] + 1);         \
			i2 += (local_idx*stride_cuda2[d] + start_cuda2[d] - dboxmin2[d]) * hypre_boxD2; \
			hypre_boxD2 *= hypre_max(0, dboxmax2[d] - dboxmin2[d] + 1);         \
		}


#define zypre_newBoxLoop2ReductionEnd(i1, i2, sum)	\
	});											\
	cudaDeviceSynchronize();					\
}

#define zypre_newBoxLoop0For() {}

#define zypre_newBoxLoop1For(i1) {}

#define zypre_newBoxLoop2For(i1, i2) {}

#define zypre_newBoxLoop3For(i1, i2, i3) {}

#define zypre_newBoxLoop4For(i1, i2, i3, i4) {}

#define zypre_newBoxLoopSetOneBlock() {}

/*
//#define hypre_BoxLoopGetIndex    zypre_newBoxLoopGetIndex
#define hypre_BoxLoopSetOneBlock zypre_newBoxLoopSetOneBlock
//#define hypre_BoxLoopBlock       zypre_newBoxLoopBlock

#define hypre_BoxLoop0Begin      zypre_BoxLoop0Begin
#define hypre_BoxLoop0For        zypre_BoxLoop0For
#define hypre_BoxLoop0End        zypre_BoxLoop0End
#define hypre_BoxLoop1Begin      zypre_newBoxLoop1Begin
#define hypre_BoxLoop1For        zypre_newBoxLoop1For
#define hypre_BoxLoop1End        zypre_newBoxLoop1End
#define hypre_BoxLoop2Begin      zypre_newBoxLoop2Begin
#define hypre_BoxLoop2For        zypre_newBoxLoop2For
#define hypre_BoxLoop2End        zypre_newBoxLoop2End
#define hypre_BoxLoop3Begin      zypre_newBoxLoop3Begin
#define hypre_BoxLoop3For        zypre_newBoxLoop3For
#define hypre_BoxLoop3End        zypre_newBoxLoop3End
#define hypre_BoxLoop4Begin      zypre_newBoxLoop4Begin
#define hypre_BoxLoop4For        zypre_newBoxLoop4For
#define hypre_BoxLoop4End        zypre_newBoxLoop4End
*/
#define hypre_BoxLoopSetOneBlock zypre_BoxLoopSetOneBlock
#define hypre_BoxLoop0Begin      zypre_BoxLoop0Begin
#define hypre_BoxLoop0For        zypre_BoxLoop0For
#define hypre_BoxLoop0End        zypre_BoxLoop0End
#define hypre_BoxLoop1Begin      zypre_BoxLoop1Begin
#define hypre_BoxLoop1For        zypre_BoxLoop1For
#define hypre_BoxLoop1End        zypre_BoxLoop1End
#define hypre_BoxLoop2Begin      zypre_BoxLoop2Begin
#define hypre_BoxLoop2For        zypre_BoxLoop2For
#define hypre_BoxLoop2End        zypre_BoxLoop2End
#define hypre_BoxLoop3Begin      zypre_BoxLoop3Begin
#define hypre_BoxLoop3For        zypre_BoxLoop3For
#define hypre_BoxLoop3End        zypre_BoxLoop3End
#define hypre_BoxLoop4Begin      zypre_BoxLoop4Begin
#define hypre_BoxLoop4For        zypre_BoxLoop4For
#define hypre_BoxLoop4End        zypre_BoxLoop4End

#endif
