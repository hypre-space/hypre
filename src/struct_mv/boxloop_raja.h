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
#include <RAJA/RAJA.hxx>
}
using namespace RAJA;

//__device__ __managed__ HYPRE_Int loop_size_cuda[3];
//__device__ __managed__ HYPRE_Int cdim;
//__device__ __managed__ HYPRE_Int stride_cuda1[3],start_cuda1[3],dboxmin1[3],dboxmax1[3];
//__device__ __managed__ HYPRE_Int stride_cuda2[3],start_cuda2[3],dboxmin2[3],dboxmax2[3];
//__device__ __managed__ HYPRE_Int stride_cuda3[3],start_cuda3[3],dboxmin3[3],dboxmax3[3];
//__device__ __managed__ HYPRE_Int stride_cuda4[3],start_cuda4[3],dboxmin4[3],dboxmax4[3];

//__device__ __managed__ HYPRE_Int loop_size_cuda[3];
//__device__ __managed__ HYPRE_Int cdim = 3;
//__device__ __managed__ HYPRE_Int stride_cuda1[3],start_cuda1[3],dboxmin1[3],dboxmax1[3];
//__device__ __managed__ HYPRE_Int stride_cuda2[3],start_cuda2[3],dboxmin2[3],dboxmax2[3];
//__device__ __managed__ HYPRE_Int stride_cuda3[3],start_cuda3[3],dboxmin3[3],dboxmax3[3];
//__device__ __managed__ HYPRE_Int stride_cuda4[3],start_cuda4[3],dboxmin4[3],dboxmax4[3];
#define BLOCKSIZE 128
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
#define MAXBLOCKS 32
#define NTHREADS 256 // must be a power of 2

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
	for (HYPRE_Int d = 0;d < ndim;d ++)						\
	{														\
		loop_size_cuda[d] = loop_size[d];					\
		hypre__tot *= loop_size[d];							\
    }														

//	cdim = ndim;								\

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

/*
__global__ void dot (HYPRE_Real * a, HYPRE_Real * b, HYPRE_Real *c, HYPRE_Int hypre__tot,
                     HYPRE_Int *loop_size_cuda,HYPRE_Int ndim,
                     HYPRE_Int *stride_cuda1,HYPRE_Int *start_cuda1,HYPRE_Int *dboxmin1,HYPRE_Int *dboxmax1,
                     HYPRE_Int *stride_cuda2,HYPRE_Int *start_cuda2,HYPRE_Int *dboxmin2,HYPRE_Int *dboxmax2)
{
    int id = (blockIdx.x * blockDim.x) + threadIdx.x;
	HYPRE_Int local_idx;
    HYPRE_Int d,idx_local = id;
    HYPRE_Int hypre_boxD1 = 1.0,hypre_boxD2 = 1.0;
    HYPRE_Int i1 = 0, i2 = 0;
    
    //// reducted output
    __shared__ HYPRE_Real shared_cache [NTHREADS];
	HYPRE_Real sum = 0;
    
    for (d = 0;d < ndim;d ++)
    {
        local_idx  = idx_local % loop_size_cuda[d];
        idx_local  = idx_local / loop_size_cuda[d];
        i1 += (local_idx*stride_cuda1[d] + start_cuda1[d] - dboxmin1[d]) * hypre_boxD1;
        hypre_boxD1 *= hypre_max(0, dboxmax1[d] - dboxmin1[d] + 1);
        i2 += (local_idx*stride_cuda2[d] + start_cuda2[d] - dboxmin2[d]) * hypre_boxD2;
        hypre_boxD2 *= hypre_max(0, dboxmax2[d] - dboxmin2[d] + 1);
    }
    //for (;id < size ;){
    //    sum += (*(a+id)) * (*(b+id));
    //    id+= nextid;
    //}
	if (id < hypre__tot)
		sum = a[i1] * hypre_conj(b[i2]);
    *(shared_cache + threadIdx.x) = sum;
	
    __syncthreads();
	
    ///////// sum of internal cache
	
    int i;    
    
    for (i=(NTHREADS /2); i>0 ; i= i/2){
		if (threadIdx.x < i){
			*(shared_cache + threadIdx.x) += *(shared_cache + threadIdx.x + i);
		}
		__syncthreads();
    }
	
    if ( threadIdx.x == 0){
        *(c+ blockIdx.x) = shared_cache[0];
    }
}
*/
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
printf("zypre_newBoxLoop1Begin, %s(%d): %s\n",__FILE__,__LINE__,__FUNCTION__);\
	HYPRE_Int hypre__tot = 1.0;								\
	const size_t block_size = 256;				  			\
	HYPRE_Int *CUDA_data, HOST_data[ndim*4+1];							\
	HOST_data[0] = ndim;												\
	for (HYPRE_Int d = 0;d < ndim;d ++)									\
	{																	\
	    HOST_data[d+1] = loop_size[d];									\
		HOST_data[d+ndim+1] = stride1[d];								\
		HOST_data[d+2*ndim+1] = start1[d] - dbox1->imin[d];				\
		HOST_data[d+3*ndim+1] = dbox1->imax[d]-dbox1->imin[d];			\
		hypre__tot *= loop_size[d];										\
	}																	\
	cudaMalloc((int**)&CUDA_data,sizeof(HYPRE_Int)*(ndim*4+1));			\
	cudaMemcpy(CUDA_data, HOST_data, sizeof(HYPRE_Int)*(ndim*4+1), cudaMemcpyHostToDevice);\
	forall< cuda_exec<block_size> >(0, hypre__tot, [=] __device__ (int i) \
	{																	\
	    zypre_BoxLoopCUDADeclare()										\
	    HYPRE_Int hypre_boxD1 = 1.0;									\
	    HYPRE_Int i1 = 0;												\
		HYPRE_Int nd = CUDA_data[0];								\
		for (d = 0;d < nd;d ++)										\
		{																\
			local_idx  = idx_local % CUDA_data[d+1];					\
			idx_local  = idx_local / CUDA_data[d+1];					\
			i1 += (local_idx*CUDA_data[d+nd+1] + CUDA_data[d+2*nd+1]) * hypre_boxD1; \
			hypre_boxD1 *= hypre_max(0, CUDA_data[d+3*nd+1] + 1);	\
		}															\
//    printf("zypre_newBoxLoop1Begin, %s(%d): %s\n",__FILE__,__LINE__,__FUNCTION__);\
//	zypre_BoxLoopCUDAInit(ndim,loop_size)								\
//	zypre_newBoxLoopInitK(ndim,dbox1,loop_size,start1,stride1,1)		\

#define zypre_newBoxLoop1End(i1)				\
	});											\
	cudaFree(CUDA_data);						\
    cudaDeviceSynchronize();					\
}

#define zypre_newBoxLoop2Begin(ndim, loop_size,				\
                                dbox1, start1, stride1, i1,	\
                                dbox2, start2, stride2, i2)	\
{\
	HYPRE_Int hypre__tot = 1.0;								\
	const size_t block_size = 256;				  			\
	HYPRE_Int *CUDA_data, HOST_data[ndim*7+1];							\
	HOST_data[0] = ndim;												\
	for (HYPRE_Int d = 0;d < ndim;d ++)									\
	{																	\
	    HOST_data[d+1] = loop_size[d];									\
		HOST_data[d+ndim+1] = stride1[d];								\
		HOST_data[d+2*ndim+1] = start1[d] - dbox1->imin[d];				\
		HOST_data[d+3*ndim+1] = dbox1->imax[d]-dbox1->imin[d];			\
		HOST_data[d+4*ndim+1] = stride2[d];								\
		HOST_data[d+5*ndim+1] = start2[d] - dbox2->imin[d];				\
		HOST_data[d+6*ndim+1] = dbox2->imax[d]-dbox2->imin[d];			\
		hypre__tot *= loop_size[d];										\
	}																	\
	cudaMalloc((int**)&CUDA_data,sizeof(HYPRE_Int)*(ndim*7+1));			\
	cudaMemcpy(CUDA_data, HOST_data, sizeof(HYPRE_Int)*(ndim*7+1), cudaMemcpyHostToDevice);\
    forall< cuda_exec<block_size> >(0, hypre__tot, [=] __device__ (int i) \
	{	\
	    zypre_BoxLoopCUDADeclare()											\
	    HYPRE_Int hypre_boxD1 = 1.0,hypre_boxD2 = 1.0;						\
	    HYPRE_Int i1 = 0, i2 = 0;											\
		HYPRE_Int nd = CUDA_data[0];								\
		for (d = 0;d < nd;d ++)										\
		{																\
			local_idx  = idx_local % CUDA_data[d+1];					\
			idx_local  = idx_local / CUDA_data[d+1];					\
			i1 += (local_idx*CUDA_data[d+nd+1] + CUDA_data[d+2*nd+1]) * hypre_boxD1; \
			hypre_boxD1 *= hypre_max(0, CUDA_data[d+3*nd+1] + 1);         \
			i2 += (local_idx*CUDA_data[d+4*nd+1] + CUDA_data[d+5*nd+1]) * hypre_boxD2; \
			hypre_boxD2 *= hypre_max(0, CUDA_data[d+6*nd+1] + 1);         \
		}
//printf("zypre_newBoxLoop2Begin,%s(%d): %s\n",__FILE__,__LINE__,__FUNCTION__); \


#define zypre_newBoxLoop2End(i1, i2)			\
	});											\
	cudaFree(CUDA_data);						\
	cudaDeviceSynchronize();					\
}


#define zypre_newBoxLoop3Begin(ndim, loop_size,\
							   dbox1, start1, stride1, i1,	\
							   dbox2, start2, stride2, i2,	\
							   dbox3, start3, stride3, i3)	\
{																	\
    HYPRE_Int hypre__tot = 1.0;											\
	const size_t block_size = 256;				  			\
	HYPRE_Int *CUDA_data, HOST_data[ndim*10+1];							\
	HOST_data[0] = ndim;												\
	for (HYPRE_Int d = 0;d < ndim;d ++)									\
	{																	\
	    HOST_data[d+1] = loop_size[d];									\
		HOST_data[d+ndim+1] = stride1[d];								\
		HOST_data[d+2*ndim+1] = start1[d] - dbox1->imin[d];				\
		HOST_data[d+3*ndim+1] = dbox1->imax[d]-dbox1->imin[d];			\
		HOST_data[d+4*ndim+1] = stride2[d];								\
		HOST_data[d+5*ndim+1] = start2[d] - dbox2->imin[d];				\
		HOST_data[d+6*ndim+1] = dbox2->imax[d]-dbox2->imin[d];			\
		HOST_data[d+7*ndim+1] = stride3[d];								\
		HOST_data[d+8*ndim+1] = start3[d] - dbox3->imin[d];				\
		HOST_data[d+9*ndim+1] = dbox3->imax[d]-dbox3->imin[d];			\
		hypre__tot *= loop_size[d];										\
	}																	\
	cudaMalloc((int**)&CUDA_data,sizeof(HYPRE_Int)*(ndim*10+1));			\
	cudaMemcpy(CUDA_data, HOST_data, sizeof(HYPRE_Int)*(ndim*10+1), cudaMemcpyHostToDevice);\
	forall< cuda_exec<block_size> >(0, hypre__tot, [=] __device__ (int i) \
	{																	\
	    zypre_BoxLoopCUDADeclare()											\
	    HYPRE_Int hypre_boxD1 = 1.0,hypre_boxD2 = 1.0,hypre_boxD3 = 1.0;	\
	    HYPRE_Int i1 = 0, i2 = 0, i3 = 0;											\
		HYPRE_Int nd = CUDA_data[0];								\
		for (d = 0;d < nd;d ++)										\
		{																\
			local_idx  = idx_local % CUDA_data[d+1];					\
			idx_local  = idx_local / CUDA_data[d+1];					\
			i1 += (local_idx*CUDA_data[d+nd+1] + CUDA_data[d+2*nd+1]) * hypre_boxD1; \
			hypre_boxD1 *= hypre_max(0, CUDA_data[d+3*nd+1] + 1);         \
			i2 += (local_idx*CUDA_data[d+4*nd+1] + CUDA_data[d+5*nd+1]) * hypre_boxD2; \
			hypre_boxD2 *= hypre_max(0, CUDA_data[d+6*nd+1] + 1);         \
			i3 += (local_idx*CUDA_data[d+7*nd+1] + CUDA_data[d+8*nd+1]) * hypre_boxD3; \
			hypre_boxD3 *= hypre_max(0, CUDA_data[d+9*nd+1] + 1);         \
		}		

#define zypre_newBoxLoop3End(i1, i2,i3)			\
	});											\
	cudaFree(CUDA_data);						\
	cudaDeviceSynchronize();					\
}
//    printf("zypre_newBoxLoop3Begin, %s(%d): %s\n",__FILE__,__LINE__,__FUNCTION__);\


#define zypre_newBoxLoop4Begin(ndim, loop_size,				\
							   dbox1, start1, stride1, i1,	\
							   dbox2, start2, stride2, i2,	\
							   dbox3, start3, stride3, i3,			\
							   dbox4, start4, stride4, i4)			\
{																	\
	HYPRE_Int hypre__tot = 1.0;											\
	const size_t block_size = 256;				  			\
	HYPRE_Int *CUDA_data, HOST_data[ndim*13+1];							\
	HOST_data[0] = ndim;												\
	for (HYPRE_Int d = 0;d < ndim;d ++)									\
	{																	\
	    HOST_data[d+1] = loop_size[d];									\
		HOST_data[d+ndim+1] = stride1[d];								\
		HOST_data[d+2*ndim+1] = start1[d] - dbox1->imin[d];				\
		HOST_data[d+3*ndim+1] = dbox1->imax[d]-dbox1->imin[d];			\
		HOST_data[d+4*ndim+1] = stride2[d];								\
		HOST_data[d+5*ndim+1] = start2[d] - dbox2->imin[d];				\
		HOST_data[d+6*ndim+1] = dbox2->imax[d]-dbox2->imin[d];			\
		HOST_data[d+7*ndim+1] = stride3[d];								\
		HOST_data[d+8*ndim+1] = start3[d] - dbox3->imin[d];				\
		HOST_data[d+9*ndim+1] = dbox3->imax[d]-dbox3->imin[d];			\
		HOST_data[d+10*ndim+1] = stride4[d];								\
		HOST_data[d+11*ndim+1] = start4[d] - dbox4->imin[d];				\
		HOST_data[d+12*ndim+1] = dbox4->imax[d]-dbox4->imin[d];			\
		hypre__tot *= loop_size[d];										\
	}																	\
	cudaMalloc((int**)&CUDA_data,sizeof(HYPRE_Int)*(ndim*13+1));			\
	cudaMemcpy(CUDA_data, HOST_data, sizeof(HYPRE_Int)*(ndim*13+1), cudaMemcpyHostToDevice);\
	forall< cuda_exec<block_size> >(0, hypre__tot, [=] __device__ (int i) \
	{																	\
	    zypre_BoxLoopCUDADeclare()											\
	    HYPRE_Int hypre_boxD1 = 1.0,hypre_boxD2 = 1.0,hypre_boxD3 = 1.0,hypre_boxD4 = 1.0; \
	    HYPRE_Int i1 = 0, i2 = 0, i3 = 0,i4 = 0;								\
		HYPRE_Int nd = CUDA_data[0];								\
		for (d = 0;d < nd;d ++)										\
		{																\
			local_idx  = idx_local % CUDA_data[d+1];					\
			idx_local  = idx_local / CUDA_data[d+1];					\
			i1 += (local_idx*CUDA_data[d+nd+1] + CUDA_data[d+2*nd+1]) * hypre_boxD1; \
			hypre_boxD1 *= hypre_max(0, CUDA_data[d+3*nd+1] + 1);         \
			i2 += (local_idx*CUDA_data[d+4*nd+1] + CUDA_data[d+5*nd+1]) * hypre_boxD2; \
			hypre_boxD2 *= hypre_max(0, CUDA_data[d+6*nd+1] + 1);         \
			i3 += (local_idx*CUDA_data[d+7*nd+1] + CUDA_data[d+8*nd+1]) * hypre_boxD3; \
			hypre_boxD3 *= hypre_max(0, CUDA_data[d+9*nd+1] + 1);         \
			i4 += (local_idx*CUDA_data[d+10*nd+1] + CUDA_data[d+11*nd+1]) * hypre_boxD4; \
			hypre_boxD4 *= hypre_max(0, CUDA_data[d+12*nd+1] + 1);         \
		}		
//    printf("zypre_newBoxLoop4Begin, %s(%d): %s\n",__FILE__,__LINE__,__FUNCTION__);\


#define zypre_newBoxLoop4End(i1, i2, i3, i4)	\
	});											\
	cudaFree(CUDA_data);						\
	cudaDeviceSynchronize();					\
}


#define zypre_newBoxLoop1Begin1(ndim, loop_size,				\
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
//    printf("zypre_newBoxLoop1Begin, %s(%d): %s\n",__FILE__,__LINE__,__FUNCTION__);\


#define zypre_newBoxLoop1End1(i1)				\
	});											\
    cudaDeviceSynchronize();					\
}

#define zypre_newBoxLoop2Begin1(ndim, loop_size,				\
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
//printf("zypre_newBoxLoop2Begin,%s(%d): %s\n",__FILE__,__LINE__,__FUNCTION__); \


#define zypre_newBoxLoop2End1(i1, i2)			\
	});											\
	cudaDeviceSynchronize();					\
}


#define zypre_newBoxLoop3Begin1(ndim, loop_size,\
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

#define zypre_newBoxLoop3End1(i1, i2,i3)			\
	});											\
	cudaDeviceSynchronize();					\
}
//    printf("zypre_newBoxLoop3Begin, %s(%d): %s\n",__FILE__,__LINE__,__FUNCTION__);\


#define zypre_newBoxLoop4Begin1(ndim, loop_size,				\
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
//    printf("zypre_newBoxLoop4Begin, %s(%d): %s\n",__FILE__,__LINE__,__FUNCTION__);\


#define zypre_newBoxLoop4End1(i1, i2, i3, i4)	\
	});											\
	cudaDeviceSynchronize();					\
}

#ifdef HYPRE_USE_RAJA
#define zypre_Reductioninit(local_result) \
ReduceSum< cuda_reduce<BLOCKSIZE>, HYPRE_Real> local_result(0.0);
#else
#define zypre_Reductioninit(local_result) \
HYPRE_Real       local_result;\
local_result = 0.0;
#endif

#define zypre_newBoxLoop2ReductionBegin(ndim, loop_size,				\
										dbox1, start1, stride1, i1,		\
										dbox2, start2, stride2, i2,sum)	\
{    																	\
	HYPRE_Int hypre__tot = 1.0;								\
	const size_t block_size = 256;				  			\
	HYPRE_Int *CUDA_data, HOST_data[ndim*7+1];							\
	HOST_data[0] = ndim;												\
	for (HYPRE_Int d = 0;d < ndim;d ++)									\
	{																	\
	    HOST_data[d+1] = loop_size[d];									\
		HOST_data[d+ndim+1] = stride1[d];								\
		HOST_data[d+2*ndim+1] = start1[d] - dbox1->imin[d];				\
		HOST_data[d+3*ndim+1] = dbox1->imax[d]-dbox1->imin[d];			\
		HOST_data[d+4*ndim+1] = stride2[d];								\
		HOST_data[d+5*ndim+1] = start2[d] - dbox2->imin[d];				\
		HOST_data[d+6*ndim+1] = dbox2->imax[d]-dbox2->imin[d];			\
		hypre__tot *= loop_size[d];										\
	}																	\
	cudaMalloc((int**)&CUDA_data,sizeof(HYPRE_Int)*(ndim*7+1));			\
	cudaMemcpy(CUDA_data, HOST_data, sizeof(HYPRE_Int)*(ndim*7+1), cudaMemcpyHostToDevice);\
	forall< cuda_exec<block_size> >(0, hypre__tot, [=] __device__ (int i) \
	{																	\
	    zypre_BoxLoopCUDADeclare()											\
	    HYPRE_Int hypre_boxD1 = 1.0,hypre_boxD2 = 1.0;						\
	    HYPRE_Int i1 = 0, i2 = 0;											\
		HYPRE_Int nd = CUDA_data[0];									\
		for (d = 0;d < nd;d ++)										\
		{																\
			local_idx  = idx_local % CUDA_data[d+1];					\
			idx_local  = idx_local / CUDA_data[d+1];					\
			i1 += (local_idx*CUDA_data[d+nd+1] + CUDA_data[d+2*nd+1]) * hypre_boxD1; \
			hypre_boxD1 *= hypre_max(0, CUDA_data[d+3*nd+1] + 1);         \
			i2 += (local_idx*CUDA_data[d+4*nd+1] + CUDA_data[d+5*nd+1]) * hypre_boxD2; \
			hypre_boxD2 *= hypre_max(0, CUDA_data[d+6*nd+1] + 1);         \
		}
//    printf("zypre_newBoxLoop2ReductionBegin, %s(%d): %s\n",__FILE__,__LINE__,__FUNCTION__);\


#define zypre_newBoxLoop2ReductionEnd(i1, i2, sum)	\
	});											\
	cudaFree(CUDA_data);						\
	cudaDeviceSynchronize();					\
}

#define zypre_newBoxLoop0For() {}

#define zypre_newBoxLoop1For(i1) {}

#define zypre_newBoxLoop2For(i1, i2) {}

#define zypre_newBoxLoop3For(i1, i2, i3) {}

#define zypre_newBoxLoop4For(i1, i2, i3, i4) {}

#define zypre_newBoxLoopSetOneBlock() {}


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
