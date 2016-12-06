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
}

struct cuda_traversal {};
struct omp_traversal  {};
struct cuda_reduce {};

#define AxCheckError(err) CheckError(err, __FUNCTION__, __LINE__)
inline void CheckError(cudaError_t const err, char const* const fun, const int line)
{
    if (err)
    {
        printf("CUDA Error Code[%d]: %s\n%s() Line:%d\n", err, cudaGetErrorString(err), fun, line);
		int *p = NULL; *p = 1;
    }
}
#define BLOCKSIZE 128

#define hypre_rand(val) \
{\
    curandState_t state;\
    curand_init(0,0,0,&state);\
    val = curand(&state);\
}
extern "C++" {
template <typename LOOP_BODY>
__global__ void forall_kernel(LOOP_BODY loop_body, int length)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < length)
		loop_body(idx);
}

template<typename LOOP_BODY>
void BoxLoopforall (cuda_traversal, HYPRE_Int length, LOOP_BODY loop_body)
{	
	size_t const blockSize = 128;
	size_t gridSize  = (length + blockSize - 1) / blockSize;
	if (gridSize == 0) gridSize = 1;
	
	//hypre_printf("length= %d, blocksize = %d, gridsize = %d\n",length,blockSize,gridSize);
	forall_kernel<<<gridSize, blockSize>>>(loop_body,length);
}

template<typename LOOP_BODY>
void BoxLoopforall (omp_traversal, HYPRE_Int length, LOOP_BODY loop_body)
{

#pragma omp parallel for schedule(static)
	for (int idx = 0;idx < length;idx++)
		loop_body(idx);
}

template<class T>
__global__ void dot (T * a, T * b, T *c, HYPRE_Int hypre__tot,
                     HYPRE_Int *CUDA_data)
{
    int id = (blockIdx.x * blockDim.x) + threadIdx.x;
	HYPRE_Int local_idx;
    HYPRE_Int d,idx_local = id;
    HYPRE_Int hypre_boxD1 = 1.0,hypre_boxD2 = 1.0;
    HYPRE_Int i1 = 0, i2 = 0;
    HYPRE_Int nd = CUDA_data[0];
    //// reducted output
    __shared__ T shared_cache [BLOCKSIZE];
	T sum = 0;
    
    for (d = 0;d < nd;d ++)
    {
        local_idx  = idx_local % CUDA_data[d+1];
		idx_local  = idx_local / CUDA_data[d+1];
		i1 += (local_idx*CUDA_data[d+nd+1] + CUDA_data[d+2*nd+1]) * hypre_boxD1;
		hypre_boxD1 *= hypre_max(0, CUDA_data[d+3*nd+1] + 1);
		i2 += (local_idx*CUDA_data[d+4*nd+1] + CUDA_data[d+5*nd+1]) * hypre_boxD2;
		hypre_boxD2 *= hypre_max(0, CUDA_data[d+6*nd+1] + 1);
    }

	if (id < hypre__tot)
		sum = a[i1] * hypre_conj(b[i2]);
    *(shared_cache + threadIdx.x) = sum;
	
    __syncthreads();
	
    ///////// sum of internal cache
	
    int i;    
    
    for (i=(BLOCKSIZE /2); i>0 ; i= i/2){
		if (threadIdx.x < i){
			*(shared_cache + threadIdx.x) += *(shared_cache + threadIdx.x + i);
		}
		__syncthreads();
    }
	
    if ( threadIdx.x == 0){
        *(c+ blockIdx.x) = shared_cache[0];
    }
}

template<class T>
__global__ void reduction_mult (T * a, T * b, HYPRE_Int hypre__tot,
                     HYPRE_Int *CUDA_data)
{
    int id = (blockIdx.x * blockDim.x) + threadIdx.x;
	HYPRE_Int local_idx;
    HYPRE_Int d,idx_local = id;
    HYPRE_Int hypre_boxD1 = 1.0;
    HYPRE_Int i1 = 0;
    HYPRE_Int nd = CUDA_data[0];
    //// reducted output
    __shared__ T shared_cache [BLOCKSIZE];
	T sum = 0;
    
    for (d = 0;d < nd;d ++)
    {
        local_idx  = idx_local % CUDA_data[d+1];
		idx_local  = idx_local / CUDA_data[d+1];
		i1 += (local_idx*CUDA_data[d+nd+1] + CUDA_data[d+2*nd+1]) * hypre_boxD1;
		hypre_boxD1 *= hypre_max(0, CUDA_data[d+3*nd+1] + 1);
    }

	if (id < hypre__tot)
		sum = a[i1];
    *(shared_cache + threadIdx.x) = sum;
	
    __syncthreads();
	
    ///////// sum of internal cache
	
    int i;    
    
    for (i=(BLOCKSIZE /2); i>0 ; i= i/2){
		if (threadIdx.x < i){
			*(shared_cache + threadIdx.x) *= *(shared_cache + threadIdx.x + i);
		}
		__syncthreads();
    }
	
    if ( threadIdx.x == 0){
        *(b+ blockIdx.x) = shared_cache[0];
    }
}
}

#define hypre_CallocIndex(indexVar, dim)									\
	cudaMallocManaged((void**)&indexVar,sizeof(HYPRE_Int)*dim, cudaMemAttachGlobal);

#define hypre_CallocReal(indexVar, dim)								\
	cudaMallocManaged((void**)&indexVar,sizeof(HYPRE_Real)*dim, cudaMemAttachGlobal);


#define zypre_BoxLoopCUDAInit(ndim)											\
	long hypre__tot = 1.0;											\
	for (HYPRE_Int i = 0;i < ndim;i ++)									\
		hypre__tot *= loop_size[i];


#define zypre_BoxLoopCUDADeclare()										\
	HYPRE_Int local_idx;												\
	HYPRE_Int d,idx_local = idx;

#define zypre_newBoxLoop0Begin(loop_size,idx)					\
{    														\
	BoxLoopforall(cuda_traversal(),loop_size,[=] __device__ (HYPRE_Int idx) \
	{


#define zypre_newBoxLoop0End()					\
	});											\
	cudaDeviceSynchronize();					\
}


typedef struct hypre_Boxloop_struct
{
	HYPRE_Int lsize0,lsize1,lsize2;
	HYPRE_Int strides0,strides1,strides2;
	HYPRE_Int bstart0,bstart1,bstart2;
	HYPRE_Int bsize0,bsize1,bsize2;
} hypre_Boxloop;

extern "C++" {
template <typename LOOP_BODY>
__global__ void forall_kernel1(LOOP_BODY loop_body, hypre_Boxloop box1,int length)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < length)
		loop_body(idx,box1);
}

template<typename LOOP_BODY>
void BoxLoop1forall (cuda_traversal, HYPRE_Int length, hypre_Boxloop box1, LOOP_BODY loop_body)
{	
	size_t const blockSize = 128;
	size_t gridSize  = (length + blockSize - 1) / blockSize;
	if (gridSize == 0) gridSize = 1;
	
	//hypre_printf("length= %d, blocksize = %d, gridsize = %d\n",length,blockSize,gridSize);
	forall_kernel1<<<gridSize, blockSize>>>(loop_body,box1,length);
}
}

#define zypre_newBoxLoop1Begin(ndim, loop_size,				\
							   dbox1, start1, stride1, i1) 	\
{    														\
    HYPRE_Int hypre__tot = 1.0;											\
	const size_t block_size = 256;										\
	hypre_Boxloop databox1;												\
	HYPRE_Int nd = ndim;												\
	databox1.lsize0 = loop_size[0];									\
    databox1.lsize1 = loop_size[1];									\
	databox1.lsize2 = loop_size[2];									\
	databox1.strides0 = stride1[0];										\
	databox1.strides1 = stride1[1];										\
	databox1.strides2 = stride1[2];										\
	databox1.bstart0  = start1[0] - dbox1->imin[0];					\
	databox1.bstart1  = start1[1] - dbox1->imin[1];					\
	databox1.bstart2  = start1[2] - dbox1->imin[2];						\
	databox1.bsize0   = dbox1->imax[0]-dbox1->imin[0];					\
	databox1.bsize1   = dbox1->imax[1]-dbox1->imin[1];							\
	databox1.bsize2   = dbox1->imax[2]-dbox1->imin[2];					\
	for (HYPRE_Int d = 0;d < ndim;d ++)									\
	{																	\
		hypre__tot *= loop_size[d];										\
	}																	\
	BoxLoop1forall(cuda_traversal(),hypre__tot,databox1,[=] __device__ (HYPRE_Int idx,hypre_Boxloop box1) \
	{																	\
	    zypre_BoxLoopCUDADeclare()											\
     	HYPRE_Int hypre_boxD1 = 1.0;						\
	    HYPRE_Int i1 = 0;											\
		local_idx  = idx_local % box1.lsize0;							\
		idx_local  = idx_local / box1.lsize0;							\
		i1 += (local_idx*box1.strides0 + box1.bstart0) * hypre_boxD1;			\
		hypre_boxD1 *= hypre_max(0, box1.bsize0 + 1);					\
		local_idx  = idx_local % box1.lsize1;							\
		idx_local  = idx_local / box1.lsize1;							\
		i1 += (local_idx*box1.strides1 + box1.bstart1) * hypre_boxD1;			\
		hypre_boxD1 *= hypre_max(0, box1.bsize1 + 1);					\
		local_idx  = idx_local % box1.lsize2;							\
		idx_local  = idx_local / box1.lsize2;							\
		i1 += (local_idx*box1.strides2 + box1.bstart2) * hypre_boxD1;			\
		hypre_boxD1 *= hypre_max(0, box1.bsize2 + 1);					\
		
#define zypre_newBoxLoop1End(i1)				\
	});											\
cudaError err = cudaGetLastError();\
if ( cudaSuccess != err ) {\
printf("\n ERROR zypre_newBoxLoop1End: %s in %s(%d) function %s\n",cudaGetErrorString(err),__FILE__,__LINE__,__FUNCTION__); \
int *p = NULL; *p = 1;\
}\
}


#define zypre_newBoxLoop1Begin1(ndim, loop_tot,loop_size,		\
								dbox1, start1, stride1, i1) 	\
{    														\
    HYPRE_Int hypre__tot = 1.0;											\
	const size_t block_size = 256;										\
    HYPRE_Int * imin1 = dbox1->iminD;\
	HYPRE_Int * imax1 = dbox1->imaxD;									\
	BoxLoopforall(cuda_traversal(),loop_tot,[=] __device__ (HYPRE_Int idx) \
	{																	\
	    zypre_BoxLoopCUDADeclare()											\
     	HYPRE_Int hypre_boxD1 = 1;						\
	    HYPRE_Int i1 = 0;											\
		HYPRE_Int nd = ndim_global;								\
		for (d = 0;d < nd;d ++)										\
		{																\
			local_idx  = idx_local % loop_size[d];					\
			idx_local  = idx_local / loop_size[d];					\
			i1 += (local_idx*stride1[d] + (start1[d] - imin1[d])) * hypre_boxD1;	\
			hypre_boxD1 *= hypre_max(0, imax1[d]-imin1[d] + 1);			\
		}

#define zypre_newBoxLoop1End1(i1)				\
	});											\
cudaError err = cudaGetLastError();\
if ( cudaSuccess != err ) {\
printf("\n ERROR zypre_newBoxLoop1End: %s in %s(%d) function %s\n",cudaGetErrorString(err),__FILE__,__LINE__,__FUNCTION__); \
int *p = NULL; *p = 1;\
}\
}

extern "C++" {
template <typename LOOP_BODY>
__global__ void forall_kernel2(LOOP_BODY loop_body, hypre_Boxloop box1,hypre_Boxloop box2,int length)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < length)
		loop_body(idx,box1,box2);
}

template<typename LOOP_BODY>
void BoxLoop2forall (cuda_traversal, HYPRE_Int length, hypre_Boxloop box1,hypre_Boxloop box2, LOOP_BODY loop_body)
{	
	size_t const blockSize = 128;
	size_t gridSize  = (length + blockSize - 1) / blockSize;
	if (gridSize == 0) gridSize = 1;
	
	forall_kernel2<<<gridSize, blockSize>>>(loop_body,box1,box2,length);
}
}
	
#define zypre_newBoxLoop2Begin(ndim, loop_size,				\
                                dbox1, start1, stride1, i1,	\
                                dbox2, start2, stride2, i2)	\
{    														\
    HYPRE_Int hypre__tot = 1.0;											\
	const size_t block_size = 256;										\
	hypre_Boxloop databox1,databox2;									\
	HYPRE_Int nd = ndim;												\
	databox1.lsize0 = loop_size[0];									\
    databox1.lsize1 = loop_size[1];									\
	databox1.lsize2 = loop_size[2];									\
	databox1.strides0 = stride1[0];										\
	databox1.strides1 = stride1[1];										\
	databox1.strides2 = stride1[2];										\
	databox1.bstart0  = start1[0] - dbox1->imin[0];					\
	databox1.bstart1  = start1[1] - dbox1->imin[1];					\
	databox1.bstart2  = start1[2] - dbox1->imin[2];						\
	databox1.bsize0   = dbox1->imax[0]-dbox1->imin[0];					\
	databox1.bsize1   = dbox1->imax[1]-dbox1->imin[1];							\
	databox1.bsize2   = dbox1->imax[2]-dbox1->imin[2];					\
	databox2.lsize0 = loop_size[0];									\
    databox2.lsize1 = loop_size[1];									\
	databox2.lsize2 = loop_size[2];									\
	databox2.strides0 = stride2[0];										\
	databox2.strides1 = stride2[1];										\
	databox2.strides2 = stride2[2];										\
	databox2.bstart0  = start2[0] - dbox2->imin[0];					\
	databox2.bstart1  = start2[1] - dbox2->imin[1];					\
	databox2.bstart2  = start2[2] - dbox2->imin[2];						\
	databox2.bsize0   = dbox2->imax[0]-dbox2->imin[0];					\
	databox2.bsize1   = dbox2->imax[1]-dbox2->imin[1];							\
	databox2.bsize2   = dbox2->imax[2]-dbox2->imin[2];					\
	for (HYPRE_Int d = 0;d < ndim;d ++)									\
	{																	\
		hypre__tot *= loop_size[d];										\
	}																	\
	BoxLoop2forall(cuda_traversal(),hypre__tot,databox1,databox2,[=] __device__ (HYPRE_Int idx,hypre_Boxloop box1,hypre_Boxloop box2) \
	{																	\
	    zypre_BoxLoopCUDADeclare()											\
	    HYPRE_Int hypre_boxD1 = 1.0,hypre_boxD2 = 1.0;						\
	    HYPRE_Int i1 = 0, i2 = 0;											\
		local_idx  = idx_local % box1.lsize0;							\
		idx_local  = idx_local / box1.lsize0;							\
		i1 += (local_idx*box1.strides0 + box1.bstart0) * hypre_boxD1;			\
		hypre_boxD1 *= hypre_max(0, box1.bsize0 + 1);					\
		i2 += (local_idx*box2.strides0 + box2.bstart0) * hypre_boxD2;			\
		hypre_boxD2 *= hypre_max(0, box2.bsize0 + 1);					\
		local_idx  = idx_local % box1.lsize1;							\
		idx_local  = idx_local / box1.lsize1;							\
		i1 += (local_idx*box1.strides1 + box1.bstart1) * hypre_boxD1;			\
		hypre_boxD1 *= hypre_max(0, box1.bsize1 + 1);					\
		i2 += (local_idx*box2.strides1 + box2.bstart1) * hypre_boxD2;			\
		hypre_boxD2 *= hypre_max(0, box2.bsize1 + 1);					\
		local_idx  = idx_local % box1.lsize2;							\
		idx_local  = idx_local / box1.lsize2;							\
		i1 += (local_idx*box1.strides2 + box1.bstart2) * hypre_boxD1;			\
		hypre_boxD1 *= hypre_max(0, box1.bsize2 + 1);					\
		i2 += (local_idx*box2.strides2 + box2.bstart2) * hypre_boxD2;			\
		hypre_boxD2 *= hypre_max(0, box2.bsize2 + 1);					\



#define zypre_newBoxLoop2End(i1, i2)			\
	});											\
cudaError err = cudaGetLastError();\
if ( cudaSuccess != err ) {\
printf("\n ERROR zypre_newBoxLoop2End: %s in %s(%d) function %s\n",cudaGetErrorString(err),__FILE__,__LINE__,__FUNCTION__); \
int *p = NULL; *p = 1;\
}\
AxCheckError(cudaDeviceSynchronize());\
}

extern "C++" {
template <typename LOOP_BODY>
__global__ void forall_kernel3(LOOP_BODY loop_body, hypre_Boxloop box1,hypre_Boxloop box2,hypre_Boxloop box3,int length)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < length)
		loop_body(idx,box1,box2,box3);
}

template<typename LOOP_BODY>
void BoxLoop3forall (cuda_traversal, HYPRE_Int length, hypre_Boxloop box1,hypre_Boxloop box2,hypre_Boxloop box3, LOOP_BODY loop_body)
{	
	size_t const blockSize = 128;
	size_t gridSize  = (length + blockSize - 1) / blockSize;
	if (gridSize == 0) gridSize = 1;
	
	//hypre_printf("length= %d, blocksize = %d, gridsize = %d\n",length,blockSize,gridSize);
	forall_kernel3<<<gridSize, blockSize>>>(loop_body,box1,box2,box3,length);
}
}

#define zypre_newBoxLoop3Begin(ndim, loop_size,\
							   dbox1, start1, stride1, i1,	\
							   dbox2, start2, stride2, i2,	\
							   dbox3, start3, stride3, i3)	\
{																	\
    HYPRE_Int hypre__tot = 1.0;											\
	const size_t block_size = 256;				  			\
	hypre_Boxloop databox1,databox2,databox3;							\
	HYPRE_Int nd = ndim;												\
	databox1.lsize0 = loop_size[0];									\
    databox1.lsize1 = loop_size[1];									\
	databox1.lsize2 = loop_size[2];									\
	databox1.strides0 = stride1[0];										\
	databox1.strides1 = stride1[1];										\
	databox1.strides2 = stride1[2];										\
	databox1.bstart0  = start1[0] - dbox1->imin[0];					\
	databox1.bstart1  = start1[1] - dbox1->imin[1];					\
	databox1.bstart2  = start1[2] - dbox1->imin[2];						\
	databox1.bsize0   = dbox1->imax[0]-dbox1->imin[0];					\
	databox1.bsize1   = dbox1->imax[1]-dbox1->imin[1];							\
	databox1.bsize2   = dbox1->imax[2]-dbox1->imin[2];					\
	databox2.lsize0 = loop_size[0];									\
    databox2.lsize1 = loop_size[1];									\
	databox2.lsize2 = loop_size[2];									\
	databox2.strides0 = stride2[0];										\
	databox2.strides1 = stride2[1];										\
	databox2.strides2 = stride2[2];										\
	databox2.bstart0  = start2[0] - dbox2->imin[0];					\
	databox2.bstart1  = start2[1] - dbox2->imin[1];					\
	databox2.bstart2  = start2[2] - dbox2->imin[2];						\
	databox2.bsize0   = dbox2->imax[0]-dbox2->imin[0];					\
	databox2.bsize1   = dbox2->imax[1]-dbox2->imin[1];							\
	databox2.bsize2   = dbox2->imax[2]-dbox2->imin[2];					\
	databox3.lsize0 = loop_size[0];									\
    databox3.lsize1 = loop_size[1];									\
	databox3.lsize2 = loop_size[2];									\
	databox3.strides0 = stride3[0];										\
	databox3.strides1 = stride3[1];										\
	databox3.strides2 = stride3[2];										\
	databox3.bstart0  = start3[0] - dbox3->imin[0];					\
	databox3.bstart1  = start3[1] - dbox3->imin[1];					\
	databox3.bstart2  = start3[2] - dbox3->imin[2];						\
	databox3.bsize0   = dbox3->imax[0]-dbox3->imin[0];					\
	databox3.bsize1   = dbox3->imax[1]-dbox3->imin[1];							\
	databox3.bsize2   = dbox3->imax[2]-dbox3->imin[2];					\
	for (HYPRE_Int d = 0;d < ndim;d ++)									\
	{																	\
		hypre__tot *= loop_size[d];										\
	}																	\
	BoxLoop3forall(cuda_traversal(),hypre__tot,databox1,databox2,databox3,[=] __device__ (HYPRE_Int idx,hypre_Boxloop box1,hypre_Boxloop box2,hypre_Boxloop box3) \
	{																	\
	    zypre_BoxLoopCUDADeclare()											\
	    HYPRE_Int hypre_boxD1 = 1.0,hypre_boxD2 = 1.0,hypre_boxD3 = 1.0;	\
	    HYPRE_Int i1 = 0, i2 = 0, i3 = 0;											\
		local_idx  = idx_local % box1.lsize0;							\
		idx_local  = idx_local / box1.lsize0;							\
		i1 += (local_idx*box1.strides0 + box1.bstart0) * hypre_boxD1;			\
		hypre_boxD1 *= hypre_max(0, box1.bsize0 + 1);					\
		i2 += (local_idx*box2.strides0 + box2.bstart0) * hypre_boxD2;			\
		hypre_boxD2 *= hypre_max(0, box2.bsize0 + 1);					\
		i3 += (local_idx*box3.strides0 + box3.bstart0) * hypre_boxD3;			\
		hypre_boxD3 *= hypre_max(0, box3.bsize0 + 1);					\
		local_idx  = idx_local % box1.lsize1;							\
		idx_local  = idx_local / box1.lsize1;							\
		i1 += (local_idx*box1.strides1 + box1.bstart1) * hypre_boxD1;			\
		hypre_boxD1 *= hypre_max(0, box1.bsize1 + 1);					\
		i2 += (local_idx*box2.strides1 + box2.bstart1) * hypre_boxD2;			\
		hypre_boxD2 *= hypre_max(0, box2.bsize1 + 1);					\
		i3 += (local_idx*box3.strides1 + box3.bstart1) * hypre_boxD3;			\
		hypre_boxD3 *= hypre_max(0, box3.bsize1 + 1);					\
		local_idx  = idx_local % box1.lsize2;							\
		idx_local  = idx_local / box1.lsize2;							\
		i1 += (local_idx*box1.strides2 + box1.bstart2) * hypre_boxD1;			\
		hypre_boxD1 *= hypre_max(0, box1.bsize2 + 1);					\
		i2 += (local_idx*box2.strides2 + box2.bstart2) * hypre_boxD2;			\
		hypre_boxD2 *= hypre_max(0, box2.bsize2 + 1);					\
		i3 += (local_idx*box3.strides2 + box3.bstart2) * hypre_boxD3;			\
		hypre_boxD3 *= hypre_max(0, box3.bsize2 + 1);					\
		

#define zypre_newBoxLoop3End(i1, i2,i3)			\
	});											\
cudaError err = cudaGetLastError();\
if ( cudaSuccess != err ) {\
printf("\n ERROR zypre_newBoxLoop3End: %s in %s(%d) function %s\n",cudaGetErrorString(err),__FILE__,__LINE__,__FUNCTION__); \
int *p = NULL; *p = 1;\
}\
}

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
	AxCheckError(cudaMalloc((int**)&CUDA_data,sizeof(HYPRE_Int)*(ndim*13+1))); \
	AxCheckError(cudaMemcpy(CUDA_data, HOST_data, sizeof(HYPRE_Int)*(ndim*13+1), cudaMemcpyHostToDevice)); \
	BoxLoopforall(cuda_traversal(),hypre__tot,[=] __device__ (HYPRE_Int idx) \
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

#define zypre_newBoxLoop4End(i1, i2, i3, i4)	\
	});											\
cudaError err = cudaGetLastError();\
if ( cudaSuccess != err ) {\
printf("\n ERROR zypre_newBoxLoop4End: %s in %s(%d) function %s\n",cudaGetErrorString(err),__FILE__,__LINE__,__FUNCTION__); \
int *p = NULL; *p = 1;\
}\
	cudaFree(CUDA_data);						\
}

#define zypre_newBoxLoop2ReductionBegin(ndim, loop_size,				\
										dbox1, start1, stride1, i1,xp,	\
										dbox2, start2, stride2, i2,yp,sum) \
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
	AxCheckError(cudaMalloc((int**)&CUDA_data,sizeof(HYPRE_Int)*(ndim*7+1)));			\
	AxCheckError(cudaMemcpy(CUDA_data, HOST_data, sizeof(HYPRE_Int)*(ndim*7+1), cudaMemcpyHostToDevice));\
	int n_blocks = (hypre__tot+BLOCKSIZE-1)/BLOCKSIZE;					\
	HYPRE_Real *d_c;													\
	HYPRE_Real * c = new HYPRE_Real[n_blocks];							\
	cudaMalloc((void**) &d_c, n_blocks * sizeof(HYPRE_Real));			\
	dot<HYPRE_Real><<< n_blocks ,BLOCKSIZE>>>(xp,yp,d_c,hypre__tot,CUDA_data);		\
cudaError err = cudaGetLastError();\
if ( cudaSuccess != err ) {\
printf("\n ERROR zypre_newBoxLoop2Reduction: %s in %s(%d) function %s\n",cudaGetErrorString(err),__FILE__,__LINE__,__FUNCTION__); \
int *p = NULL; *p = 1;\
}\
	AxCheckError(cudaMemcpy(c,d_c,n_blocks*sizeof(HYPRE_Real),cudaMemcpyDeviceToHost)); \
	cudaFree(CUDA_data);												\
	for (int j = 0 ; j< n_blocks ; ++j){								\
		sum += c[j];													\
	}																	\
	delete c;															\
}

#define zypre_newBoxLoop1ReductionBegin(ndim, loop_size,				\
										dbox1, start1, stride1, i1,xp,sum) \
{    																	\
	HYPRE_Int hypre__tot = 1.0;											\
	const size_t block_size = 256;										\
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
	AxCheckError(cudaMalloc((int**)&CUDA_data,sizeof(HYPRE_Int)*(ndim*4+1))); \
	AxCheckError(cudaMemcpy(CUDA_data, HOST_data, sizeof(HYPRE_Int)*(ndim*4+1), cudaMemcpyHostToDevice));	\
	int n_blocks = (hypre__tot+BLOCKSIZE-1)/BLOCKSIZE;					\
	HYPRE_Real *d_b;													\
	HYPRE_Real * b = new HYPRE_Real[n_blocks];							\
	cudaMalloc((void**) &d_b, n_blocks * sizeof(HYPRE_Real));			\
	reduction_mult<HYPRE_Real><<< n_blocks ,BLOCKSIZE>>>(xp,d_b,hypre__tot,CUDA_data);		\
cudaError err = cudaGetLastError();\
if ( cudaSuccess != err ) {\
printf("\n ERROR zypre_newBoxLoop1Reduction: %s in %s(%d) function %s\n",cudaGetErrorString(err),__FILE__,__LINE__,__FUNCTION__); \
int *p = NULL; *p = 1;\
}\
AxCheckError(cudaMemcpy(b,d_b,n_blocks*sizeof(HYPRE_Real),cudaMemcpyDeviceToHost)); \
	cudaFree(CUDA_data);												\
	for (int j = 0 ; j< n_blocks ; ++j){								\
		sum += b[j];											\
	}																	\
	delete b;															\
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

#define hypre_BoxLoop1Begin1      zypre_newBoxLoop1Begin1
#define hypre_BoxLoop1End1        zypre_newBoxLoop1End1
