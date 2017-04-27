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
extern "C++" {
#include <Kokkos_Core.hpp>
}
#if defined( KOKKOS_HAVE_MPI )
#include <mpi.h>
#endif

#define AxCheckError(err) CheckError(err, __FUNCTION__, __LINE__)
inline void CheckError(cudaError_t const err, char const* const fun, const HYPRE_Int line)
{
    if (err)
    {
        printf("CUDA Error Code[%d]: %s\n%s() Line:%d\n", err, cudaGetErrorString(err), fun, line);
	//HYPRE_Int *p = NULL; *p = 1;
    }
}
#define BLOCKSIZE 256

#if defined(HYPRE_MEMORY_GPU)
#define hypre_rand(val) \
{\
    curandState_t state;\
    curand_init(0,0,0,&state);\
    val = curand(&state);\
}
#else
#define hypre_rand(val) \
{\
    val = rand();\
}
#endif


#define zypre_BoxLoopCUDAInit(ndim)											\
	HYPRE_Int hypre__tot = 1.0;											\
	for (HYPRE_Int i = 0;i < ndim;i ++)									\
		hypre__tot *= loop_size[i];


#define zypre_BoxLoopIncK(k,box,i)					\
{									\
   HYPRE_Int idx = idx_local;						\
   local_idx  = idx % box.lsize0;					\
   idx        = idx / box.lsize0;					\
   i += (local_idx*box.strides0 + box.bstart0) * hypre_boxD##k;		\
   hypre_boxD##k *= hypre_max(0, box.bsize0 + 1);			\
   local_idx  = idx % box.lsize1;					\
   idx        = idx / box.lsize1;					\
   i += (local_idx*box.strides1 + box.bstart1) * hypre_boxD##k;		\
   hypre_boxD##k *= hypre_max(0, box.bsize1 + 1);			\
   local_idx  = idx % box.lsize2;					\
   idx  = idx / box.lsize2;					\
   i += (local_idx*box.strides2 + box.bstart2) * hypre_boxD##k;		\
   hypre_boxD##k *= hypre_max(0, box.bsize2 + 1);			\
}

#define zypre_BoxLoopDataDeclareK(k,ndim,loop_size,dbox,start,stride)	\
	hypre_Boxloop databox##k;											\
	databox##k.lsize0 = loop_size[0];									\
	databox##k.strides0 = stride[0];									\
	databox##k.bstart0  = start[0] - dbox->imin[0];					\
	databox##k.bsize0   = dbox->imax[0]-dbox->imin[0];				\
	databox##k.lsize1 = loop_size[1];									\
	databox##k.strides1 = stride[1];									\
	databox##k.bstart1  = start[1] - dbox->imin[1];					\
	databox##k.bsize1   = dbox->imax[1]-dbox->imin[1];				\
	if (ndim == 3)														\
	{																	\
		databox##k.lsize2 = loop_size[2];								\
		databox##k.strides2 = stride[2];								\
		databox##k.bstart2  = start[2] - dbox->imin[2];				\
		databox##k.bsize2   = dbox->imax[2]-dbox->imin[2];			\
	}																	\
	else																\
	{																	\
		databox##k.lsize2 = 0;											\
		databox##k.strides2 = 0;									\
		databox##k.bstart2  = 0;									\
		databox##k.bsize2   = 0;							\
	}

#define zypre_BoxLoopCUDADeclare()										\
	HYPRE_Int local_idx;												\
	HYPRE_Int idx_local = idx;

#define zypre_newBoxLoop0Begin(ndim, loop_size) 	\
{																	\
    zypre_BoxLoopCUDAInit(ndim);											\
    Kokkos::parallel_for (hypre__tot, [=] (HYPRE_Int idx)		\
    {																		\
	    zypre_newBoxLoopDeclare(idx)										\
	    HYPRE_Int hypre_boxD1 = 1.0;										\
	    HYPRE_Int i1 = idx;


#define zypre_newBoxLoop0End(i1)				\
	});											\
}


#define zypre_newBoxLoop1Begin(ndim, loop_size,				\
			       dbox1, start1, stride1, i1)		\
{									\
    zypre_newBoxLoopInit(ndim)						\
    zypre_BoxLoopDataDeclareK(1,ndim,loop_size,dbox1,start1,stride1);	\
    Kokkos::parallel_for (hypre__tot, KOKKOS_LAMBDA (HYPRE_Int idx)		\
    {									\
      zypre_newBoxLoopInit(ndim);					\
      HYPRE_Int hypre_boxD1 = 1.0;					\
      HYPRE_Int i1 = 0;							\
      local_idx  = idx_local % databox1.lsize0;				\
      idx_local  = idx_local / databox1.lsize0;				\
      i1 += (local_idx*databox1.strides0 + databox1.bstart0) * hypre_boxD1; \
      hypre_boxD1 *= hypre_max(0, databox1.bsize0 + 1);			\
      local_idx  = idx_local % databox1.lsize1;				\
      idx_local  = idx_local / databox1.lsize1;				\
      i1 += (local_idx*databox1.strides1 + databox1.bstart1) * hypre_boxD1; \
      hypre_boxD1 *= hypre_max(0, databox1.bsize1 + 1);			\
      local_idx  = idx_local % databox1.lsize2;				\
      idx_local  = idx_local / databox1.lsize2;				\
      i1 += (local_idx*databox1.strides2 + databox1.bstart2) * hypre_boxD1; \
      hypre_boxD1 *= hypre_max(0, databox1.bsize2 + 1);


#define zypre_newBoxLoop1End(i1)					\
    });									\
    cudaError err = cudaGetLastError();					\
    if ( cudaSuccess != err ) {						\
      printf("\n ERROR zypre_newBoxLoop1End: %s in %s(%d) function %s\n",cudaGetErrorString(err),__FILE__,__LINE__,__FUNCTION__); \
    }									\
    AxCheckError(cudaDeviceSynchronize());				\
}


#define zypre_newBoxLoop2Begin(ndim, loop_size,				\
			       dbox1, start1, stride1, i1,		\
			       dbox2, start2, stride2, i2)		\
{    														\
    zypre_BoxLoopCUDAInit(ndim);						\
    zypre_BoxLoopDataDeclareK(1,ndim,loop_size,dbox1,start1,stride1);	\
    zypre_BoxLoopDataDeclareK(2,ndim,loop_size,dbox2,start2,stride2);	\
    Kokkos::parallel_for (hypre__tot, KOKKOS_LAMBDA (HYPRE_Int idx)		\
    {									\
        zypre_BoxLoopCUDADeclare()					\
        HYPRE_Int hypre_boxD1 = 1.0,hypre_boxD2 = 1.0;			\
	HYPRE_Int i1 = 0, i2 = 0;					\
	local_idx  = idx_local % databox1.lsize0;			\
	idx_local  = idx_local / databox1.lsize0;			\
	i1 += (local_idx*databox1.strides0 + databox1.bstart0) * hypre_boxD1; \
	hypre_boxD1 *= hypre_max(0, databox1.bsize0 + 1);		\
	i2 += (local_idx*databox2.strides0 + databox2.bstart0) * hypre_boxD2; \
	hypre_boxD2 *= hypre_max(0, databox2.bsize0 + 1);		\
	local_idx  = idx_local % databox1.lsize1;			\
	idx_local  = idx_local / databox1.lsize1;			\
	i1 += (local_idx*databox1.strides1 + databox1.bstart1) * hypre_boxD1; \
	hypre_boxD1 *= hypre_max(0, databox1.bsize1 + 1);		\
	i2 += (local_idx*databox2.strides1 + databox2.bstart1) * hypre_boxD2; \
	hypre_boxD2 *= hypre_max(0, databox2.bsize1 + 1);		\
	local_idx  = idx_local % databox1.lsize2;			\
	idx_local  = idx_local / databox1.lsize2;			\
	i1 += (local_idx*databox1.strides2 + databox1.bstart2) * hypre_boxD1; \
	hypre_boxD1 *= hypre_max(0, databox1.bsize2 + 1);		\
	i2 += (local_idx*databox2.strides2 + databox2.bstart2) * hypre_boxD2; \
	hypre_boxD2 *= hypre_max(0, databox2.bsize2 + 1);		\

#define zypre_newBoxLoop2End(i1, i2)			\
     });							\
     cudaError err = cudaGetLastError();			\
     if ( cudaSuccess != err ) {					\
       printf("\n ERROR zypre_newBoxLoop2End: %s in %s(%d) function %s\n",cudaGetErrorString(err),__FILE__,__LINE__,__FUNCTION__); \
     }									\
     AxCheckError(cudaDeviceSynchronize());				\
}


#define zypre_newBoxLoop3Begin(ndim, loop_size,\
			       dbox1, start1, stride1, i1,		\
			       dbox2, start2, stride2, i2,		\
			       dbox3, start3, stride3, i3)		\
{																	\
     zypre_BoxLoopCUDAInit(ndim);						\
     zypre_BoxLoopDataDeclareK(1,ndim,loop_size,dbox1,start1,stride1);	\
     zypre_BoxLoopDataDeclareK(2,ndim,loop_size,dbox2,start2,stride2);	\
     zypre_BoxLoopDataDeclareK(3,ndim,loop_size,dbox3,start3,stride3);	\
     Kokkos::parallel_for (hypre__tot, KOKKOS_LAMBDA (HYPRE_Int idx)		\
     {									\
	zypre_BoxLoopCUDADeclare();					\
	HYPRE_Int hypre_boxD1 = 1.0,hypre_boxD2 = 1.0,hypre_boxD3 = 1.0; \
	HYPRE_Int i1 = 0, i2 = 0, i3 = 0;				\
	local_idx  = idx_local % databox1.lsize0;				\
	idx_local  = idx_local / databox1.lsize0;				\
	i1 += (local_idx*databox1.strides0 + databox1.bstart0) * hypre_boxD1;	\
	hypre_boxD1 *= hypre_max(0, databox1.bsize0 + 1);			\
	i2 += (local_idx*databox2.strides0 + databox2.bstart0) * hypre_boxD2;	\
	hypre_boxD2 *= hypre_max(0, databox2.bsize0 + 1);			\
	i3 += (local_idx*databox3.strides0 + databox3.bstart0) * hypre_boxD3;	\
	hypre_boxD3 *= hypre_max(0, databox3.bsize0 + 1);			\
	local_idx  = idx_local % databox1.lsize1;				\
	idx_local  = idx_local / databox1.lsize1;				\
	i1 += (local_idx*databox1.strides1 + databox1.bstart1) * hypre_boxD1;	\
	hypre_boxD1 *= hypre_max(0, databox1.bsize1 + 1);			\
	i2 += (local_idx*databox2.strides1 + databox2.bstart1) * hypre_boxD2;	\
	hypre_boxD2 *= hypre_max(0, databox2.bsize1 + 1);			\
	i3 += (local_idx*databox3.strides1 + databox3.bstart1) * hypre_boxD3;	\
	hypre_boxD3 *= hypre_max(0, databox3.bsize1 + 1);			\
	local_idx  = idx_local % databox1.lsize2;				\
	idx_local  = idx_local / databox1.lsize2;				\
	i1 += (local_idx*databox1.strides2 + databox1.bstart2) * hypre_boxD1;	\
	hypre_boxD1 *= hypre_max(0, databox1.bsize2 + 1);			\
	i2 += (local_idx*databox2.strides2 + databox2.bstart2) * hypre_boxD2;	\
	hypre_boxD2 *= hypre_max(0, databox2.bsize2 + 1);			\
	i3 += (local_idx*databox3.strides2 +databox3.bstart2) * hypre_boxD3;	\
	hypre_boxD3 *= hypre_max(0, databox3.bsize2 + 1);

#define zypre_newBoxLoop3End(i1, i2, i3)			\
    });							\
    cudaError err = cudaGetLastError();			\
    if ( cudaSuccess != err ) {						\
      printf("\n ERROR zypre_newBoxLoop3End: %s in %s(%d) function %s\n",cudaGetErrorString(err),__FILE__,__LINE__,__FUNCTION__); \
    }									\
    AxCheckError(cudaDeviceSynchronize());				\
}

#define zypre_newBoxLoop4Begin(ndim, loop_size,\
			       dbox1, start1, stride1, i1,		\
			       dbox2, start2, stride2, i2,		\
			       dbox3, start3, stride3, i3,		\
			       dbox4, start4, stride4, i4)		\
{									\
    zypre_BoxLoopCUDAInit(ndim);						\
    zypre_BoxLoopDataDeclareK(1,ndim,loop_size,dbox1,start1,stride1);	\
    zypre_BoxLoopDataDeclareK(2,ndim,loop_size,dbox2,start2,stride2);	\
    zypre_BoxLoopDataDeclareK(3,ndim,loop_size,dbox3,start3,stride3);	\
    zypre_BoxLoopDataDeclareK(4,ndim,loop_size,dbox4,start4,stride4);	\
    Kokkos::parallel_for (hypre__tot, KOKKOS_LAMBDA (HYPRE_Int idx)		\
    {									\
        zypre_BoxLoopCUDADeclare();					\
	HYPRE_Int hypre_boxD1 = 1.0,hypre_boxD2 = 1.0,hypre_boxD3 = 1.0,hypre_boxD4 = 1.0; \
	HYPRE_Int i1 = 0, i2 = 0, i3 = 0,i4 = 0;			\
	local_idx  = idx_local % databox1.lsize0;			\
	idx_local  = idx_local / databox1.lsize0;			\
	i1 += (local_idx*databox1.strides0 + databox1.bstart0) * hypre_boxD1; \
	hypre_boxD1 *= hypre_max(0, databox1.bsize0 + 1);		\
	i2 += (local_idx*databox2.strides0 + databox2.bstart0) * hypre_boxD2; \
	hypre_boxD2 *= hypre_max(0, databox2.bsize0 + 1);		\
	i3 += (local_idx*databox3.strides0 + databox3.bstart0) * hypre_boxD3; \
	hypre_boxD3 *= hypre_max(0, databox3.bsize0 + 1);		\
	i4 += (local_idx*databox4.strides0 + databox4.bstart0) * hypre_boxD4; \
	hypre_boxD4 *= hypre_max(0, databox4.bsize0 + 1);		\
	local_idx  = idx_local % databox1.lsize1;			\
	idx_local  = idx_local / databox1.lsize1;			\
	i1 += (local_idx*databox1.strides1 + databox1.bstart1) * hypre_boxD1; \
	hypre_boxD1 *= hypre_max(0, databox1.bsize1 + 1);		\
	i2 += (local_idx*databox2.strides1 + databox2.bstart1) * hypre_boxD2; \
	hypre_boxD2 *= hypre_max(0, databox2.bsize1 + 1);		\
	i3 += (local_idx*databox3.strides1 + databox3.bstart1) * hypre_boxD3; \
	hypre_boxD3 *= hypre_max(0, databox3.bsize1 + 1);		\
	i4 += (local_idx*databox4.strides1 + databox4.bstart1) * hypre_boxD4; \
	hypre_boxD4 *= hypre_max(0, databox4.bsize1 + 1);		\
	local_idx  = idx_local % databox1.lsize2;			\
	idx_local  = idx_local / databox1.lsize2;			\
	i1 += (local_idx*databox1.strides2 + databox1.bstart2) * hypre_boxD1; \
	hypre_boxD1 *= hypre_max(0, databox1.bsize2 + 1);		\
	i2 += (local_idx*databox2.strides2 + databox2.bstart2) * hypre_boxD2; \
	hypre_boxD2 *= hypre_max(0, databox2.bsize2 + 1);		\
	i3 += (local_idx*databox3.strides2 + databox3.bstart2) * hypre_boxD3; \
	hypre_boxD3 *= hypre_max(0, databox3.bsize2 + 1);		\
	i4 += (local_idx*databox4.strides2 + databox4.bstart2) * hypre_boxD4; \
	hypre_boxD4 *= hypre_max(0, databox4.bsize2 + 1);		\


#define zypre_newBoxLoop4End(i1, i2, i3, i4)		\
    });							\
    cudaError err = cudaGetLastError();					\
    if ( cudaSuccess != err ) {						\
      printf("\n ERROR zypre_newBoxLoop4End: %s in %s(%d) function %s\n",cudaGetErrorString(err),__FILE__,__LINE__,__FUNCTION__); \
    }									\
    AxCheckError(cudaDeviceSynchronize());				\
}

#define zypre_newBoxLoop1ReductionBegin(ndim, loop_size,		\
					dbox1, start1, stride1, i1, sum) \
{									\
    zypre_BoxLoopCUDAInit(ndim);						\
    zypre_BoxLoopDataDeclareK(1,ndim,loop_size,dbox1,start1,stride1);	\
    Kokkos::parallel_reduce (hypre__tot, KOKKOS_LAMBDA (HYPRE_Int idx,HYPRE_Real &sum) \
    {									\
        zypre_BoxLoopCUDADeclare()					\
        HYPRE_Int hypre_boxD1 = 1.0;					\
	HYPRE_Int i1 = 0;						\
	local_idx  = idx_local % databox1.lsize0;			\
	idx_local  = idx_local / databox1.lsize0;			\
	i1 += (local_idx*databox1.strides0 + databox1.bstart0) * hypre_boxD1; \
	hypre_boxD1 *= hypre_max(0, databox1.bsize0 + 1);		\
	local_idx  = idx_local % databox1.lsize1;			\
	idx_local  = idx_local / databox1.lsize1;			\
	i1 += (local_idx*databox1.strides1 + databox1.bstart1) * hypre_boxD1; \
	hypre_boxD1 *= hypre_max(0, databox1.bsize1 + 1);		\
	local_idx  = idx_local % databox1.lsize2;			\
	idx_local  = idx_local / databox1.lsize2;			\
	i1 += (local_idx*databox1.strides2 + databox1.bstart2) * hypre_boxD1; \
	hypre_boxD1 *= hypre_max(0, databox1.bsize2 + 1);		\



#define zypre_newBoxLoop2ReductionEnd(i1, sum)				\
    },sum);								\
    cudaError err = cudaGetLastError();					\
    if ( cudaSuccess != err ) {						\
      printf("\n ERROR zypre_newBoxLoop2End: %s in %s(%d) function %s\n",cudaGetErrorString(err),__FILE__,__LINE__,__FUNCTION__); \
    }									\
    AxCheckError(cudaDeviceSynchronize());				\
}

#define zypre_newBoxLoop2ReductionBegin(ndim, loop_size,		\
					dbox1, start1, stride1, i1,	\
					dbox2, start2, stride2, i2, sum) \
{									\
    zypre_BoxLoopCUDAInit(ndim);						\
    zypre_BoxLoopDataDeclareK(1,ndim,loop_size,dbox1,start1,stride1);	\
    zypre_BoxLoopDataDeclareK(2,ndim,loop_size,dbox2,start2,stride2);	\
    Kokkos::parallel_reduce (hypre__tot, KOKKOS_LAMBDA (HYPRE_Int idx,HYPRE_Real &sum) \
    {									\
        zypre_BoxLoopCUDADeclare()					\
        HYPRE_Int hypre_boxD1 = 1.0,hypre_boxD2 = 1.0;			\
	HYPRE_Int i1 = 0, i2 = 0;					\
	local_idx  = idx_local % databox1.lsize0;			\
	idx_local  = idx_local / databox1.lsize0;			\
	i1 += (local_idx*databox1.strides0 + databox1.bstart0) * hypre_boxD1; \
	hypre_boxD1 *= hypre_max(0, databox1.bsize0 + 1);		\
	i2 += (local_idx*databox2.strides0 + databox2.bstart0) * hypre_boxD2; \
	hypre_boxD2 *= hypre_max(0, databox2.bsize0 + 1);		\
	local_idx  = idx_local % databox1.lsize1;			\
	idx_local  = idx_local / databox1.lsize1;			\
	i1 += (local_idx*databox1.strides1 + databox1.bstart1) * hypre_boxD1; \
	hypre_boxD1 *= hypre_max(0, databox1.bsize1 + 1);		\
	i2 += (local_idx*databox2.strides1 + databox2.bstart1) * hypre_boxD2; \
	hypre_boxD2 *= hypre_max(0, databox2.bsize1 + 1);		\
	local_idx  = idx_local % databox1.lsize2;			\
	idx_local  = idx_local / databox1.lsize2;			\
	i1 += (local_idx*databox1.strides2 + databox1.bstart2) * hypre_boxD1; \
	hypre_boxD1 *= hypre_max(0, databox1.bsize2 + 1);		\
	i2 += (local_idx*databox2.strides2 + databox2.bstart2) * hypre_boxD2; \
	hypre_boxD2 *= hypre_max(0, databox2.bsize2 + 1);		\


#define zypre_newBoxLoop2ReductionEnd(i1, i2, sum)			\
    },sum);								\
    cudaError err = cudaGetLastError();					\
    if ( cudaSuccess != err ) {						\
      printf("\n ERROR zypre_newBoxLoop2End: %s in %s(%d) function %s\n",cudaGetErrorString(err),__FILE__,__LINE__,__FUNCTION__); \
    }									\
    AxCheckError(cudaDeviceSynchronize());				\
}

#define zypre_newBoxLoop1ReductionMult(ndim, loop_size,		\
				       dbox1, start1, stride1, i1, sum) \
{									\
    zypre_BoxLoopCUDAInit(ndim);						\
    zypre_BoxLoopDataDeclareK(1,ndim,loop_size,dbox1,start1,stride1);	\
    Kokkos::parallel_reduce (hypre__tot, KOKKOS_LAMBDA (HYPRE_Int idx,HYPRE_Real &sum) \
    {									\
        zypre_BoxLoopCUDADeclare()					\
        HYPRE_Int hypre_boxD1 = 1.0;					\
	HYPRE_Int i1 = 0;						\
	local_idx  = idx_local % databox1.lsize0;			\
	idx_local  = idx_local / databox1.lsize0;			\
	i1 += (local_idx*databox1.strides0 + databox1.bstart0) * hypre_boxD1; \
	hypre_boxD1 *= hypre_max(0, databox1.bsize0 + 1);		\
	local_idx  = idx_local % databox1.lsize1;			\
	idx_local  = idx_local / databox1.lsize1;			\
	i1 += (local_idx*databox1.strides1 + databox1.bstart1) * hypre_boxD1; \
	hypre_boxD1 *= hypre_max(0, databox1.bsize1 + 1);		\
	local_idx  = idx_local % databox1.lsize2;			\
	idx_local  = idx_local / databox1.lsize2;			\
	i1 += (local_idx*databox1.strides2 + databox1.bstart2) * hypre_boxD1; \
	hypre_boxD1 *= hypre_max(0, databox1.bsize2 + 1);		\
    },sum);								\
    cudaError err = cudaGetLastError();					\
    if ( cudaSuccess != err ) {						\
      printf("\n ERROR zypre_newBoxLoop2End: %s in %s(%d) function %s\n",cudaGetErrorString(err),__FILE__,__LINE__,__FUNCTION__); \
    }									\
    AxCheckError(cudaDeviceSynchronize());				\
}


#define hypre_LoopBegin(size,idx)					\
{    														\
    Kokkos::parallel_for(size, KOKKOS_LAMBDA (HYPRE_Int idx)	\
    {

#define hypre_LoopEnd()							\
    });									\
    AxCheckError(cudaDeviceSynchronize());				\
}
  
#define zypre_BoxBoundaryCopyBegin(ndim, loop_size, stride1, i1, idx) 	\
{    														\
    HYPRE_Int hypre__tot = 1.0;											\
    hypre_Boxloop databox1;						\
    databox1.lsize0 = loop_size[0];					\
    databox1.lsize1 = loop_size[1];					\
    databox1.lsize2 = loop_size[2];					\
    databox1.strides0 = stride1[0];					\
    databox1.strides1 = stride1[1];					\
    databox1.strides2 = stride1[2];					\
    for (HYPRE_Int d = 0;d < ndim;d ++)					\
    {									\
       hypre__tot *= loop_size[d];					\
    }									\
    Kokkos::parallel_for(hypre__tot, KOKKOS_LAMBDA (HYPRE_Int idx)	\
    {									\
        zypre_BoxLoopCUDADeclare()					\
        HYPRE_Int i1 = 0;						\
	local_idx  = idx_local % databox1.lsize0;			\
	idx_local  = idx_local / databox1.lsize0;			\
	i1 += local_idx*databox1.strides0;				\
	local_idx  = idx_local % databox1.lsize1;			\
	idx_local  = idx_local / databox1.lsize1;			\
	i1 += local_idx*databox1.strides1;				\
	local_idx  = idx_local % databox1.lsize2;			\
	idx_local  = idx_local / databox1.lsize2;			\
	i1 += local_idx*databox1.strides2;				\
		
#define zypre_BoxBoundaryCopyEnd()				\
	});											\
cudaError err = cudaGetLastError();\
if ( cudaSuccess != err ) {\
printf("\n ERROR zypre_newBoxLoop1End: %s in %s(%d) function %s\n",cudaGetErrorString(err),__FILE__,__LINE__,__FUNCTION__); \
}\
 AxCheckError(cudaDeviceSynchronize());		\
}

#define zypre_BoxDataExchangeBegin(ndim, loop_size,				\
                                   stride1, i1,	\
                                   stride2, i2)	\
{    														\
    HYPRE_Int hypre__tot = 1.0;											\
    const size_t block_size = 256;					\
    hypre_Boxloop databox1,databox2;					\
    databox1.lsize0 = loop_size[0];					\
    databox1.lsize1 = loop_size[1];					\
    databox1.lsize2 = loop_size[2];					\
    databox1.strides0 = stride1[0];					\
    databox1.strides1 = stride1[1];					\
    databox1.strides2 = stride1[2];					\
    databox2.lsize0 = loop_size[0];					\
    databox2.lsize1 = loop_size[1];					\
    databox2.lsize2 = loop_size[2];					\
    databox2.strides0 = stride2[0];					\
    databox2.strides1 = stride2[1];					\
    databox2.strides2 = stride2[2];					\
    for (HYPRE_Int d = 0;d < ndim;d ++)					\
      {									\
	hypre__tot *= loop_size[d];					\
      }									\
    Kokkos::parallel_for(hypre__tot, KOKKOS_LAMBDA (HYPRE_Int idx)	\
    {									\
        zypre_BoxLoopCUDADeclare()					\
	HYPRE_Int i1 = 0, i2 = 0;					\
	local_idx  = idx_local % databox1.lsize0;			\
	idx_local  = idx_local / databox1.lsize0;			\
	i1 += local_idx*databox1.strides0;				\
	i2 += local_idx*databox2.strides0;				\
	local_idx  = idx_local % databox1.lsize1;			\
	idx_local  = idx_local / databox1.lsize1;			\
	i1 += local_idx*databox1.strides1;				\
	i2 += local_idx*databox2.strides1;				\
	local_idx  = idx_local % databox1.lsize2;			\
	idx_local  = idx_local / databox1.lsize2;			\
	i1 += local_idx*databox1.strides2;				\
	i2 += local_idx*databox2.strides2;



#define zypre_BoxDataExchangeEnd()				\
	});											\
cudaError err = cudaGetLastError();\
if ( cudaSuccess != err ) {\
printf("\n ERROR zypre_newBoxLoop2End: %s in %s(%d) function %s\n",cudaGetErrorString(err),__FILE__,__LINE__,__FUNCTION__); \
}\
AxCheckError(cudaDeviceSynchronize());\
}

#define zypre_newBoxLoop0For() {}

#define zypre_newBoxLoop1For(i1) {}

#define zypre_newBoxLoop2For(i1, i2) {}

#define zypre_newBoxLoop3For(i1, i2, i3) {}

#define zypre_newBoxLoop4For(i1, i2, i3, i4) {}

#define zypre_newBoxLoopSetOneBlock() {}

#define hypre_newBoxLoopGetIndex(index)					\
  index[0] = hypre__i; index[1] = hypre__j; index[2] = hypre__k

#define hypre_BoxLoopGetIndex    zypre_BoxLoopGetIndex
#define hypre_BoxLoopSetOneBlock zypre_newBoxLoopSetOneBlock
#define hypre_BoxLoopBlock()       1
#define hypre_BoxLoop0Begin      zypre_newBoxLoop0Begin
#define hypre_BoxLoop0For        zypre_newBoxLoop0For
#define hypre_BoxLoop0End        zypre_newBoxLoop0End
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

#define hypre_newBoxLoop1ReductionBegin zypre_newBoxLoop1ReductionBegin
#define hypre_newBoxLoop1ReductionEnd   zypre_newBoxLoop1ReductionEnd
#define hypre_newBoxLoop2ReductionBegin zypre_newBoxLoop2ReductionBegin
#define hypre_newBoxLoop2ReductionEnd   zypre_newBoxLoop2ReductionEnd
#define hypre_newBoxLoop1ReductionMult zypre_newBoxLoop1ReductionMult
#define hypre_BoxBoundaryCopyBegin zypre_BoxBoundaryCopyBegin
#define hypre_BoxBoundaryCopyEnd zypre_BoxBoundaryCopyEnd
#define hypre_BoxDataExchangeBegin zypre_BoxDataExchangeBegin
#define hypre_BoxDataExchangeEnd zypre_BoxDataExchangeEnd

#endif
