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
#ifdef HYPRE_USING_OPENMP
#include <omp.h>
#endif

#define HYPRE_LAMBDA [=] __host__  __device__
#define BLOCKSIZE 512

typedef struct hypre_Boxloop_struct
{
   HYPRE_Int lsize0,lsize1,lsize2;
   HYPRE_Int strides0,strides1,strides2;
   HYPRE_Int bstart0,bstart1,bstart2;
   HYPRE_Int bsize0,bsize1,bsize2;
} hypre_Boxloop;

#if 1
#define hypre_fence()
/*printf("\n hypre_newBoxLoop in %s(%d) function %s\n",__FILE__,__LINE__,__FUNCTION__);*/
#else
#define hypre_fence()                                                                                                       \
{                                                                                                                           \
  cudaError err = cudaGetLastError();                                                                                       \
  if ( cudaSuccess != err )                                                                                                 \
  {                                                                                                                         \
    printf("\n ERROR hypre_newBoxLoop: %s in %s(%d) function %s\n",cudaGetErrorString(err),__FILE__,__LINE__,__FUNCTION__); \
    /* HYPRE_Int *p = NULL; *p = 1; */                                                                                      \
  }                                                                                                                         \
  hypre_CheckErrorDevice(cudaDeviceSynchronize());                                                                          \
}
#endif

/* #define hypre_reduce_policy  cuda_reduce<BLOCKSIZE> */

extern "C++" {

template <typename LOOP_BODY>
__global__ void forall_kernel(LOOP_BODY loop_body, HYPRE_Int length)
{
   HYPRE_Int idx = blockDim.x * blockIdx.x + threadIdx.x;
   if (idx < length)
   {
      loop_body(idx);
   }
}

template<typename LOOP_BODY>
void BoxLoopforall(HYPRE_Int policy, HYPRE_Int length, LOOP_BODY loop_body)
{
   if (policy == HYPRE_MEMORY_HOST)
   {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for HYPRE_SMP_SCHEDULE
#endif
      for (HYPRE_Int idx = 0; idx < length; idx++)
      {
         loop_body(idx);
      }
   }
   else if (policy == HYPRE_MEMORY_DEVICE)
   {
      HYPRE_Int gridSize = (length + BLOCKSIZE - 1) / BLOCKSIZE;
      if (gridSize == 0)
      {
         gridSize = 1;
      }
      forall_kernel<<<gridSize, BLOCKSIZE>>>(loop_body, length);
   }
   else if (policy == 2)
   {
   }
}


template <typename LOOP_BODY>
__global__ void reductionforall_kernel(LOOP_BODY ReductionLoop,
                                       HYPRE_Int length)
{
   ReductionLoop(blockDim.x*blockIdx.x+threadIdx.x, blockDim.x*gridDim.x, length);
}

template<typename LOOP_BODY>
void ReductionBoxLoopforall(HYPRE_Int policy, HYPRE_Int length, LOOP_BODY ReductionLoop)
{
   if (length <= 0)
   {
      return;
   }

   if (policy == HYPRE_MEMORY_HOST)
   {
   }
   else if (policy == HYPRE_MEMORY_DEVICE)
   {
      HYPRE_Int gridSize = (length + BLOCKSIZE - 1) / BLOCKSIZE;
      gridSize = hypre_min(gridSize, 1024);

      /*
      hypre_printf("length= %d, blocksize = %d, gridsize = %d\n",
                   length, BLOCKSIZE, gridSize);
      */
      reductionforall_kernel<<<gridSize, BLOCKSIZE>>>(ReductionLoop, length);
   }
}

}


#define hypre_BoxLoopIncK(k,box,hypre__i)                                               \
   HYPRE_Int hypre_boxD##k = 1;                                                         \
   HYPRE_Int hypre__i = 0;                                                              \
   hypre__i += (hypre_IndexD(local_idx, 0)*box.strides0 + box.bstart0) * hypre_boxD##k; \
   hypre_boxD##k *= hypre_max(0, box.bsize0 + 1);                                       \
   hypre__i += (hypre_IndexD(local_idx, 1)*box.strides1 + box.bstart1) * hypre_boxD##k; \
   hypre_boxD##k *= hypre_max(0, box.bsize1 + 1);                                       \
   hypre__i += (hypre_IndexD(local_idx, 2)*box.strides2 + box.bstart2) * hypre_boxD##k; \
   hypre_boxD##k *= hypre_max(0, box.bsize2 + 1);

#define hypre_newBoxLoopInit(ndim,loop_size)            \
  HYPRE_Int hypre__tot = 1;                             \
  for (HYPRE_Int hypre_d = 0;hypre_d < ndim;hypre_d ++) \
    hypre__tot *= loop_size[hypre_d];

#define hypre_BasicBoxLoopInit(ndim,loop_size)          \
  HYPRE_Int hypre__tot = 1;                             \
  for (HYPRE_Int hypre_d = 0;hypre_d < ndim;hypre_d ++) \
    hypre__tot *= loop_size[hypre_d];                   \

#define hypre_newBoxLoopDeclare(box)                    \
  hypre_Index local_idx;                                \
  HYPRE_Int idx_local = idx;                            \
  hypre_IndexD(local_idx, 0)  = idx_local % box.lsize0; \
  idx_local = idx_local / box.lsize0;                   \
  hypre_IndexD(local_idx, 1)  = idx_local % box.lsize1; \
  idx_local = idx_local / box.lsize1;                   \
  hypre_IndexD(local_idx, 2)  = idx_local % box.lsize2; \

#define hypre_newBoxLoop0Begin(ndim, loop_size)                            \
{                                                                          \
   hypre_newBoxLoopInit(ndim,loop_size);                                   \
   BoxLoopforall(hypre_exec_policy,hypre__tot,HYPRE_LAMBDA (HYPRE_Int idx) \
   {

#define hypre_newBoxLoop0End() \
    });                        \
    hypre_fence();             \
}

#define hypre_BoxLoopDataDeclareK(k,ndim,loop_size,dbox,start,stride) \
hypre_Boxloop databox##k;                                             \
databox##k.lsize0 = loop_size[0];                                     \
databox##k.strides0 = stride[0];                                      \
databox##k.bstart0  = start[0] - dbox->imin[0];                       \
databox##k.bsize0   = dbox->imax[0]-dbox->imin[0];                    \
if (ndim > 1)                                                         \
{                                                                     \
   databox##k.lsize1 = loop_size[1];                                  \
   databox##k.strides1 = stride[1];                                   \
   databox##k.bstart1  = start[1] - dbox->imin[1];                    \
   databox##k.bsize1   = dbox->imax[1]-dbox->imin[1];                 \
}                                                                     \
else                                                                  \
{                                                                     \
   databox##k.lsize1 = 1;                                             \
   databox##k.strides1 = 0;                                           \
   databox##k.bstart1  = 0;                                           \
   databox##k.bsize1   = 0;                                           \
}                                                                     \
if (ndim == 3)                                                        \
{                                                                     \
   databox##k.lsize2 = loop_size[2];                                  \
   databox##k.strides2 = stride[2];                                   \
   databox##k.bstart2  = start[2] - dbox->imin[2];                    \
   databox##k.bsize2   = dbox->imax[2]-dbox->imin[2];                 \
}                                                                     \
else                                                                  \
{                                                                     \
   databox##k.lsize2 = 1;                                             \
   databox##k.strides2 = 0;                                           \
   databox##k.bstart2  = 0;                                           \
   databox##k.bsize2   = 0;                                           \
}

#define hypre_newBoxLoop1Begin(ndim, loop_size,                             \
                               dbox1, start1, stride1, i1)                  \
{                                                                           \
    hypre_newBoxLoopInit(ndim,loop_size);                                   \
    hypre_BoxLoopDataDeclareK(1,ndim,loop_size,dbox1,start1,stride1);       \
    BoxLoopforall(hypre_exec_policy,hypre__tot,HYPRE_LAMBDA (HYPRE_Int idx) \
    {                                                                       \
      hypre_newBoxLoopDeclare(databox1);                                    \
      hypre_BoxLoopIncK(1,databox1,i1);

#define hypre_newBoxLoop1End(i1) \
    });                          \
    hypre_fence();               \
}

#define hypre_newBoxLoop2Begin(ndim, loop_size,                             \
                               dbox1, start1, stride1, i1,                  \
                               dbox2, start2, stride2, i2)                  \
{                                                                           \
    hypre_newBoxLoopInit(ndim,loop_size);                                   \
    hypre_BoxLoopDataDeclareK(1,ndim,loop_size,dbox1,start1,stride1);       \
    hypre_BoxLoopDataDeclareK(2,ndim,loop_size,dbox2,start2,stride2);       \
    BoxLoopforall(hypre_exec_policy,hypre__tot,HYPRE_LAMBDA (HYPRE_Int idx) \
    {                                                                       \
       hypre_newBoxLoopDeclare(databox1);                                   \
       hypre_BoxLoopIncK(1,databox1,i1);                                    \
       hypre_BoxLoopIncK(2,databox2,i2);

#define hypre_newBoxLoop2End(i1, i2) \
    });                              \
    hypre_fence();                   \
}

#define hypre_newBoxLoop3Begin(ndim, loop_size,                             \
                               dbox1, start1, stride1, i1,                  \
                               dbox2, start2, stride2, i2,                  \
                               dbox3, start3, stride3, i3)                  \
{                                                                           \
   hypre_newBoxLoopInit(ndim,loop_size);                                    \
   hypre_BoxLoopDataDeclareK(1,ndim,loop_size,dbox1,start1,stride1);        \
   hypre_BoxLoopDataDeclareK(2,ndim,loop_size,dbox2,start2,stride2);        \
   hypre_BoxLoopDataDeclareK(3,ndim,loop_size,dbox3,start3,stride3);        \
   BoxLoopforall(hypre_exec_policy,hypre__tot,HYPRE_LAMBDA (HYPRE_Int idx)  \
   {                                                                        \
         hypre_newBoxLoopDeclare(databox1);                                 \
         hypre_BoxLoopIncK(1,databox1,i1);                                  \
         hypre_BoxLoopIncK(2,databox2,i2);                                  \
         hypre_BoxLoopIncK(3,databox3,i3);


#define hypre_newBoxLoop3End(i1, i2,i3) \
    });                                 \
    hypre_fence();                      \
}

#define hypre_newBoxLoop4Begin(ndim, loop_size,                            \
                               dbox1, start1, stride1, i1,                 \
                               dbox2, start2, stride2, i2,                 \
                               dbox3, start3, stride3, i3,                 \
                               dbox4, start4, stride4, i4)                 \
{                                                                          \
   hypre_newBoxLoopInit(ndim,loop_size);                                   \
   hypre_BoxLoopDataDeclareK(1,ndim,loop_size,dbox1,start1,stride1);       \
   hypre_BoxLoopDataDeclareK(2,ndim,loop_size,dbox2,start2,stride2);       \
   hypre_BoxLoopDataDeclareK(3,ndim,loop_size,dbox3,start3,stride3);       \
   hypre_BoxLoopDataDeclareK(4,ndim,loop_size,dbox4,start4,stride4);       \
   BoxLoopforall(hypre_exec_policy,hypre__tot,HYPRE_LAMBDA (HYPRE_Int idx) \
   {                                                                       \
         hypre_newBoxLoopDeclare(databox1);                                \
         hypre_BoxLoopIncK(1,databox1,i1);                                 \
         hypre_BoxLoopIncK(2,databox2,i2);                                 \
         hypre_BoxLoopIncK(3,databox3,i3);                                 \
         hypre_BoxLoopIncK(4,databox4,i4);

#define hypre_newBoxLoop4End(i1, i2, i3, i4) \
    });                                      \
    hypre_fence();                           \
}

#define zypre_BasicBoxLoopDataDeclareK(k,ndim,loop_size,stride) \
hypre_Boxloop databox##k;                                       \
databox##k.lsize0   = loop_size[0];                             \
databox##k.strides0 = stride[0];                                \
databox##k.bstart0  = 0;                                        \
databox##k.bsize0   = 0;                                        \
if (ndim > 1)                                                   \
{                                                               \
   databox##k.lsize1   = loop_size[1];                          \
   databox##k.strides1 = stride[1];                             \
   databox##k.bstart1  = 0;                                     \
   databox##k.bsize1   = 0;                                     \
}                                                               \
else                                                            \
{                                                               \
   databox##k.lsize1   = 1;                                     \
   databox##k.strides1 = 0;                                     \
   databox##k.bstart1  = 0;                                     \
   databox##k.bsize1   = 0;                                     \
}                                                               \
if (ndim == 3)                                                  \
{                                                               \
   databox##k.lsize2   = loop_size[2];                          \
   databox##k.strides2 = stride[2];                             \
   databox##k.bstart2  = 0;                                     \
   databox##k.bsize2   = 0;                                     \
}                                                               \
else                                                            \
{                                                               \
    databox##k.lsize2   = 1;                                    \
    databox##k.strides2 = 0;                                    \
    databox##k.bstart2  = 0;                                    \
    databox##k.bsize2   = 0;                                    \
}

#define zypre_newBasicBoxLoop1Begin(ndim, loop_size,                        \
                                    stride1, i1)                            \
{                                                                           \
    hypre_BasicBoxLoopInit(ndim,loop_size);                                 \
    zypre_BasicBoxLoopDataDeclareK(1,ndim,loop_size,stride1);               \
    BoxLoopforall(hypre_exec_policy,hypre__tot,HYPRE_LAMBDA (HYPRE_Int idx) \
    {                                                                       \
        hypre_newBoxLoopDeclare(databox1);                                  \
        hypre_BoxLoopIncK(1,databox1,i1);                                   \

#define zypre_newBasicBoxLoop2Begin(ndim, loop_size,                        \
                                    stride1, i1,                            \
                                    stride2, i2)                            \
{                                                                           \
    hypre_BasicBoxLoopInit(ndim,loop_size);                                 \
    zypre_BasicBoxLoopDataDeclareK(1,ndim,loop_size,stride1);               \
    zypre_BasicBoxLoopDataDeclareK(2,ndim,loop_size,stride2);               \
    BoxLoopforall(hypre_exec_policy,hypre__tot,HYPRE_LAMBDA (HYPRE_Int idx) \
    {                                                                       \
        hypre_newBoxLoopDeclare(databox1);                                  \
        hypre_BoxLoopIncK(1,databox1,i1);                                   \
        hypre_BoxLoopIncK(2,databox2,i2);                                   \


#define hypre_LoopBegin(size,idx)                                    \
{                                                                    \
   BoxLoopforall(hypre_exec_policy,size,HYPRE_LAMBDA (HYPRE_Int idx) \
   {

#define hypre_LoopEnd() \
   });                  \
   hypre_fence();       \
}

#define hypre_newBoxLoopGetIndex(index)                                                                                \
  index[0] = hypre_IndexD(local_idx, 0); index[1] = hypre_IndexD(local_idx, 1); index[2] = hypre_IndexD(local_idx, 2);

#define hypre_BoxLoopGetIndex    zypre_BoxLoopGetIndex

#define hypre_BoxLoopSetOneBlock() ;
#define hypre_BoxLoopBlock()       0

#define hypre_BoxLoop0Begin      hypre_newBoxLoop0Begin
#define hypre_BoxLoop0For        hypre_newBoxLoop0For
#define hypre_BoxLoop0End        hypre_newBoxLoop0End
#define hypre_BoxLoop1Begin      hypre_newBoxLoop1Begin
#define hypre_BoxLoop1For        hypre_newBoxLoop1For
#define hypre_BoxLoop1End        hypre_newBoxLoop1End
#define hypre_BoxLoop2Begin      hypre_newBoxLoop2Begin
#define hypre_BoxLoop2For        hypre_newBoxLoop2For
#define hypre_BoxLoop2End        hypre_newBoxLoop2End
#define hypre_BoxLoop3Begin      hypre_newBoxLoop3Begin
#define hypre_BoxLoop3For        hypre_newBoxLoop3For
#define hypre_BoxLoop3End        hypre_newBoxLoop3End
#define hypre_BoxLoop4Begin      hypre_newBoxLoop4Begin
#define hypre_BoxLoop4For        hypre_newBoxLoop4For
#define hypre_BoxLoop4End        hypre_newBoxLoop4End

#define hypre_BasicBoxLoop1Begin zypre_newBasicBoxLoop1Begin
#define hypre_BasicBoxLoop2Begin zypre_newBasicBoxLoop2Begin

/* Reduction BoxLoop1*/
#define hypre_BoxLoop1ReductionBegin(ndim, loop_size,                         \
                                     dbox1, start1, stride1, i1,              \
                                     reducesum)                               \
{                                                                             \
   hypre_newBoxLoopInit(ndim,loop_size);                                      \
   hypre_BoxLoopDataDeclareK(1,ndim,loop_size,dbox1,start1,stride1);          \
   reducesum.nblocks = hypre_min( (hypre__tot+BLOCKSIZE-1)/BLOCKSIZE, 1024 ); \
   ReductionBoxLoopforall(hypre_exec_policy, hypre__tot,                      \
                          HYPRE_LAMBDA (HYPRE_Int tid, HYPRE_Int nthreads,    \
                                        HYPRE_Int len)                        \
   {                                                                          \
       for (HYPRE_Int idx = tid;                                              \
                      idx < len;                                              \
                      idx += nthreads)                                        \
       {                                                                      \
          hypre_newBoxLoopDeclare(databox1);                                  \
          hypre_BoxLoopIncK(1,databox1,i1);

#define hypre_BoxLoop1ReductionEnd(i1, reducesum) \
       }                                          \
       reducesum.BlockReduce();                   \
    });                                           \
    hypre_fence();                                \
}

/* Reduction BoxLoop2 */
#define hypre_BoxLoop2ReductionBegin(ndim, loop_size,                         \
                                     dbox1, start1, stride1, i1,              \
                                     dbox2, start2, stride2, i2,              \
                                     reducesum)                               \
{                                                                             \
   hypre_newBoxLoopInit(ndim,loop_size);                                      \
   hypre_BoxLoopDataDeclareK(1,ndim,loop_size,dbox1,start1,stride1);          \
   hypre_BoxLoopDataDeclareK(2,ndim,loop_size,dbox2,start2,stride2);          \
   reducesum.nblocks = hypre_min( (hypre__tot+BLOCKSIZE-1)/BLOCKSIZE, 1024 ); \
   ReductionBoxLoopforall(hypre_exec_policy, hypre__tot,                      \
                          HYPRE_LAMBDA (HYPRE_Int tid, HYPRE_Int nthreads,    \
                                        HYPRE_Int len)                        \
   {                                                                          \
       for (HYPRE_Int idx = tid;                                              \
                      idx < len;                                              \
                      idx += nthreads)                                        \
       {                                                                      \
          hypre_newBoxLoopDeclare(databox1);                                  \
          hypre_BoxLoopIncK(1,databox1,i1);                                   \
          hypre_BoxLoopIncK(2,databox2,i2);


#define hypre_BoxLoop2ReductionEnd(i1, i2, reducesum) \
       }                                              \
       reducesum.BlockReduce();                       \
    });                                               \
    hypre_fence();                                    \
}

#endif
