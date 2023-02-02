/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Header info for the BoxLoop
 *
 *****************************************************************************/

/*--------------------------------------------------------------------------
 * BoxLoop macros:
 *--------------------------------------------------------------------------*/

#ifndef HYPRE_BOXLOOP_RAJA_HEADER
#define HYPRE_BOXLOOP_RAJA_HEADER

#if defined(HYPRE_USING_RAJA)

#ifdef __cplusplus
extern "C++"
{
#endif

#include <RAJA/RAJA.hpp>
   using namespace RAJA;

#ifdef __cplusplus
}
#endif

typedef struct hypre_Boxloop_struct
{
   HYPRE_Int lsize0, lsize1, lsize2;
   HYPRE_Int strides0, strides1, strides2;
   HYPRE_Int bstart0, bstart1, bstart2;
   HYPRE_Int bsize0, bsize1, bsize2;
} hypre_Boxloop;


#if defined(HYPRE_USING_CUDA) /* RAJA with CUDA, running on device */

#define BLOCKSIZE                HYPRE_1D_BLOCK_SIZE
#define hypre_RAJA_DEVICE        RAJA_DEVICE
#define hypre_raja_exec_policy   cuda_exec<BLOCKSIZE>
/* #define hypre_raja_reduce_policy cuda_reduce_atomic<BLOCKSIZE> */
#define hypre_raja_reduce_policy cuda_reduce //<BLOCKSIZE>
#define hypre_fence()
/*
#define hypre_fence() \
cudaError err = cudaGetLastError();\
if ( cudaSuccess != err ) {\
printf("\n ERROR zypre_newBoxLoop: %s in %s(%d) function %s\n",cudaGetErrorString(err),__FILE__,__LINE__,__FUNCTION__); \
}\
hypre_CheckErrorDevice(cudaDeviceSynchronize());
*/

#elif defined(HYPRE_USING_DEVICE_OPENMP) /* RAJA with OpenMP (>4.5), running on device */

#define hypre_RAJA_DEVICE
#define hypre_raja_exec_policy   omp_target_parallel_for_exec<BLOCKSIZE>
#define hypre_raja_reduce_policy omp_target_reduce
#define hypre_fence()

#elif defined(HYPRE_USING_OPENMP) /* RAJA with OpenMP, running on host (CPU) */

#define hypre_RAJA_DEVICE
#define hypre_raja_exec_policy   omp_for_exec
#define hypre_raja_reduce_policy omp_reduce
#define hypre_fence()

#else /* RAJA, running on host (CPU) */

#define hypre_RAJA_DEVICE
#define hypre_raja_exec_policy   seq_exec
#define hypre_raja_reduce_policy seq_reduce
#define hypre_fence()

#endif /* #if defined(HYPRE_USING_CUDA) */




#define zypre_BoxLoopIncK(k,box,hypre__i)                                               \
   HYPRE_Int hypre_boxD##k = 1;                                                         \
   HYPRE_Int hypre__i = 0;                                                              \
   hypre__i += (hypre_IndexD(local_idx, 0)*box.strides0 + box.bstart0) * hypre_boxD##k; \
   hypre_boxD##k *= hypre_max(0, box.bsize0 + 1);                                       \
   hypre__i += (hypre_IndexD(local_idx, 1)*box.strides1 + box.bstart1) * hypre_boxD##k; \
   hypre_boxD##k *= hypre_max(0, box.bsize1 + 1);                                       \
   hypre__i += (hypre_IndexD(local_idx, 2)*box.strides2 + box.bstart2) * hypre_boxD##k; \
   hypre_boxD##k *= hypre_max(0, box.bsize2 + 1);                                       \


#define zypre_newBoxLoopInit(ndim,loop_size)                                \
  HYPRE_Int hypre__tot = 1;                                                 \
  for (HYPRE_Int d = 0;d < ndim;d ++)                                       \
      hypre__tot *= loop_size[d];


#define zypre_newBoxLoopDeclare(box)                                    \
  hypre_Index local_idx;                                                \
  HYPRE_Int idx_local = idx;                                            \
  hypre_IndexD(local_idx, 0)  = idx_local % box.lsize0;                 \
  idx_local = idx_local / box.lsize0;                                   \
  hypre_IndexD(local_idx, 1)  = idx_local % box.lsize1;                 \
  idx_local = idx_local / box.lsize1;                                   \
  hypre_IndexD(local_idx, 2)  = idx_local % box.lsize2;

#define zypre_BoxLoopDataDeclareK(k,ndim,loop_size,dbox,start,stride)   \
   hypre_Boxloop databox##k;                                            \
   databox##k.lsize0 = loop_size[0];                                    \
   databox##k.strides0 = stride[0];                                     \
   databox##k.bstart0  = start[0] - dbox->imin[0];                      \
   databox##k.bsize0   = dbox->imax[0]-dbox->imin[0];                   \
   if (ndim > 1)                                                        \
   {                                                                    \
      databox##k.lsize1 = loop_size[1];                                 \
      databox##k.strides1 = stride[1];                                  \
      databox##k.bstart1  = start[1] - dbox->imin[1];                   \
      databox##k.bsize1   = dbox->imax[1]-dbox->imin[1];                \
   }                                                                    \
   else                                                                 \
   {                                                                    \
      databox##k.lsize1 = 1;                                            \
      databox##k.strides1 = 0;                                          \
      databox##k.bstart1  = 0;                                          \
      databox##k.bsize1   = 0;                                          \
   }                                                                    \
   if (ndim == 3)                                                       \
   {                                                                    \
      databox##k.lsize2 = loop_size[2];                                 \
      databox##k.strides2 = stride[2];                                  \
      databox##k.bstart2  = start[2] - dbox->imin[2];                   \
      databox##k.bsize2   = dbox->imax[2]-dbox->imin[2];                \
   }                                                                    \
   else                                                                 \
   {                                                                    \
      databox##k.lsize2 = 1;                                            \
      databox##k.strides2 = 0;                                          \
      databox##k.bstart2  = 0;                                          \
      databox##k.bsize2   = 0;                                          \
   }

#define zypre_newBoxLoop0Begin(ndim, loop_size)                                                   \
{                                                                                                 \
   zypre_newBoxLoopInit(ndim,loop_size);                                                          \
   forall< hypre_raja_exec_policy >(RangeSegment(0, hypre__tot), [=] hypre_RAJA_DEVICE (HYPRE_Int idx) \
   {


#define zypre_newBoxLoop0End()    \
        });                       \
        hypre_fence();            \
}

#define zypre_newBoxLoop1Begin(ndim, loop_size,                                                    \
                               dbox1, start1, stride1, i1)                                         \
{                                                                                                  \
    zypre_newBoxLoopInit(ndim,loop_size);                                                          \
    zypre_BoxLoopDataDeclareK(1,ndim,loop_size,dbox1,start1,stride1);                              \
    forall< hypre_raja_exec_policy >(RangeSegment(0, hypre__tot), [=] hypre_RAJA_DEVICE (HYPRE_Int idx) \
    {                                                                                              \
       zypre_newBoxLoopDeclare(databox1);                                                          \
       zypre_BoxLoopIncK(1,databox1,i1);


#define zypre_newBoxLoop1End(i1) \
    });                          \
    hypre_fence();               \
}

#define zypre_newBoxLoop2Begin(ndim, loop_size,                                                  \
                               dbox1, start1, stride1, i1,                                       \
                               dbox2, start2, stride2, i2)                                       \
{                                                                                                \
  zypre_newBoxLoopInit(ndim,loop_size);                                                          \
  zypre_BoxLoopDataDeclareK(1,ndim,loop_size,dbox1,start1,stride1);                              \
  zypre_BoxLoopDataDeclareK(2,ndim,loop_size,dbox2,start2,stride2);                              \
  forall< hypre_raja_exec_policy >(RangeSegment(0, hypre__tot), [=] hypre_RAJA_DEVICE (HYPRE_Int idx) \
  {                                                                                              \
     zypre_newBoxLoopDeclare(databox1);                                                          \
     zypre_BoxLoopIncK(1,databox1,i1);                                                           \
     zypre_BoxLoopIncK(2,databox2,i2);


#define zypre_newBoxLoop2End(i1, i2) \
  });                                \
  hypre_fence();                     \
}

#define zypre_newBoxLoop3Begin(ndim, loop_size,                                                   \
                               dbox1, start1, stride1, i1,                                        \
                               dbox2, start2, stride2, i2,                                        \
                               dbox3, start3, stride3, i3)                                        \
{                                                                                                 \
   zypre_newBoxLoopInit(ndim,loop_size);                                                          \
   zypre_BoxLoopDataDeclareK(1,ndim,loop_size,dbox1,start1,stride1);                              \
   zypre_BoxLoopDataDeclareK(2,ndim,loop_size,dbox2,start2,stride2);                              \
   zypre_BoxLoopDataDeclareK(3,ndim,loop_size,dbox3,start3,stride3);                              \
   forall< hypre_raja_exec_policy >(RangeSegment(0, hypre__tot), [=] hypre_RAJA_DEVICE (HYPRE_Int idx) \
   {                                                                                              \
      zypre_newBoxLoopDeclare(databox1);                                                          \
      zypre_BoxLoopIncK(1,databox1,i1);                                                           \
      zypre_BoxLoopIncK(2,databox2,i2);                                                           \
      zypre_BoxLoopIncK(3,databox3,i3);

#define zypre_newBoxLoop3End(i1, i2, i3)                                      \
    });                                                                       \
    hypre_fence();                                                            \
}

#define zypre_newBoxLoop4Begin(ndim, loop_size,                                                   \
                               dbox1, start1, stride1, i1,                                        \
                               dbox2, start2, stride2, i2,                                        \
                               dbox3, start3, stride3, i3,                                        \
                               dbox4, start4, stride4, i4)                                        \
{                                                                                                 \
   zypre_newBoxLoopInit(ndim,loop_size);                                                          \
   zypre_BoxLoopDataDeclareK(1,ndim,loop_size,dbox1,start1,stride1);                              \
   zypre_BoxLoopDataDeclareK(2,ndim,loop_size,dbox2,start2,stride2);                              \
   zypre_BoxLoopDataDeclareK(3,ndim,loop_size,dbox3,start3,stride3);                              \
   zypre_BoxLoopDataDeclareK(4,ndim,loop_size,dbox4,start4,stride4);                              \
   forall< hypre_raja_exec_policy >(RangeSegment(0, hypre__tot), [=] hypre_RAJA_DEVICE (HYPRE_Int idx) \
   {                                                                                              \
      zypre_newBoxLoopDeclare(databox1);                                                          \
      zypre_BoxLoopIncK(1,databox1,i1);                                                           \
      zypre_BoxLoopIncK(2,databox2,i2);                                                           \
      zypre_BoxLoopIncK(3,databox3,i3);                                                           \
      zypre_BoxLoopIncK(4,databox4,i4);

#define zypre_newBoxLoop4End(i1, i2, i3, i4)                                  \
   });                                                                        \
   hypre_fence();                                                             \
}

#define zypre_BasicBoxLoopDataDeclareK(k,ndim,loop_size,stride)               \
   hypre_Boxloop databox##k;                                                  \
   databox##k.lsize0   = loop_size[0];                                        \
   databox##k.strides0 = stride[0];                                           \
   databox##k.bstart0  = 0;                                                   \
   databox##k.bsize0   = 0;                                                   \
   if (ndim > 1)                                                              \
   {                                                                          \
      databox##k.lsize1   = loop_size[1];                                     \
      databox##k.strides1 = stride[1];                                        \
      databox##k.bstart1  = 0;                                                \
      databox##k.bsize1   = 0;                                                \
   }                                                                          \
   else                                                                       \
   {                                                                          \
      databox##k.lsize1   = 1;                                                \
      databox##k.strides1 = 0;                                                \
      databox##k.bstart1  = 0;                                                \
      databox##k.bsize1   = 0;                                                \
   }                                                                          \
   if (ndim == 3)                                                             \
   {                                                                          \
      databox##k.lsize2   = loop_size[2];                                     \
      databox##k.strides2 = stride[2];                                        \
      databox##k.bstart2  = 0;                                                \
      databox##k.bsize2   = 0;                                                \
   }                                                                          \
   else                                                                       \
   {                                                                          \
      databox##k.lsize2   = 1;                                                \
      databox##k.strides2 = 0;                                                \
      databox##k.bstart2  = 0;                                                \
      databox##k.bsize2   = 0;                                                \
   }

#define zypre_newBasicBoxLoop2Begin(ndim, loop_size,                                               \
                                    stride1, i1,                                                   \
                                    stride2, i2)                                                   \
{                                                                                                  \
    zypre_newBoxLoopInit(ndim,loop_size);                                                          \
    zypre_BasicBoxLoopDataDeclareK(1,ndim,loop_size,stride1);                                      \
    zypre_BasicBoxLoopDataDeclareK(2,ndim,loop_size,stride2);                                      \
    forall< hypre_raja_exec_policy >(RangeSegment(0, hypre__tot), [=] hypre_RAJA_DEVICE (HYPRE_Int idx) \
    {                                                                                              \
       zypre_newBoxLoopDeclare(databox1);                                                          \
       zypre_BoxLoopIncK(1,databox1,i1);                                                           \
       zypre_BoxLoopIncK(2,databox2,i2);                                                           \

#define hypre_LoopBegin(size,idx)                                                           \
{                                                                                           \
   forall< hypre_raja_exec_policy >(RangeSegment(0, size), [=] hypre_RAJA_DEVICE (HYPRE_Int idx) \
   {

#define hypre_LoopEnd()                                                        \
   });                                                                         \
   hypre_fence();                                                              \
}

#define hypre_BoxLoopGetIndex(index)                                           \
  index[0] = hypre_IndexD(local_idx, 0);                                       \
  index[1] = hypre_IndexD(local_idx, 1);                                       \
  index[2] = hypre_IndexD(local_idx, 2);

#define hypre_BoxLoopBlock()       0
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
#define hypre_newBoxLoopInit     zypre_newBoxLoopInit

#define hypre_BasicBoxLoop2Begin zypre_newBasicBoxLoop2Begin

/* Reduction */
#define hypre_BoxLoop1ReductionBegin(ndim, loop_size, dbox1, start1, stride1, i1, reducesum) \
        hypre_BoxLoop1Begin(ndim, loop_size, dbox1, start1, stride1, i1)

#define hypre_BoxLoop1ReductionEnd(i1, reducesum) \
        hypre_BoxLoop1End(i1)

#define hypre_BoxLoop2ReductionBegin(ndim, loop_size, dbox1, start1, stride1, i1, \
                                                      dbox2, start2, stride2, i2, reducesum) \
        hypre_BoxLoop2Begin(ndim, loop_size, dbox1, start1, stride1, i1, \
                                             dbox2, start2, stride2, i2)

#define hypre_BoxLoop2ReductionEnd(i1, i2, reducesum) \
        hypre_BoxLoop2End(i1, i2)

#endif

#endif /* #ifndef HYPRE_BOXLOOP_RAJA_HEADER */

