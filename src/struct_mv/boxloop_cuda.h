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

#ifndef HYPRE_BOXLOOP_CUDA_HEADER
#define HYPRE_BOXLOOP_CUDA_HEADER

#if (defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)) && !defined(HYPRE_USING_RAJA) && !defined(HYPRE_USING_KOKKOS)

#define HYPRE_LAMBDA [=] __host__  __device__

/* TODO: RL: support 4-D */
typedef struct hypre_Boxloop_struct
{
   HYPRE_Int lsize0, lsize1, lsize2;
   HYPRE_Int strides0, strides1, strides2;
   HYPRE_Int bstart0, bstart1, bstart2;
   HYPRE_Int bsize0, bsize1, bsize2;
} hypre_Boxloop;

#ifdef __cplusplus
extern "C++"
{
#endif

   /* -------------------------
    *     parfor-loop
    * ------------------------*/

   template <typename LOOP_BODY>
   __global__ void
   forall_kernel( hypre_DeviceItem & item,
                  LOOP_BODY loop_body,
                  HYPRE_Int length )
   {
      const HYPRE_Int idx = hypre_gpu_get_grid_thread_id<1, 1>(item);
      /* const HYPRE_Int number_threads = hypre_gpu_get_grid_num_threads<1,1>(item); */

      if (idx < length)
      {
         loop_body(idx);
      }
   }

   template<typename LOOP_BODY>
   void
   BoxLoopforall( HYPRE_Int length,
                  LOOP_BODY loop_body )
   {
      const HYPRE_MemoryLocation memory_location = hypre_HandleMemoryLocation(hypre_handle());
      const HYPRE_ExecutionPolicy exec_policy = hypre_GetExecPolicy1(memory_location);

      if (exec_policy == HYPRE_EXEC_HOST)
      {
#ifdef HYPRE_USING_OPENMP
         #pragma omp parallel for HYPRE_SMP_SCHEDULE
#endif
         for (HYPRE_Int idx = 0; idx < length; idx++)
         {
            loop_body(idx);
         }
      }
      else if (exec_policy == HYPRE_EXEC_DEVICE)
      {
         const dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
         const dim3 gDim = hypre_GetDefaultDeviceGridDimension(length, "thread", bDim);

         HYPRE_GPU_LAUNCH( forall_kernel, gDim, bDim, loop_body, length );
      }
   }

   /* ------------------------------
    *     parforreduction-loop
    * -----------------------------*/

   template <typename LOOP_BODY, typename REDUCER>
   __global__ void
   reductionforall_kernel( hypre_DeviceItem & item,
                           HYPRE_Int length,
                           REDUCER   reducer,
                           LOOP_BODY loop_body )
   {
      const HYPRE_Int thread_id = hypre_gpu_get_grid_thread_id<1, 1>(item);
      const HYPRE_Int n_threads = hypre_gpu_get_grid_num_threads<1, 1>(item);

      for (HYPRE_Int idx = thread_id; idx < length; idx += n_threads)
      {
         loop_body(idx, reducer);
      }

      /* reduction in block-level and the save the results in reducer */
      reducer.BlockReduce();
   }

   template<typename LOOP_BODY, typename REDUCER>
   void
   ReductionBoxLoopforall( HYPRE_Int  length,
                           REDUCER   & reducer,
                           LOOP_BODY  loop_body )
   {
      if (length <= 0)
      {
         return;
      }

      const HYPRE_MemoryLocation memory_location = hypre_HandleMemoryLocation(hypre_handle());
      const HYPRE_ExecutionPolicy exec_policy = hypre_GetExecPolicy1(memory_location);

      if (exec_policy == HYPRE_EXEC_HOST)
      {
         for (HYPRE_Int idx = 0; idx < length; idx++)
         {
            loop_body(idx, reducer);
         }
      }
      else if (exec_policy == HYPRE_EXEC_DEVICE)
      {
         /* Assume gDim cannot exceed HYPRE_MAX_NTHREADS_BLOCK (the max size for the 2nd reduction)
          * and bDim <= WARP * WARP (because we use 1 warp fro the block-level reduction) */
         const dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
         dim3 gDim = hypre_GetDefaultDeviceGridDimension(length, "thread", bDim);
         gDim.x = hypre_min(gDim.x, HYPRE_MAX_NTHREADS_BLOCK);

         reducer.nblocks = gDim.x;

         reducer.Allocate2ndPhaseBuffer();

         /*
         hypre_printf("length= %d, blocksize = %d, gridsize = %d\n", length, bDim.x, gDim.x);
         */

         HYPRE_GPU_LAUNCH( reductionforall_kernel, gDim, bDim, length, reducer, loop_body );
      }
   }

#ifdef __cplusplus
}
#endif

/* Get 1-D length of the loop, in hypre__tot */
#define hypre_newBoxLoopInit(ndim, loop_size)              \
   HYPRE_Int hypre__tot = 1;                               \
   for (HYPRE_Int hypre_d = 0; hypre_d < ndim; hypre_d ++) \
   {                                                       \
      hypre__tot *= loop_size[hypre_d];                    \
   }

/* Initialize struct for box-k */
#define hypre_BoxLoopDataDeclareK(k, ndim, loop_size, dbox, start, stride) \
   hypre_Boxloop databox##k;                                               \
   /* dim 0 */                                                             \
   databox##k.lsize0   = loop_size[0];                                     \
   databox##k.strides0 = stride[0];                                        \
   databox##k.bstart0  = start[0] - dbox->imin[0];                         \
   databox##k.bsize0   = dbox->imax[0] - dbox->imin[0];                    \
   /* dim 1 */                                                             \
   if (ndim > 1)                                                           \
   {                                                                       \
      databox##k.lsize1   = loop_size[1];                                  \
      databox##k.strides1 = stride[1];                                     \
      databox##k.bstart1  = start[1] - dbox->imin[1];                      \
      databox##k.bsize1   = dbox->imax[1] - dbox->imin[1];                 \
   }                                                                       \
   else                                                                    \
   {                                                                       \
      databox##k.lsize1   = 1;                                             \
      databox##k.strides1 = 0;                                             \
      databox##k.bstart1  = 0;                                             \
      databox##k.bsize1   = 0;                                             \
   }                                                                       \
   /* dim 2 */                                                             \
   if (ndim == 3)                                                          \
   {                                                                       \
      databox##k.lsize2   = loop_size[2];                                  \
      databox##k.strides2 = stride[2];                                     \
      databox##k.bstart2  = start[2] - dbox->imin[2];                      \
      databox##k.bsize2   = dbox->imax[2] - dbox->imin[2];                 \
   }                                                                       \
   else                                                                    \
   {                                                                       \
      databox##k.lsize2   = 1;                                             \
      databox##k.strides2 = 0;                                             \
      databox##k.bstart2  = 0;                                             \
      databox##k.bsize2   = 0;                                             \
   }

#define hypre_BasicBoxLoopDataDeclareK(k,ndim,loop_size,stride) \
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

/* RL: TODO loop_size out of box struct, bsize +1 */
/* Given input 1-D 'idx' in box, get 3-D 'local_idx' in loop_size */
#define hypre_newBoxLoopDeclare(box)                     \
   hypre_Index local_idx;                                \
   HYPRE_Int idx_local = idx;                            \
   hypre_IndexD(local_idx, 0)  = idx_local % box.lsize0; \
   idx_local = idx_local / box.lsize0;                   \
   hypre_IndexD(local_idx, 1)  = idx_local % box.lsize1; \
   idx_local = idx_local / box.lsize1;                   \
   hypre_IndexD(local_idx, 2)  = idx_local % box.lsize2; \

/* Given input 3-D 'local_idx', get 1-D 'hypre__i' in 'box' */
#define hypre_BoxLoopIncK(k, box, hypre__i)                                               \
   HYPRE_Int hypre_boxD##k = 1;                                                           \
   HYPRE_Int hypre__i = 0;                                                                \
   hypre__i += (hypre_IndexD(local_idx, 0) * box.strides0 + box.bstart0) * hypre_boxD##k; \
   hypre_boxD##k *= hypre_max(0, box.bsize0 + 1);                                         \
   hypre__i += (hypre_IndexD(local_idx, 1) * box.strides1 + box.bstart1) * hypre_boxD##k; \
   hypre_boxD##k *= hypre_max(0, box.bsize1 + 1);                                         \
   hypre__i += (hypre_IndexD(local_idx, 2) * box.strides2 + box.bstart2) * hypre_boxD##k; \
   hypre_boxD##k *= hypre_max(0, box.bsize2 + 1);

/* get 3-D local_idx into 'index' */
#define hypre_BoxLoopGetIndexCUDA(index)                                                              \
   index[0] = hypre_IndexD(local_idx, 0);                                                             \
   index[1] = hypre_IndexD(local_idx, 1);                                                             \
   index[2] = hypre_IndexD(local_idx, 2);

/* BoxLoop 0 */
#define hypre_BoxLoop0BeginCUDA(ndim, loop_size)                                                      \
{                                                                                                     \
   hypre_newBoxLoopInit(ndim, loop_size);                                                             \
   BoxLoopforall(hypre__tot, HYPRE_LAMBDA (HYPRE_Int idx)                                             \
   {

#define hypre_BoxLoop0EndCUDA()                                                                       \
   });                                                                                                \
}

/* BoxLoop 1 */
#define hypre_BoxLoop1BeginCUDA(ndim, loop_size, dbox1, start1, stride1, i1)                          \
{                                                                                                     \
   hypre_newBoxLoopInit(ndim, loop_size);                                                             \
   hypre_BoxLoopDataDeclareK(1, ndim, loop_size, dbox1, start1, stride1);                             \
   BoxLoopforall(hypre__tot, HYPRE_LAMBDA (HYPRE_Int idx)                                             \
   {                                                                                                  \
      hypre_newBoxLoopDeclare(databox1);                                                              \
      hypre_BoxLoopIncK(1, databox1, i1);

#define hypre_BoxLoop1EndCUDA(i1)                                                                     \
   });                                                                                                \
}

/* BoxLoop 2 */
#define hypre_BoxLoop2BeginCUDA(ndim, loop_size, dbox1, start1, stride1, i1,                          \
                                                 dbox2, start2, stride2, i2)                          \
{                                                                                                     \
   hypre_newBoxLoopInit(ndim, loop_size);                                                             \
   hypre_BoxLoopDataDeclareK(1, ndim, loop_size, dbox1, start1, stride1);                             \
   hypre_BoxLoopDataDeclareK(2, ndim, loop_size, dbox2, start2, stride2);                             \
   BoxLoopforall(hypre__tot, HYPRE_LAMBDA (HYPRE_Int idx)                                             \
   {                                                                                                  \
      hypre_newBoxLoopDeclare(databox1);                                                              \
      hypre_BoxLoopIncK(1, databox1, i1);                                                             \
      hypre_BoxLoopIncK(2, databox2, i2);

#define hypre_BoxLoop2EndCUDA(i1, i2)                                                                 \
   });                                                                                                \
}

/* BoxLoop 3 */
#define hypre_BoxLoop3BeginCUDA(ndim, loop_size, dbox1, start1, stride1, i1,                          \
                                                 dbox2, start2, stride2, i2,                          \
                                                 dbox3, start3, stride3, i3)                          \
{                                                                                                     \
   hypre_newBoxLoopInit(ndim, loop_size);                                                             \
   hypre_BoxLoopDataDeclareK(1, ndim,loop_size, dbox1, start1, stride1);                              \
   hypre_BoxLoopDataDeclareK(2, ndim,loop_size, dbox2, start2, stride2);                              \
   hypre_BoxLoopDataDeclareK(3, ndim,loop_size, dbox3, start3, stride3);                              \
   BoxLoopforall(hypre__tot, HYPRE_LAMBDA (HYPRE_Int idx)                                             \
   {                                                                                                  \
      hypre_newBoxLoopDeclare(databox1);                                                              \
      hypre_BoxLoopIncK(1, databox1, i1);                                                             \
      hypre_BoxLoopIncK(2, databox2, i2);                                                             \
      hypre_BoxLoopIncK(3, databox3, i3);

#define hypre_BoxLoop3EndCUDA(i1, i2, i3)                                                             \
   });                                                                                                \
}

/* BoxLoop 4 */
#define hypre_BoxLoop4BeginCUDA(ndim, loop_size, dbox1, start1, stride1, i1,                          \
                                                 dbox2, start2, stride2, i2,                          \
                                                 dbox3, start3, stride3, i3,                          \
                                                 dbox4, start4, stride4, i4)                          \
{                                                                                                     \
   hypre_newBoxLoopInit(ndim, loop_size);                                                             \
   hypre_BoxLoopDataDeclareK(1, ndim, loop_size, dbox1, start1, stride1);                             \
   hypre_BoxLoopDataDeclareK(2, ndim, loop_size, dbox2, start2, stride2);                             \
   hypre_BoxLoopDataDeclareK(3, ndim, loop_size, dbox3, start3, stride3);                             \
   hypre_BoxLoopDataDeclareK(4, ndim, loop_size, dbox4, start4, stride4);                             \
   BoxLoopforall(hypre__tot, HYPRE_LAMBDA (HYPRE_Int idx)                                             \
   {                                                                                                  \
      hypre_newBoxLoopDeclare(databox1);                                                              \
      hypre_BoxLoopIncK(1, databox1, i1);                                                             \
      hypre_BoxLoopIncK(2, databox2, i2);                                                             \
      hypre_BoxLoopIncK(3, databox3, i3);                                                             \
      hypre_BoxLoopIncK(4, databox4, i4);

#define hypre_BoxLoop4EndCUDA(i1, i2, i3, i4)                                                         \
   });                                                                                                \
}

/* Basic BoxLoops have no boxes */
/* BoxLoop 1 */
#define hypre_BasicBoxLoop1BeginCUDA(ndim, loop_size, stride1, i1)                                    \
{                                                                                                     \
   hypre_newBoxLoopInit(ndim, loop_size);                                                             \
   hypre_BasicBoxLoopDataDeclareK(1, ndim, loop_size, stride1);                                       \
   BoxLoopforall(hypre__tot, HYPRE_LAMBDA (HYPRE_Int idx)                                             \
   {                                                                                                  \
      hypre_newBoxLoopDeclare(databox1);                                                              \
      hypre_BoxLoopIncK(1, databox1, i1);

/* BoxLoop 2 */
#define hypre_BasicBoxLoop2BeginCUDA(ndim, loop_size, stride1, i1, stride2, i2)                       \
{                                                                                                     \
   hypre_newBoxLoopInit(ndim, loop_size);                                                             \
   hypre_BasicBoxLoopDataDeclareK(1, ndim, loop_size, stride1);                                       \
   hypre_BasicBoxLoopDataDeclareK(2, ndim, loop_size, stride2);                                       \
   BoxLoopforall(hypre__tot, HYPRE_LAMBDA (HYPRE_Int idx)                                             \
   {                                                                                                  \
      hypre_newBoxLoopDeclare(databox1);                                                              \
      hypre_BoxLoopIncK(1, databox1, i1);                                                             \
      hypre_BoxLoopIncK(2, databox2, i2);                                                             \

/* Parallel for-loop */
#define hypre_LoopBeginCUDA(size, idx)                                                                \
{                                                                                                     \
   BoxLoopforall(size, HYPRE_LAMBDA (HYPRE_Int idx)                                                   \
   {

#define hypre_LoopEndCUDA()                                                                           \
   });                                                                                                \
}

/* Reduction BoxLoop1 */
#define hypre_BoxLoop1ReductionBeginCUDA(ndim, loop_size, dbox1, start1, stride1, i1, reducesum)                 \
{                                                                                                                \
   hypre_newBoxLoopInit(ndim, loop_size);                                                                        \
   hypre_BoxLoopDataDeclareK(1, ndim, loop_size, dbox1, start1, stride1);                                        \
   ReductionBoxLoopforall(hypre__tot, reducesum, HYPRE_LAMBDA (HYPRE_Int idx, decltype(reducesum) &reducesum)    \
   {                                                                                                             \
      hypre_newBoxLoopDeclare(databox1);                                                                         \
      hypre_BoxLoopIncK(1, databox1, i1);

#define hypre_BoxLoop1ReductionEndCUDA(i1, reducesum)                                                            \
   });                                                                                                           \
}

/* Reduction BoxLoop2 */
#define hypre_BoxLoop2ReductionBeginCUDA(ndim, loop_size, dbox1, start1, stride1, i1,                            \
                                                          dbox2, start2, stride2, i2, reducesum)                 \
{                                                                                                                \
   hypre_newBoxLoopInit(ndim, loop_size);                                                                        \
   hypre_BoxLoopDataDeclareK(1, ndim, loop_size, dbox1, start1, stride1);                                        \
   hypre_BoxLoopDataDeclareK(2, ndim, loop_size, dbox2, start2, stride2);                                        \
   ReductionBoxLoopforall(hypre__tot, reducesum, HYPRE_LAMBDA (HYPRE_Int idx, decltype(reducesum) &reducesum)    \
   {                                                                                                             \
      hypre_newBoxLoopDeclare(databox1);                                                                         \
      hypre_BoxLoopIncK(1, databox1, i1);                                                                        \
      hypre_BoxLoopIncK(2, databox2, i2);

#define hypre_BoxLoop2ReductionEndCUDA(i1, i2, reducesum)                                                        \
   });                                                                                                           \
}

/* Renamings */
#define hypre_BoxLoopGetIndexDevice          hypre_BoxLoopGetIndexCUDA
#define hypre_BoxLoopBlockDevice()           0
#define hypre_BoxLoop0BeginDevice            hypre_BoxLoop0BeginCUDA
#define hypre_BoxLoop0EndDevice              hypre_BoxLoop0EndCUDA
#define hypre_BoxLoop1BeginDevice            hypre_BoxLoop1BeginCUDA
#define hypre_BoxLoop1EndDevice              hypre_BoxLoop1EndCUDA
#define hypre_BoxLoop2BeginDevice            hypre_BoxLoop2BeginCUDA
#define hypre_BoxLoop2EndDevice              hypre_BoxLoop2EndCUDA
#define hypre_BoxLoop3BeginDevice            hypre_BoxLoop3BeginCUDA
#define hypre_BoxLoop3EndDevice              hypre_BoxLoop3EndCUDA
#define hypre_BoxLoop4BeginDevice            hypre_BoxLoop4BeginCUDA
#define hypre_BoxLoop4EndDevice              hypre_BoxLoop4EndCUDA
#define hypre_BasicBoxLoop1BeginDevice       hypre_BasicBoxLoop1BeginCUDA
#define hypre_BasicBoxLoop2BeginDevice       hypre_BasicBoxLoop2BeginCUDA
#define hypre_LoopBeginDevice                hypre_LoopBeginCUDA
#define hypre_LoopEndDevice                  hypre_LoopEndCUDA
#define hypre_BoxLoop1ReductionBeginDevice   hypre_BoxLoop1ReductionBeginCUDA
#define hypre_BoxLoop1ReductionEndDevice     hypre_BoxLoop1ReductionEndCUDA
#define hypre_BoxLoop2ReductionBeginDevice   hypre_BoxLoop2ReductionBeginCUDA
#define hypre_BoxLoop2ReductionEndDevice     hypre_BoxLoop2ReductionEndCUDA


//TODO TEMP FIX
#define hypre_BoxLoopGetIndex          hypre_BoxLoopGetIndexDevice
#define hypre_BoxLoopBlock()           0
#define hypre_BoxLoop0Begin            hypre_BoxLoop0BeginDevice
#define hypre_BoxLoop0End              hypre_BoxLoop0EndDevice
#define hypre_BoxLoop1Begin            hypre_BoxLoop1BeginDevice
#define hypre_BoxLoop1End              hypre_BoxLoop1EndDevice
#define hypre_BoxLoop2Begin            hypre_BoxLoop2BeginDevice
#define hypre_BoxLoop2End              hypre_BoxLoop2EndDevice
#define hypre_BoxLoop3Begin            hypre_BoxLoop3BeginDevice
#define hypre_BoxLoop3End              hypre_BoxLoop3EndDevice
#define hypre_BoxLoop4Begin            hypre_BoxLoop4BeginDevice
#define hypre_BoxLoop4End              hypre_BoxLoop4EndDevice
#define hypre_BasicBoxLoop1Begin       hypre_BasicBoxLoop1BeginDevice
#define hypre_BasicBoxLoop2Begin       hypre_BasicBoxLoop2BeginDevice
#define hypre_LoopBegin                hypre_LoopBeginDevice
#define hypre_LoopEnd                  hypre_LoopEndDevice
#define hypre_BoxLoop1ReductionBegin   hypre_BoxLoop1ReductionBeginDevice
#define hypre_BoxLoop1ReductionEnd     hypre_BoxLoop1ReductionEndDevice
#define hypre_BoxLoop2ReductionBegin   hypre_BoxLoop2ReductionBeginDevice
#define hypre_BoxLoop2ReductionEnd     hypre_BoxLoop2ReductionEndDevice

#endif

#endif /* #ifndef HYPRE_BOXLOOP_CUDA_HEADER */

