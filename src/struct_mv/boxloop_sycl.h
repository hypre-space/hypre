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

#ifndef HYPRE_BOXLOOP_SYCL_HEADER
#define HYPRE_BOXLOOP_SYCL_HEADER

#if defined(HYPRE_USING_SYCL) && !defined(HYPRE_USING_RAJA) && !defined(HYPRE_USING_KOKKOS)

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

   /*********************************************************************
    * wrapper functions calling sycl parallel_for
    * WM: todo - add runtime switch between CPU/GPU execution
    *********************************************************************/

   template<typename LOOP_BODY>
   void
   BoxLoopforall( HYPRE_Int length,
                  LOOP_BODY loop_body)
   {
      if (length <= 0)
      {
         return;
      }
      dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
      dim3 gDim = hypre_GetDefaultDeviceGridDimension(length, "thread", bDim);

      hypre_HandleComputeStream(hypre_handle())->submit([&] (sycl::handler & cgh)
      {
         cgh.parallel_for(sycl::nd_range<3>(gDim * bDim, bDim), loop_body);
      }).wait_and_throw();
   }

   template<typename LOOP_BODY>
   void
   ReductionBoxLoopforall( LOOP_BODY  loop_body,
                           HYPRE_Int length,
                           HYPRE_Real * shared_sum_var )
   {
      if (length <= 0)
      {
         return;
      }
      dim3 bDim = hypre_GetDefaultDeviceBlockDimension();
      dim3 gDim = hypre_GetDefaultDeviceGridDimension(length, "thread", bDim);

      hypre_HandleComputeStream(hypre_handle())->submit([&] (sycl::handler & cgh)
      {
         cgh.parallel_for(sycl::nd_range<3>(gDim * bDim, bDim), sycl::reduction(shared_sum_var,
                                                                                std::plus<>()), loop_body);
      }).wait_and_throw();
   }

#ifdef __cplusplus
}
#endif


/*********************************************************************
 * Init/Declare/IncK etc.
 *********************************************************************/

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
   databox##k.lsize0   = loop_size[0];                                     \
   databox##k.strides0 = stride[0];                                        \
   databox##k.bstart0  = start[0] - dbox->imin[0];                         \
   databox##k.bsize0   = dbox->imax[0] - dbox->imin[0];                    \
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
#define hypre_BoxLoopGetIndex(index)      \
   index[0] = hypre_IndexD(local_idx, 0); \
   index[1] = hypre_IndexD(local_idx, 1); \
   index[2] = hypre_IndexD(local_idx, 2);


/*********************************************************************
 * Boxloops
 *********************************************************************/

/* BoxLoop 0 */
#define hypre_newBoxLoop0Begin(ndim, loop_size)                                                       \
{                                                                                                     \
   hypre_newBoxLoopInit(ndim, loop_size);                                                             \
   BoxLoopforall(hypre__tot, [=] (sycl::nd_item<3> item)                                              \
   {                                                                                                  \
      HYPRE_Int idx = (HYPRE_Int) item.get_global_linear_id();                                        \
      if (idx < hypre__tot)                                                                           \
      {                                                                                               \

#define hypre_newBoxLoop0End()                                                                        \
      }                                                                                               \
   });                                                                                                \
}

/* BoxLoop 1 */
#define hypre_newBoxLoop1Begin(ndim, loop_size, dbox1, start1, stride1, i1)                           \
{                                                                                                     \
   hypre_newBoxLoopInit(ndim, loop_size);                                                             \
   hypre_BoxLoopDataDeclareK(1, ndim, loop_size, dbox1, start1, stride1);                             \
   BoxLoopforall(hypre__tot, [=] (sycl::nd_item<3> item)                                              \
   {                                                                                                  \
      HYPRE_Int idx = (HYPRE_Int) item.get_global_linear_id();                                        \
      if (idx < hypre__tot)                                                                           \
      {                                                                                               \
         hypre_newBoxLoopDeclare(databox1);                                                           \
         hypre_BoxLoopIncK(1, databox1, i1);

#define hypre_newBoxLoop1End(i1)                                                                      \
      }                                                                                               \
   });                                                                                                \
}

/* BoxLoop 2 */
#define hypre_newBoxLoop2Begin(ndim, loop_size, dbox1, start1, stride1, i1,                           \
                                                dbox2, start2, stride2, i2)                           \
{                                                                                                     \
   hypre_newBoxLoopInit(ndim, loop_size);                                                             \
   hypre_BoxLoopDataDeclareK(1, ndim, loop_size, dbox1, start1, stride1);                             \
   hypre_BoxLoopDataDeclareK(2, ndim, loop_size, dbox2, start2, stride2);                             \
   BoxLoopforall(hypre__tot, [=] (sycl::nd_item<3> item)                                              \
   {                                                                                                  \
      HYPRE_Int idx = (HYPRE_Int) item.get_global_linear_id();                                        \
      if (idx < hypre__tot)                                                                           \
      {                                                                                               \
         hypre_newBoxLoopDeclare(databox1);                                                           \
         hypre_BoxLoopIncK(1, databox1, i1);                                                          \
         hypre_BoxLoopIncK(2, databox2, i2);

#define hypre_newBoxLoop2End(i1, i2)                                                                  \
      }                                                                                               \
   });                                                                                                \
}

/* BoxLoop 3 */
#define hypre_newBoxLoop3Begin(ndim, loop_size, dbox1, start1, stride1, i1,                           \
                                                dbox2, start2, stride2, i2,                           \
                                                dbox3, start3, stride3, i3)                           \
{                                                                                                     \
   hypre_newBoxLoopInit(ndim, loop_size);                                                             \
   hypre_BoxLoopDataDeclareK(1, ndim,loop_size, dbox1, start1, stride1);                              \
   hypre_BoxLoopDataDeclareK(2, ndim,loop_size, dbox2, start2, stride2);                              \
   hypre_BoxLoopDataDeclareK(3, ndim,loop_size, dbox3, start3, stride3);                              \
   BoxLoopforall(hypre__tot, [=] (sycl::nd_item<3> item)                                              \
   {                                                                                                  \
      HYPRE_Int idx = (HYPRE_Int) item.get_global_linear_id();                                        \
      if (idx < hypre__tot)                                                                           \
      {                                                                                               \
         hypre_newBoxLoopDeclare(databox1);                                                           \
         hypre_BoxLoopIncK(1, databox1, i1);                                                          \
         hypre_BoxLoopIncK(2, databox2, i2);                                                          \
         hypre_BoxLoopIncK(3, databox3, i3);

#define hypre_newBoxLoop3End(i1, i2, i3)                                                              \
      }                                                                                               \
   });                                                                                                \
}

/* BoxLoop 4 */
#define hypre_newBoxLoop4Begin(ndim, loop_size, dbox1, start1, stride1, i1,                           \
                                                dbox2, start2, stride2, i2,                           \
                                                dbox3, start3, stride3, i3,                           \
                                                dbox4, start4, stride4, i4)                           \
{                                                                                                     \
   hypre_newBoxLoopInit(ndim, loop_size);                                                             \
   hypre_BoxLoopDataDeclareK(1, ndim, loop_size, dbox1, start1, stride1);                             \
   hypre_BoxLoopDataDeclareK(2, ndim, loop_size, dbox2, start2, stride2);                             \
   hypre_BoxLoopDataDeclareK(3, ndim, loop_size, dbox3, start3, stride3);                             \
   hypre_BoxLoopDataDeclareK(4, ndim, loop_size, dbox4, start4, stride4);                             \
   BoxLoopforall(hypre__tot, [=] (sycl::nd_item<3> item)                                              \
   {                                                                                                  \
      HYPRE_Int idx = (HYPRE_Int) item.get_global_linear_id();                                        \
      if (idx < hypre__tot)                                                                           \
      {                                                                                               \
         hypre_newBoxLoopDeclare(databox1);                                                           \
         hypre_BoxLoopIncK(1, databox1, i1);                                                          \
         hypre_BoxLoopIncK(2, databox2, i2);                                                          \
         hypre_BoxLoopIncK(3, databox3, i3);                                                          \
         hypre_BoxLoopIncK(4, databox4, i4);

#define hypre_newBoxLoop4End(i1, i2, i3, i4)                                                          \
      }                                                                                               \
   });                                                                                                \
}


/* Basic BoxLoops have no boxes */
/* BoxLoop 1 */
#define hypre_newBasicBoxLoop1Begin(ndim, loop_size, stride1, i1)                                     \
{                                                                                                     \
   hypre_newBoxLoopInit(ndim, loop_size);                                                             \
   hypre_BasicBoxLoopDataDeclareK(1, ndim, loop_size, stride1);                                       \
   BoxLoopforall(hypre__tot, [=] (sycl::nd_item<3> item)                                              \
   {                                                                                                  \
      HYPRE_Int idx = (HYPRE_Int) item.get_global_linear_id();                                        \
      if (idx < hypre__tot)                                                                           \
      {                                                                                               \
         hypre_newBoxLoopDeclare(databox1);                                                           \
         hypre_BoxLoopIncK(1, databox1, i1);

/* BoxLoop 2 */
#define hypre_newBasicBoxLoop2Begin(ndim, loop_size, stride1, i1, stride2, i2)                        \
{                                                                                                     \
   hypre_newBoxLoopInit(ndim, loop_size);                                                             \
   hypre_BasicBoxLoopDataDeclareK(1, ndim, loop_size, stride1);                                       \
   hypre_BasicBoxLoopDataDeclareK(2, ndim, loop_size, stride2);                                       \
   BoxLoopforall(hypre__tot, [=] (sycl::nd_item<3> item)                                              \
   {                                                                                                  \
      HYPRE_Int idx = (HYPRE_Int) item.get_global_linear_id();                                        \
      if (idx < hypre__tot)                                                                           \
      {                                                                                               \
         hypre_newBoxLoopDeclare(databox1);                                                           \
         hypre_BoxLoopIncK(1, databox1, i1);                                                          \
         hypre_BoxLoopIncK(2, databox2, i2);


/* Reduction BoxLoop1 */
#define hypre_newBoxLoop1ReductionBegin(ndim, loop_size, dbox1, start1, stride1, i1, sum_var)         \
{                                                                                                     \
   hypre_newBoxLoopInit(ndim, loop_size);                                                             \
   hypre_BoxLoopDataDeclareK(1, ndim, loop_size, dbox1, start1, stride1);                             \
   HYPRE_Real *shared_sum_var = hypre_CTAlloc(HYPRE_Real, 1, HYPRE_MEMORY_DEVICE);                    \
   hypre_TMemcpy(shared_sum_var, &sum_var, HYPRE_Real, 1, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);    \
   ReductionBoxLoopforall( [=,hypre_unused_var=sum_var] (sycl::nd_item<3> item, auto &sum_var)        \
   {                                                                                                  \
      HYPRE_Int idx = (HYPRE_Int) item.get_global_linear_id();                                        \
      if (idx < hypre__tot)                                                                           \
      {                                                                                               \
         hypre_newBoxLoopDeclare(databox1);                                                           \
         hypre_BoxLoopIncK(1, databox1, i1);

#define hypre_newBoxLoop1ReductionEnd(i1, sum_var)                                                    \
      }                                                                                               \
   }, hypre__tot, shared_sum_var);                                                                    \
   hypre_TMemcpy(&sum_var, shared_sum_var, HYPRE_Real, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);    \
   hypre_TFree(shared_sum_var, HYPRE_MEMORY_DEVICE);                                                  \
}

/* Reduction BoxLoop2 */
#define hypre_newBoxLoop2ReductionBegin(ndim, loop_size, dbox1, start1, stride1, i1,                  \
                                                      dbox2, start2, stride2, i2, sum_var)            \
{                                                                                                     \
   hypre_newBoxLoopInit(ndim, loop_size);                                                             \
   hypre_BoxLoopDataDeclareK(1, ndim, loop_size, dbox1, start1, stride1);                             \
   hypre_BoxLoopDataDeclareK(2, ndim, loop_size, dbox2, start2, stride2);                             \
   HYPRE_Real *shared_sum_var = hypre_CTAlloc(HYPRE_Real, 1, HYPRE_MEMORY_DEVICE);                    \
   hypre_TMemcpy(shared_sum_var, &sum_var, HYPRE_Real, 1, HYPRE_MEMORY_DEVICE, HYPRE_MEMORY_HOST);    \
   ReductionBoxLoopforall( [=,hypre_unused_var=sum_var] (sycl::nd_item<3> item, auto &sum_var)        \
   {                                                                                                  \
      HYPRE_Int idx = (HYPRE_Int) item.get_global_linear_id();                                        \
      if (idx < hypre__tot)                                                                           \
      {                                                                                               \
         hypre_newBoxLoopDeclare(databox1);                                                           \
         hypre_BoxLoopIncK(1, databox1, i1);                                                          \
         hypre_BoxLoopIncK(2, databox2, i2);

#define hypre_newBoxLoop2ReductionEnd(i1, i2, sum_var)                                                \
      }                                                                                               \
   }, hypre__tot, shared_sum_var);                                                                    \
   hypre_TMemcpy(&sum_var, shared_sum_var, HYPRE_Real, 1, HYPRE_MEMORY_HOST, HYPRE_MEMORY_DEVICE);    \
   hypre_TFree(shared_sum_var, HYPRE_MEMORY_DEVICE);                                                  \
}

/* Plain parallel_for loop */
#define hypre_LoopBegin(size, idx)                                                                    \
{                                                                                                     \
   HYPRE_Int hypre__tot = size;                                                                       \
   BoxLoopforall(hypre__tot, [=] (sycl::nd_item<3> item)                                              \
   {                                                                                                  \
      HYPRE_Int idx = (HYPRE_Int) item.get_global_linear_id();                                        \
      if (idx < hypre__tot)                                                                           \
      {                                                                                               \

#define hypre_LoopEnd()                                                                               \
      }                                                                                               \
   });                                                                                                \
}


/*********************************************************************
 * renamings
 *********************************************************************/

#define hypre_BoxLoopBlock()       0

#define hypre_BoxLoop0Begin      hypre_newBoxLoop0Begin
#define hypre_BoxLoop0End        hypre_newBoxLoop0End
#define hypre_BoxLoop1Begin      hypre_newBoxLoop1Begin
#define hypre_BoxLoop1End        hypre_newBoxLoop1End
#define hypre_BoxLoop2Begin      hypre_newBoxLoop2Begin
#define hypre_BoxLoop2End        hypre_newBoxLoop2End
#define hypre_BoxLoop3Begin      hypre_newBoxLoop3Begin
#define hypre_BoxLoop3End        hypre_newBoxLoop3End
#define hypre_BoxLoop4Begin      hypre_newBoxLoop4Begin
#define hypre_BoxLoop4End        hypre_newBoxLoop4End

#define hypre_BasicBoxLoop1Begin hypre_newBasicBoxLoop1Begin
#define hypre_BasicBoxLoop2Begin hypre_newBasicBoxLoop2Begin

/* Reduction */
#define hypre_BoxLoop1ReductionBegin(ndim, loop_size, dbox1, start1, stride1, i1, reducesum) \
        hypre_newBoxLoop1ReductionBegin(ndim, loop_size, dbox1, start1, stride1, i1, reducesum)

#define hypre_BoxLoop1ReductionEnd(i1, reducesum) \
        hypre_newBoxLoop1ReductionEnd(i1, reducesum)

#define hypre_BoxLoop2ReductionBegin(ndim, loop_size, dbox1, start1, stride1, i1, \
                                                      dbox2, start2, stride2, i2, reducesum) \
        hypre_newBoxLoop2ReductionBegin(ndim, loop_size, dbox1, start1, stride1, i1, \
                                                         dbox2, start2, stride2, i2, reducesum)

#define hypre_BoxLoop2ReductionEnd(i1, i2, reducesum) \
        hypre_newBoxLoop2ReductionEnd(i1, i2, reducesum)

#endif

#endif /* #ifndef HYPRE_BOXLOOP_SYCL_HEADER */

