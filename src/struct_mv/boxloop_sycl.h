/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
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

typedef struct hypre_Boxloop_struct
{
   HYPRE_Int lsize0,lsize1,lsize2;
   HYPRE_Int strides0,strides1,strides2;
   HYPRE_Int bstart0,bstart1,bstart2;
   HYPRE_Int bsize0,bsize1,bsize2;
} hypre_Boxloop;


#ifdef __cplusplus
extern "C++" {
#endif

/*********************************************************************
 * forall function
 *********************************************************************/

template<typename LOOP_BODY>
void
BoxLoopforall( LOOP_BODY loop_body,
               HYPRE_Int length )
{
   /* HYPRE_ExecutionPolicy exec_policy = hypre_HandleStructExecPolicy(hypre_handle()); */
   /* WM: TODO: uncomment above and remove below */
   HYPRE_ExecutionPolicy exec_policy = HYPRE_EXEC_DEVICE;

   if (exec_policy == HYPRE_EXEC_HOST)
   {
/* WM: todo - is this really necessary, even? */
/* #ifdef HYPRE_USING_OPENMP */
/* #pragma omp parallel for HYPRE_SMP_SCHEDULE */
/* #endif */
/*       for (HYPRE_Int idx = 0; idx < length; idx++) */
/*       { */
/*          loop_body(idx); */
/*       } */
   }
   else if (exec_policy == HYPRE_EXEC_DEVICE)
   {
      /* WM: question - is it better in sycl to launch parallel_for with blocks in this way as we do for cuda? */
      const sycl::range<1> bDim = hypre_GetDefaultCUDABlockDimension();
      const sycl::range<1> gDim = hypre_GetDefaultCUDAGridDimension(length, "thread", bDim);

      hypre_HandleComputeStream(hypre_handle())->submit([&] (sycl::handler& cgh)
         {
            cgh.parallel_for(sycl::nd_range<1>(gDim*bDim, bDim), loop_body);
         }).wait_and_throw();
   }
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

/* Given input 1-D 'idx' in box, get 3-D 'local_idx' in loop_size */
/* WM: todo - double check that item.get_local_id(0) is actually what you want below */
#define hypre_newBoxLoopDeclare(box)                     \
   hypre_Index local_idx;                                \
   HYPRE_Int idx_local = (HYPRE_Int) idx;              \
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



/* BoxLoop 1 */
#define hypre_newBoxLoop1Begin(ndim, loop_size, dbox1, start1, stride1, i1)                           \
{                                                                                                     \
   hypre_newBoxLoopInit(ndim, loop_size);                                                             \
   hypre_BoxLoopDataDeclareK(1, ndim, loop_size, dbox1, start1, stride1);                             \
   BoxLoopforall( [=] (sycl::nd_item<1> item)                                                         \
   {                                                                                                  \
      size_t idx = item.get_global_linear_id();                                                       \
      if (idx < hypre__tot)                                                                           \
      {                                                                                               \
         hypre_newBoxLoopDeclare(databox1);                                                           \
         hypre_BoxLoopIncK(1, databox1, i1);

#define hypre_newBoxLoop1End(i1)                                                                      \
      }                                                                                               \
   }, hypre__tot);                                                                                    \
}









/*********************************************************************
 * HOST IMPLEMENTATION
 *********************************************************************/

#ifdef HYPRE_USING_OPENMP
#define HYPRE_BOX_REDUCTION
#if defined(WIN32) && defined(_MSC_VER)
#define Pragma(x) __pragma(HYPRE_XSTR(x))
#else
#define Pragma(x) _Pragma(HYPRE_XSTR(x))
#endif
#define OMP0 Pragma(omp parallel for HYPRE_BOX_REDUCTION HYPRE_SMP_SCHEDULE)
#define OMP1 Pragma(omp parallel for private(HYPRE_BOX_PRIVATE) HYPRE_BOX_REDUCTION HYPRE_SMP_SCHEDULE)
#else /* #ifdef HYPRE_USING_OPENMP */
#define OMP0
#define OMP1
#endif /* #ifdef HYPRE_USING_OPENMP */

#define zypre_newBoxLoop0Begin(ndim, loop_size)                               \
{                                                                             \
   zypre_BoxLoopDeclare();                                                    \
   zypre_BoxLoopInit(ndim, loop_size);                                        \
   OMP1                                                                       \
   for (hypre__block = 0; hypre__block < hypre__num_blocks; hypre__block++)   \
   {                                                                          \
      zypre_BoxLoopSet();                                                     \
      for (hypre__J = 0; hypre__J < hypre__JN; hypre__J++)                    \
      {                                                                       \
         for (hypre__I = 0; hypre__I < hypre__IN; hypre__I++)                 \
         {

#define zypre_newBoxLoop0End()                                                \
         }                                                                    \
         zypre_BoxLoopInc1();                                                 \
         zypre_BoxLoopInc2();                                                 \
      }                                                                       \
   }                                                                          \
}

#define zypre_newBoxLoop1Begin(ndim, loop_size,                               \
                               dbox1, start1, stride1, i1)                    \
{                                                                             \
   HYPRE_Int i1;                                                              \
   zypre_BoxLoopDeclare();                                                    \
   zypre_BoxLoopDeclareK(1);                                                  \
   zypre_BoxLoopInit(ndim, loop_size);                                        \
   zypre_BoxLoopInitK(1, dbox1, start1, stride1, i1);                         \
   OMP1                                                                       \
   for (hypre__block = 0; hypre__block < hypre__num_blocks; hypre__block++)   \
   {                                                                          \
      HYPRE_Int i1;                                                           \
      zypre_BoxLoopSet();                                                     \
      zypre_BoxLoopSetK(1, i1);                                               \
      for (hypre__J = 0; hypre__J < hypre__JN; hypre__J++)                    \
      {                                                                       \
         for (hypre__I = 0; hypre__I < hypre__IN; hypre__I++)                 \
         {

#define zypre_newBoxLoop1End(i1)                                              \
            i1 += hypre__i0inc1;                                              \
         }                                                                    \
         zypre_BoxLoopInc1();                                                 \
         i1 += hypre__ikinc1[hypre__d];                                       \
         zypre_BoxLoopInc2();                                                 \
      }                                                                       \
   }                                                                          \
}


#define zypre_newBoxLoop2Begin(ndim, loop_size,                               \
                               dbox1, start1, stride1, i1,                    \
                               dbox2, start2, stride2, i2)                    \
{                                                                             \
   HYPRE_Int i1, i2;                                                          \
   zypre_BoxLoopDeclare();                                                    \
   zypre_BoxLoopDeclareK(1);                                                  \
   zypre_BoxLoopDeclareK(2);                                                  \
   zypre_BoxLoopInit(ndim, loop_size);                                        \
   zypre_BoxLoopInitK(1, dbox1, start1, stride1, i1);                         \
   zypre_BoxLoopInitK(2, dbox2, start2, stride2, i2);                         \
   OMP1                                                                       \
   for (hypre__block = 0; hypre__block < hypre__num_blocks; hypre__block++)   \
   {                                                                          \
      HYPRE_Int i1, i2;                                                       \
      zypre_BoxLoopSet();                                                     \
      zypre_BoxLoopSetK(1, i1);                                               \
      zypre_BoxLoopSetK(2, i2);                                               \
      for (hypre__J = 0; hypre__J < hypre__JN; hypre__J++)                    \
      {                                                                       \
         for (hypre__I = 0; hypre__I < hypre__IN; hypre__I++)                 \
         {

#define zypre_newBoxLoop2End(i1, i2)                                          \
            i1 += hypre__i0inc1;                                              \
            i2 += hypre__i0inc2;                                              \
         }                                                                    \
         zypre_BoxLoopInc1();                                                 \
         i1 += hypre__ikinc1[hypre__d];                                       \
         i2 += hypre__ikinc2[hypre__d];                                       \
         zypre_BoxLoopInc2();                                                 \
      }                                                                       \
   }                                                                          \
}


#define zypre_newBoxLoop3Begin(ndim, loop_size,                               \
                               dbox1, start1, stride1, i1,                    \
                               dbox2, start2, stride2, i2,                    \
                               dbox3, start3, stride3, i3)                    \
{                                                                             \
   HYPRE_Int i1, i2, i3;                                                      \
   zypre_BoxLoopDeclare();                                                    \
   zypre_BoxLoopDeclareK(1);                                                  \
   zypre_BoxLoopDeclareK(2);                                                  \
   zypre_BoxLoopDeclareK(3);                                                  \
   zypre_BoxLoopInit(ndim, loop_size);                                        \
   zypre_BoxLoopInitK(1, dbox1, start1, stride1, i1);                         \
   zypre_BoxLoopInitK(2, dbox2, start2, stride2, i2);                         \
   zypre_BoxLoopInitK(3, dbox3, start3, stride3, i3);                         \
   OMP1                                                                       \
   for (hypre__block = 0; hypre__block < hypre__num_blocks; hypre__block++)   \
   {                                                                          \
      HYPRE_Int i1, i2, i3;                                                   \
      zypre_BoxLoopSet();                                                     \
      zypre_BoxLoopSetK(1, i1);                                               \
      zypre_BoxLoopSetK(2, i2);                                               \
      zypre_BoxLoopSetK(3, i3);                                               \
      for (hypre__J = 0; hypre__J < hypre__JN; hypre__J++)                    \
      {                                                                       \
         for (hypre__I = 0; hypre__I < hypre__IN; hypre__I++)                 \
         {

#define zypre_newBoxLoop3End(i1, i2, i3)                                      \
            i1 += hypre__i0inc1;                                              \
            i2 += hypre__i0inc2;                                              \
            i3 += hypre__i0inc3;                                              \
         }                                                                    \
         zypre_BoxLoopInc1();                                                 \
         i1 += hypre__ikinc1[hypre__d];                                       \
         i2 += hypre__ikinc2[hypre__d];                                       \
         i3 += hypre__ikinc3[hypre__d];                                       \
         zypre_BoxLoopInc2();                                                 \
      }                                                                       \
   }                                                                          \
}

#define zypre_newBoxLoop4Begin(ndim, loop_size,                               \
                            dbox1, start1, stride1, i1,                       \
                            dbox2, start2, stride2, i2,                       \
                            dbox3, start3, stride3, i3,                       \
                            dbox4, start4, stride4, i4)                       \
{                                                                             \
   HYPRE_Int i1, i2, i3, i4;                                                  \
   zypre_BoxLoopDeclare();                                                    \
   zypre_BoxLoopDeclareK(1);                                                  \
   zypre_BoxLoopDeclareK(2);                                                  \
   zypre_BoxLoopDeclareK(3);                                                  \
   zypre_BoxLoopDeclareK(4);                                                  \
   zypre_BoxLoopInit(ndim, loop_size);                                        \
   zypre_BoxLoopInitK(1, dbox1, start1, stride1, i1);                         \
   zypre_BoxLoopInitK(2, dbox2, start2, stride2, i2);                         \
   zypre_BoxLoopInitK(3, dbox3, start3, stride3, i3);                         \
   zypre_BoxLoopInitK(4, dbox4, start4, stride4, i4);                         \
   OMP1                                                                       \
   for (hypre__block = 0; hypre__block < hypre__num_blocks; hypre__block++)   \
   {                                                                          \
      HYPRE_Int i1, i2, i3, i4;                                               \
      zypre_BoxLoopSet();                                                     \
      zypre_BoxLoopSetK(1, i1);                                               \
      zypre_BoxLoopSetK(2, i2);                                               \
      zypre_BoxLoopSetK(3, i3);                                               \
      zypre_BoxLoopSetK(4, i4);                                               \
      for (hypre__J = 0; hypre__J < hypre__JN; hypre__J++)                    \
      {                                                                       \
         for (hypre__I = 0; hypre__I < hypre__IN; hypre__I++)                 \
         {

#define zypre_newBoxLoop4End(i1, i2, i3, i4)                                  \
            i1 += hypre__i0inc1;                                              \
            i2 += hypre__i0inc2;                                              \
            i3 += hypre__i0inc3;                                              \
            i4 += hypre__i0inc4;                                              \
         }                                                                    \
         zypre_BoxLoopInc1();                                                 \
         i1 += hypre__ikinc1[hypre__d];                                       \
         i2 += hypre__ikinc2[hypre__d];                                       \
         i3 += hypre__ikinc3[hypre__d];                                       \
         i4 += hypre__ikinc4[hypre__d];                                       \
         zypre_BoxLoopInc2();                                                 \
      }                                                                       \
   }                                                                          \
}

#define zypre_newBasicBoxLoop2Begin(ndim, loop_size,                          \
                                    stride1, i1,                              \
                                    stride2, i2)                              \
{                                                                             \
   zypre_BoxLoopDeclare();                                                    \
   zypre_BoxLoopDeclareK(1);                                                  \
   zypre_BoxLoopDeclareK(2);                                                  \
   zypre_BoxLoopInit(ndim, loop_size);                                        \
   zypre_BasicBoxLoopInitK(1, stride1);                                       \
   zypre_BasicBoxLoopInitK(2, stride2);                                       \
   OMP1                                                                       \
   for (hypre__block = 0; hypre__block < hypre__num_blocks; hypre__block++)   \
   {                                                                          \
      HYPRE_Int i1, i2;                                                       \
      zypre_BoxLoopSet();                                                     \
      zypre_BoxLoopSetK(1, i1);                                               \
      zypre_BoxLoopSetK(2, i2);                                               \
      for (hypre__J = 0; hypre__J < hypre__JN; hypre__J++)                    \
      {                                                                       \
         for (hypre__I = 0; hypre__I < hypre__IN; hypre__I++)                 \
         {


#define hypre_LoopBegin(size, idx)                                            \
{                                                                             \
   HYPRE_Int idx;                                                             \
   OMP0                                                                       \
   for (idx = 0; idx < size; idx ++)                                          \
   {

#define hypre_LoopEnd()                                                       \
  }                                                                           \
}

#define hypre_BoxLoopGetIndex    zypre_BoxLoopGetIndex

#define hypre_BoxLoopBlock       zypre_BoxLoopBlock
#define hypre_BoxLoop0Begin      zypre_newBoxLoop0Begin
#define hypre_BoxLoop0End        zypre_newBoxLoop0End
/* WM: replacing boxloops one at a time starting with boxloop1 */
#define hypre_BoxLoop1Begin      hypre_newBoxLoop1Begin
#define hypre_BoxLoop1End        hypre_newBoxLoop1End
/* #define hypre_BoxLoop1Begin      zypre_newBoxLoop1Begin */
/* #define hypre_BoxLoop1End        zypre_newBoxLoop1End */
#define hypre_BoxLoop2Begin      zypre_newBoxLoop2Begin
#define hypre_BoxLoop2End        zypre_newBoxLoop2End
#define hypre_BoxLoop3Begin      zypre_newBoxLoop3Begin
#define hypre_BoxLoop3End        zypre_newBoxLoop3End
#define hypre_BoxLoop4Begin      zypre_newBoxLoop4Begin
#define hypre_BoxLoop4End        zypre_newBoxLoop4End
#define hypre_BasicBoxLoop2Begin zypre_newBasicBoxLoop2Begin

/* Reduction */
/* WM: todo */
#define hypre_BoxLoop1ReductionBegin(ndim, loop_size, dbox1, start1, stride1, i1, reducesum) \
        zypre_newBoxLoop1Begin(ndim, loop_size, dbox1, start1, stride1, i1)

#define hypre_BoxLoop1ReductionEnd(i1, reducesum) \
        zypre_newBoxLoop1End(i1)

#define hypre_BoxLoop2ReductionBegin(ndim, loop_size, dbox1, start1, stride1, i1, \
                                                      dbox2, start2, stride2, i2, reducesum) \
        hypre_BoxLoop2Begin(ndim, loop_size, dbox1, start1, stride1, i1, \
                                             dbox2, start2, stride2, i2)

#define hypre_BoxLoop2ReductionEnd(i1, i2, reducesum) \
        hypre_BoxLoop2End(i1, i2)

#endif
