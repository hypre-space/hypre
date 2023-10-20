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

#ifndef HYPRE_BOXLOOP_HOST_HEADER
#define HYPRE_BOXLOOP_HOST_HEADER

#if defined(HYPRE_USING_OPENMP)
#define HYPRE_BOX_REDUCTION
#define HYPRE_OMP_CLAUSE
#if defined(WIN32) && defined(_MSC_VER)
#define Pragma(x) __pragma(x)
#else
#define Pragma(x) _Pragma(HYPRE_XSTR(x))
#endif
#define OMP0 Pragma(omp parallel for HYPRE_OMP_CLAUSE HYPRE_BOX_REDUCTION HYPRE_SMP_SCHEDULE)
#define OMP1 Pragma(omp parallel for private(HYPRE_BOX_PRIVATE) HYPRE_OMP_CLAUSE HYPRE_BOX_REDUCTION HYPRE_SMP_SCHEDULE)
#else /* #if defined(HYPRE_USING_OPENMP) */
#define OMP0
#define OMP1
#endif /* #if defined(HYPRE_USING_OPENMP) */

#define zypre_BoxLoop0Begin(ndim, loop_size)                                  \
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

#define zypre_BoxLoop0End()                                                   \
         }                                                                    \
         zypre_BoxLoopInc1();                                                 \
         zypre_BoxLoopInc2();                                                 \
      }                                                                       \
   }                                                                          \
}

#define zypre_BoxLoop1Begin(ndim, loop_size,                                  \
                            dbox1, start1, stride1, i1)                       \
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

#define zypre_BoxLoop1End(i1)                                                 \
            i1 += hypre__i0inc1;                                              \
         }                                                                    \
         zypre_BoxLoopInc1();                                                 \
         i1 += hypre__ikinc1[hypre__d];                                       \
         zypre_BoxLoopInc2();                                                 \
      }                                                                       \
   }                                                                          \
}


#define zypre_BoxLoop2Begin(ndim, loop_size,                                  \
                            dbox1, start1, stride1, i1,                       \
                            dbox2, start2, stride2, i2)                       \
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

#define zypre_BoxLoop2End(i1, i2)                                             \
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


#define zypre_BoxLoop3Begin(ndim, loop_size,                                  \
                            dbox1, start1, stride1, i1,                       \
                            dbox2, start2, stride2, i2,                       \
                            dbox3, start3, stride3, i3)                       \
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

#define zypre_BoxLoop3End(i1, i2, i3)                                         \
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

#define zypre_BoxLoop4Begin(ndim, loop_size,                                  \
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

#define zypre_BoxLoop4End(i1, i2, i3, i4)                                     \
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

#define zypre_BasicBoxLoop1Begin(ndim, loop_size,                             \
                                 stride1, i1)                                 \
{                                                                             \
   zypre_BoxLoopDeclare();                                                    \
   zypre_BoxLoopDeclareK(1);                                                  \
   zypre_BoxLoopInit(ndim, loop_size);                                        \
   zypre_BasicBoxLoopInitK(1, stride1);                                       \
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

#define zypre_BasicBoxLoop2Begin(ndim, loop_size,                             \
                                 stride1, i1,                                 \
                                 stride2, i2)                                 \
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


#define zypre_LoopBegin(size, idx)                                            \
{                                                                             \
   HYPRE_Int idx;                                                             \
   OMP0                                                                       \
   for (idx = 0; idx < size; idx ++)                                          \
   {

#define zypre_LoopEnd()                                                       \
   }                                                                          \
}

/*--------------------------------------------------------------------------
 * Serial BoxLoop macros:
 * [same as the ones above (without OMP and with SetOneBlock)]
 * TODO: combine them
 *--------------------------------------------------------------------------*/
#define hypre_SerialBoxLoop0Begin(ndim, loop_size)                            \
{                                                                             \
   zypre_BoxLoopDeclare();                                                    \
   zypre_BoxLoopInit(ndim, loop_size);                                        \
   zypre_BoxLoopSetOneBlock();                                                \
   for (hypre__block = 0; hypre__block < hypre__num_blocks; hypre__block++)   \
   {                                                                          \
      zypre_BoxLoopSet();                                                     \
      for (hypre__J = 0; hypre__J < hypre__JN; hypre__J++)                    \
      {                                                                       \
         for (hypre__I = 0; hypre__I < hypre__IN; hypre__I++)                 \
         {

#define hypre_SerialBoxLoop0End()                                             \
         }                                                                    \
         zypre_BoxLoopInc1();                                                 \
         zypre_BoxLoopInc2();                                                 \
      }                                                                       \
   }                                                                          \
}

#define hypre_SerialBoxLoop1Begin(ndim, loop_size,                            \
                                  dbox1, start1, stride1, i1)                 \
{                                                                             \
   HYPRE_Int i1;                                                              \
   zypre_BoxLoopDeclare();                                                    \
   zypre_BoxLoopDeclareK(1);                                                  \
   zypre_BoxLoopInit(ndim, loop_size);                                        \
   zypre_BoxLoopInitK(1, dbox1, start1, stride1, i1);                         \
   zypre_BoxLoopSetOneBlock();                                                \
   for (hypre__block = 0; hypre__block < hypre__num_blocks; hypre__block++)   \
   {                                                                          \
      zypre_BoxLoopSet();                                                     \
      zypre_BoxLoopSetK(1, i1);                                               \
      for (hypre__J = 0; hypre__J < hypre__JN; hypre__J++)                    \
      {                                                                       \
         for (hypre__I = 0; hypre__I < hypre__IN; hypre__I++)                 \
         {

#define hypre_SerialBoxLoop1End(i1)  zypre_BoxLoop1End(i1)

#define hypre_SerialBoxLoop2Begin(ndim, loop_size,                            \
                                  dbox1, start1, stride1, i1,                 \
                                  dbox2, start2, stride2, i2)                 \
{                                                                             \
   HYPRE_Int i1,i2;                                                           \
   zypre_BoxLoopDeclare();                                                    \
   zypre_BoxLoopDeclareK(1);                                                  \
   zypre_BoxLoopDeclareK(2);                                                  \
   zypre_BoxLoopInit(ndim, loop_size);                                        \
   zypre_BoxLoopInitK(1, dbox1, start1, stride1, i1);                         \
   zypre_BoxLoopInitK(2, dbox2, start2, stride2, i2);                         \
   zypre_BoxLoopSetOneBlock();                                                \
   for (hypre__block = 0; hypre__block < hypre__num_blocks; hypre__block++)   \
   {                                                                          \
      zypre_BoxLoopSet();                                                     \
      zypre_BoxLoopSetK(1, i1);                                               \
      zypre_BoxLoopSetK(2, i2);                                               \
      for (hypre__J = 0; hypre__J < hypre__JN; hypre__J++)                    \
      {                                                                       \
         for (hypre__I = 0; hypre__I < hypre__IN; hypre__I++)                 \
         {

#define hypre_SerialBoxLoop2End(i1, i2) zypre_BoxLoop2End(i1, i2)

/* Reduction BoxLoop1 */
#define zypre_BoxLoop1ReductionBegin(ndim, loop_size, dbox1, start1, stride1, i1, reducesum) \
        zypre_BoxLoop1Begin(ndim, loop_size, dbox1, start1, stride1, i1)

#define zypre_BoxLoop1ReductionEnd(i1, reducesum) zypre_BoxLoop1End(i1)

/* Reduction BoxLoop2 */
#define zypre_BoxLoop2ReductionBegin(ndim, loop_size, dbox1, start1, stride1, i1,            \
                                                      dbox2, start2, stride2, i2, reducesum) \
        zypre_BoxLoop2Begin(ndim, loop_size, dbox1, start1, stride1, i1,                     \
                                             dbox2, start2, stride2, i2)

#define zypre_BoxLoop2ReductionEnd(i1, i2, reducesum) zypre_BoxLoop2End(i1, i2)


/* Renaming */
#define hypre_BoxLoopGetIndexHost          zypre_BoxLoopGetIndex
#define hypre_BoxLoopBlockHost             zypre_BoxLoopBlock
#define hypre_BoxLoop0BeginHost            zypre_BoxLoop0Begin
#define hypre_BoxLoop0EndHost              zypre_BoxLoop0End
#define hypre_BoxLoop1BeginHost            zypre_BoxLoop1Begin
#define hypre_BoxLoop1EndHost              zypre_BoxLoop1End
#define hypre_BoxLoop2BeginHost            zypre_BoxLoop2Begin
#define hypre_BoxLoop2EndHost              zypre_BoxLoop2End
#define hypre_BoxLoop3BeginHost            zypre_BoxLoop3Begin
#define hypre_BoxLoop3EndHost              zypre_BoxLoop3End
#define hypre_BoxLoop4BeginHost            zypre_BoxLoop4Begin
#define hypre_BoxLoop4EndHost              zypre_BoxLoop4End
#define hypre_BasicBoxLoop1BeginHost       zypre_BasicBoxLoop1Begin
#define hypre_BasicBoxLoop2BeginHost       zypre_BasicBoxLoop2Begin
#define hypre_LoopBeginHost                zypre_LoopBegin
#define hypre_LoopEndHost                  zypre_LoopEnd
#define hypre_BoxLoop1ReductionBeginHost   zypre_BoxLoop1ReductionBegin
#define hypre_BoxLoop1ReductionEndHost     zypre_BoxLoop1ReductionEnd
#define hypre_BoxLoop2ReductionBeginHost   zypre_BoxLoop2ReductionBegin
#define hypre_BoxLoop2ReductionEndHost     zypre_BoxLoop2ReductionEnd

//TODO TEMP FIX
#if !defined(HYPRE_USING_RAJA) && !defined(HYPRE_USING_KOKKOS) && !defined(HYPRE_USING_CUDA) && !defined(HYPRE_USING_HIP) && !defined(HYPRE_USING_DEVICE_OPENMP) && !defined(HYPRE_USING_SYCL)
#define hypre_BoxLoopGetIndex          hypre_BoxLoopGetIndexHost
#define hypre_BoxLoopBlock()           0
#define hypre_BoxLoop0Begin            hypre_BoxLoop0BeginHost
#define hypre_BoxLoop0End              hypre_BoxLoop0EndHost
#define hypre_BoxLoop1Begin            hypre_BoxLoop1BeginHost
#define hypre_BoxLoop1End              hypre_BoxLoop1EndHost
#define hypre_BoxLoop2Begin            hypre_BoxLoop2BeginHost
#define hypre_BoxLoop2End              hypre_BoxLoop2EndHost
#define hypre_BoxLoop3Begin            hypre_BoxLoop3BeginHost
#define hypre_BoxLoop3End              hypre_BoxLoop3EndHost
#define hypre_BoxLoop4Begin            hypre_BoxLoop4BeginHost
#define hypre_BoxLoop4End              hypre_BoxLoop4EndHost
#define hypre_BasicBoxLoop1Begin       hypre_BasicBoxLoop1BeginHost
#define hypre_BasicBoxLoop2Begin       hypre_BasicBoxLoop2BeginHost
#define hypre_LoopBegin                hypre_LoopBeginHost
#define hypre_LoopEnd                  hypre_LoopEndHost
#define hypre_BoxLoop1ReductionBegin   hypre_BoxLoop1ReductionBeginHost
#define hypre_BoxLoop1ReductionEnd     hypre_BoxLoop1ReductionEndHost
#define hypre_BoxLoop2ReductionBegin   hypre_BoxLoop2ReductionBeginHost
#define hypre_BoxLoop2ReductionEnd     hypre_BoxLoop2ReductionEndHost
#endif

#endif /* #ifndef HYPRE_BOXLOOP_HOST_HEADER */

