/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/*--------------------------------------------------------------------------
 * hypre_RedBlackGSData data structure
 *--------------------------------------------------------------------------*/

typedef struct
{
   MPI_Comm                comm;

   HYPRE_Real              tol;                /* not yet used */
   HYPRE_Int               max_iter;
   HYPRE_Int               rel_change;         /* not yet used */
   HYPRE_Int               zero_guess;
   HYPRE_Int               rb_start;

   hypre_StructMatrix     *A;
   hypre_StructVector     *b;
   hypre_StructVector     *x;

   HYPRE_Int               diag_rank;

   hypre_ComputePkg       *compute_pkg;

   /* log info (always logged) */
   HYPRE_Int               num_iterations;
   HYPRE_Int               time_index;
   HYPRE_Int               flops;

} hypre_RedBlackGSData;

#ifdef HYPRE_USING_RAJA

#define hypre_RedBlackLoopInit()
#define hypre_RedBlackLoopBegin(ni,nj,nk,redblack,     \
                                Astart,Ani,Anj,Ai,     \
                                bstart,bni,bnj,bi,     \
                                xstart,xni,xnj,xi)     \
{                                                      \
   HYPRE_Int hypre__tot = nk*nj*((ni+1)/2);            \
   forall< hypre_raja_exec_policy >(RangeSegment(0, hypre__tot), [=] hypre_RAJA_DEVICE (HYPRE_Int idx) \
   {                                                   \
      HYPRE_Int idx_local = idx;                       \
      HYPRE_Int ii,jj,kk,Ai,bi,xi;                     \
      HYPRE_Int local_ii;                              \
      kk = idx_local % nk;                             \
      idx_local = idx_local / nk;                      \
      jj = idx_local % nj;                             \
      idx_local = idx_local / nj;                      \
      local_ii = (kk + jj + redblack) % 2;             \
      ii = 2*idx_local + local_ii;                     \
      if (ii < ni)                                     \
      {                                                \
         Ai = Astart + kk*Anj*Ani + jj*Ani + ii;       \
         bi = bstart + kk*bnj*bni + jj*bni + ii;       \
         xi = xstart + kk*xnj*xni + jj*xni + ii;       \

#define hypre_RedBlackLoopEnd()                        \
      }                                                \
   });                                                 \
   hypre_fence();                                      \
}

#define hypre_RedBlackConstantcoefLoopBegin(ni,nj,nk,redblack, \
                                            bstart,bni,bnj,bi, \
                                            xstart,xni,xnj,xi) \
{                                                              \
   HYPRE_Int hypre__tot = nk*nj*((ni+1)/2);                    \
   forall< hypre_raja_exec_policy >(RangeSegment(0, hypre__tot), [=] hypre_RAJA_DEVICE (HYPRE_Int idx) \
   {                                                           \
      HYPRE_Int idx_local = idx;                               \
      HYPRE_Int ii,jj,kk,bi,xi;                                \
      HYPRE_Int local_ii;                                      \
      kk = idx_local % nk;                                     \
      idx_local = idx_local / nk;                              \
      jj = idx_local % nj;                                     \
      idx_local = idx_local / nj;                              \
      local_ii = (kk + jj + redblack) % 2;                     \
      ii = 2*idx_local + local_ii;                             \
      if (ii < ni)                                             \
      {                                                        \
          bi = bstart + kk*bnj*bni + jj*bni + ii;              \
          xi = xstart + kk*xnj*xni + jj*xni + ii;              \

#define hypre_RedBlackConstantcoefLoopEnd()                    \
      }                                                        \
   });                                                         \
   hypre_fence();                                              \
}

#elif defined(HYPRE_USING_KOKKOS)

#define hypre_RedBlackLoopInit()
#define hypre_RedBlackLoopBegin(ni,nj,nk,redblack,                  \
                                Astart,Ani,Anj,Ai,                  \
                                bstart,bni,bnj,bi,                  \
                                xstart,xni,xnj,xi)                  \
{                                                                   \
   HYPRE_Int hypre__tot = nk*nj*((ni+1)/2);                         \
   Kokkos::parallel_for (hypre__tot, KOKKOS_LAMBDA (HYPRE_Int idx)  \
   {                                                                \
      HYPRE_Int idx_local = idx;                                    \
      HYPRE_Int ii,jj,kk,Ai,bi,xi;                                  \
      HYPRE_Int local_ii;                                           \
      kk = idx_local % nk;                                          \
      idx_local = idx_local / nk;                                   \
      jj = idx_local % nj;                                          \
      idx_local = idx_local / nj;                                   \
      local_ii = (kk + jj + redblack) % 2;                          \
      ii = 2*idx_local + local_ii;                                  \
      if (ii < ni)                                                  \
      {                                                             \
         Ai = Astart + kk*Anj*Ani + jj*Ani + ii;                    \
         bi = bstart + kk*bnj*bni + jj*bni + ii;                    \
         xi = xstart + kk*xnj*xni + jj*xni + ii;                    \

#define hypre_RedBlackLoopEnd()                                     \
      }                                                             \
   });                                                              \
   hypre_fence();                                                   \
}

#define hypre_RedBlackConstantcoefLoopBegin(ni,nj,nk,redblack,      \
                                            bstart,bni,bnj,bi,      \
                                            xstart,xni,xnj,xi)      \
{                                                                   \
   HYPRE_Int hypre__tot = nk*nj*((ni+1)/2);                         \
   Kokkos::parallel_for (hypre__tot, KOKKOS_LAMBDA (HYPRE_Int idx)  \
   {                                                                \
      HYPRE_Int idx_local = idx;                                    \
      HYPRE_Int ii,jj,kk,bi,xi;                                     \
      HYPRE_Int local_ii;                                           \
      kk = idx_local % nk;                                          \
      idx_local = idx_local / nk;                                   \
      jj = idx_local % nj;                                          \
      idx_local = idx_local / nj;                                   \
      local_ii = (kk + jj + redblack) % 2;                          \
      ii = 2*idx_local + local_ii;                                  \
      if (ii < ni)                                                  \
      {                                                             \
         bi = bstart + kk*bnj*bni + jj*bni + ii;                    \
         xi = xstart + kk*xnj*xni + jj*xni + ii;                    \

#define hypre_RedBlackConstantcoefLoopEnd()                         \
      }                                                             \
   });                                                              \
   hypre_fence();                                                   \
}

#elif defined(HYPRE_USING_CUDA) || defined(HYPRE_USING_HIP)

#define hypre_RedBlackLoopInit()
#define hypre_RedBlackLoopBegin(ni,nj,nk,redblack,        \
                                Astart,Ani,Anj,Ai,        \
                                bstart,bni,bnj,bi,        \
                                xstart,xni,xnj,xi)        \
{                                                         \
   HYPRE_Int hypre__tot = nk*nj*((ni+1)/2);               \
   BoxLoopforall(hypre__tot, HYPRE_LAMBDA (HYPRE_Int idx) \
   {                                                      \
      HYPRE_Int idx_local = idx;                          \
      HYPRE_Int ii,jj,kk,Ai,bi,xi;                        \
      HYPRE_Int local_ii;                                 \
      kk = idx_local % nk;                                \
      idx_local = idx_local / nk;                         \
      jj = idx_local % nj;                                \
      idx_local = idx_local / nj;                         \
      local_ii = (kk + jj + redblack) % 2;                \
      ii = 2*idx_local + local_ii;                        \
      if (ii < ni)                                        \
      {                                                   \
         Ai = Astart + kk*Anj*Ani + jj*Ani + ii;          \
         bi = bstart + kk*bnj*bni + jj*bni + ii;          \
         xi = xstart + kk*xnj*xni + jj*xni + ii;          \

#define hypre_RedBlackLoopEnd()                           \
      }                                                   \
   });                                                    \
}

#define hypre_RedBlackConstantcoefLoopBegin(ni,nj,nk,redblack,      \
                                            bstart,bni,bnj,bi,      \
                                            xstart,xni,xnj,xi)      \
{                                                                   \
   HYPRE_Int hypre__tot = nk*nj*((ni+1)/2);                         \
   BoxLoopforall(hypre__tot, HYPRE_LAMBDA (HYPRE_Int idx)           \
   {                                                                \
      HYPRE_Int idx_local = idx;                                    \
      HYPRE_Int ii,jj,kk,bi,xi;                                     \
      HYPRE_Int local_ii;                                           \
      kk = idx_local % nk;                                          \
      idx_local = idx_local / nk;                                   \
      jj = idx_local % nj;                                          \
      idx_local = idx_local / nj;                                   \
      local_ii = (kk + jj + redblack) % 2;                          \
      ii = 2*idx_local + local_ii;                                  \
      if (ii < ni)                                                  \
      {                                                             \
         bi = bstart + kk*bnj*bni + jj*bni + ii;                    \
         xi = xstart + kk*xnj*xni + jj*xni + ii;                    \

#define hypre_RedBlackConstantcoefLoopEnd()                         \
      }                                                             \
   });                                                              \
}

#elif defined(HYPRE_USING_SYCL)

#define hypre_RedBlackLoopInit()
#define hypre_RedBlackLoopBegin(ni,nj,nk,redblack,                  \
                                Astart,Ani,Anj,Ai,                  \
                                bstart,bni,bnj,bi,                  \
                                xstart,xni,xnj,xi)                  \
{                                                                   \
   HYPRE_Int hypre__tot = nk*nj*((ni+1)/2);                         \
   BoxLoopforall(hypre__tot, [=] (sycl::nd_item<3> item)            \
   {                                                                \
      HYPRE_Int idx = (HYPRE_Int) item.get_global_linear_id();      \
      HYPRE_Int idx_local = idx;                                    \
      HYPRE_Int ii,jj,kk,Ai,bi,xi;                                  \
      HYPRE_Int local_ii;                                           \
      kk = idx_local % nk;                                          \
      idx_local = idx_local / nk;                                   \
      jj = idx_local % nj;                                          \
      idx_local = idx_local / nj;                                   \
      local_ii = (kk + jj + redblack) % 2;                          \
      ii = 2*idx_local + local_ii;                                  \
      if (ii < ni)                                                  \
      {                                                             \
         Ai = Astart + kk*Anj*Ani + jj*Ani + ii;                    \
         bi = bstart + kk*bnj*bni + jj*bni + ii;                    \
         xi = xstart + kk*xnj*xni + jj*xni + ii;                    \

#define hypre_RedBlackLoopEnd()                                     \
      }                                                             \
   });                                                              \
}

#define hypre_RedBlackConstantcoefLoopBegin(ni,nj,nk,redblack,      \
                                            bstart,bni,bnj,bi,      \
                                            xstart,xni,xnj,xi)      \
{                                                                   \
   HYPRE_Int hypre__tot = nk*nj*((ni+1)/2);                         \
   BoxLoopforall(hypre__tot, [=] (sycl::nd_item<3> item)            \
   {                                                                \
      HYPRE_Int idx = (HYPRE_Int) item.get_global_linear_id();      \
      HYPRE_Int idx_local = idx;                                    \
      HYPRE_Int ii,jj,kk,bi,xi;                                     \
      HYPRE_Int local_ii;                                           \
      kk = idx_local % nk;                                          \
      idx_local = idx_local / nk;                                   \
      jj = idx_local % nj;                                          \
      idx_local = idx_local / nj;                                   \
      local_ii = (kk + jj + redblack) % 2;                          \
      ii = 2*idx_local + local_ii;                                  \
      if (ii < ni)                                                  \
      {                                                             \
         bi = bstart + kk*bnj*bni + jj*bni + ii;                    \
         xi = xstart + kk*xnj*xni + jj*xni + ii;                    \

#define hypre_RedBlackConstantcoefLoopEnd()                         \
      }                                                             \
   });                                                              \
}

#elif defined(HYPRE_USING_DEVICE_OPENMP)

/* BEGIN OF OMP 4.5 */
/* #define IF_CLAUSE if (hypre__global_offload) */

/* stringification:
 * _Pragma(string-literal), so we need to cast argument to a string
 * The three dots as last argument of the macro tells compiler that this is a variadic macro.
 * I.e. this is a macro that receives variable number of arguments.
 */
//#define HYPRE_STR(s...) #s
//#define HYPRE_XSTR(s...) HYPRE_STR(s)

#define hypre_RedBlackLoopInit()

#define hypre_RedBlackLoopBegin(ni,nj,nk,redblack,                      \
                                Astart,Ani,Anj,Ai,                      \
                                bstart,bni,bnj,bi,                      \
                                xstart,xni,xnj,xi)                      \
{                                                                       \
   HYPRE_Int hypre__thread, hypre__tot = nk*nj*((ni+1)/2);              \
   HYPRE_BOXLOOP_ENTRY_PRINT                                            \
   /* device code: */                                                   \
   _Pragma (HYPRE_XSTR(omp target teams distribute parallel for IF_CLAUSE IS_DEVICE_CLAUSE)) \
   for (hypre__thread=0; hypre__thread<hypre__tot; hypre__thread++)     \
   {                                                                    \
        HYPRE_Int idx_local = hypre__thread;                            \
        HYPRE_Int ii,jj,kk,Ai,bi,xi;                                    \
        HYPRE_Int local_ii;                                             \
        kk = idx_local % nk;                                            \
        idx_local = idx_local / nk;                                     \
        jj = idx_local % nj;                                            \
        idx_local = idx_local / nj;                                     \
        local_ii = (kk + jj + redblack) % 2;                            \
        ii = 2*idx_local + local_ii;                                    \
        if (ii < ni)                                                    \
        {                                                               \
            Ai = Astart + kk*Anj*Ani + jj*Ani + ii;                     \
            bi = bstart + kk*bnj*bni + jj*bni + ii;                     \
            xi = xstart + kk*xnj*xni + jj*xni + ii;                     \

#define hypre_RedBlackLoopEnd()                                         \
        }                                                               \
     }                                                                  \
}



#define hypre_RedBlackConstantcoefLoopBegin(ni,nj,nk,redblack,        \
                                            bstart,bni,bnj,bi,        \
                                            xstart,xni,xnj,xi)        \
{                                                                     \
   HYPRE_Int hypre__thread, hypre__tot = nk*nj*((ni+1)/2);            \
   HYPRE_BOXLOOP_ENTRY_PRINT                                          \
   /* device code: */                                                 \
   _Pragma (HYPRE_XSTR(omp target teams distribute parallel for IF_CLAUSE IS_DEVICE_CLAUSE)) \
   for (hypre__thread=0; hypre__thread<hypre__tot; hypre__thread++)   \
   {                                                                  \
        HYPRE_Int idx_local = hypre__thread;                          \
        HYPRE_Int ii,jj,kk,bi,xi;                                     \
        HYPRE_Int local_ii;                                           \
        kk = idx_local % nk;                                          \
        idx_local = idx_local / nk;                                   \
        jj = idx_local % nj;                                          \
        idx_local = idx_local / nj;                                   \
        local_ii = (kk + jj + redblack) % 2;                          \
        ii = 2*idx_local + local_ii;                                  \
        if (ii < ni)                                                  \
        {                                                             \
            bi = bstart + kk*bnj*bni + jj*bni + ii;                   \
            xi = xstart + kk*xnj*xni + jj*xni + ii;                   \

#define hypre_RedBlackConstantcoefLoopEnd()                           \
         }                                                            \
     }                                                                \
}
/* END OF OMP 4.5 */

#else

/* CPU */
#define HYPRE_REDBLACK_PRIVATE hypre__kk

#define hypre_RedBlackLoopInit()\
{\
   HYPRE_Int hypre__kk;

#ifdef HYPRE_USING_OPENMP
#define HYPRE_BOX_REDUCTION
#ifndef Pragma
#if defined(WIN32) && defined(_MSC_VER)
#define Pragma(x) __pragma(x)
#else
#define Pragma(x) _Pragma(HYPRE_XSTR(x))
#endif
#endif
#define OMPRB1 Pragma(omp parallel for private(HYPRE_REDBLACK_PRIVATE) HYPRE_BOX_REDUCTION HYPRE_SMP_SCHEDULE)
#else
#define OMPRB1
#endif

#define hypre_RedBlackLoopBegin(ni,nj,nk,redblack,  \
                                Astart,Ani,Anj,Ai,  \
                                bstart,bni,bnj,bi,  \
                                xstart,xni,xnj,xi)  \
   OMPRB1 \
   for (hypre__kk = 0; hypre__kk < nk; hypre__kk++) \
   {\
      HYPRE_Int ii,jj,Ai,bi,xi;\
      for (jj = 0; jj < nj; jj++)\
      {\
         ii = (hypre__kk + jj + redblack) % 2;\
         Ai = Astart + hypre__kk*Anj*Ani + jj*Ani + ii; \
         bi = bstart + hypre__kk*bnj*bni + jj*bni + ii; \
         xi = xstart + hypre__kk*xnj*xni + jj*xni + ii; \
         for (; ii < ni; ii+=2, Ai+=2, bi+=2, xi+=2)\
         {

#define hypre_RedBlackLoopEnd()\
         }\
      }\
   }\
}

#define hypre_RedBlackConstantcoefLoopBegin(ni,nj,nk,redblack, \
                                            bstart,bni,bnj,bi, \
                                            xstart,xni,xnj,xi) \
   OMPRB1 \
   for (hypre__kk = 0; hypre__kk < nk; hypre__kk++)\
   {\
      HYPRE_Int ii,jj,bi,xi;\
      for (jj = 0; jj < nj; jj++)\
      {\
         ii = (hypre__kk + jj + redblack) % 2;\
         bi = bstart + hypre__kk*bnj*bni + jj*bni + ii;\
         xi = xstart + hypre__kk*xnj*xni + jj*xni + ii;\
         for (; ii < ni; ii+=2, Ai+=2, bi+=2, xi+=2)\
         {

#define hypre_RedBlackConstantcoefLoopEnd()\
         }\
      }\
   }\
}
#endif
