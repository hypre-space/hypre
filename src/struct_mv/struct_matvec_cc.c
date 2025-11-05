/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_struct_mv.h"
#include "_hypre_struct_mv.hpp"
#include "struct_matvec_core.h"

/*--------------------------------------------------------------------------
 * z = beta * y + alpha * A*x
 *
 * StructMatrix/Vector multiplication core function for constant coeficients.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatvecCompute_core_CC( HYPRE_Complex       alpha,
                                   hypre_StructMatrix *A,
                                   hypre_StructVector *x,
                                   HYPRE_Complex       beta,
                                   hypre_StructVector *y,
                                   hypre_StructVector *z,
                                   HYPRE_Int           Ab,
                                   HYPRE_Int           xb,
                                   HYPRE_Int           yb,
                                   HYPRE_Int           zb,
                                   HYPRE_Int           transpose,
                                   HYPRE_Int           nentries,
                                   HYPRE_Int          *entries,
                                   hypre_IndexRef      start,
                                   hypre_IndexRef      stride,
                                   hypre_IndexRef      loop_size,
                                   hypre_IndexRef      xfstride,
                                   hypre_IndexRef      ran_stride,
                                   hypre_IndexRef      xdstride,
                                   hypre_IndexRef      ydstride,
                                   hypre_IndexRef      zdstride,
                                   hypre_Box          *x_data_box,
                                   hypre_Box          *y_data_box,
                                   hypre_Box          *z_data_box)
{
#define DEVICE_VAR is_device_ptr(yp,xp,Ap0,Ap1,Ap2,Ap3,Ap4,Ap5,Ap6,Ap7,Ap8,Ap9,Ap10,Ap11,Ap12,Ap13,Ap14,Ap15,Ap16,Ap17,Ap18,Ap19,Ap20,Ap21,Ap22,Ap23,Ap24,Ap25,Ap26)
   HYPRE_Int             ndim = hypre_StructMatrixNDim(A);
   hypre_StructStencil  *stencil = hypre_StructMatrixStencil(A);
   hypre_Index          *stencil_shape = hypre_StructStencilShape(stencil);

   hypre_Index           xdstart, ydstart, zdstart;
   hypre_Index           offset;
   HYPRE_Int             si = 0, depth;
   HYPRE_Complex        *xp, *yp, *zp;
   HYPRE_DECLARE_OFFSETS_UP_TO_26;
   HYPRE_DECLARE_OFFSETS(26);

   /* Exit early if no stencil entries fall in this category */
   if (!nentries)
   {
      return hypre_error_flag;
   }

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("CC");

   xp = hypre_StructVectorBoxData(x, xb);
   yp = hypre_StructVectorBoxData(y, yb);
   zp = hypre_StructVectorBoxData(z, zb);

   hypre_CopyToIndex(start, ndim, xdstart);

   /* The next line is only used to avoid 'print error' messages.  It ensures
    * that xdstart is aligned with the vector data space before mapping.  The
    * choice, Neg vs Pos, doesn't matter because an offset will be used to index
    * into the vector x (xoff = index - xdstart). */
   hypre_SnapIndexNeg(xdstart, NULL, stride, ndim);
   hypre_MapToFineIndex(xdstart, NULL, xfstride, ndim);
   hypre_StructVectorMapDataIndex(x, xdstart);
   hypre_CopyToIndex(start, ndim, ydstart);
   hypre_MapToCoarseIndex(ydstart, NULL, ran_stride, ndim);
   hypre_CopyToIndex(start, ndim, zdstart);
   hypre_MapToCoarseIndex(zdstart, NULL, ran_stride, ndim);

   /* Initialize output vector (z = beta * y + alpha * A*x) with a first pass */
   depth = hypre_min(HYPRE_UNROLL_MAXDEPTH, nentries);
   hypre_StructMatvecCompute_core_ICC(A, x, Ab, depth, alpha, beta, xp, yp, zp,
                                      ndim, transpose, nentries, entries,
                                      stencil_shape, loop_size, xfstride,
                                      start, xdstart, ydstart, zdstart,
                                      xdstride, ydstride, zdstride,
                                      x_data_box, y_data_box, z_data_box);

   /* Update output vector with remaining A*x components if any */
#ifdef HYPRE_CORE_CASE
#undef HYPRE_CORE_CASE
#endif
#define HYPRE_CORE_CASE(n)                                                     \
   case n:                                                                     \
      HYPRE_LOAD_CAX_UP_TO_##n(transpose);                                     \
      hypre_BoxLoop2Begin(ndim, loop_size,                                     \
                          x_data_box, xdstart, xdstride, xi,                   \
                          z_data_box, zdstart, zdstride, zi);                  \
      {                                                                        \
         zp[zi] += alpha * (HYPRE_CALC_CAX_ADD_UP_TO_##n);                     \
      }                                                                        \
      hypre_BoxLoop2End(xi, zi);                                               \
      break;

   for (si = depth; si < nentries; si += HYPRE_UNROLL_MAXDEPTH)
   {
      depth = hypre_min(HYPRE_UNROLL_MAXDEPTH, (nentries - si));

      switch (depth)
      {
            HYPRE_CORE_CASE(27);
            HYPRE_CORE_CASE(26);
            HYPRE_CORE_CASE(25);
            HYPRE_CORE_CASE(24);
            HYPRE_CORE_CASE(23);
            HYPRE_CORE_CASE(22);
            HYPRE_CORE_CASE(21);
            HYPRE_CORE_CASE(20);
            HYPRE_CORE_CASE(19);
            HYPRE_CORE_CASE(18);
            HYPRE_CORE_CASE(17);
            HYPRE_CORE_CASE(16);
            HYPRE_CORE_CASE(15);
            HYPRE_CORE_CASE(14);
            HYPRE_CORE_CASE(13);
            HYPRE_CORE_CASE(12);
            HYPRE_CORE_CASE(11);
            HYPRE_CORE_CASE(10);
            HYPRE_CORE_CASE(9);
            HYPRE_CORE_CASE(8);
            HYPRE_CORE_CASE(7);
            HYPRE_CORE_CASE(6);
            HYPRE_CORE_CASE(5);
            HYPRE_CORE_CASE(4);
            HYPRE_CORE_CASE(3);
            HYPRE_CORE_CASE(2);
            HYPRE_CORE_CASE(1);

         case 0:
            break;
      } /* switch (depth) */
   } /* for si */
#undef DEVICE_VAR

   hypre_GpuProfilingPopRange();
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}
