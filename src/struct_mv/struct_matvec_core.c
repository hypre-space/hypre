/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Structured matrix-vector multiply routine
 *
 *****************************************************************************/

#include "_hypre_struct_mv.h"
#include "_hypre_struct_mv.hpp"

#define UNROLL_MAXDEPTH 27

/*--------------------------------------------------------------------------
 * Macros used in the kernel loops below
 *--------------------------------------------------------------------------*/

#define HYPRE_MAP_A_OFFSET(offset)                       \
   hypre_StructMatrixMapDataIndex(A, offset);            \
   hypre_SubtractIndexes(offset, Adstart, ndim, offset);

#define HYPRE_MAP_X_OFFSET(offset)                       \
   hypre_MapToFineIndex(offset, NULL, xfstride, ndim);   \
   hypre_StructVectorMapDataIndex(x, offset);            \
   hypre_SubtractIndexes(offset, xdstart, ndim, offset);

#define HYPRE_SET_CAX(Ap, xoff, entry)                               \
   Ap = hypre_StructMatrixBoxData(A, Ab, entry);                     \
   hypre_AddIndexes(start, stencil_shape[entry], ndim, offset);      \
   HYPRE_MAP_X_OFFSET(offset);                                       \
   xoff = hypre_BoxOffsetDistance(x_data_box, offset);

#define HYPRE_SET_CAX_TRANS(Ap, xoff, entry)                         \
   Ap = hypre_StructMatrixBoxData(A, Ab, entry);                     \
   hypre_SubtractIndexes(start, stencil_shape[entry], ndim, offset); \
   HYPRE_MAP_X_OFFSET(offset);                                       \
   xoff = hypre_BoxOffsetDistance(x_data_box, offset);

#define HYPRE_SET_AX(Ap, xoff, entry)                                \
   HYPRE_SET_CAX(Ap, xoff, entry)

#define HYPRE_SET_AX_TRANS(Ap, xoff, entry)                          \
   HYPRE_SET_CAX_TRANS(Ap, xoff, entry)                              \
   hypre_SubtractIndexes(start, stencil_shape[entry], ndim, offset); \
   HYPRE_MAP_A_OFFSET(offset);                                       \
   Ap += hypre_BoxOffsetDistance(A_data_box, offset);

/* Individual */
#define HYPRE_CALC_AX(n)        \
   Ap##n[Ai] * xp[xi + xoff##n]

#define HYPRE_CALC_AX_ADD(n)    \
   Ap##n[Ai] * xp[xi + xoff##n] +

#define HYPRE_CALC_CAX(n)        \
   Ap##n[0] * xp[xi + xoff##n]

#define HYPRE_CALC_CAX_ADD(n)    \
   Ap##n[0] * xp[xi + xoff##n] +

/* Sequence macros for various matrix/vector multiplication components */
#define HYPRE_CALC_AX_ADD_UP_TO_1 \
   HYPRE_CALC_AX(0) \

#define HYPRE_CALC_AX_ADD_UP_TO_2 \
   HYPRE_CALC_AX_ADD(1) \
   HYPRE_CALC_AX_ADD_UP_TO_1

#define HYPRE_CALC_AX_ADD_UP_TO_3 \
   HYPRE_CALC_AX_ADD(2) \
   HYPRE_CALC_AX_ADD_UP_TO_2

#define HYPRE_CALC_AX_ADD_UP_TO_4 \
   HYPRE_CALC_AX_ADD(3) \
   HYPRE_CALC_AX_ADD_UP_TO_3

#define HYPRE_CALC_AX_ADD_UP_TO_5 \
   HYPRE_CALC_AX_ADD(4) \
   HYPRE_CALC_AX_ADD_UP_TO_4

#define HYPRE_CALC_AX_ADD_UP_TO_6 \
   HYPRE_CALC_AX_ADD(5) \
   HYPRE_CALC_AX_ADD_UP_TO_5

#define HYPRE_CALC_AX_ADD_UP_TO_7 \
   HYPRE_CALC_AX_ADD(6) \
   HYPRE_CALC_AX_ADD_UP_TO_6

#define HYPRE_CALC_AX_ADD_UP_TO_8 \
   HYPRE_CALC_AX_ADD(7) \
   HYPRE_CALC_AX_ADD_UP_TO_7

#define HYPRE_CALC_AX_ADD_UP_TO_9 \
   HYPRE_CALC_AX_ADD(8) \
   HYPRE_CALC_AX_ADD_UP_TO_8

#define HYPRE_CALC_AX_ADD_UP_TO_10 \
   HYPRE_CALC_AX_ADD(9) \
   HYPRE_CALC_AX_ADD_UP_TO_9

#define HYPRE_CALC_AX_ADD_UP_TO_11 \
   HYPRE_CALC_AX_ADD(10) \
   HYPRE_CALC_AX_ADD_UP_TO_10

#define HYPRE_CALC_AX_ADD_UP_TO_12 \
   HYPRE_CALC_AX_ADD(11) \
   HYPRE_CALC_AX_ADD_UP_TO_11

#define HYPRE_CALC_AX_ADD_UP_TO_13 \
   HYPRE_CALC_AX_ADD(12) \
   HYPRE_CALC_AX_ADD_UP_TO_12

#define HYPRE_CALC_AX_ADD_UP_TO_14 \
   HYPRE_CALC_AX_ADD(13) \
   HYPRE_CALC_AX_ADD_UP_TO_13

#define HYPRE_CALC_AX_ADD_UP_TO_15 \
   HYPRE_CALC_AX_ADD(14) \
   HYPRE_CALC_AX_ADD_UP_TO_14

#define HYPRE_CALC_AX_ADD_UP_TO_16 \
   HYPRE_CALC_AX_ADD(15) \
   HYPRE_CALC_AX_ADD_UP_TO_15

#define HYPRE_CALC_AX_ADD_UP_TO_17 \
   HYPRE_CALC_AX_ADD(16) \
   HYPRE_CALC_AX_ADD_UP_TO_16

#define HYPRE_CALC_AX_ADD_UP_TO_18 \
   HYPRE_CALC_AX_ADD(17) \
   HYPRE_CALC_AX_ADD_UP_TO_17

#define HYPRE_CALC_AX_ADD_UP_TO_19 \
   HYPRE_CALC_AX_ADD(18) \
   HYPRE_CALC_AX_ADD_UP_TO_18

#define HYPRE_CALC_AX_ADD_UP_TO_20 \
   HYPRE_CALC_AX_ADD(19) \
   HYPRE_CALC_AX_ADD_UP_TO_19

#define HYPRE_CALC_AX_ADD_UP_TO_21 \
   HYPRE_CALC_AX_ADD(20) \
   HYPRE_CALC_AX_ADD_UP_TO_20

#define HYPRE_CALC_AX_ADD_UP_TO_22 \
   HYPRE_CALC_AX_ADD(21) \
   HYPRE_CALC_AX_ADD_UP_TO_21

#define HYPRE_CALC_AX_ADD_UP_TO_23 \
   HYPRE_CALC_AX_ADD(22) \
   HYPRE_CALC_AX_ADD_UP_TO_22

#define HYPRE_CALC_AX_ADD_UP_TO_24 \
   HYPRE_CALC_AX_ADD(23) \
   HYPRE_CALC_AX_ADD_UP_TO_23

#define HYPRE_CALC_AX_ADD_UP_TO_25 \
   HYPRE_CALC_AX_ADD(24) \
   HYPRE_CALC_AX_ADD_UP_TO_24

#define HYPRE_CALC_AX_ADD_UP_TO_26 \
   HYPRE_CALC_AX_ADD(25) \
   HYPRE_CALC_AX_ADD_UP_TO_25

#define HYPRE_CALC_AX_ADD_UP_TO_27 \
   HYPRE_CALC_AX_ADD(26) \
   HYPRE_CALC_AX_ADD_UP_TO_26

#define HYPRE_CALC_CAX_ADD_UP_TO_1 \
   HYPRE_CALC_CAX(0) \

#define HYPRE_CALC_CAX_ADD_UP_TO_2 \
   HYPRE_CALC_CAX_ADD(1) \
   HYPRE_CALC_CAX_ADD_UP_TO_1

#define HYPRE_CALC_CAX_ADD_UP_TO_3 \
   HYPRE_CALC_CAX_ADD(2) \
   HYPRE_CALC_CAX_ADD_UP_TO_2

#define HYPRE_CALC_CAX_ADD_UP_TO_4 \
   HYPRE_CALC_CAX_ADD(3) \
   HYPRE_CALC_CAX_ADD_UP_TO_3

#define HYPRE_CALC_CAX_ADD_UP_TO_5 \
   HYPRE_CALC_CAX_ADD(4) \
   HYPRE_CALC_CAX_ADD_UP_TO_4

#define HYPRE_CALC_CAX_ADD_UP_TO_6 \
   HYPRE_CALC_CAX_ADD(5) \
   HYPRE_CALC_CAX_ADD_UP_TO_5

#define HYPRE_CALC_CAX_ADD_UP_TO_7 \
   HYPRE_CALC_CAX_ADD(6) \
   HYPRE_CALC_CAX_ADD_UP_TO_6

#define HYPRE_CALC_CAX_ADD_UP_TO_8 \
   HYPRE_CALC_CAX_ADD(7) \
   HYPRE_CALC_CAX_ADD_UP_TO_7

#define HYPRE_CALC_CAX_ADD_UP_TO_9 \
   HYPRE_CALC_CAX_ADD(8) \
   HYPRE_CALC_CAX_ADD_UP_TO_8

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
   HYPRE_Int             ndim = hypre_StructMatrixNDim(A);
   hypre_StructStencil  *stencil = hypre_StructMatrixStencil(A);
   hypre_Index          *stencil_shape = hypre_StructStencilShape(stencil);

   hypre_Index           xdstart, ydstart, zdstart;
   hypre_Index           offset;

   HYPRE_Complex        *Ap0 = NULL, *Ap1 = NULL, *Ap2 = NULL;
   HYPRE_Complex        *Ap3 = NULL, *Ap4 = NULL, *Ap5 = NULL;
   HYPRE_Complex        *Ap6 = NULL, *Ap7 = NULL, *Ap8 = NULL;
   HYPRE_Complex        *xp = NULL, *yp = NULL, *zp = NULL;
   HYPRE_Int             xoff0 = 0, xoff1 = 0, xoff2 = 0;
   HYPRE_Int             xoff3 = 0, xoff4 = 0, xoff5 = 0;
   HYPRE_Int             xoff6 = 0, xoff7 = 0, xoff8 = 0;

   HYPRE_Int             si, depth;

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

#define DEVICE_VAR is_device_ptr(xp,yp,zp)
   for (si = 0; si < nentries; si += 9)
   {
      depth = hypre_min(9, (nentries - si));

      if (!transpose)
      {
         switch (depth)
         {
            case 9:
               HYPRE_SET_CAX(Ap8, xoff8, entries[si + 8]);
               HYPRE_FALLTHROUGH;

            case 8:
               HYPRE_SET_CAX(Ap7, xoff7, entries[si + 7]);
               HYPRE_FALLTHROUGH;

            case 7:
               HYPRE_SET_CAX(Ap6, xoff6, entries[si + 6]);
               HYPRE_FALLTHROUGH;

            case 6:
               HYPRE_SET_CAX(Ap5, xoff5, entries[si + 5]);
               HYPRE_FALLTHROUGH;

            case 5:
               HYPRE_SET_CAX(Ap4, xoff4, entries[si + 4]);
               HYPRE_FALLTHROUGH;

            case 4:
               HYPRE_SET_CAX(Ap3, xoff3, entries[si + 3]);
               HYPRE_FALLTHROUGH;

            case 3:
               HYPRE_SET_CAX(Ap2, xoff2, entries[si + 2]);
               HYPRE_FALLTHROUGH;

            case 2:
               HYPRE_SET_CAX(Ap1, xoff1, entries[si + 1]);
               HYPRE_FALLTHROUGH;

            case 1:
               HYPRE_SET_CAX(Ap0, xoff0, entries[si + 0]);
               HYPRE_FALLTHROUGH;

            case 0:
               break;
         }
      }
      else
      {
         switch (depth)
         {
            case 9:
               HYPRE_SET_CAX_TRANS(Ap8, xoff8, entries[si + 8]);
               HYPRE_FALLTHROUGH;

            case 8:
               HYPRE_SET_CAX_TRANS(Ap7, xoff7, entries[si + 7]);
               HYPRE_FALLTHROUGH;

            case 7:
               HYPRE_SET_CAX_TRANS(Ap6, xoff6, entries[si + 6]);
               HYPRE_FALLTHROUGH;

            case 6:
               HYPRE_SET_CAX_TRANS(Ap5, xoff5, entries[si + 5]);
               HYPRE_FALLTHROUGH;

            case 5:
               HYPRE_SET_CAX_TRANS(Ap4, xoff4, entries[si + 4]);
               HYPRE_FALLTHROUGH;

            case 4:
               HYPRE_SET_CAX_TRANS(Ap3, xoff3, entries[si + 3]);
               HYPRE_FALLTHROUGH;

            case 3:
               HYPRE_SET_CAX_TRANS(Ap2, xoff2, entries[si + 2]);
               HYPRE_FALLTHROUGH;

            case 2:
               HYPRE_SET_CAX_TRANS(Ap1, xoff1, entries[si + 1]);
               HYPRE_FALLTHROUGH;

            case 1:
               HYPRE_SET_CAX_TRANS(Ap0, xoff0, entries[si + 0]);
               HYPRE_FALLTHROUGH;

            case 0:
               break;
         }
      }

      switch (depth)
      {
         case 9:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                x_data_box, xdstart, xdstride, xi,
                                y_data_box, ydstart, ydstride, yi,
                                z_data_box, zdstart, zdstride, zi);
            {
               zp[zi] = beta * yp[yi] + alpha * (HYPRE_CALC_CAX_ADD_UP_TO_9);
            }
            hypre_BoxLoop3End(xi, yi, zi);

         case 8:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                x_data_box, xdstart, xdstride, xi,
                                y_data_box, ydstart, ydstride, yi,
                                z_data_box, zdstart, zdstride, zi);
            {
               zp[zi] = beta * yp[yi] + alpha * (HYPRE_CALC_CAX_ADD_UP_TO_8);
            }
            hypre_BoxLoop3End(xi, yi, zi);

         case 7:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                x_data_box, xdstart, xdstride, xi,
                                y_data_box, ydstart, ydstride, yi,
                                z_data_box, zdstart, zdstride, zi);
            {
               zp[zi] = beta * yp[yi] + alpha * (HYPRE_CALC_CAX_ADD_UP_TO_7);
            }
            hypre_BoxLoop3End(xi, yi, zi);

         case 6:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                x_data_box, xdstart, xdstride, xi,
                                y_data_box, ydstart, ydstride, yi,
                                z_data_box, zdstart, zdstride, zi);
            {
               zp[zi] = beta * yp[yi] + alpha * (HYPRE_CALC_CAX_ADD_UP_TO_6);
            }
            hypre_BoxLoop3End(xi, yi, zi);

         case 5:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                x_data_box, xdstart, xdstride, xi,
                                y_data_box, ydstart, ydstride, yi,
                                z_data_box, zdstart, zdstride, zi);
            {
               zp[zi] = beta * yp[yi] + alpha * (HYPRE_CALC_CAX_ADD_UP_TO_5);
            }
            hypre_BoxLoop3End(xi, yi, zi);

         case 4:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                x_data_box, xdstart, xdstride, xi,
                                y_data_box, ydstart, ydstride, yi,
                                z_data_box, zdstart, zdstride, zi);
            {
               zp[zi] = beta * yp[yi] + alpha * (HYPRE_CALC_CAX_ADD_UP_TO_4);
            }
            hypre_BoxLoop3End(xi, yi, zi);

         case 3:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                x_data_box, xdstart, xdstride, xi,
                                y_data_box, ydstart, ydstride, yi,
                                z_data_box, zdstart, zdstride, zi);
            {
               zp[zi] = beta * yp[yi] + alpha * (HYPRE_CALC_CAX_ADD_UP_TO_3);
            }
            hypre_BoxLoop3End(xi, yi, zi);

         case 2:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                x_data_box, xdstart, xdstride, xi,
                                y_data_box, ydstart, ydstride, yi,
                                z_data_box, zdstart, zdstride, zi);
            {
               zp[zi] = beta * yp[yi] + alpha * (HYPRE_CALC_CAX_ADD_UP_TO_2);
            }
            hypre_BoxLoop3End(xi, yi, zi);

         case 1:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                x_data_box, xdstart, xdstride, xi,
                                y_data_box, ydstart, ydstride, yi,
                                z_data_box, zdstart, zdstride, zi);
            {
               zp[zi] = beta * yp[yi] + alpha * (HYPRE_CALC_CAX_ADD_UP_TO_1);
            }
            hypre_BoxLoop3End(xi, yi, zi);

         case 0:
            break;
      } /* switch (depth) */
   } /* for si */
#undef DEVICE_VAR

   hypre_GpuProfilingPopRange();
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * z = beta * y + alpha * A*x
 *
 * StructMatrix/Vector multiplication core routine for variable coeficients.
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_StructMatvecCompute_core_VC( HYPRE_Complex       alpha,
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
                                   hypre_IndexRef      Adstride,
                                   hypre_IndexRef      xdstride,
                                   hypre_IndexRef      ydstride,
                                   hypre_IndexRef      zdstride,
                                   hypre_Box          *A_data_box,
                                   hypre_Box          *x_data_box,
                                   hypre_Box          *y_data_box,
                                   hypre_Box          *z_data_box )
{
   HYPRE_Int             ndim = hypre_StructMatrixNDim(A);
   hypre_StructStencil  *stencil = hypre_StructMatrixStencil(A);
   hypre_Index          *stencil_shape = hypre_StructStencilShape(stencil);

   hypre_Index           Adstart, xdstart, ydstart, zdstart;
   hypre_Index           offset;

   HYPRE_Complex        *xp, *yp, *zp;
   HYPRE_Complex        *Ap0 = NULL, *Ap1 = NULL, *Ap2 = NULL;
   HYPRE_Complex        *Ap3 = NULL, *Ap4 = NULL, *Ap5 = NULL;
   HYPRE_Complex        *Ap6 = NULL, *Ap7 = NULL, *Ap8 = NULL;
   HYPRE_Complex        *Ap9 = NULL, *Ap10 = NULL, *Ap11 = NULL;
   HYPRE_Complex        *Ap12 = NULL, *Ap13 = NULL, *Ap14 = NULL;
   HYPRE_Complex        *Ap15 = NULL, *Ap16 = NULL, *Ap17 = NULL;
   HYPRE_Complex        *Ap18 = NULL, *Ap19 = NULL, *Ap20 = NULL;
   HYPRE_Complex        *Ap21 = NULL, *Ap22 = NULL, *Ap23 = NULL;
   HYPRE_Complex        *Ap24 = NULL, *Ap25 = NULL, *Ap26 = NULL;
   HYPRE_Int             xoff0 = 0, xoff1 = 0, xoff2 = 0;
   HYPRE_Int             xoff3 = 0, xoff4 = 0, xoff5 = 0;
   HYPRE_Int             xoff6 = 0, xoff7 = 0, xoff8 = 0;
   HYPRE_Int             xoff9 = 0, xoff10 = 0, xoff11 = 0;
   HYPRE_Int             xoff12 = 0, xoff13 = 0, xoff14 = 0;
   HYPRE_Int             xoff15 = 0, xoff16 = 0, xoff17 = 0;
   HYPRE_Int             xoff18 = 0, xoff19 = 0, xoff20 = 0;
   HYPRE_Int             xoff21 = 0, xoff22 = 0, xoff23 = 0;
   HYPRE_Int             xoff24 = 0, xoff25 = 0, xoff26 = 0;
   HYPRE_Int             si, depth;

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("VC");

   xp = hypre_StructVectorBoxData(x, xb);
   yp = hypre_StructVectorBoxData(y, yb);
   zp = hypre_StructVectorBoxData(z, zb);

   hypre_CopyToIndex(start, ndim, Adstart);

   /* The next line is only used to avoid 'print error' messages.  It ensures
    * that Adstart is aligned with the matrix data space before mapping.  The
    * choice, Neg vs Pos, doesn't matter because the coefficient pointer will be
    * offset (Ap += BoxOffsetDistance(index - Adstart)). */
   hypre_SnapIndexNeg(Adstart, NULL, stride, ndim);
   hypre_StructMatrixMapDataIndex(A, Adstart);
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

#define DEVICE_VAR is_device_ptr(yp,xp,Ap0,Ap1,Ap2,Ap3,Ap4,Ap5,Ap6,Ap7,Ap8,Ap9,Ap10,Ap11,Ap12,Ap13,Ap14,Ap15,Ap16,Ap17,Ap18,Ap19,Ap20,Ap21,Ap22,Ap23,Ap24,Ap25,Ap26)
   for (si = 0; si < nentries; si += UNROLL_MAXDEPTH)
   {
      depth = hypre_min(UNROLL_MAXDEPTH, (nentries - si));

      if (!transpose)
      {
         switch (depth)
         {
            case 27:
               HYPRE_SET_AX(Ap26, xoff26, entries[si + 26]);
               HYPRE_FALLTHROUGH;

            case 26:
               HYPRE_SET_AX(Ap25, xoff25, entries[si + 25]);
               HYPRE_FALLTHROUGH;

            case 25:
               HYPRE_SET_AX(Ap24, xoff24, entries[si + 24]);
               HYPRE_FALLTHROUGH;

            case 24:
               HYPRE_SET_AX(Ap23, xoff23, entries[si + 23]);
               HYPRE_FALLTHROUGH;

            case 23:
               HYPRE_SET_AX(Ap22, xoff22, entries[si + 22]);
               HYPRE_FALLTHROUGH;

            case 22:
               HYPRE_SET_AX(Ap21, xoff21, entries[si + 21]);
               HYPRE_FALLTHROUGH;

            case 21:
               HYPRE_SET_AX(Ap20, xoff20, entries[si + 20]);
               HYPRE_FALLTHROUGH;

            case 20:
               HYPRE_SET_AX(Ap19, xoff19, entries[si + 19]);
               HYPRE_FALLTHROUGH;

            case 19:
               HYPRE_SET_AX(Ap18, xoff18, entries[si + 18]);
               HYPRE_FALLTHROUGH;

            case 18:
               HYPRE_SET_AX(Ap17, xoff17, entries[si + 17]);
               HYPRE_FALLTHROUGH;

            case 17:
               HYPRE_SET_AX(Ap16, xoff16, entries[si + 16]);
               HYPRE_FALLTHROUGH;

            case 16:
               HYPRE_SET_AX(Ap15, xoff15, entries[si + 15]);
               HYPRE_FALLTHROUGH;

            case 15:
               HYPRE_SET_AX(Ap14, xoff14, entries[si + 14]);
               HYPRE_FALLTHROUGH;

            case 14:
               HYPRE_SET_AX(Ap13, xoff13, entries[si + 13]);
               HYPRE_FALLTHROUGH;

            case 13:
               HYPRE_SET_AX(Ap12, xoff12, entries[si + 12]);
               HYPRE_FALLTHROUGH;

            case 12:
               HYPRE_SET_AX(Ap11, xoff11, entries[si + 11]);
               HYPRE_FALLTHROUGH;

            case 11:
               HYPRE_SET_AX(Ap10, xoff10, entries[si + 10]);
               HYPRE_FALLTHROUGH;

            case 10:
               HYPRE_SET_AX(Ap9, xoff9, entries[si + 9]);
               HYPRE_FALLTHROUGH;

            case 9:
               HYPRE_SET_AX(Ap8, xoff8, entries[si + 8]);
               HYPRE_FALLTHROUGH;

            case 8:
               HYPRE_SET_AX(Ap7, xoff7, entries[si + 7]);
               HYPRE_FALLTHROUGH;

            case 7:
               HYPRE_SET_AX(Ap6, xoff6, entries[si + 6]);
               HYPRE_FALLTHROUGH;

            case 6:
               HYPRE_SET_AX(Ap5, xoff5, entries[si + 5]);
               HYPRE_FALLTHROUGH;

            case 5:
               HYPRE_SET_AX(Ap4, xoff4, entries[si + 4]);
               HYPRE_FALLTHROUGH;

            case 4:
               HYPRE_SET_AX(Ap3, xoff3, entries[si + 3]);
               HYPRE_FALLTHROUGH;

            case 3:
               HYPRE_SET_AX(Ap2, xoff2, entries[si + 2]);
               HYPRE_FALLTHROUGH;

            case 2:
               HYPRE_SET_AX(Ap1, xoff1, entries[si + 1]);
               HYPRE_FALLTHROUGH;

            case 1:
               HYPRE_SET_AX(Ap0, xoff0, entries[si + 0]);
               HYPRE_FALLTHROUGH;

            case 0:
               break;
         }
      }
      else
      {
         switch (depth)
         {
            case 27:
               HYPRE_SET_AX_TRANS(Ap26, xoff26, entries[si + 26]);
               HYPRE_FALLTHROUGH;

            case 26:
               HYPRE_SET_AX_TRANS(Ap25, xoff25, entries[si + 25]);
               HYPRE_FALLTHROUGH;

            case 25:
               HYPRE_SET_AX_TRANS(Ap24, xoff24, entries[si + 24]);
               HYPRE_FALLTHROUGH;

            case 24:
               HYPRE_SET_AX_TRANS(Ap23, xoff23, entries[si + 23]);
               HYPRE_FALLTHROUGH;

            case 23:
               HYPRE_SET_AX_TRANS(Ap22, xoff22, entries[si + 22]);
               HYPRE_FALLTHROUGH;

            case 22:
               HYPRE_SET_AX_TRANS(Ap21, xoff21, entries[si + 21]);
               HYPRE_FALLTHROUGH;

            case 21:
               HYPRE_SET_AX_TRANS(Ap20, xoff20, entries[si + 20]);
               HYPRE_FALLTHROUGH;

            case 20:
               HYPRE_SET_AX_TRANS(Ap19, xoff19, entries[si + 19]);
               HYPRE_FALLTHROUGH;

            case 19:
               HYPRE_SET_AX_TRANS(Ap18, xoff18, entries[si + 18]);
               HYPRE_FALLTHROUGH;

            case 18:
               HYPRE_SET_AX_TRANS(Ap17, xoff17, entries[si + 17]);
               HYPRE_FALLTHROUGH;

            case 17:
               HYPRE_SET_AX_TRANS(Ap16, xoff16, entries[si + 16]);
               HYPRE_FALLTHROUGH;

            case 16:
               HYPRE_SET_AX_TRANS(Ap15, xoff15, entries[si + 15]);
               HYPRE_FALLTHROUGH;

            case 15:
               HYPRE_SET_AX_TRANS(Ap14, xoff14, entries[si + 14]);
               HYPRE_FALLTHROUGH;

            case 14:
               HYPRE_SET_AX_TRANS(Ap13, xoff13, entries[si + 13]);
               HYPRE_FALLTHROUGH;

            case 13:
               HYPRE_SET_AX_TRANS(Ap12, xoff12, entries[si + 12]);
               HYPRE_FALLTHROUGH;

            case 12:
               HYPRE_SET_AX_TRANS(Ap11, xoff11, entries[si + 11]);
               HYPRE_FALLTHROUGH;

            case 11:
               HYPRE_SET_AX_TRANS(Ap10, xoff10, entries[si + 10]);
               HYPRE_FALLTHROUGH;

            case 10:
               HYPRE_SET_AX_TRANS(Ap9, xoff9, entries[si + 9]);
               HYPRE_FALLTHROUGH;

            case 9:
               HYPRE_SET_AX_TRANS(Ap8, xoff8, entries[si + 8]);
               HYPRE_FALLTHROUGH;

            case 8:
               HYPRE_SET_AX_TRANS(Ap7, xoff7, entries[si + 7]);
               HYPRE_FALLTHROUGH;

            case 7:
               HYPRE_SET_AX_TRANS(Ap6, xoff6, entries[si + 6]);
               HYPRE_FALLTHROUGH;

            case 6:
               HYPRE_SET_AX_TRANS(Ap5, xoff5, entries[si + 5]);
               HYPRE_FALLTHROUGH;

            case 5:
               HYPRE_SET_AX_TRANS(Ap4, xoff4, entries[si + 4]);
               HYPRE_FALLTHROUGH;

            case 4:
               HYPRE_SET_AX_TRANS(Ap3, xoff3, entries[si + 3]);
               HYPRE_FALLTHROUGH;

            case 3:
               HYPRE_SET_AX_TRANS(Ap2, xoff2, entries[si + 2]);
               HYPRE_FALLTHROUGH;

            case 2:
               HYPRE_SET_AX_TRANS(Ap1, xoff1, entries[si + 1]);
               HYPRE_FALLTHROUGH;

            case 1:
               HYPRE_SET_AX_TRANS(Ap0, xoff0, entries[si + 0]);
               HYPRE_FALLTHROUGH;

            case 0:
               break;
         }
      }

      switch (depth)
      {
         case 27:
            hypre_BoxLoop4Begin(ndim, loop_size,
                                A_data_box, Adstart, Adstride, Ai,
                                x_data_box, xdstart, xdstride, xi,
                                y_data_box, ydstart, ydstride, yi,
                                z_data_box, zdstart, zdstride, zi);
            {
               zp[zi] = beta * yp[yi] + alpha * (HYPRE_CALC_AX_ADD_UP_TO_27);
            }
            hypre_BoxLoop4End(Ai, xi, yi, zi);
            break;

         case 26:
            hypre_BoxLoop4Begin(ndim, loop_size,
                                A_data_box, Adstart, Adstride, Ai,
                                x_data_box, xdstart, xdstride, xi,
                                y_data_box, ydstart, ydstride, yi,
                                z_data_box, zdstart, zdstride, zi);
            {
               zp[zi] = beta * yp[yi] + alpha * (HYPRE_CALC_AX_ADD_UP_TO_26);
            }
            hypre_BoxLoop4End(Ai, xi, yi, zi);
            break;

         case 25:
            hypre_BoxLoop4Begin(ndim, loop_size,
                                A_data_box, Adstart, Adstride, Ai,
                                x_data_box, xdstart, xdstride, xi,
                                y_data_box, ydstart, ydstride, yi,
                                z_data_box, zdstart, zdstride, zi);
            {
               zp[zi] = beta * yp[yi] + alpha * (HYPRE_CALC_AX_ADD_UP_TO_25);
            }
            hypre_BoxLoop4End(Ai, xi, yi, zi);
            break;

         case 24:
            hypre_BoxLoop4Begin(ndim, loop_size,
                                A_data_box, Adstart, Adstride, Ai,
                                x_data_box, xdstart, xdstride, xi,
                                y_data_box, ydstart, ydstride, yi,
                                z_data_box, zdstart, zdstride, zi);
            {
               zp[zi] = beta * yp[yi] + alpha * (HYPRE_CALC_AX_ADD_UP_TO_24);
            }
            hypre_BoxLoop4End(Ai, xi, yi, zi);
            break;

         case 23:
            hypre_BoxLoop4Begin(ndim, loop_size,
                                A_data_box, Adstart, Adstride, Ai,
                                x_data_box, xdstart, xdstride, xi,
                                y_data_box, ydstart, ydstride, yi,
                                z_data_box, zdstart, zdstride, zi);
            {
               zp[zi] = beta * yp[yi] + alpha * (HYPRE_CALC_AX_ADD_UP_TO_23);
            }
            hypre_BoxLoop4End(Ai, xi, yi, zi);
            break;

         case 22:
            hypre_BoxLoop4Begin(ndim, loop_size,
                                A_data_box, Adstart, Adstride, Ai,
                                x_data_box, xdstart, xdstride, xi,
                                y_data_box, ydstart, ydstride, yi,
                                z_data_box, zdstart, zdstride, zi);
            {
               zp[zi] = beta * yp[yi] + alpha * (HYPRE_CALC_AX_ADD_UP_TO_22);
            }
            hypre_BoxLoop4End(Ai, xi, yi, zi);
            break;

         case 21:
            hypre_BoxLoop4Begin(ndim, loop_size,
                                A_data_box, Adstart, Adstride, Ai,
                                x_data_box, xdstart, xdstride, xi,
                                y_data_box, ydstart, ydstride, yi,
                                z_data_box, zdstart, zdstride, zi);
            {
               zp[zi] = beta * yp[yi] + alpha * (HYPRE_CALC_AX_ADD_UP_TO_21);
            }
            hypre_BoxLoop4End(Ai, xi, yi, zi);
            break;

         case 20:
            hypre_BoxLoop4Begin(ndim, loop_size,
                                A_data_box, Adstart, Adstride, Ai,
                                x_data_box, xdstart, xdstride, xi,
                                y_data_box, ydstart, ydstride, yi,
                                z_data_box, zdstart, zdstride, zi);
            {
               zp[zi] = beta * yp[yi] + alpha * (HYPRE_CALC_AX_ADD_UP_TO_20);
            }
            hypre_BoxLoop4End(Ai, xi, yi, zi);
            break;

         case 19:
            hypre_BoxLoop4Begin(ndim, loop_size,
                                A_data_box, Adstart, Adstride, Ai,
                                x_data_box, xdstart, xdstride, xi,
                                y_data_box, ydstart, ydstride, yi,
                                z_data_box, zdstart, zdstride, zi);
            {
               zp[zi] = beta * yp[yi] + alpha * (HYPRE_CALC_AX_ADD_UP_TO_19);
            }
            hypre_BoxLoop4End(Ai, xi, yi, zi);
            break;

         case 18:
            hypre_BoxLoop4Begin(ndim, loop_size,
                                A_data_box, Adstart, Adstride, Ai,
                                x_data_box, xdstart, xdstride, xi,
                                y_data_box, ydstart, ydstride, yi,
                                z_data_box, zdstart, zdstride, zi);
            {
               zp[zi] = beta * yp[yi] + alpha * (HYPRE_CALC_AX_ADD_UP_TO_18);
            }
            hypre_BoxLoop4End(Ai, xi, yi, zi);
            break;

         case 17:
            hypre_BoxLoop4Begin(ndim, loop_size,
                                A_data_box, Adstart, Adstride, Ai,
                                x_data_box, xdstart, xdstride, xi,
                                y_data_box, ydstart, ydstride, yi,
                                z_data_box, zdstart, zdstride, zi);
            {
               zp[zi] = beta * yp[yi] + alpha * (HYPRE_CALC_AX_ADD_UP_TO_17);
            }
            hypre_BoxLoop4End(Ai, xi, yi, zi);
            break;

         case 16:
            hypre_BoxLoop4Begin(ndim, loop_size,
                                A_data_box, Adstart, Adstride, Ai,
                                x_data_box, xdstart, xdstride, xi,
                                y_data_box, ydstart, ydstride, yi,
                                z_data_box, zdstart, zdstride, zi);
            {
               zp[zi] = beta * yp[yi] + alpha * (HYPRE_CALC_AX_ADD_UP_TO_16);
            }
            hypre_BoxLoop4End(Ai, xi, yi, zi);
            break;

         case 15:
            hypre_BoxLoop4Begin(ndim, loop_size,
                                A_data_box, Adstart, Adstride, Ai,
                                x_data_box, xdstart, xdstride, xi,
                                y_data_box, ydstart, ydstride, yi,
                                z_data_box, zdstart, zdstride, zi);
            {
               zp[zi] = beta * yp[yi] + alpha * (HYPRE_CALC_AX_ADD_UP_TO_15);
            }
            hypre_BoxLoop4End(Ai, xi, yi, zi);
            break;

         case 14:
            hypre_BoxLoop4Begin(ndim, loop_size,
                                A_data_box, Adstart, Adstride, Ai,
                                x_data_box, xdstart, xdstride, xi,
                                y_data_box, ydstart, ydstride, yi,
                                z_data_box, zdstart, zdstride, zi);
            {
               zp[zi] = beta * yp[yi] + alpha * (HYPRE_CALC_AX_ADD_UP_TO_14);
            }
            hypre_BoxLoop4End(Ai, xi, yi, zi);
            break;

         case 13:
            hypre_BoxLoop4Begin(ndim, loop_size,
                                A_data_box, Adstart, Adstride, Ai,
                                x_data_box, xdstart, xdstride, xi,
                                y_data_box, ydstart, ydstride, yi,
                                z_data_box, zdstart, zdstride, zi);
            {
               zp[zi] = beta * yp[yi] + alpha * (HYPRE_CALC_AX_ADD_UP_TO_13);
            }
            hypre_BoxLoop4End(Ai, xi, yi, zi);
            break;

         case 12:
            hypre_BoxLoop4Begin(ndim, loop_size,
                                A_data_box, Adstart, Adstride, Ai,
                                x_data_box, xdstart, xdstride, xi,
                                y_data_box, ydstart, ydstride, yi,
                                z_data_box, zdstart, zdstride, zi);
            {
               zp[zi] = beta * yp[yi] + alpha * (HYPRE_CALC_AX_ADD_UP_TO_12);
            }
            hypre_BoxLoop4End(Ai, xi, yi, zi);
            break;

         case 11:
            hypre_BoxLoop4Begin(ndim, loop_size,
                                A_data_box, Adstart, Adstride, Ai,
                                x_data_box, xdstart, xdstride, xi,
                                y_data_box, ydstart, ydstride, yi,
                                z_data_box, zdstart, zdstride, zi);
            {
               zp[zi] = beta * yp[yi] + alpha * (HYPRE_CALC_AX_ADD_UP_TO_11);
            }
            hypre_BoxLoop4End(Ai, xi, yi, zi);
            break;

         case 10:
            hypre_BoxLoop4Begin(ndim, loop_size,
                                A_data_box, Adstart, Adstride, Ai,
                                x_data_box, xdstart, xdstride, xi,
                                y_data_box, ydstart, ydstride, yi,
                                z_data_box, zdstart, zdstride, zi);
            {
               zp[zi] = beta * yp[yi] + alpha * (HYPRE_CALC_AX_ADD_UP_TO_10);
            }
            hypre_BoxLoop4End(Ai, xi, yi, zi);
            break;

         case 9:
            hypre_BoxLoop4Begin(ndim, loop_size,
                                A_data_box, Adstart, Adstride, Ai,
                                x_data_box, xdstart, xdstride, xi,
                                y_data_box, ydstart, ydstride, yi,
                                z_data_box, zdstart, zdstride, zi);
            {
               zp[zi] = beta * yp[yi] + alpha * (HYPRE_CALC_AX_ADD_UP_TO_9);
            }
            hypre_BoxLoop4End(Ai, xi, yi, zi);
            break;

         case 8:
            hypre_BoxLoop4Begin(ndim, loop_size,
                                A_data_box, Adstart, Adstride, Ai,
                                x_data_box, xdstart, xdstride, xi,
                                y_data_box, ydstart, ydstride, yi,
                                z_data_box, zdstart, zdstride, zi);
            {
               zp[zi] = beta * yp[yi] + alpha * (HYPRE_CALC_AX_ADD_UP_TO_8);
            }
            hypre_BoxLoop4End(Ai, xi, yi, zi);
            break;

         case 7:
            hypre_BoxLoop4Begin(ndim, loop_size,
                                A_data_box, Adstart, Adstride, Ai,
                                x_data_box, xdstart, xdstride, xi,
                                y_data_box, ydstart, ydstride, yi,
                                z_data_box, zdstart, zdstride, zi);
            {
               zp[zi] = beta * yp[yi] + alpha * (HYPRE_CALC_AX_ADD_UP_TO_7);
            }
            hypre_BoxLoop4End(Ai, xi, yi, zi);
            break;

         case 6:
            hypre_BoxLoop4Begin(ndim, loop_size,
                                A_data_box, Adstart, Adstride, Ai,
                                x_data_box, xdstart, xdstride, xi,
                                y_data_box, ydstart, ydstride, yi,
                                z_data_box, zdstart, zdstride, zi);
            {
               zp[zi] = beta * yp[yi] + alpha * (HYPRE_CALC_AX_ADD_UP_TO_6);
            }
            hypre_BoxLoop4End(Ai, xi, yi, zi);
            break;

         case 5:
            hypre_BoxLoop4Begin(ndim, loop_size,
                                A_data_box, Adstart, Adstride, Ai,
                                x_data_box, xdstart, xdstride, xi,
                                y_data_box, ydstart, ydstride, yi,
                                z_data_box, zdstart, zdstride, zi);
            {
               zp[zi] = beta * yp[yi] + alpha * (HYPRE_CALC_AX_ADD_UP_TO_5);
            }
            hypre_BoxLoop4End(Ai, xi, yi, zi);
            break;

         case 4:
            hypre_BoxLoop4Begin(ndim, loop_size,
                                A_data_box, Adstart, Adstride, Ai,
                                x_data_box, xdstart, xdstride, xi,
                                y_data_box, ydstart, ydstride, yi,
                                z_data_box, zdstart, zdstride, zi);
            {
               zp[zi] = beta * yp[yi] + alpha * (HYPRE_CALC_AX_ADD_UP_TO_4);
            }
            hypre_BoxLoop4End(Ai, xi, yi, zi);
            break;

         case 3:
            hypre_BoxLoop4Begin(ndim, loop_size,
                                A_data_box, Adstart, Adstride, Ai,
                                x_data_box, xdstart, xdstride, xi,
                                y_data_box, ydstart, ydstride, yi,
                                z_data_box, zdstart, zdstride, zi);
            {
               zp[zi] = beta * yp[yi] + alpha * (HYPRE_CALC_AX_ADD_UP_TO_3);
            }
            hypre_BoxLoop4End(Ai, xi, yi, zi);
            break;

         case 2:
            hypre_BoxLoop4Begin(ndim, loop_size,
                                A_data_box, Adstart, Adstride, Ai,
                                x_data_box, xdstart, xdstride, xi,
                                y_data_box, ydstart, ydstride, yi,
                                z_data_box, zdstart, zdstride, zi);
            {
               zp[zi] = beta * yp[yi] + alpha * (HYPRE_CALC_AX_ADD_UP_TO_2);
            }
            hypre_BoxLoop4End(Ai, xi, yi, zi);
            break;

         case 1:
            hypre_BoxLoop4Begin(ndim, loop_size,
                                A_data_box, Adstart, Adstride, Ai,
                                x_data_box, xdstart, xdstride, xi,
                                y_data_box, ydstart, ydstride, yi,
                                z_data_box, zdstart, zdstride, zi);
            {
               zp[zi] = beta * yp[yi] + alpha * (HYPRE_CALC_AX_ADD_UP_TO_1);
            }
            hypre_BoxLoop4End(Ai, xi, yi, zi);
            break;

         case 0:
            break;
      } /* switch (depth) */
   } /* for si */
#undef DEVICE_VAR

   hypre_GpuProfilingPopRange();
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * z = beta * y + alpha * A*x
 *
 * StructMatrix/Vector multiplication core routine for one constant
 * and an arbitrary number of variable coefficients
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_StructMatvecCompute_core_VCC( HYPRE_Complex       alpha,
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
                                    HYPRE_Int           centry,
                                    HYPRE_Int           nentries,
                                    HYPRE_Int          *entries,
                                    hypre_IndexRef      start,
                                    hypre_IndexRef      stride,
                                    hypre_IndexRef      loop_size,
                                    hypre_IndexRef      xfstride,
                                    hypre_IndexRef      ran_stride,
                                    hypre_IndexRef      Adstride,
                                    hypre_IndexRef      xdstride,
                                    hypre_IndexRef      ydstride,
                                    hypre_IndexRef      zdstride,
                                    hypre_Box          *A_data_box,
                                    hypre_Box          *x_data_box,
                                    hypre_Box          *y_data_box,
                                    hypre_Box          *z_data_box )
{
   HYPRE_Int             ndim = hypre_StructMatrixNDim(A);
   hypre_StructStencil  *stencil = hypre_StructMatrixStencil(A);
   hypre_Index          *stencil_shape = hypre_StructStencilShape(stencil);

   hypre_Index           Adstart, xdstart, ydstart, zdstart;
   hypre_Index           offset;

   HYPRE_Complex        *xp, *yp, *zp;
   HYPRE_Complex        *Ap0 = NULL, *Ap1 = NULL, *Ap2 = NULL;
   HYPRE_Complex        *Ap3 = NULL, *Ap4 = NULL, *Ap5 = NULL;
   HYPRE_Complex        *Ap6 = NULL;
   HYPRE_Int             xoff0 = 0, xoff1 = 0, xoff2 = 0;
   HYPRE_Int             xoff3 = 0, xoff4 = 0, xoff5 = 0;
   HYPRE_Int             xoff6 = 0;
   HYPRE_Int             si, depth;

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("VCC");

   xp = hypre_StructVectorBoxData(x, xb);
   yp = hypre_StructVectorBoxData(y, yb);
   zp = hypre_StructVectorBoxData(z, zb);

   hypre_CopyToIndex(start, ndim, Adstart);

   /* The next line is only used to avoid 'print error' messages.  It ensures
    * that Adstart is aligned with the matrix data space before mapping.  The
    * choice, Neg vs Pos, doesn't matter because the coefficient pointer will be
    * offset (Ap += BoxOffsetDistance(index - Adstart)). */
   hypre_SnapIndexNeg(Adstart, NULL, stride, ndim);
   hypre_StructMatrixMapDataIndex(A, Adstart);
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

#define DEVICE_VAR is_device_ptr(yp,xp,Ap0,Ap1,Ap2,Ap3,Ap4,Ap5,Ap6)
   for (si = 0; si < nentries; si += 6)
   {
      depth = hypre_min(6, (nentries - si));

      if (!transpose)
      {
         switch (depth)
         {
            case 6:
               HYPRE_SET_AX(Ap5, xoff5, entries[si + 5]);
               HYPRE_FALLTHROUGH;

            case 5:
               HYPRE_SET_AX(Ap4, xoff4, entries[si + 4]);
               HYPRE_FALLTHROUGH;

            case 4:
               HYPRE_SET_AX(Ap3, xoff3, entries[si + 3]);
               HYPRE_FALLTHROUGH;

            case 3:
               HYPRE_SET_AX(Ap2, xoff2, entries[si + 2]);
               HYPRE_FALLTHROUGH;

            case 2:
               HYPRE_SET_AX(Ap1, xoff1, entries[si + 1]);
               HYPRE_FALLTHROUGH;

            case 1:
               HYPRE_SET_AX(Ap0, xoff0, entries[si + 0]);
               HYPRE_FALLTHROUGH;

            case 0:
               break;
         }

         /* Load constant coefficient in the first loop */
         if (si < 6)
         {
            HYPRE_SET_CAX(Ap6, xoff6, centry);
         }
      }
      else
      {
         switch (depth)
         {
            case 7:
               HYPRE_SET_AX_TRANS(Ap6, xoff6, entries[si + 6]);
               HYPRE_FALLTHROUGH;

            case 6:
               HYPRE_SET_AX_TRANS(Ap5, xoff5, entries[si + 5]);
               HYPRE_FALLTHROUGH;

            case 5:
               HYPRE_SET_AX_TRANS(Ap4, xoff4, entries[si + 4]);
               HYPRE_FALLTHROUGH;

            case 4:
               HYPRE_SET_AX_TRANS(Ap3, xoff3, entries[si + 3]);
               HYPRE_FALLTHROUGH;

            case 3:
               HYPRE_SET_AX_TRANS(Ap2, xoff2, entries[si + 2]);
               HYPRE_FALLTHROUGH;

            case 2:
               HYPRE_SET_AX_TRANS(Ap1, xoff1, entries[si + 1]);
               HYPRE_FALLTHROUGH;

            case 1:
               HYPRE_SET_AX_TRANS(Ap0, xoff0, entries[si + 0]);
               HYPRE_FALLTHROUGH;

            case 0:
               break;
         }

         /* Load constant coefficient in the first loop */
         if (si < 6)
         {
            HYPRE_SET_CAX_TRANS(Ap6, xoff6, centry);
         }
      }

      if (si < 6)
      {
         switch (depth)
         {
            case 6:
               hypre_BoxLoop4Begin(ndim, loop_size,
                                   A_data_box, Adstart, Adstride, Ai,
                                   x_data_box, xdstart, xdstride, xi,
                                   y_data_box, ydstart, ydstride, yi,
                                   z_data_box, zdstart, zdstride, zi);
               {
                  zp[zi] = beta * yp[yi] +
                    alpha * ( Ap0[Ai] * xp[xi + xoff0] +
                              Ap1[Ai] * xp[xi + xoff1] +
                              Ap2[Ai] * xp[xi + xoff2] +
                              Ap3[Ai] * xp[xi + xoff3] +
                              Ap4[Ai] * xp[xi + xoff4] +
                              Ap5[Ai] * xp[xi + xoff5] +
                              Ap6[0]  * xp[xi + xoff6] );
               }
               hypre_BoxLoop4End(Ai, xi, yi, zi);
               break;

            case 5:
               hypre_BoxLoop4Begin(ndim, loop_size,
                                   A_data_box, Adstart, Adstride, Ai,
                                   x_data_box, xdstart, xdstride, xi,
                                   y_data_box, ydstart, ydstride, yi,
                                   z_data_box, zdstart, zdstride, zi);
               {
                  zp[zi] = beta * yp[yi] +
                    alpha * ( Ap0[Ai] * xp[xi + xoff0] +
                              Ap1[Ai] * xp[xi + xoff1] +
                              Ap2[Ai] * xp[xi + xoff2] +
                              Ap3[Ai] * xp[xi + xoff3] +
                              Ap4[Ai] * xp[xi + xoff4] +
                              Ap6[0]  * xp[xi + xoff6] );
               }
               hypre_BoxLoop4End(Ai, xi, yi, zi);
               break;

            case 4:
               hypre_BoxLoop4Begin(ndim, loop_size,
                                   A_data_box, Adstart, Adstride, Ai,
                                   x_data_box, xdstart, xdstride, xi,
                                   y_data_box, ydstart, ydstride, yi,
                                   z_data_box, zdstart, zdstride, zi);
               {
                  zp[zi] = beta * yp[yi] +
                    alpha * ( Ap0[Ai] * xp[xi + xoff0] +
                              Ap1[Ai] * xp[xi + xoff1] +
                              Ap2[Ai] * xp[xi + xoff2] +
                              Ap3[Ai] * xp[xi + xoff3] +
                              Ap6[0]  * xp[xi + xoff6] );
               }
               hypre_BoxLoop4End(Ai, xi, yi, zi);
               break;

            case 3:
               hypre_BoxLoop4Begin(ndim, loop_size,
                                   A_data_box, Adstart, Adstride, Ai,
                                   x_data_box, xdstart, xdstride, xi,
                                   y_data_box, ydstart, ydstride, yi,
                                   z_data_box, zdstart, zdstride, zi);
               {
                  zp[zi] = beta * yp[yi] +
                    alpha * ( Ap0[Ai] * xp[xi + xoff0] +
                              Ap1[Ai] * xp[xi + xoff1] +
                              Ap2[Ai] * xp[xi + xoff2] +
                              Ap6[0]  * xp[xi + xoff6] );
               }
               hypre_BoxLoop4End(Ai, xi, yi, zi);
               break;

            case 2:
               hypre_BoxLoop4Begin(ndim, loop_size,
                                   A_data_box, Adstart, Adstride, Ai,
                                   x_data_box, xdstart, xdstride, xi,
                                   y_data_box, ydstart, ydstride, yi,
                                   z_data_box, zdstart, zdstride, zi);
               {
                  zp[zi] = beta * yp[yi] +
                    alpha * ( Ap0[Ai] * xp[xi + xoff0] +
                              Ap1[Ai] * xp[xi + xoff1] +
                              Ap6[0]  * xp[xi + xoff6] );
               }
               hypre_BoxLoop4End(Ai, xi, yi, zi);
               break;

            case 1:
               hypre_BoxLoop4Begin(ndim, loop_size,
                                   A_data_box, Adstart, Adstride, Ai,
                                   x_data_box, xdstart, xdstride, xi,
                                   y_data_box, ydstart, ydstride, yi,
                                   z_data_box, zdstart, zdstride, zi);
               {
                  zp[zi] = beta * yp[yi] +
                    alpha * ( Ap0[Ai] * xp[xi + xoff0] +
                              Ap6[0]  * xp[xi + xoff6] );
               }
               hypre_BoxLoop4End(Ai, xi, yi, zi);
               break;

            case 0:
               break;
         } /* switch (depth) */
      }
      else
      {
         switch (depth)
         {
            case 6:
               hypre_BoxLoop3Begin(ndim, loop_size,
                                   A_data_box, Adstart, Adstride, Ai,
                                   x_data_box, xdstart, xdstride, xi,
                                   z_data_box, zdstart, zdstride, zi);
               {
                  zp[zi] += alpha * ( Ap0[Ai] * xp[xi + xoff0] +
                                      Ap1[Ai] * xp[xi + xoff1] +
                                      Ap2[Ai] * xp[xi + xoff2] +
                                      Ap3[Ai] * xp[xi + xoff3] +
                                      Ap4[Ai] * xp[xi + xoff4] +
                                      Ap5[Ai] * xp[xi + xoff5] );
               }
               hypre_BoxLoop3End(Ai, xi, zi);
               break;

            case 5:
               hypre_BoxLoop3Begin(ndim, loop_size,
                                   A_data_box, Adstart, Adstride, Ai,
                                   x_data_box, xdstart, xdstride, xi,
                                   z_data_box, zdstart, zdstride, zi);
               {
                  zp[zi] += alpha * ( Ap0[Ai] * xp[xi + xoff0] +
                                      Ap1[Ai] * xp[xi + xoff1] +
                                      Ap2[Ai] * xp[xi + xoff2] +
                                      Ap3[Ai] * xp[xi + xoff3] +
                                      Ap4[Ai] * xp[xi + xoff4] );
               }
               hypre_BoxLoop3End(Ai, xi, zi);
               break;

            case 4:
               hypre_BoxLoop3Begin(ndim, loop_size,
                                   A_data_box, Adstart, Adstride, Ai,
                                   x_data_box, xdstart, xdstride, xi,
                                   z_data_box, zdstart, zdstride, zi);
               {
                  zp[zi] += alpha * ( Ap0[Ai] * xp[xi + xoff0] +
                                      Ap1[Ai] * xp[xi + xoff1] +
                                      Ap2[Ai] * xp[xi + xoff2] +
                                      Ap3[Ai] * xp[xi + xoff3] );
               }
               hypre_BoxLoop3End(Ai, xi, zi);
               break;

            case 3:
               hypre_BoxLoop3Begin(ndim, loop_size,
                                   A_data_box, Adstart, Adstride, Ai,
                                   x_data_box, xdstart, xdstride, xi,
                                   z_data_box, zdstart, zdstride, zi);
               {
                  zp[zi] += alpha * ( Ap0[Ai] * xp[xi + xoff0] +
                                      Ap1[Ai] * xp[xi + xoff1] +
                                      Ap2[Ai] * xp[xi + xoff2] );
               }
               hypre_BoxLoop3End(Ai, xi, zi);
               break;

            case 2:
               hypre_BoxLoop3Begin(ndim, loop_size,
                                   A_data_box, Adstart, Adstride, Ai,
                                   x_data_box, xdstart, xdstride, xi,
                                   z_data_box, zdstart, zdstride, zi);
               {
                  zp[zi] += alpha * ( Ap0[Ai] * xp[xi + xoff0] +
                                      Ap1[Ai] * xp[xi + xoff1] );
               }
               hypre_BoxLoop3End(Ai, xi, zi);
               break;

            case 1:
               hypre_BoxLoop3Begin(ndim, loop_size,
                                   A_data_box, Adstart, Adstride, Ai,
                                   x_data_box, xdstart, xdstride, xi,
                                   z_data_box, zdstart, zdstride, zi);
               {
                  zp[zi] += alpha * Ap0[Ai] * xp[xi + xoff0];
               }
               hypre_BoxLoop3End(Ai, xi, zi);
               break;

            case 0:
               break;
         } /* switch (depth) */
      } /* if (si < 6) */
   } /* for si */
#undef DEVICE_VAR

   hypre_GpuProfilingPopRange();
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}
