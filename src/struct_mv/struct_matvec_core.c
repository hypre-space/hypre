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

#define UNROLL_MAXDEPTH 7

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

/*--------------------------------------------------------------------------
 * StructMatrix/Vector multiplication core function for constant coeficients.
 * Note: This function computes -A*x.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatvecCompute_core_CC( hypre_StructMatrix *A,
                                   hypre_StructVector *x,
                                   hypre_StructVector *y,
                                   HYPRE_Int           Ab,
                                   HYPRE_Int           xb,
                                   HYPRE_Int           yb,
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
                                   hypre_Box          *x_data_box,
                                   hypre_Box          *y_data_box )
{
   HYPRE_Int             ndim = hypre_StructMatrixNDim(A);
   hypre_StructStencil  *stencil = hypre_StructMatrixStencil(A);
   hypre_Index          *stencil_shape = hypre_StructStencilShape(stencil);

   hypre_Index           xdstart, ydstart;
   hypre_Index           offset;

   HYPRE_Complex        *Ap0, *Ap1, *Ap2;
   HYPRE_Complex        *Ap3, *Ap4, *Ap5;
   HYPRE_Complex        *Ap6, *Ap7, *Ap8;
   HYPRE_Complex        *xp,  *yp;
   HYPRE_Int             xoff0, xoff1, xoff2;
   HYPRE_Int             xoff3, xoff4, xoff5;
   HYPRE_Int             xoff6, xoff7, xoff8;

   HYPRE_Int             si, depth;

   HYPRE_ANNOTATE_FUNC_BEGIN;

   xp = hypre_StructVectorBoxData(x, xb);
   yp = hypre_StructVectorBoxData(y, yb);

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

#define DEVICE_VAR is_device_ptr(yp,xp)
   for (si = 0; si < nentries; si += UNROLL_MAXDEPTH)
   {
      depth = hypre_min(UNROLL_MAXDEPTH, (nentries - si));

      if (!transpose)
      {
         switch (depth)
         {
            case 9:
               HYPRE_SET_CAX(Ap8, xoff8, entries[si + 8]);

            case 8:
               HYPRE_SET_CAX(Ap7, xoff7, entries[si + 7]);

            case 7:
               HYPRE_SET_CAX(Ap6, xoff6, entries[si + 6]);

            case 6:
               HYPRE_SET_CAX(Ap5, xoff5, entries[si + 5]);

            case 5:
               HYPRE_SET_CAX(Ap4, xoff4, entries[si + 4]);

            case 4:
               HYPRE_SET_CAX(Ap3, xoff3, entries[si + 3]);

            case 3:
               HYPRE_SET_CAX(Ap2, xoff2, entries[si + 2]);

            case 2:
               HYPRE_SET_CAX(Ap1, xoff1, entries[si + 1]);

            case 1:
               HYPRE_SET_CAX(Ap0, xoff0, entries[si + 0]);

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

            case 8:
               HYPRE_SET_CAX_TRANS(Ap7, xoff7, entries[si + 7]);

            case 7:
               HYPRE_SET_CAX_TRANS(Ap6, xoff6, entries[si + 6]);

            case 6:
               HYPRE_SET_CAX_TRANS(Ap5, xoff5, entries[si + 5]);

            case 5:
               HYPRE_SET_CAX_TRANS(Ap4, xoff4, entries[si + 4]);

            case 4:
               HYPRE_SET_CAX_TRANS(Ap3, xoff3, entries[si + 3]);

            case 3:
               HYPRE_SET_CAX_TRANS(Ap2, xoff2, entries[si + 2]);

            case 2:
               HYPRE_SET_CAX_TRANS(Ap1, xoff1, entries[si + 1]);

            case 1:
               HYPRE_SET_CAX_TRANS(Ap0, xoff0, entries[si + 0]);

            case 0:
               break;
         }
      }

      switch (depth)
      {
         case 9:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                x_data_box, xdstart, xdstride, xi,
                                y_data_box, ydstart, ydstride, yi);
            {
               yp[yi] -=
                  Ap0[0] * xp[xi + xoff0] +
                  Ap1[0] * xp[xi + xoff1] +
                  Ap2[0] * xp[xi + xoff2] +
                  Ap3[0] * xp[xi + xoff3] +
                  Ap4[0] * xp[xi + xoff4] +
                  Ap5[0] * xp[xi + xoff5] +
                  Ap6[0] * xp[xi + xoff6] +
                  Ap7[0] * xp[xi + xoff7] +
                  Ap8[0] * xp[xi + xoff8];
            }
            hypre_BoxLoop2End(xi, yi);
            break;

         case 8:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                x_data_box, xdstart, xdstride, xi,
                                y_data_box, ydstart, ydstride, yi);

            {
               yp[yi] -=
                  Ap0[0] * xp[xi + xoff0] +
                  Ap1[0] * xp[xi + xoff1] +
                  Ap2[0] * xp[xi + xoff2] +
                  Ap3[0] * xp[xi + xoff3] +
                  Ap4[0] * xp[xi + xoff4] +
                  Ap5[0] * xp[xi + xoff5] +
                  Ap6[0] * xp[xi + xoff6] +
                  Ap7[0] * xp[xi + xoff7];
            }
            hypre_BoxLoop2End(xi, yi);
            break;

         case 7:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                x_data_box, xdstart, xdstride, xi,
                                y_data_box, ydstart, ydstride, yi);
            {
               yp[yi] -=
                  Ap0[0] * xp[xi + xoff0] +
                  Ap1[0] * xp[xi + xoff1] +
                  Ap2[0] * xp[xi + xoff2] +
                  Ap3[0] * xp[xi + xoff3] +
                  Ap4[0] * xp[xi + xoff4] +
                  Ap5[0] * xp[xi + xoff5] +
                  Ap6[0] * xp[xi + xoff6];
            }
            hypre_BoxLoop2End(xi, yi);
            break;

         case 6:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                x_data_box, xdstart, xdstride, xi,
                                y_data_box, ydstart, ydstride, yi);
            {
               yp[yi] -=
                  Ap0[0] * xp[xi + xoff0] +
                  Ap1[0] * xp[xi + xoff1] +
                  Ap2[0] * xp[xi + xoff2] +
                  Ap3[0] * xp[xi + xoff3] +
                  Ap4[0] * xp[xi + xoff4] +
                  Ap5[0] * xp[xi + xoff5];
            }
            hypre_BoxLoop2End(xi, yi);
            break;

         case 5:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                x_data_box, xdstart, xdstride, xi,
                                y_data_box, ydstart, ydstride, yi);
            {
               yp[yi] -=
                  Ap0[0] * xp[xi + xoff0] +
                  Ap1[0] * xp[xi + xoff1] +
                  Ap2[0] * xp[xi + xoff2] +
                  Ap3[0] * xp[xi + xoff3] +
                  Ap4[0] * xp[xi + xoff4];
            }
            hypre_BoxLoop2End(xi, yi);
            break;

         case 4:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                x_data_box, xdstart, xdstride, xi,
                                y_data_box, ydstart, ydstride, yi);
            {
               yp[yi] -=
                  Ap0[0] * xp[xi + xoff0] +
                  Ap1[0] * xp[xi + xoff1] +
                  Ap2[0] * xp[xi + xoff2] +
                  Ap3[0] * xp[xi + xoff3];
            }
            hypre_BoxLoop2End(xi, yi);
            break;

         case 3:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                x_data_box, xdstart, xdstride, xi,
                                y_data_box, ydstart, ydstride, yi);
            {
               yp[yi] -=
                  Ap0[0] * xp[xi + xoff0] +
                  Ap1[0] * xp[xi + xoff1] +
                  Ap2[0] * xp[xi + xoff2];
            }
            hypre_BoxLoop2End(xi, yi);
            break;

         case 2:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                x_data_box, xdstart, xdstride, xi,
                                y_data_box, ydstart, ydstride, yi);
            {
               yp[yi] -=
                  Ap0[0] * xp[xi + xoff0] +
                  Ap1[0] * xp[xi + xoff1];
            }
            hypre_BoxLoop2End(xi, yi);
            break;

         case 1:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                x_data_box, xdstart, xdstride, xi,
                                y_data_box, ydstart, ydstride, yi);
            {
               yp[yi] -=
                  Ap0[0] * xp[xi + xoff0];
            }
            hypre_BoxLoop2End(xi, yi);
            break;

         case 0:
            break;
      } /* switch (depth) */
   } /* for si */
#undef DEVICE_VAR

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * StructMatrix/Vector multiplication core routine for variable coeficients.
 * Note: This function computes -A*x.
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_StructMatvecCompute_core_VC( hypre_StructMatrix *A,
                                   hypre_StructVector *x,
                                   hypre_StructVector *y,
                                   HYPRE_Int           Ab,
                                   HYPRE_Int           xb,
                                   HYPRE_Int           yb,
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
                                   hypre_Box          *A_data_box,
                                   hypre_Box          *x_data_box,
                                   hypre_Box          *y_data_box )
{
   HYPRE_Int             ndim = hypre_StructMatrixNDim(A);
   hypre_StructStencil  *stencil = hypre_StructMatrixStencil(A);
   hypre_Index          *stencil_shape = hypre_StructStencilShape(stencil);

   hypre_Index           Adstart, xdstart, ydstart;
   hypre_Index           offset;

   HYPRE_Complex        *Ap0, *Ap1, *Ap2;
   HYPRE_Complex        *Ap3, *Ap4, *Ap5;
   HYPRE_Complex        *Ap6, *Ap7, *Ap8;
   HYPRE_Complex        *xp,  *yp;
   HYPRE_Int             xoff0, xoff1, xoff2;
   HYPRE_Int             xoff3, xoff4, xoff5;
   HYPRE_Int             xoff6, xoff7, xoff8;

   HYPRE_Int             si, depth;

   HYPRE_ANNOTATE_FUNC_BEGIN;

   xp = hypre_StructVectorBoxData(x, xb);
   yp = hypre_StructVectorBoxData(y, yb);

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

#define DEVICE_VAR is_device_ptr(yp,xp,Ap0,Ap1,Ap2,Ap3,Ap4,Ap5,Ap6,Ap7,Ap8)
   for (si = 0; si < nentries; si += UNROLL_MAXDEPTH)
   {
      depth = hypre_min(UNROLL_MAXDEPTH, (nentries - si));

      if (!transpose)
      {
         switch (depth)
         {
            case 9:
               HYPRE_SET_AX(Ap8, xoff8, entries[si + 8]);

            case 8:
               HYPRE_SET_AX(Ap7, xoff7, entries[si + 7]);

            case 7:
               HYPRE_SET_AX(Ap6, xoff6, entries[si + 6]);

            case 6:
               HYPRE_SET_AX(Ap5, xoff5, entries[si + 5]);

            case 5:
               HYPRE_SET_AX(Ap4, xoff4, entries[si + 4]);

            case 4:
               HYPRE_SET_AX(Ap3, xoff3, entries[si + 3]);

            case 3:
               HYPRE_SET_AX(Ap2, xoff2, entries[si + 2]);

            case 2:
               HYPRE_SET_AX(Ap1, xoff1, entries[si + 1]);

            case 1:
               HYPRE_SET_AX(Ap0, xoff0, entries[si + 0]);

            case 0:
               break;
         }
      }
      else
      {
         switch (depth)
         {
            case 9:
               HYPRE_SET_AX_TRANS(Ap8, xoff8, entries[si + 8]);

            case 8:
               HYPRE_SET_AX_TRANS(Ap7, xoff7, entries[si + 7]);

            case 7:
               HYPRE_SET_AX_TRANS(Ap6, xoff6, entries[si + 6]);

            case 6:
               HYPRE_SET_AX_TRANS(Ap5, xoff5, entries[si + 5]);

            case 5:
               HYPRE_SET_AX_TRANS(Ap4, xoff4, entries[si + 4]);

            case 4:
               HYPRE_SET_AX_TRANS(Ap3, xoff3, entries[si + 3]);

            case 3:
               HYPRE_SET_AX_TRANS(Ap2, xoff2, entries[si + 2]);

            case 2:
               HYPRE_SET_AX_TRANS(Ap1, xoff1, entries[si + 1]);

            case 1:
               HYPRE_SET_AX_TRANS(Ap0, xoff0, entries[si + 0]);

            case 0:
               break;
         }
      }

      switch (depth)
      {
         case 9:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                A_data_box, Adstart, Adstride, Ai,
                                x_data_box, xdstart, xdstride, xi,
                                y_data_box, ydstart, ydstride, yi);
            {
               yp[yi] -=
                  Ap0[Ai] * xp[xi + xoff0] +
                  Ap1[Ai] * xp[xi + xoff1] +
                  Ap2[Ai] * xp[xi + xoff2] +
                  Ap3[Ai] * xp[xi + xoff3] +
                  Ap4[Ai] * xp[xi + xoff4] +
                  Ap5[Ai] * xp[xi + xoff5] +
                  Ap6[Ai] * xp[xi + xoff6] +
                  Ap7[Ai] * xp[xi + xoff7] +
                  Ap8[Ai] * xp[xi + xoff8];
            }
            hypre_BoxLoop3End(Ai, xi, yi);
            break;

         case 8:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                A_data_box, Adstart, Adstride, Ai,
                                x_data_box, xdstart, xdstride, xi,
                                y_data_box, ydstart, ydstride, yi);
            {
               yp[yi] -=
                  Ap0[Ai] * xp[xi + xoff0] +
                  Ap1[Ai] * xp[xi + xoff1] +
                  Ap2[Ai] * xp[xi + xoff2] +
                  Ap3[Ai] * xp[xi + xoff3] +
                  Ap4[Ai] * xp[xi + xoff4] +
                  Ap5[Ai] * xp[xi + xoff5] +
                  Ap6[Ai] * xp[xi + xoff6] +
                  Ap7[Ai] * xp[xi + xoff7];
            }
            hypre_BoxLoop3End(Ai, xi, yi);
            break;

         case 7:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                A_data_box, Adstart, Adstride, Ai,
                                x_data_box, xdstart, xdstride, xi,
                                y_data_box, ydstart, ydstride, yi);
            {
               yp[yi] -=
                  Ap0[Ai] * xp[xi + xoff0] +
                  Ap1[Ai] * xp[xi + xoff1] +
                  Ap2[Ai] * xp[xi + xoff2] +
                  Ap3[Ai] * xp[xi + xoff3] +
                  Ap4[Ai] * xp[xi + xoff4] +
                  Ap5[Ai] * xp[xi + xoff5] +
                  Ap6[Ai] * xp[xi + xoff6];
            }
            hypre_BoxLoop3End(Ai, xi, yi);
            break;

         case 6:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                A_data_box, Adstart, Adstride, Ai,
                                x_data_box, xdstart, xdstride, xi,
                                y_data_box, ydstart, ydstride, yi);
            {
               yp[yi] -=
                  Ap0[Ai] * xp[xi + xoff0] +
                  Ap1[Ai] * xp[xi + xoff1] +
                  Ap2[Ai] * xp[xi + xoff2] +
                  Ap3[Ai] * xp[xi + xoff3] +
                  Ap4[Ai] * xp[xi + xoff4] +
                  Ap5[Ai] * xp[xi + xoff5];
            }
            hypre_BoxLoop3End(Ai, xi, yi);
            break;

         case 5:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                A_data_box, Adstart, Adstride, Ai,
                                x_data_box, xdstart, xdstride, xi,
                                y_data_box, ydstart, ydstride, yi);
            {
               yp[yi] -=
                  Ap0[Ai] * xp[xi + xoff0] +
                  Ap1[Ai] * xp[xi + xoff1] +
                  Ap2[Ai] * xp[xi + xoff2] +
                  Ap3[Ai] * xp[xi + xoff3] +
                  Ap4[Ai] * xp[xi + xoff4];
            }
            hypre_BoxLoop3End(Ai, xi, yi);
            break;

         case 4:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                A_data_box, Adstart, Adstride, Ai,
                                x_data_box, xdstart, xdstride, xi,
                                y_data_box, ydstart, ydstride, yi);
            {
               yp[yi] -=
                  Ap0[Ai] * xp[xi + xoff0] +
                  Ap1[Ai] * xp[xi + xoff1] +
                  Ap2[Ai] * xp[xi + xoff2] +
                  Ap3[Ai] * xp[xi + xoff3];
            }
            hypre_BoxLoop3End(Ai, xi, yi);
            break;

         case 3:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                A_data_box, Adstart, Adstride, Ai,
                                x_data_box, xdstart, xdstride, xi,
                                y_data_box, ydstart, ydstride, yi);
            {
               yp[yi] -=
                  Ap0[Ai] * xp[xi + xoff0] +
                  Ap1[Ai] * xp[xi + xoff1] +
                  Ap2[Ai] * xp[xi + xoff2];
            }
            hypre_BoxLoop3End(Ai, xi, yi);
            break;

         case 2:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                A_data_box, Adstart, Adstride, Ai,
                                x_data_box, xdstart, xdstride, xi,
                                y_data_box, ydstart, ydstride, yi);
            {
               yp[yi] -=
                  Ap0[Ai] * xp[xi + xoff0] +
                  Ap1[Ai] * xp[xi + xoff1];
            }
            hypre_BoxLoop3End(Ai, xi, yi);
            break;

         case 1:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                A_data_box, Adstart, Adstride, Ai,
                                x_data_box, xdstart, xdstride, xi,
                                y_data_box, ydstart, ydstride, yi);
            {
               yp[yi] -=
                  Ap0[Ai] * xp[xi + xoff0];
            }
            hypre_BoxLoop3End(Ai, xi, yi);
            break;

         case 0:
            break;
      } /* switch (depth) */
   } /* for si */
#undef DEVICE_VAR

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

