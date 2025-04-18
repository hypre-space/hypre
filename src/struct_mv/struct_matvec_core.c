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

#define UNROLL_MAXDEPTH 4

/*--------------------------------------------------------------------------
 * hypre_StructMatvecCompute_core_CC
 *
 * StructMatrix/Vector multiplication core function for constant coeficients.
 *
 * Note: This function computes -A*x.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructMatvecCompute_core_CC( hypre_StructMatrix *A,
                                   hypre_StructVector *x,
                                   hypre_StructVector *y,
                                   HYPRE_Int           Ab,
                                   HYPRE_Int           xb,
                                   HYPRE_Int           yb,
                                   HYPRE_Int           nentries,
                                   HYPRE_Int          *entries,
                                   hypre_Box          *compute_box,
                                   hypre_Box          *x_data_box,
                                   hypre_Box          *y_data_box )
{
   HYPRE_Int             ndim = hypre_StructMatrixNDim(A);
   hypre_StructStencil  *stencil = hypre_StructMatrixStencil(A);
   hypre_Index          *stencil_shape = hypre_StructStencilShape(stencil);

   hypre_Index           loop_size, ustride;
   hypre_IndexRef        start;

   HYPRE_Complex        *Ap0, *Ap1, *Ap2;
   HYPRE_Complex        *Ap3, *Ap4, *Ap5;
   HYPRE_Complex        *Ap6, *Ap7, *Ap8;
   HYPRE_Complex        *xp,  *yp;
   HYPRE_Int             xoff0, xoff1, xoff2;
   HYPRE_Int             xoff3, xoff4, xoff5;
   HYPRE_Int             xoff6, xoff7, xoff8;

   HYPRE_Int             si, depth;

   HYPRE_ANNOTATE_FUNC_BEGIN;

   start = hypre_BoxIMin(compute_box);
   hypre_BoxGetSize(compute_box, loop_size);
   hypre_SetIndex(ustride, 1);
   xp = hypre_StructVectorBoxData(x, xb);
   yp = hypre_StructVectorBoxData(y, yb);

#define DEVICE_VAR is_device_ptr(yp,xp)
   for (si = 0; si < nentries; si += UNROLL_MAXDEPTH)
   {
      depth = hypre_min(UNROLL_MAXDEPTH, (nentries - si));

      switch (depth)
      {
         case 9:
            Ap8 = hypre_StructMatrixBoxData(A, Ab, entries[si + 8]);
            xoff8 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[entries[si + 8]]);

         case 8:
            Ap7 = hypre_StructMatrixBoxData(A, Ab, entries[si + 7]);
            xoff7 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[entries[si + 7]]);

         case 7:
            Ap6 = hypre_StructMatrixBoxData(A, Ab, entries[si + 6]);
            xoff6 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[entries[si + 6]]);

         case 6:
            Ap5 = hypre_StructMatrixBoxData(A, Ab, entries[si + 5]);
            xoff5 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[entries[si + 5]]);

         case 5:
            Ap4 = hypre_StructMatrixBoxData(A, Ab, entries[si + 4]);
            xoff4 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[entries[si + 4]]);

         case 4:
            Ap3 = hypre_StructMatrixBoxData(A, Ab, entries[si + 3]);
            xoff3 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[entries[si + 3]]);

         case 3:
            Ap2 = hypre_StructMatrixBoxData(A, Ab, entries[si + 2]);
            xoff2 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[entries[si + 2]]);

         case 2:
            Ap1 = hypre_StructMatrixBoxData(A, Ab, entries[si + 1]);
            xoff1 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[entries[si + 1]]);

         case 1:
            Ap0 = hypre_StructMatrixBoxData(A, Ab, entries[si + 0]);
            xoff0 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[entries[si + 0]]);

         case 0:
            break;
      }

      switch (depth)
      {
         case 9:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                x_data_box, start, ustride, xi,
                                y_data_box, start, ustride, yi);
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
                                x_data_box, start, ustride, xi,
                                y_data_box, start, ustride, yi);

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
                                x_data_box, start, ustride, xi,
                                y_data_box, start, ustride, yi);
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
                                x_data_box, start, ustride, xi,
                                y_data_box, start, ustride, yi);
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
                                x_data_box, start, ustride, xi,
                                y_data_box, start, ustride, yi);
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
                                x_data_box, start, ustride, xi,
                                y_data_box, start, ustride, yi);
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
                                x_data_box, start, ustride, xi,
                                y_data_box, start, ustride, yi);
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
                                x_data_box, start, ustride, xi,
                                y_data_box, start, ustride, yi);
            {
               yp[yi] -=
                  Ap0[0] * xp[xi + xoff0] +
                  Ap1[0] * xp[xi + xoff1];
            }
            hypre_BoxLoop2End(xi, yi);
            break;

         case 1:
            hypre_BoxLoop2Begin(ndim, loop_size,
                                x_data_box, start, ustride, xi,
                                y_data_box, start, ustride, yi);
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
 * hypre_StructMatvecCompute_core_VC
 *
 * StructMatrix/Vector multiplication core routine for variable coeficients.
 *
 * Note: This function computes -A*x.
 *--------------------------------------------------------------------------*/
HYPRE_Int
hypre_StructMatvecCompute_core_VC( hypre_StructMatrix *A,
                                   hypre_StructVector *x,
                                   hypre_StructVector *y,
                                   HYPRE_Int           Ab,
                                   HYPRE_Int           xb,
                                   HYPRE_Int           yb,
                                   HYPRE_Int           nentries,
                                   HYPRE_Int          *entries,
                                   hypre_Box          *compute_box,
                                   hypre_Box          *A_data_box,
                                   hypre_Box          *x_data_box,
                                   hypre_Box          *y_data_box )
{
   HYPRE_Int             ndim = hypre_StructMatrixNDim(A);
   hypre_StructStencil  *stencil = hypre_StructMatrixStencil(A);
   hypre_Index          *stencil_shape = hypre_StructStencilShape(stencil);

   hypre_Index           loop_size, ustride;
   hypre_IndexRef        start;

   HYPRE_Complex        *Ap0, *Ap1, *Ap2;
   HYPRE_Complex        *Ap3, *Ap4, *Ap5;
   HYPRE_Complex        *Ap6, *Ap7, *Ap8;
   HYPRE_Complex        *xp,  *yp;
   HYPRE_Int             xoff0, xoff1, xoff2;
   HYPRE_Int             xoff3, xoff4, xoff5;
   HYPRE_Int             xoff6, xoff7, xoff8;

   HYPRE_Int             si, depth;

   HYPRE_ANNOTATE_FUNC_BEGIN;

   start = hypre_BoxIMin(compute_box);
   hypre_BoxGetSize(compute_box, loop_size);
   hypre_SetIndex(ustride, 1);
   xp = hypre_StructVectorBoxData(x, xb);
   yp = hypre_StructVectorBoxData(y, yb);

#define DEVICE_VAR is_device_ptr(yp,xp,Ap0,Ap1,Ap2,Ap3,Ap4,Ap5,Ap6,Ap7,Ap8)
   for (si = 0; si < nentries; si += UNROLL_MAXDEPTH)
   {
      depth = hypre_min(UNROLL_MAXDEPTH, (nentries - si));

      switch (depth)
      {
         case 9:
            Ap8 = hypre_StructMatrixBoxData(A, Ab, entries[si + 8]);
            xoff8 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[entries[si + 8]]);

         case 8:
            Ap7 = hypre_StructMatrixBoxData(A, Ab, entries[si + 7]);
            xoff7 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[entries[si + 7]]);

         case 7:
            Ap6 = hypre_StructMatrixBoxData(A, Ab, entries[si + 6]);
            xoff6 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[entries[si + 6]]);

         case 6:
            Ap5 = hypre_StructMatrixBoxData(A, Ab, entries[si + 5]);
            xoff5 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[entries[si + 5]]);

         case 5:
            Ap4 = hypre_StructMatrixBoxData(A, Ab, entries[si + 4]);
            xoff4 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[entries[si + 4]]);

         case 4:
            Ap3 = hypre_StructMatrixBoxData(A, Ab, entries[si + 3]);
            xoff3 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[entries[si + 3]]);

         case 3:
            Ap2 = hypre_StructMatrixBoxData(A, Ab, entries[si + 2]);
            xoff2 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[entries[si + 2]]);

         case 2:
            Ap1 = hypre_StructMatrixBoxData(A, Ab, entries[si + 1]);
            xoff1 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[entries[si + 1]]);

         case 1:
            Ap0 = hypre_StructMatrixBoxData(A, Ab, entries[si + 0]);
            xoff0 = hypre_BoxOffsetDistance(x_data_box, stencil_shape[entries[si + 0]]);

         case 0:
            break;
      }

      switch (depth)
      {
         case 9:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                A_data_box, start, ustride, Ai,
                                x_data_box, start, ustride, xi,
                                y_data_box, start, ustride, yi);
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
                                A_data_box, start, ustride, Ai,
                                x_data_box, start, ustride, xi,
                                y_data_box, start, ustride, yi);
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
                                A_data_box, start, ustride, Ai,
                                x_data_box, start, ustride, xi,
                                y_data_box, start, ustride, yi);
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
                                A_data_box, start, ustride, Ai,
                                x_data_box, start, ustride, xi,
                                y_data_box, start, ustride, yi);
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
                                A_data_box, start, ustride, Ai,
                                x_data_box, start, ustride, xi,
                                y_data_box, start, ustride, yi);
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
                                A_data_box, start, ustride, Ai,
                                x_data_box, start, ustride, xi,
                                y_data_box, start, ustride, yi);
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
                                A_data_box, start, ustride, Ai,
                                x_data_box, start, ustride, xi,
                                y_data_box, start, ustride, yi);
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
                                A_data_box, start, ustride, Ai,
                                x_data_box, start, ustride, xi,
                                y_data_box, start, ustride, yi);
            {
               yp[yi] -=
                  Ap0[Ai] * xp[xi + xoff0] +
                  Ap1[Ai] * xp[xi + xoff1];
            }
            hypre_BoxLoop3End(Ai, xi, yi);
            break;

         case 1:
            hypre_BoxLoop3Begin(ndim, loop_size,
                                A_data_box, start, ustride, Ai,
                                x_data_box, start, ustride, xi,
                                y_data_box, start, ustride, yi);
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

