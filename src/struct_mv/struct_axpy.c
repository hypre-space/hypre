/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Structured axpy routine
 *
 *****************************************************************************/

#include "_hypre_struct_mv.h"
#include "_hypre_struct_mv.hpp"

/*--------------------------------------------------------------------------
 * hypre_StructAxpy
 *
 * The vectors x and y may have different base grids, but the grid boxes for
 * each vector (defined by grid, stride, nboxes, boxnums) must be the same.
 * Only nboxes is checked, the rest is assumed to be true.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructAxpy( HYPRE_Complex       alpha,
                  hypre_StructVector *x,
                  hypre_StructVector *y     )
{
   HYPRE_Int         ndim = hypre_StructVectorNDim(x);

   hypre_Box        *x_data_box;
   hypre_Box        *y_data_box;

   HYPRE_Complex    *xp;
   HYPRE_Complex    *yp;

   HYPRE_Int         nboxes;
   hypre_Box        *loop_box;
   hypre_Index       loop_size;
   hypre_IndexRef    start;
   hypre_Index       unit_stride;

   HYPRE_Int         i;

   HYPRE_ANNOTATE_FUNC_BEGIN;

   nboxes = hypre_StructVectorNBoxes(x);

   /* Return if nboxes is not the same for x and y */
   if (nboxes != hypre_StructVectorNBoxes(y))
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "StructAxpy: nboxes for x and y do not match!");

      HYPRE_ANNOTATE_FUNC_END;
      return hypre_error_flag;
   }

   loop_box = hypre_BoxCreate(ndim);
   hypre_SetIndex(unit_stride, 1);

   for (i = 0; i < nboxes; i++)
   {
      hypre_StructVectorGridBoxCopy(x, i, loop_box);
      start = hypre_BoxIMin(loop_box);

      x_data_box = hypre_StructVectorGridDataBox(x, i);
      y_data_box = hypre_StructVectorGridDataBox(y, i);

      xp = hypre_StructVectorGridData(x, i);
      yp = hypre_StructVectorGridData(y, i);

      hypre_BoxGetSize(loop_box, loop_size);

#if 0
      HYPRE_BOXLOOP (
         hypre_BoxLoop2Begin, (ndim, loop_size,
                               x_data_box, start, unit_stride, xi,
                               y_data_box, start, unit_stride, yi),
      {
         yp[yi] += alpha * xp[xi];
      },
      hypre_BoxLoop2End, (xi, yi) )

#else

#define DEVICE_VAR is_device_ptr(yp,xp)
      hypre_BoxLoop2Begin(ndim, loop_size,
                          x_data_box, start, unit_stride, xi,
                          y_data_box, start, unit_stride, yi);
      {
         yp[yi] += alpha * xp[xi];
      }
      hypre_BoxLoop2End(xi, yi);
#undef DEVICE_VAR

#endif
   }

   hypre_BoxDestroy(loop_box);

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructVectorElmdivpy
 *
 * y = alpha*x./z + beta*y
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructVectorElmdivpy( HYPRE_Complex       alpha,
                            hypre_StructVector *x,
                            hypre_StructVector *z,
                            HYPRE_Complex       beta,
                            hypre_StructVector *y )
{
   HYPRE_Int           ndim  = hypre_StructVectorNDim(x);
   hypre_StructGrid   *xgrid = hypre_StructVectorGrid(x);
   hypre_BoxArray     *boxes = hypre_StructGridBoxes(xgrid);

   hypre_Box          *xdbox, *ydbox, *zdbox;
   HYPRE_Complex      *xp, *yp, *zp;

   hypre_Box          *box;
   hypre_Index         loop_size;
   hypre_IndexRef      start;
   hypre_Index         unit_stride;

   HYPRE_Int           i;

   HYPRE_ANNOTATE_FUNC_BEGIN;

   hypre_SetIndex(unit_stride, 1);
   hypre_ForBoxI(i, boxes)
   {
      box   = hypre_BoxArrayBox(boxes, i);
      start = hypre_BoxIMin(box);

      xdbox = hypre_BoxArrayBox(hypre_StructVectorDataSpace(x), i);
      ydbox = hypre_BoxArrayBox(hypre_StructVectorDataSpace(y), i);
      zdbox = hypre_BoxArrayBox(hypre_StructVectorDataSpace(z), i);

      xp = hypre_StructVectorBoxData(x, i);
      yp = hypre_StructVectorBoxData(y, i);
      zp = hypre_StructVectorBoxData(z, i);

      hypre_BoxGetSize(box, loop_size);
#define DEVICE_VAR is_device_ptr(zp, yp,xp)
      hypre_BoxLoop3Begin(ndim, loop_size,
                          xdbox, start, unit_stride, xi,
                          ydbox, start, unit_stride, yi,
                          zdbox, start, unit_stride, zi);
      {
         yp[yi] = alpha * xp[xi] / zp[zi] + beta * yp[yi];
      }
      hypre_BoxLoop3End(xi, yi, zi);
#undef DEVICE_VAR
   }

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

