/******************************************************************************
 * Copyright 1998-2019 Lawrence Livermore National Security, LLC and other
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
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructAxpy( HYPRE_Complex       alpha,
                  hypre_StructVector *x,
                  hypre_StructVector *y     )
{
   hypre_Box        *x_data_box;
   hypre_Box        *y_data_box;

   HYPRE_Complex    *xp;
   HYPRE_Complex    *yp;

   hypre_BoxArray   *boxes;
   hypre_Box        *box;
   hypre_Index       loop_size;
   hypre_IndexRef    start;
   hypre_Index       unit_stride;

   HYPRE_Int         i;

   HYPRE_ANNOTATE_FUNC_BEGIN;

   hypre_SetIndex(unit_stride, 1);

   boxes = hypre_StructGridBoxes(hypre_StructVectorGrid(y));
   hypre_ForBoxI(i, boxes)
   {
      box   = hypre_BoxArrayBox(boxes, i);
      start = hypre_BoxIMin(box);

      x_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(x), i);
      y_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(y), i);

      xp = hypre_StructVectorBoxData(x, i);
      yp = hypre_StructVectorBoxData(y, i);

      hypre_BoxGetSize(box, loop_size);

#define DEVICE_VAR is_device_ptr(yp,xp)
      hypre_BoxLoop2Begin(hypre_StructVectorNDim(x), loop_size,
                          x_data_box, start, unit_stride, xi,
                          y_data_box, start, unit_stride, yi);
      {
         yp[yi] += alpha * xp[xi];
      }
      hypre_BoxLoop2End(xi, yi);
#undef DEVICE_VAR
   }

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
      hypre_BoxLoop3End(xi,yi,zi);
#undef DEVICE_VAR
   }

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}
