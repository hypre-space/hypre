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
 * z = alpha * x + beta * y
 *
 * The vectors x, y, and z may have different base grids, but the grid boxes
 * for each vector (defined by grid, stride, nboxes, boxnums) must be the same.
 * Only nboxes is checked, the rest is assumed to be true.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructVectorAxpy( HYPRE_Complex       alpha,
                        hypre_StructVector *x,
                        HYPRE_Complex       beta,
                        hypre_StructVector *y,
                        hypre_StructVector *z )
{
   HYPRE_Int         ndim   = hypre_StructVectorNDim(x);
   HYPRE_Int         nboxes = hypre_StructVectorNBoxes(x);

   hypre_Box        *x_data_box;
   hypre_Box        *y_data_box;
   hypre_Box        *z_data_box;

   HYPRE_Complex    *xp;
   HYPRE_Complex    *yp;
   HYPRE_Complex    *zp;

   hypre_Box        *loop_box;
   hypre_Index       loop_size;
   hypre_IndexRef    start;
   hypre_Index       ustride;

   HYPRE_Int         i;

   /* Return if x, y, or z do not have the same numbers of boxes */
   if (hypre_StructVectorNBoxes(x) != hypre_StructVectorNBoxes(y) ||
       hypre_StructVectorNBoxes(x) != hypre_StructVectorNBoxes(z))
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "nboxes for x, y or z do not match!");
      return hypre_error_flag;
   }

   HYPRE_ANNOTATE_FUNC_BEGIN;
   hypre_GpuProfilingPushRange("Axpy");

   loop_box = hypre_BoxCreate(ndim);
   hypre_SetIndex(ustride, 1);
   for (i = 0; i < nboxes; i++)
   {
      hypre_StructVectorGridBoxCopy(x, i, loop_box);
      start = hypre_BoxIMin(loop_box);

      x_data_box = hypre_StructVectorGridDataBox(x, i);
      y_data_box = hypre_StructVectorGridDataBox(y, i);
      z_data_box = hypre_StructVectorGridDataBox(z, i);

      xp = hypre_StructVectorGridData(x, i);
      yp = hypre_StructVectorGridData(y, i);
      zp = hypre_StructVectorGridData(z, i);

      hypre_BoxGetSize(loop_box, loop_size);

#define DEVICE_VAR is_device_ptr(xp,yp,zp)
      if (hypre_BoxesEqual(x_data_box, y_data_box) &&
          hypre_BoxesEqual(x_data_box, z_data_box))
      {
         if (alpha != 0.0 && beta != 0.0)
         {
            hypre_BoxLoop1Begin(ndim, loop_size, x_data_box, start, ustride, xi)
            {
               zp[xi] = alpha * xp[xi] + beta * yp[xi];
            }
            hypre_BoxLoop1End(xi);
         }
         else if (alpha != 0.0 && beta == 0.0)
         {
            hypre_BoxLoop1Begin(ndim, loop_size, x_data_box, start, ustride, xi)
            {
               zp[xi] = alpha * xp[xi];
            }
            hypre_BoxLoop1End(xi);
         }
         else if (alpha == 0.0 && beta != 0.0)
         {
            hypre_BoxLoop1Begin(ndim, loop_size, x_data_box, start, ustride, xi)
            {
               zp[xi] = beta * yp[xi];
            }
            hypre_BoxLoop1End(xi);
         }
         else if (alpha == 0.0 && beta == 0.0)
         {
            hypre_BoxLoop1Begin(ndim, loop_size, x_data_box, start, ustride, xi)
            {
               zp[xi] = 0.0;
            }
            hypre_BoxLoop1End(xi);
         }
      }
      else
      {
         hypre_BoxLoop3Begin(ndim, loop_size,
                             x_data_box, start, ustride, xi,
                             y_data_box, start, ustride, yi,
                             z_data_box, start, ustride, zi)
         {
            zp[zi] = alpha * xp[xi] + beta * yp[yi];
         }
         hypre_BoxLoop3End(xi, yi, zi);
      }
#undef DEVICE_VAR
   }

   hypre_BoxDestroy(loop_box);

   hypre_GpuProfilingPopRange();
   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * y = alpha*x./z + beta*y
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructVectorPointwiseDivpy( HYPRE_Complex       alpha,
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
   hypre_Index         ustride;

   HYPRE_Int           i;

   HYPRE_ANNOTATE_FUNC_BEGIN;

   hypre_SetIndex(ustride, 1);
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
                          xdbox, start, ustride, xi,
                          ydbox, start, ustride, yi,
                          zdbox, start, ustride, zi);
      {
         yp[yi] = alpha * xp[xi] / zp[zi] + beta * yp[yi];
      }
      hypre_BoxLoop3End(xi, yi, zi);
#undef DEVICE_VAR
   }

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructVectorPointwiseDivision( hypre_StructVector  *x,
                                     hypre_StructVector  *y,
                                     hypre_StructVector **z_ptr )
{
   HYPRE_UNUSED_VAR(x);
   HYPRE_UNUSED_VAR(y);
   HYPRE_UNUSED_VAR(z_ptr);

   /* Not implemented yet */

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructVectorPointwiseProduct( hypre_StructVector  *x,
                                    hypre_StructVector  *y,
                                    hypre_StructVector **z_ptr )
{
   HYPRE_UNUSED_VAR(x);
   HYPRE_UNUSED_VAR(y);
   HYPRE_UNUSED_VAR(z_ptr);

   /* Not implemented yet */

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructVectorPointwiseInverse( hypre_StructVector  *x,
                                    hypre_StructVector **y_ptr )
{
   HYPRE_UNUSED_VAR(x);
   HYPRE_UNUSED_VAR(y_ptr);

   /* Not implemented yet */

   return hypre_error_flag;
}
