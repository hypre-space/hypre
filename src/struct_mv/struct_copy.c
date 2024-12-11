/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Structured copy routine
 *
 * RDF TODO: The names for the vector class functions needs to be revisited.
 * Should this be hypre_StructVectorCopy?  What should our user interface
 * conventions be for these routines?
 *****************************************************************************/

#include "_hypre_struct_mv.h"
#include "_hypre_struct_mv.hpp"

/*--------------------------------------------------------------------------
 * hypre_StructCopy
 *
 * The vectors x and y may have different base grids, but the grid boxes for
 * each vector (defined by grid, stride, nboxes, boxnums) must be the same.
 * Only nboxes is checked, the rest is assumed to be true.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructCopy( hypre_StructVector *x,
                  hypre_StructVector *y     )
{
   HYPRE_Int        ndim = hypre_StructVectorNDim(x);

   hypre_Box       *x_data_box;
   hypre_Box       *y_data_box;

   HYPRE_Complex   *xp;
   HYPRE_Complex   *yp;

   HYPRE_Int        nboxes;
   hypre_Box       *loop_box;
   hypre_Index      loop_size;
   hypre_IndexRef   start;
   hypre_Index      unit_stride;

   HYPRE_Int        i;

   HYPRE_ANNOTATE_FUNC_BEGIN;

   nboxes = hypre_StructVectorNBoxes(x);

   /* Return if nboxes is not the same for x and y */
   if (nboxes != hypre_StructVectorNBoxes(y))
   {
      hypre_error_w_msg(HYPRE_ERROR_GENERIC, "StructCopy: nboxes for x and y do not match!");

      HYPRE_ANNOTATE_FUNC_END;
      return hypre_error_flag;
   }

   /* Return if x and y point to the same hypre_StructVector */
   if (x == y)
   {
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

#define DEVICE_VAR is_device_ptr(yp,xp)
      hypre_BoxLoop2Begin(hypre_StructVectorNDim(x), loop_size,
                          x_data_box, start, unit_stride, xi,
                          y_data_box, start, unit_stride, yi);
      {
         yp[yi] = xp[xi];
      }
      hypre_BoxLoop2End(xi, yi);
#undef DEVICE_VAR
   }

   hypre_BoxDestroy(loop_box);

   HYPRE_ANNOTATE_FUNC_END;

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructPartialCopy: copy only the components on a subset of the grid.
 * A BoxArrayArray of boxes are needed- for each box of x, only an array
 * of subboxes (i.e., a boxarray for each box of x) are copied.
 *
 * RDF TODO: Remove this routine and the SStruct routine that calls it.  Need to
 * remove the FAC solver as well (it's the only thing that uses it).
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructPartialCopy( hypre_StructVector  *x,
                         hypre_StructVector  *y,
                         hypre_BoxArrayArray *array_boxes )
{
   hypre_Box       *x_data_box;
   hypre_Box       *y_data_box;

   HYPRE_Complex   *xp;
   HYPRE_Complex   *yp;

   hypre_BoxArray  *boxes;
   hypre_Box       *box;
   hypre_Index      loop_size;
   hypre_IndexRef   start;
   hypre_Index      unit_stride;

   HYPRE_Int        i, j ;

   hypre_SetIndex(unit_stride, 1);

   hypre_ForBoxArrayI(i, array_boxes)
   {
      boxes = hypre_BoxArrayArrayBoxArray(array_boxes, i);

      x_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(x), i);
      y_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(y), i);

      xp = hypre_StructVectorBoxData(x, i);
      yp = hypre_StructVectorBoxData(y, i);

      /* array of sub_boxes of box_i of the vector */
      hypre_ForBoxI(j, boxes)
      {
         box = hypre_BoxArrayBox(boxes, j);

         start = hypre_BoxIMin(box);
         hypre_BoxGetSize(box, loop_size);

#define DEVICE_VAR is_device_ptr(yp,xp)
         hypre_BoxLoop2Begin(hypre_StructVectorNDim(x), loop_size,
                             x_data_box, start, unit_stride, xi,
                             y_data_box, start, unit_stride, yi);
         {
            yp[yi] = xp[xi];
         }
         hypre_BoxLoop2End(xi, yi);
#undef DEVICE_VAR
      }
   }

   return hypre_error_flag;
}
