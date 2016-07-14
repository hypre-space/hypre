/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision$
 ***********************************************************************EHEADER*/

/******************************************************************************
 *
 * Structured copy routine
 *
 *****************************************************************************/

#include "_hypre_struct_mv.h"

/*--------------------------------------------------------------------------
 * hypre_StructCopy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructCopy( hypre_StructVector *x,
                  hypre_StructVector *y     )
{
   hypre_Box       *x_data_box;
   hypre_Box       *y_data_box;
                   
   HYPRE_Int        xi;
   HYPRE_Int        yi;
                   
   HYPRE_Complex   *xp;
   HYPRE_Complex   *yp;
                   
   hypre_BoxArray  *boxes;
   hypre_Box       *box;
   hypre_Index      loop_size;
   hypre_IndexRef   start;
   hypre_Index      unit_stride;
                   
   HYPRE_Int        i;

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

      hypre_BoxLoop2Begin(hypre_StructVectorNDim(x), loop_size,
                          x_data_box, start, unit_stride, xi,
                          y_data_box, start, unit_stride, yi);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,xi,yi) HYPRE_SMP_SCHEDULE
#endif
      hypre_BoxLoop2For(xi, yi)
      {
         yp[yi] = xp[xi];
      }
      hypre_BoxLoop2End(xi, yi);
   }

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 * hypre_StructPartialCopy: copy only the components on a subset of the grid.
 * A BoxArrayArray of boxes are needed- for each box of x, only an array
 * of subboxes (i.e., a boxarray for each box of x) are copied.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructPartialCopy( hypre_StructVector  *x,
                         hypre_StructVector  *y,    
                         hypre_BoxArrayArray *array_boxes )
{
   hypre_Box       *x_data_box;
   hypre_Box       *y_data_box;

   HYPRE_Int        xi;
   HYPRE_Int        yi;

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

         hypre_BoxLoop2Begin(hypre_StructVectorNDim(x), loop_size,
                             x_data_box, start, unit_stride, xi,
                             y_data_box, start, unit_stride, yi);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,xi,yi) HYPRE_SMP_SCHEDULE
#endif
         hypre_BoxLoop2For(xi, yi)
         {
            yp[yi] = xp[xi];
         }
         hypre_BoxLoop2End(xi, yi);
      }
   }

   return hypre_error_flag;
}
