/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_struct_ls.h"
#include "_hypre_struct_mv.hpp"

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SMGAxpy( HYPRE_Real          alpha,
               hypre_StructVector *x,
               hypre_StructVector *y,
               hypre_Index         base_index,
               hypre_Index         base_stride )
{
   HYPRE_Int         ndim = hypre_StructVectorNDim(x);
   hypre_Box        *x_data_box;
   hypre_Box        *y_data_box;

   HYPRE_Real       *xp;
   HYPRE_Real       *yp;

   hypre_BoxArray   *boxes;
   hypre_Box        *box;
   hypre_Index       loop_size;
   hypre_IndexRef    start;

   HYPRE_Int         i;

   box = hypre_BoxCreate(ndim);
   boxes = hypre_StructGridBoxes(hypre_StructVectorGrid(y));
   hypre_ForBoxI(i, boxes)
   {
      hypre_CopyBox(hypre_BoxArrayBox(boxes, i), box);
      hypre_ProjectBox(box, base_index, base_stride);
      start = hypre_BoxIMin(box);

      x_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(x), i);
      y_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(y), i);

      xp = hypre_StructVectorBoxData(x, i);
      yp = hypre_StructVectorBoxData(y, i);

      hypre_BoxGetStrideSize(box, base_stride, loop_size);

#define DEVICE_VAR is_device_ptr(yp,xp)
      hypre_BoxLoop2Begin(hypre_StructVectorNDim(x), loop_size,
                          x_data_box, start, base_stride, xi,
                          y_data_box, start, base_stride, yi);
      {
         yp[yi] += alpha * xp[xi];
      }
      hypre_BoxLoop2End(xi, yi);
#undef DEVICE_VAR
   }
   hypre_BoxDestroy(box);

   return hypre_error_flag;
}
