/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Structured scale routine
 *
 *****************************************************************************/

#include "_hypre_struct_mv.h"
#include "_hypre_struct_mv.hpp"

/*--------------------------------------------------------------------------
 * hypre_StructScale
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructScale( HYPRE_Complex       alpha,
                   hypre_StructVector *y     )
{
   hypre_Box       *y_data_box;

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

      y_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(y), i);
      yp = hypre_StructVectorBoxData(y, i);

      hypre_BoxGetSize(box, loop_size);

#define DEVICE_VAR is_device_ptr(yp)
      hypre_BoxLoop1Begin(hypre_StructVectorNDim(y), loop_size,
                          y_data_box, start, unit_stride, yi);
      {
         yp[yi] *= alpha;
      }
      hypre_BoxLoop1End(yi);
#undef DEVICE_VAR
   }

   return hypre_error_flag;
}
