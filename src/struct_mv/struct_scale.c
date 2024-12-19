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
   HYPRE_Int        ndim = hypre_StructVectorNDim(y);

   hypre_Box       *y_data_box;

   HYPRE_Complex   *yp;

   HYPRE_Int        nboxes;
   hypre_Box       *loop_box;
   hypre_Index      loop_size;
   hypre_IndexRef   start;
   hypre_Index      ustride;

   HYPRE_Int        i;

   /* If alpha is 1.0, y does not change */
   if (alpha == 1.0)
   {
      return hypre_error_flag;
   }

   nboxes = hypre_StructVectorNBoxes(y);

   loop_box = hypre_BoxCreate(ndim);
   hypre_SetIndex(ustride, 1);

   for (i = 0; i < nboxes; i++)
   {
      hypre_StructVectorGridBoxCopy(y, i, loop_box);
      start = hypre_BoxIMin(loop_box);

      y_data_box = hypre_StructVectorGridDataBox(y, i);
      yp = hypre_StructVectorGridData(y, i);

      hypre_BoxGetSize(loop_box, loop_size);

#define DEVICE_VAR is_device_ptr(yp)
      hypre_BoxLoop1Begin(hypre_StructVectorNDim(y), loop_size,
                          y_data_box, start, ustride, yi);
      {
         yp[yi] *= alpha;
      }
      hypre_BoxLoop1End(yi);
#undef DEVICE_VAR
   }

   hypre_BoxDestroy(loop_box);

   return hypre_error_flag;
}
