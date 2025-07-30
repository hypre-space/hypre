/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_struct_ls.h"
#include "_hypre_struct_mv.hpp"

/*==========================================================================*/

HYPRE_Int
hypre_StructDiagScale( hypre_StructMatrix   *A,
                       hypre_StructVector   *y,
                       hypre_StructVector   *x )
{
   hypre_BoxArray       *boxes;
   hypre_Box            *box;

   hypre_Box            *A_data_box;
   hypre_Box            *y_data_box;
   hypre_Box            *x_data_box;

   HYPRE_Real           *Ap;
   HYPRE_Real           *yp;
   HYPRE_Real           *xp;

   hypre_Index           index;
   hypre_IndexRef        start;
   hypre_Index           stride;
   hypre_Index           loop_size;

   HYPRE_Int             i;

   /* x = D^{-1} y */
   hypre_SetIndex(stride, 1);
   boxes = hypre_StructGridBoxes(hypre_StructMatrixGrid(A));
   hypre_ForBoxI(i, boxes)
   {
      box = hypre_BoxArrayBox(boxes, i);

      A_data_box = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), i);
      x_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(x), i);
      y_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(y), i);

      hypre_SetIndex(index, 0);
      Ap = hypre_StructMatrixExtractPointerByIndex(A, i, index);
      xp = hypre_StructVectorBoxData(x, i);
      yp = hypre_StructVectorBoxData(y, i);

      start  = hypre_BoxIMin(box);

      hypre_BoxGetSize(box, loop_size);

#define DEVICE_VAR is_device_ptr(xp,yp,Ap)
      hypre_BoxLoop3Begin(hypre_StructVectorNDim(x), loop_size,
                          A_data_box, start, stride, Ai,
                          x_data_box, start, stride, xi,
                          y_data_box, start, stride, yi);
      {
         xp[xi] = yp[yi] / Ap[Ai];
      }
      hypre_BoxLoop3End(Ai, xi, yi);
#undef DEVICE_VAR
   }

   return hypre_error_flag;
}
