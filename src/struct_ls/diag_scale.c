/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_struct_ls.h"
#include "_hypre_struct_mv.hpp"

/*--------------------------------------------------------------------------
 * x = D^{-1} y
 *
 * RDF TODO: This is partially updated to support non-unitary strides.
 * Need to fix the matrix part of the code.
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructDiagScale( hypre_StructMatrix   *A,
                       hypre_StructVector   *y,
                       hypre_StructVector   *x )
{
   HYPRE_Int             ndim = hypre_StructVectorNDim(x);

   hypre_Box            *A_data_box;
   hypre_Box            *y_data_box;
   hypre_Box            *x_data_box;

   HYPRE_Real           *Ap;
   HYPRE_Real           *yp;
   HYPRE_Real           *xp;

   hypre_Index           index;

   HYPRE_Int             nboxes;
   hypre_Box            *loop_box;
   hypre_Index           loop_size;
   hypre_IndexRef        start;
   hypre_Index           ustride;

   HYPRE_Int             i;

   nboxes = hypre_StructVectorNBoxes(x);

   loop_box = hypre_BoxCreate(ndim);
   hypre_SetIndex(ustride, 1);

   for (i = 0; i < nboxes; i++)
   {
      hypre_StructVectorGridBoxCopy(x, i, loop_box);
      start = hypre_BoxIMin(loop_box);

      A_data_box = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), i);
      x_data_box = hypre_StructVectorGridDataBox(x, i);
      y_data_box = hypre_StructVectorGridDataBox(y, i);

      hypre_SetIndex(index, 0);
      Ap = hypre_StructMatrixExtractPointerByIndex(A, i, index);
      xp = hypre_StructVectorGridData(x, i);
      yp = hypre_StructVectorGridData(y, i);

      hypre_BoxGetSize(loop_box, loop_size);

#define DEVICE_VAR is_device_ptr(xp,yp,Ap)
      hypre_BoxLoop3Begin(ndim, loop_size,
                          A_data_box, start, ustride, Ai,
                          x_data_box, start, ustride, xi,
                          y_data_box, start, ustride, yi);
      {
         xp[xi] = yp[yi] / Ap[Ai];
      }
      hypre_BoxLoop3End(Ai, xi, yi);
#undef DEVICE_VAR
   }

   hypre_BoxDestroy(loop_box);

   return hypre_error_flag;
}
