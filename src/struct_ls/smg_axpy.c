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

#include "_hypre_struct_ls.h"

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
                 
   HYPRE_Int         xi;
   HYPRE_Int         yi;
                    
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
      hypre_BoxLoop2Begin(hypre_StructVectorNDim(x), loop_size,
                          x_data_box, start, base_stride, xi,
                          y_data_box, start, base_stride, yi);
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(HYPRE_BOX_PRIVATE,xi,yi) HYPRE_SMP_SCHEDULE
#endif
      hypre_BoxLoop2For(xi, yi)
      {
         yp[yi] += alpha * xp[xi];
      }
      hypre_BoxLoop2End(xi, yi);
   }
   hypre_BoxDestroy(box);

   return hypre_error_flag;
}
