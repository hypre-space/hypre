/*BHEADER**********************************************************************
 * Copyright (c) 2008,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * This file is part of HYPRE.  See file COPYRIGHT for details.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * $Revision: 2.6 $
 ***********************************************************************EHEADER*/




/******************************************************************************
 *
 * SMG axpy routine
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_SMGAxpy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_SMGAxpy( double              alpha,
               hypre_StructVector *x,
               hypre_StructVector *y,
               hypre_Index         base_index,
               hypre_Index         base_stride )
{
   HYPRE_Int ierr = 0;

   hypre_Box        *x_data_box;
   hypre_Box        *y_data_box;
                 
   HYPRE_Int         xi;
   HYPRE_Int         yi;
                    
   double           *xp;
   double           *yp;
                    
   hypre_BoxArray   *boxes;
   hypre_Box        *box;
   hypre_Index       loop_size;
   hypre_IndexRef    start;
                    
   HYPRE_Int         i;
   HYPRE_Int         loopi, loopj, loopk;

   box = hypre_BoxCreate();
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
         hypre_BoxLoop2Begin(loop_size,
                             x_data_box, start, base_stride, xi,
                             y_data_box, start, base_stride, yi);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,xi,yi
#include "hypre_box_smp_forloop.h"
	 hypre_BoxLoop2For(loopi, loopj, loopk, xi, yi)
            {
               yp[yi] += alpha * xp[xi];
            }
         hypre_BoxLoop2End(xi, yi);
      }
   hypre_BoxDestroy(box);

   return ierr;
}
