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
 * Structured scale routine
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * hypre_StructScale
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructScale( double              alpha,
                   hypre_StructVector *y     )
{
   HYPRE_Int ierr = 0;

   hypre_Box       *y_data_box;
                   
   HYPRE_Int        yi;
   double          *yp;
                   
   hypre_BoxArray  *boxes;
   hypre_Box       *box;
   hypre_Index      loop_size;
   hypre_IndexRef   start;
   hypre_Index      unit_stride;
                   
   HYPRE_Int        i;
   HYPRE_Int        loopi, loopj, loopk;

   hypre_SetIndex(unit_stride, 1, 1, 1);

   boxes = hypre_StructGridBoxes(hypre_StructVectorGrid(y));
   hypre_ForBoxI(i, boxes)
      {
         box   = hypre_BoxArrayBox(boxes, i);
         start = hypre_BoxIMin(box);

         y_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(y), i);
         yp = hypre_StructVectorBoxData(y, i);

         hypre_BoxGetSize(box, loop_size);

	 hypre_BoxLoop1Begin(loop_size,
                             y_data_box, start, unit_stride, yi);
#define HYPRE_BOX_SMP_PRIVATE loopk,loopi,loopj,yi
#include "hypre_box_smp_forloop.h"
	 hypre_BoxLoop1For(loopi, loopj, loopk, yi)
            {
               yp[yi] *= alpha;
            }
	 hypre_BoxLoop1End(yi);
      }

   return ierr;
}
