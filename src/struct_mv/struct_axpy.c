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
 * Structured axpy routine
 *
 *****************************************************************************/

#include "_hypre_struct_mv.h"

/*--------------------------------------------------------------------------
 * hypre_StructAxpy
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_StructAxpy( HYPRE_Complex       alpha,
                  hypre_StructVector *x,
                  hypre_StructVector *y     )
{
   hypre_Box        *x_data_box;
   hypre_Box        *y_data_box;
                 
   HYPRE_Int         xi;
   HYPRE_Int         yi;
                    
   HYPRE_Complex    *xp;
   HYPRE_Complex    *yp;
                    
   hypre_BoxArray   *boxes;
   hypre_Box        *box;
 
   HYPRE_Int        *loop_size, *stride, * start;
   HYPRE_Int         ndim = hypre_StructVectorNDim(x);
   
   HYPRE_Int         i,d;

   /* Allocate data for boxloop */
   hypre_DataCTAlloc(loop_size,HYPRE_Int,ndim);
   hypre_DataCTAlloc(   stride,HYPRE_Int,ndim);
   hypre_DataCTAlloc(    start,HYPRE_Int,ndim);
   x_data_box = hypre_BoxCreate(ndim);
   y_data_box = hypre_BoxCreate(ndim);
   
   hypre_SetIndex(stride, 1);

   boxes = hypre_StructGridBoxes(hypre_StructVectorGrid(y));
   hypre_ForBoxI(i, boxes)
   {
      box   = hypre_BoxArrayBox(boxes, i);
      //start = hypre_BoxIMin(box);

      //x_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(x), i);
      //y_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(y), i);
	  for (d = 0; d < ndim; d++)
	  {
		  hypre_IndexD(start, d) = hypre_BoxIMinD(box, d);
		  hypre_IndexD(loop_size, d) = hypre_BoxSizeD(box, d);
	  }
	  
	  hypre_CopyBox(hypre_BoxArrayBox(hypre_StructVectorDataSpace(x), i), x_data_box);
	  hypre_CopyBox(hypre_BoxArrayBox(hypre_StructVectorDataSpace(y), i), y_data_box);

	  xp = hypre_StructVectorBoxData(x, i);
      yp = hypre_StructVectorBoxData(y, i);

#ifdef HYPRE_BOX_PRIVATE_VAR
#undef HYPRE_BOX_PRIVATE_VAR
#endif
#define HYPRE_BOX_PRIVATE_VAR xi,yi
	  zypre_newBoxLoop2Begin(ndim, loop_size,
		                     x_data_box, start, stride, xi,
		                     y_data_box, start, stride, yi);
      {
         yp[yi] += alpha * xp[xi];
      }
      zypre_newBoxLoop2End(xi,yi);
   }

   hypre_BoxDestroy(x_data_box);
   hypre_BoxDestroy(y_data_box);
   hypre_DataTFree(loop_size);
   hypre_DataTFree(   stride);
   hypre_DataTFree(    start);
   
   return hypre_error_flag;
}
