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
 * Structured inner product routine
 *
 *****************************************************************************/

#include "_hypre_struct_mv.h"

/*--------------------------------------------------------------------------
 * hypre_StructInnerProd
 *--------------------------------------------------------------------------*/

HYPRE_Real
hypre_StructInnerProd( hypre_StructVector *x,
                       hypre_StructVector *y )
{
   HYPRE_Real       final_innerprod_result;
   HYPRE_Real       process_result;
                   
   hypre_Box       *x_data_box;
   hypre_Box       *y_data_box;
                   
   HYPRE_Complex   *xp;
   HYPRE_Complex   *yp;
                   
   hypre_BoxArray  *boxes;
   hypre_Box       *box;
   hypre_Index      loop_size;
   hypre_IndexRef   start;
   hypre_Index      unit_stride;
    
   HYPRE_Int         ndim = hypre_StructVectorNDim(x);               
   HYPRE_Int        i;

   hypre_Reductioninit(local_result);
   
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

#ifdef HYPRE_BOX_PRIVATE_VAR
#undef HYPRE_BOX_PRIVATE_VAR
#endif
#define HYPRE_BOX_PRIVATE_VAR xi,yi
#ifdef HYPRE_BOX_REDUCTION
#undef HYPRE_BOX_REDUCTION
#endif
#define HYPRE_BOX_REDUCTION reduction(+:local_result)
      hypre_newBoxLoop2ReductionBegin(ndim, loop_size,
                                      x_data_box, start, unit_stride, xi,
                                      y_data_box, start, unit_stride, yi,local_result);
      {
         local_result += xp[xi] * hypre_conj(yp[yi]);		 
      }
      hypre_newBoxLoop2ReductionEnd(xi, yi, local_result);
   }
   process_result = local_result;
   
   hypre_MPI_Allreduce(&process_result, &final_innerprod_result, 1,
                       HYPRE_MPI_REAL, hypre_MPI_SUM, hypre_StructVectorComm(x));

   hypre_IncFLOPCount(2*hypre_StructVectorGlobalSize(x));

   return final_innerprod_result;
}
