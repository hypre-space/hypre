/*BHEADER**********************************************************************
 * (c) 1997   The Regents of the University of California
 *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright
 * notice, contact person, and disclaimer.
 *
 * $Revision$
 *********************************************************************EHEADER*/
/******************************************************************************
 *
 * Structured inner product routine
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * zzz_StructInnerProd
 *--------------------------------------------------------------------------*/

double
zzz_StructInnerProd(  zzz_StructVector *x,
                      zzz_StructVector *y )
{
   double                result;
   double                local_result;

   zzz_Box              *x_data_box;
   zzz_Box              *y_data_box;

   int                   xi;
   int                   yi;

   double               *xp;
   double               *yp;

   zzz_BoxArray         *boxes;
   zzz_Box              *box;
   zzz_Index             loop_size;
   zzz_IndexRef          start;
   zzz_Index             unit_stride;

   int                   i;
   int                   loopi, loopj, loopk;

   local_result = 0.0;

   zzz_SetIndex(unit_stride, 1, 1, 1);

   boxes = zzz_StructGridBoxes(zzz_StructVectorGrid(y));
   zzz_ForBoxI(i, boxes)
   {
      box   = zzz_BoxArrayBox(boxes, i);
      start = zzz_BoxIMin(box);

      x_data_box = zzz_BoxArrayBox(zzz_StructVectorDataSpace(x), i);
      y_data_box = zzz_BoxArrayBox(zzz_StructVectorDataSpace(y), i);

      xp = zzz_StructVectorBoxData(x, i);
      yp = zzz_StructVectorBoxData(y, i);

      zzz_GetBoxSize(box, loop_size);
      zzz_BoxLoop2(loopi, loopj, loopk, loop_size,
                   x_data_box, start, unit_stride, xi,
                   y_data_box, start, unit_stride, yi,
                   {
                      local_result += xp[xi] * yp[yi];
                   });
   }

   MPI_Allreduce(&local_result, &result, 1,
                 MPI_DOUBLE, MPI_SUM, *zzz_StructVectorComm(x));

   zzz_IncFLOPCount(2*zzz_StructVectorGlobalSize(x));

   return result;
}

