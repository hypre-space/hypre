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
 * Structured axpy routine
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * zzz_StructAxpy
 *--------------------------------------------------------------------------*/

int
zzz_StructAxpy( double            alpha,
                zzz_StructVector *x,
                zzz_StructVector *y     )
{
   int ierr;

   zzz_Box              *x_data_box;
   zzz_Box              *y_data_box;

   int                   xi;
   int                   yi;

   double               *xp;
   double               *yp;

   zzz_BoxArray         *boxes;
   zzz_Box              *box;
   zzz_Index            *loop_index;
   zzz_Index            *loop_size;
   zzz_Index            *start;
   zzz_Index            *unit_stride;

   int                   i;

   loop_index = zzz_NewIndex();
   loop_size  = zzz_NewIndex();

   unit_stride = zzz_NewIndex();
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
      zzz_BoxLoop2(loop_index, loop_size,
                   x_data_box, start, unit_stride, xi,
                   y_data_box, start, unit_stride, yi,
                   {
                      yp[yi] += alpha * xp[xi];
                   });
   }

   zzz_FreeIndex(loop_index);
   zzz_FreeIndex(loop_size);
   zzz_FreeIndex(unit_stride);

   return ierr;
}
