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
 * Structured scale routine
 *
 *****************************************************************************/

#include "headers.h"

/*--------------------------------------------------------------------------
 * zzz_StructScale
 *--------------------------------------------------------------------------*/

int
zzz_StructScale( double            alpha,
                 zzz_StructVector *y     )
{
   int ierr;

   zzz_Box              *y_data_box;

   int                   yi;
   double               *yp;

   zzz_BoxArray         *boxes;
   zzz_Box              *box;
   zzz_Index            *index;
   zzz_Index            *start;
   zzz_Index            *unit_stride;

   int                   i;

   index = zzz_NewIndex();

   unit_stride = zzz_NewIndex();
   zzz_SetIndex(unit_stride, 1, 1, 1);

   boxes = zzz_StructGridBoxes(zzz_StructVectorGrid(y));
   zzz_ForBoxI(i, boxes)
   {
      box   = zzz_BoxArrayBox(boxes, i);
      start = zzz_BoxIMin(box);

      y_data_box = zzz_BoxArrayBox(zzz_StructVectorDataSpace(y), i);
      yp = zzz_StructVectorBoxData(y, i);

      zzz_BoxLoop1(box, index,
                   y_data_box, start, unit_stride, yi,
                   {
                      yp[yi] *= alpha;
                   });
   }

   zzz_FreeIndex(index);
   zzz_FreeIndex(unit_stride);

   return ierr;
}
