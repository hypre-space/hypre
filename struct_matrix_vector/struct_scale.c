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
#ifdef HYPRE_USE_PTHREADS
#include "box_pthreads.h"
#endif
/*--------------------------------------------------------------------------
 * hypre_StructScale
 *--------------------------------------------------------------------------*/

int
hypre_StructScale( double              alpha,
                   hypre_StructVector *y     )
{
   int ierr = 0;

   hypre_Box       *y_data_box;
                   
   int              yi;
   double          *yp;
                   
   hypre_BoxArray  *boxes;
   hypre_Box       *box;
   hypre_Index      loop_size;
   hypre_IndexRef   start;
   hypre_Index      unit_stride;
                   
   int              i;
   int              loopi, loopj, loopk;

   hypre_SetIndex(unit_stride, 1, 1, 1);

   boxes = hypre_StructGridBoxes(hypre_StructVectorGrid(y));
   hypre_ForBoxI(i, boxes)
      {
         box   = hypre_BoxArrayBox(boxes, i);
         start = hypre_BoxIMin(box);

         y_data_box = hypre_BoxArrayBox(hypre_StructVectorDataSpace(y), i);
         yp = hypre_StructVectorBoxData(y, i);

         hypre_GetBoxSize(box, loop_size);
         hypre_BoxLoop1(loopi, loopj, loopk, loop_size,
                        y_data_box, start, unit_stride, yi,
                        {
                           yp[yi] *= alpha;
                        });
      }

   return ierr;
}
